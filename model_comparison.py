"""
Model Comparison Experiment
===========================

Compare 3 Machine Learning algorithms:

1. SVM (LinearSVC + CalibratedClassifierCV)
2. Logistic Regression
3. Random Forest

Tasks:
- Risk
- Emotion
- Problem
- Support Need
- Intent
- Conversation Style

The same:
- Dataset
- Text preprocessing
- Tokenization
- TF-IDF
- Train/Test split
- Random State
- Cross Validation

are used for all algorithms.

Purpose:
Experimental comparison for university project / research report.
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from pythainlp.tokenize import word_tokenize


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5

OUTPUT_DIR = "evaluation/model_comparison"

MODEL_DIR = "models/comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "risk": "dataset/risk.csv",
    "emotion": "dataset/emotion.csv",
    "problem": "dataset/problem.csv",
    "support_need": "dataset/support_need.csv",
    "intent": "dataset/intent.csv",
    "conversation_style": "dataset/conversation_style.csv",
}


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    """
    Basic text cleaning.

    Keep this preprocessing consistent with the existing
    project pipeline.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove unnecessary whitespace
    text = " ".join(text.split())

    return text.strip()


def tokenize(text):
    """
    Thai tokenization using PyThaiNLP newmm.
    """

    text = clean_text(text)

    return word_tokenize(
        text,
        engine="newmm"
    )


# ============================================================
# TF-IDF
# ============================================================

def create_vectorizer():

    return TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True
    )


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {

        "SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            method="sigmoid",
            cv=5
        ),

        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print("\nDataset:", path)
    print("Columns:", list(df.columns))
    print("Dataset Size:", len(df))

    return df


# ============================================================
# FIND TEXT / LABEL COLUMNS
# ============================================================

def detect_columns(df):

    text_candidates = [
        "text",
        "message",
        "input",
        "sentence",
        "question",
        "utterance"
    ]

    label_candidates = [
        "label",
        "class",
        "category",
        "target",
        "intent"
    ]

    text_column = None
    label_column = None

    for column in text_candidates:

        if column in df.columns:

            text_column = column
            break

    for column in label_candidates:

        if column in df.columns:

            label_column = column
            break

    # Fallback:
    # first column = text
    # second column = label

    if text_column is None:

        text_column = df.columns[0]

    if label_column is None:

        if len(df.columns) >= 2:

            label_column = df.columns[1]

        else:

            raise ValueError(
                "Cannot detect label column."
            )

    return text_column, label_column


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    task_name,
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 70)

    print(
        f"Training {model_name} - {task_name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Create Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline([

        (
            "tfidf",
            create_vectorizer()
        ),

        (
            "classifier",
            model
        )
    ])

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_train = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter()
        - start_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    start_predict = time.perf_counter()

    y_pred = pipeline.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - start_predict
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    labels = sorted(
        list(
            set(y_test)
            |
            set(y_pred)
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    # --------------------------------------------------------
    # Create Task Output Directory
    # --------------------------------------------------------

    task_dir = os.path.join(
        OUTPUT_DIR,
        task_name
    )

    os.makedirs(
        task_dir,
        exist_ok=True
    )

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    # ========================================================
    # SAVE PER-CLASS METRICS
    # ========================================================

    excluded_rows = {
        "accuracy",
        "macro avg",
        "weighted avg",
        "micro avg"
    }

    class_metrics = [
        {
            "Task": task_name,
            "Model": model_name,
            "Class": class_name,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1-score"],
            "Support": values["support"]
        }
        for class_name, values in report_dict.items()
        if (
            class_name not in excluded_rows
            and isinstance(values, dict)
            and "f1-score" in values
        )
    ]

    class_metrics_path = os.path.join(
        task_dir,
        f"{safe_model_name}_class_metrics.csv"
    )

    pd.DataFrame(
        class_metrics,
        columns=[
            "Task",
            "Model",
            "Class",
            "Precision",
            "Recall",
            "F1",
            "Support"
        ]
    ).to_csv(
        class_metrics_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    y_test_values = y_test.to_numpy()

    y_pred_values = np.asarray(
        y_pred
    )

    error_mask = (
        y_test_values
        !=
        y_pred_values
    )

    error_positions = np.flatnonzero(
        error_mask
    )

    errors_df = pd.DataFrame({

        "Task": task_name,

        "Model": model_name,

        "Text": X_test.iloc[
            error_positions
        ].to_numpy(),

        "True_Label": y_test_values[
            error_mask
        ],

        "Predicted_Label": y_pred_values[
            error_mask
        ]

    }, columns=[
        "Task",
        "Model",
        "Text",
        "True_Label",
        "Predicted_Label"
    ])

    errors_path = os.path.join(
        task_dir,
        f"{safe_model_name}_errors.csv"
    )

    errors_df.to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SAVE CLASSIFICATION REPORT
    # ========================================================

    report_path = os.path.join(
        task_dir,
        f"{safe_model_name}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Task: {task_name}\n"
        )

        f.write(
            f"Model: {model_name}\n\n"
        )

        f.write(report)

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    cm_path = os.path.join(
        task_dir,
        f"{safe_model_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"{task_name}_{safe_model_name}.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "Task": task_name,

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision_Macro": precision_macro,

        "Recall_Macro": recall_macro,

        "F1_Macro": f1_macro,

        "Precision_Weighted": precision_weighted,

        "Recall_Weighted": recall_weighted,

        "F1_Weighted": f1_weighted,

        "Training_Time_sec": training_time,

        "Prediction_Time_sec": prediction_time,

        "Train_Size": len(X_train),

        "Test_Size": len(X_test),

        "Dataset_Size": (
            len(X_train)
            +
            len(X_test)
        ),

        "Number_of_Classes": len(
            set(y_train)
            |
            set(y_test)
        ),

        "Error_Count": int(
            error_mask.sum()
        )

    }

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print()

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Precision Macro: {precision_macro:.4f}"
    )

    print(
        f"Recall Macro   : {recall_macro:.4f}"
    )

    print(
        f"F1 Macro       : {f1_macro:.4f}"
    )

    print(
        f"F1 Weighted    : {f1_weighted:.4f}"
    )

    print(
        f"Training Time  : {training_time:.4f}s"
    )

    print(
        f"Prediction Time: {prediction_time:.4f}s"
    )

    print(
        f"Errors         : {error_mask.sum()}"
    )

    return result


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    task_name,
    model_name,
    model,
    X,
    y
):

    print(
        f"\nRunning 5-Fold CV: "
        f"{task_name} - {model_name}"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline([

        (
            "tfidf",
            create_vectorizer()
        ),

        (
            "classifier",
            model
        )
    ])

    # --------------------------------------------------------
    # Stratified K-Fold
    # --------------------------------------------------------

    cv = StratifiedKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------

    scoring = {

        "accuracy":
            "accuracy",

        "precision_macro":
            "precision_macro",

        "recall_macro":
            "recall_macro",

        "f1_macro":
            "f1_macro",

        "f1_weighted":
            "f1_weighted"
    }

    # --------------------------------------------------------
    # Run CV
    # --------------------------------------------------------

    start = time.perf_counter()

    scores = cross_validate(

        pipeline,

        X,

        y,

        cv=cv,

        scoring=scoring,

        return_train_score=False,

        n_jobs=1
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "Task":
            task_name,

        "Model":
            model_name,

        # Accuracy
        "CV_Accuracy_Mean":
            scores[
                "test_accuracy"
            ].mean(),

        "CV_Accuracy_Std":
            scores[
                "test_accuracy"
            ].std(),

        # Precision
        "CV_Precision_Macro_Mean":
            scores[
                "test_precision_macro"
            ].mean(),

        "CV_Precision_Macro_Std":
            scores[
                "test_precision_macro"
            ].std(),

        # Recall
        "CV_Recall_Macro_Mean":
            scores[
                "test_recall_macro"
            ].mean(),

        "CV_Recall_Macro_Std":
            scores[
                "test_recall_macro"
            ].std(),

        # F1 Macro
        "CV_F1_Macro_Mean":
            scores[
                "test_f1_macro"
            ].mean(),

        "CV_F1_Macro_Std":
            scores[
                "test_f1_macro"
            ].std(),

        # F1 Weighted
        "CV_F1_Weighted_Mean":
            scores[
                "test_f1_weighted"
            ].mean(),

        "CV_F1_Weighted_Std":
            scores[
                "test_f1_weighted"
            ].std(),

        # Keep old-compatible names
        "CV_Mean_Accuracy":
            scores[
                "test_accuracy"
            ].mean(),

        "CV_Std":
            scores[
                "test_accuracy"
            ].std(),

        # CV Time
        "CV_Time_sec":
            elapsed
    }

    # ========================================================
    # SAVE INDIVIDUAL FOLD RESULTS
    # ========================================================

    for fold in range(N_SPLITS):

        fold_number = fold + 1

        result[
            f"Accuracy_Fold_{fold_number}"
        ] = scores[
            "test_accuracy"
        ][fold]

        result[
            f"Precision_Macro_Fold_{fold_number}"
        ] = scores[
            "test_precision_macro"
        ][fold]

        result[
            f"Recall_Macro_Fold_{fold_number}"
        ] = scores[
            "test_recall_macro"
        ][fold]

        result[
            f"F1_Macro_Fold_{fold_number}"
        ] = scores[
            "test_f1_macro"
        ][fold]

        result[
            f"F1_Weighted_Fold_{fold_number}"
        ] = scores[
            "test_f1_weighted"
        ][fold]

    # --------------------------------------------------------
    # Print CV Results
    # --------------------------------------------------------

    print(
        f"Mean Accuracy: "
        f"{scores['test_accuracy'].mean():.4f}"
    )

    print(
        f"Mean Precision Macro: "
        f"{scores['test_precision_macro'].mean():.4f}"
    )

    print(
        f"Mean Recall Macro: "
        f"{scores['test_recall_macro'].mean():.4f}"
    )

    print(
        f"Mean F1 Macro: "
        f"{scores['test_f1_macro'].mean():.4f}"
    )

    print(
        f"Mean F1 Weighted: "
        f"{scores['test_f1_weighted'].mean():.4f}"
    )

    print(
        f"Accuracy Std: "
        f"{scores['test_accuracy'].std():.4f}"
    )

    return result


# ============================================================
# RUN ONE TASK
# ============================================================

def run_task(
    task_name,
    dataset_path
):

    print("\n\n")

    print("#" * 80)

    print(
        f"# TASK: {task_name.upper()}"
    )

    print("#" * 80)

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    df = load_dataset(
        dataset_path
    )

    # --------------------------------------------------------
    # Detect Columns
    # --------------------------------------------------------

    text_column, label_column = detect_columns(
        df
    )

    print(
        "Text Column :",
        text_column
    )

    print(
        "Label Column:",
        label_column
    )

    # --------------------------------------------------------
    # Remove Missing Values
    # --------------------------------------------------------

    df = df[
        [
            text_column,
            label_column
        ]
    ].dropna()

    # --------------------------------------------------------
    # Clean Text
    # --------------------------------------------------------

    df[text_column] = df[
        text_column
    ].apply(clean_text)

    # --------------------------------------------------------
    # Remove Empty Text
    # --------------------------------------------------------

    df = df[
        df[text_column] != ""
    ]

    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X = df[
        text_column
    ].astype(str)

    y = df[
        label_column
    ].astype(str)

    print(
        "Final Dataset:",
        len(df)
    )

    print(
        "Number of Classes:",
        y.nunique()
    )

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print(
        "Training Size:",
        len(X_train)
    )

    print(
        "Testing Size:",
        len(X_test)
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = create_models()

    holdout_results = []

    cv_results = []

    # --------------------------------------------------------
    # Train Each Model
    # --------------------------------------------------------

    for model_name, model in models.items():

        # ----------------------------------------------------
        # Holdout Evaluation
        # ----------------------------------------------------

        result = train_model(

            task_name,

            model_name,

            model,

            X_train,

            X_test,

            y_train,

            y_test
        )

        holdout_results.append(
            result
        )

        # ----------------------------------------------------
        # Cross Validation
        # ----------------------------------------------------

        cv_result = run_cross_validation(

            task_name,

            model_name,

            model,

            X_train,

            y_train
        )

        cv_results.append(
            cv_result
        )

    return (
        holdout_results,
        cv_results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    print("\n")

    print("=" * 80)

    print(
        "MACHINE LEARNING MODEL COMPARISON"
    )

    print("=" * 80)

    print()

    print(
        "Algorithms:"
    )

    print(
        "1. SVM"
    )

    print(
        "2. Logistic Regression"
    )

    print(
        "3. Random Forest"
    )

    print()

    print(
        "Tasks:"
    )

    for task in DATASETS:

        print(
            f"- {task}"
        )

    print()

    print(
        "Random State:",
        RANDOM_STATE
    )

    print(
        "Test Size:",
        TEST_SIZE
    )

    print(
        "Cross Validation:",
        f"{N_SPLITS}-Fold"
    )

    # --------------------------------------------------------
    # All Results
    # --------------------------------------------------------

    all_holdout_results = []

    all_cv_results = []

    # --------------------------------------------------------
    # Run All Tasks
    # --------------------------------------------------------

    for task_name, dataset_path in DATASETS.items():

        try:

            (
                holdout_results,
                cv_results
            ) = run_task(

                task_name,

                dataset_path
            )

            all_holdout_results.extend(
                holdout_results
            )

            all_cv_results.extend(
                cv_results
            )

        except Exception as e:

            print()

            print(
                f"ERROR in task: {task_name}"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

            print()

    # --------------------------------------------------------
    # Save Holdout Results
    # --------------------------------------------------------

    holdout_df = pd.DataFrame(
        all_holdout_results
    )

    holdout_path = os.path.join(

        OUTPUT_DIR,

        "holdout_results.csv"
    )

    holdout_df.to_csv(

        holdout_path,

        index=False,

        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save Cross Validation Results
    # --------------------------------------------------------

    cv_df = pd.DataFrame(
        all_cv_results
    )

    cv_path = os.path.join(

        OUTPUT_DIR,

        "cross_validation_results.csv"
    )

    cv_df.to_csv(

        cv_path,

        index=False,

        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Merge Results
    # --------------------------------------------------------

    combined_df = pd.merge(

        holdout_df,

        cv_df,

        on=[
            "Task",
            "Model"
        ],

        how="left"
    )

    combined_path = os.path.join(

        OUTPUT_DIR,

        "model_comparison_all_results.csv"
    )

    combined_df.to_csv(

        combined_path,

        index=False,

        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = os.path.join(

        OUTPUT_DIR,

        "comparison_summary.txt"
    )

    with open(

        summary_path,

        "w",

        encoding="utf-8"
    ) as f:

        f.write(
            "MACHINE LEARNING MODEL COMPARISON\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Algorithms:\n"
        )

        f.write(
            "- SVM\n"
        )

        f.write(
            "- Logistic Regression\n"
        )

        f.write(
            "- Random Forest\n\n"
        )

        f.write(
            "Tasks:\n"
        )

        for task in DATASETS:

            f.write(
                f"- {task}\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "Experimental Configuration:\n"
        )

        f.write(
            f"- Random State: {RANDOM_STATE}\n"
        )

        f.write(
            f"- Test Size: {TEST_SIZE}\n"
        )

        f.write(
            f"- Cross Validation: {N_SPLITS}-Fold Stratified\n"
        )

        f.write(
            "- TF-IDF ngram_range: (1, 3)\n"
        )

        f.write(
            "- TF-IDF min_df: 1\n"
        )

        f.write(
            "- TF-IDF sublinear_tf: True\n"
        )

        f.write(
            "- Thai Tokenization: PyThaiNLP newmm\n\n"
        )

        f.write(
            "Results\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            combined_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Print Final Results
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 80)

    print(
        "FINAL MODEL COMPARISON"
    )

    print("=" * 80)

    print()

    if len(combined_df) > 0:

        display_columns = [

            "Task",

            "Model",

            "Accuracy",

            "Precision_Macro",

            "Recall_Macro",

            "F1_Macro",

            "F1_Weighted",

            "CV_Accuracy_Mean",

            "CV_Precision_Macro_Mean",

            "CV_Recall_Macro_Mean",

            "CV_F1_Macro_Mean",

            "CV_F1_Weighted_Mean",

            "CV_Accuracy_Std"
        ]

        # Only display columns that actually exist
        display_columns = [
            column
            for column in display_columns
            if column in combined_df.columns
        ]

        print(
            combined_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()

    print(
        "Total Experiments:",
        len(combined_df)
    )

    print(
        "Expected Experiments:",
        len(DATASETS) * 3
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Models saved to:"
    )

    print(
        MODEL_DIR
    )

    print()

    print(
        "Experiment completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()