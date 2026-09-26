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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

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

    # Create Pipeline

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
    # Save Classification Report
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

    # --------------------------------------------------------
    # Save Confusion Matrix
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"{task_name}_{safe_model_name}.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

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

        "Test_Size": len(X_test)
    }

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

    cv = StratifiedKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    start = time.perf_counter()

    scores = cross_val_score(

        pipeline,

        X,

        y,

        cv=cv,

        scoring="accuracy",

        n_jobs=1
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    result = {

        "Task": task_name,

        "Model": model_name,

        "CV_Mean_Accuracy": scores.mean(),

        "CV_Std": scores.std(),

        "CV_Fold_1": scores[0],

        "CV_Fold_2": scores[1],

        "CV_Fold_3": scores[2],

        "CV_Fold_4": scores[3],

        "CV_Fold_5": scores[4],

        "CV_Time_sec": elapsed
    }

    print(
        f"Mean Accuracy: "
        f"{scores.mean():.4f}"
    )

    print(
        f"Std: "
        f"{scores.std():.4f}"
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

    # Load Dataset

    df = load_dataset(
        dataset_path
    )

    # Detect columns

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

    # Remove missing values

    df = df[
        [
            text_column,
            label_column
        ]
    ].dropna()

    # Clean

    df[text_column] = df[
        text_column
    ].apply(clean_text)

    # Remove empty text

    df = df[
        df[text_column] != ""
    ]

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
    # Train each model
    # --------------------------------------------------------

    for model_name, model in models.items():

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

        # Cross Validation

        cv_result = run_cross_validation(

            task_name,

            model_name,

            model,

            X,

            y
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
    # Run all tasks
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

            print(e)

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
    # Save CV Results
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

            "F1_Macro",

            "F1_Weighted",

            "CV_Mean_Accuracy",

            "CV_Std"
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