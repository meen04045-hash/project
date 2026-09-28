import re
import numpy as np
import pandas as pd

from pythainlp.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

RANDOM_STATE = 42

# ---------------------------------------------------------
# Thresholds to test
# ---------------------------------------------------------
THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------
DATASETS = {
    "risk": "dataset/risk.csv",
    "emotion": "dataset/emotion.csv",
    "problem": "dataset/problem.csv",
    "support_need": "dataset/support_need.csv",
    "intent": "dataset/intent.csv",
    "conversation_style": "dataset/conversation_style.csv",
}

OUTPUT_DIR = "evaluation/threshold"


# ---------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tokenize(text):
    """
    Production-style tokenization.

    Returns a whitespace-separated string so that the
    representation is consistent with ai_model.py.
    """
    return " ".join(
        word_tokenize(
            clean_text(text),
            engine="newmm"
        )
    )


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------
def create_vectorizer():
    return TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        ngram_range=(1, 3),
        min_df=1
    )


# ---------------------------------------------------------
# Calibrated SVM
# ---------------------------------------------------------
def create_classifier(min_class_samples):
    folds = max(
        2,
        min(5, int(min_class_samples))
    )

    base = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    return CalibratedClassifierCV(
        base,
        cv=folds
    )


# ---------------------------------------------------------
# Calculate threshold metrics
# ---------------------------------------------------------
def calculate_threshold_metrics(
    y_true,
    predictions,
    confidence,
    threshold
):
    """
    Evaluate only samples whose confidence >= threshold.

    coverage:
        proportion of samples for which the model gives
        an accepted prediction.

    precision / recall / f1:
        calculated among accepted predictions.

    rejected samples are not treated as a new class.
    """

    accepted = confidence >= threshold

    total = len(y_true)
    accepted_count = int(accepted.sum())
    rejected_count = total - accepted_count

    coverage = (
        accepted_count / total
        if total > 0
        else 0.0
    )

    if accepted_count == 0:
        return {
            "threshold": threshold,
            "coverage": coverage,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "accuracy": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
            "f1_weighted": np.nan,
            "fp": np.nan,
            "fn": np.nan,
        }

    y_accepted = np.asarray(y_true)[accepted]
    pred_accepted = np.asarray(predictions)[accepted]

    accuracy = accuracy_score(
        y_accepted,
        pred_accepted
    )

    precision = precision_score(
        y_accepted,
        pred_accepted,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_accepted,
        pred_accepted,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_accepted,
        pred_accepted,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_accepted,
        pred_accepted,
        average="weighted",
        zero_division=0
    )

    # Number of incorrect accepted predictions.
    errors = int(
        (y_accepted != pred_accepted).sum()
    )

    # For a multiclass problem, these are total FP/FN
    # across all classes.
    labels = np.unique(
        np.concatenate(
            [y_accepted, pred_accepted]
        )
    )

    cm = confusion_matrix(
        y_accepted,
        pred_accepted,
        labels=labels
    )

    fp = int(
        (cm.sum(axis=0) - np.diag(cm)).sum()
    )

    fn = int(
        (cm.sum(axis=1) - np.diag(cm)).sum()
    )

    return {
        "threshold": threshold,
        "coverage": coverage,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "fp": fp,
        "fn": fn,
        "errors": errors,
    }


# ---------------------------------------------------------
# Run threshold sweep
# ---------------------------------------------------------
def run_threshold_sweep(
    name,
    csv_path
):
    print("\n" + "=" * 70)
    print(f"THRESHOLD EXPERIMENT: {name.upper()}")
    print("=" * 70)

    df = pd.read_csv(csv_path)

    df = df.dropna(
        subset=["text", "label"]
    ).copy()

    texts = df["text"]
    labels = df["label"]

    print(f"Total samples: {len(df)}")
    print(f"Classes: {labels.nunique()}")

    # -----------------------------------------------------
    # Step 1: Hold out final Test Set = 20%
    # -----------------------------------------------------
    x_temp, x_test, y_temp, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=labels
    )

    # -----------------------------------------------------
    # Step 2:
    # Remaining 80%
    # → Train 64%
    # → Validation 16%
    # -----------------------------------------------------
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_temp
    )

    print(f"Train: {len(x_train)}")
    print(f"Validation: {len(x_val)}")
    print(f"Test: {len(x_test)}")

    # -----------------------------------------------------
    # Step 3: Fit TF-IDF ONLY on training data
    # -----------------------------------------------------
    vectorizer = create_vectorizer()

    x_train_v = vectorizer.fit_transform(
        x_train
    )

    x_val_v = vectorizer.transform(
        x_val
    )

    x_test_v = vectorizer.transform(
        x_test
    )

    # -----------------------------------------------------
    # Step 4: Train calibrated SVM
    # -----------------------------------------------------
    min_class_samples = int(
        y_train.value_counts().min()
    )

    classifier = create_classifier(
        min_class_samples
    )

    classifier.fit(
        x_train_v,
        y_train
    )

    # -----------------------------------------------------
    # Step 5: Validation probabilities
    # -----------------------------------------------------
    val_probs = classifier.predict_proba(
        x_val_v
    )

    val_predictions = classifier.classes_[
        np.argmax(
            val_probs,
            axis=1
        )
    ]

    val_confidence = val_probs.max(
        axis=1
    )

    # -----------------------------------------------------
    # Step 6: Sweep thresholds on VALIDATION ONLY
    # -----------------------------------------------------
    validation_results = []

    for threshold in THRESHOLDS:

        result = calculate_threshold_metrics(
            y_val.to_numpy(),
            val_predictions,
            val_confidence,
            threshold
        )

        result["task"] = name
        result["dataset"] = "validation"

        validation_results.append(result)

    validation_df = pd.DataFrame(
        validation_results
    )

    # -----------------------------------------------------
    # Select threshold using validation Macro F1
    # -----------------------------------------------------
    valid_selection = validation_df.dropna(
        subset=["f1_macro"]
    )

    if valid_selection.empty:
        print(
            "WARNING: No valid threshold result."
        )
        return None

    best_row = valid_selection.loc[
        valid_selection["f1_macro"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print("\nValidation Threshold Results:")
    print(
        validation_df[
            [
                "threshold",
                "coverage",
                "accepted",
                "rejected",
                "precision_macro",
                "recall_macro",
                "f1_macro"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSelected threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_row['f1_macro']:.4f}"
    )

    # -----------------------------------------------------
    # Step 7: Final evaluation on untouched TEST set
    # -----------------------------------------------------
    test_probs = classifier.predict_proba(
        x_test_v
    )

    test_predictions = classifier.classes_[
        np.argmax(
            test_probs,
            axis=1
        )
    ]

    test_confidence = test_probs.max(
        axis=1
    )

    test_result = calculate_threshold_metrics(
        y_test.to_numpy(),
        test_predictions,
        test_confidence,
        best_threshold
    )

    test_result["task"] = name
    test_result["dataset"] = "test"
    test_result["selected_threshold"] = best_threshold

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    import os

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    validation_path = (
        f"{OUTPUT_DIR}/"
        f"{name}_validation_thresholds.csv"
    )

    test_path = (
        f"{OUTPUT_DIR}/"
        f"{name}_final_test.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        [test_result]
    ).to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFinal Test Result:")
    print(
        f"Threshold : "
        f"{best_threshold:.2f}"
    )
    print(
        f"Coverage  : "
        f"{test_result['coverage']:.2%}"
    )
    print(
        f"Accepted  : "
        f"{test_result['accepted']}"
    )
    print(
        f"Rejected  : "
        f"{test_result['rejected']}"
    )
    print(
        f"Accuracy  : "
        f"{test_result['accuracy']:.4f}"
    )
    print(
        f"Precision : "
        f"{test_result['precision_macro']:.4f}"
    )
    print(
        f"Recall    : "
        f"{test_result['recall_macro']:.4f}"
    )
    print(
        f"Macro F1  : "
        f"{test_result['f1_macro']:.4f}"
    )
    print(
        f"Weighted F1: "
        f"{test_result['f1_weighted']:.4f}"
    )
    print(
        f"FP        : "
        f"{test_result['fp']}"
    )
    print(
        f"FN        : "
        f"{test_result['fn']}"
    )

    print(
        f"\nSaved:"
        f"\n  {validation_path}"
        f"\n  {test_path}"
    )

    return {
        "task": name,
        "threshold": best_threshold,
        "validation_f1": best_row["f1_macro"],
        "test_coverage": test_result["coverage"],
        "test_accuracy": test_result["accuracy"],
        "test_precision": test_result["precision_macro"],
        "test_recall": test_result["recall_macro"],
        "test_f1": test_result["f1_macro"],
        "test_f1_weighted": test_result["f1_weighted"],
        "test_fp": test_result["fp"],
        "test_fn": test_result["fn"],
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    import os

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    all_results = []

    for task, csv_path in DATASETS.items():

        if not os.path.exists(csv_path):
            print(
                f"\nWARNING: Dataset not found: "
                f"{csv_path}"
            )
            continue

        result = run_threshold_sweep(
            task,
            csv_path
        )

        if result is not None:
            all_results.append(result)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    if all_results:

        summary_df = pd.DataFrame(
            all_results
        )

        summary_path = (
            f"{OUTPUT_DIR}/"
            "threshold_summary.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 70)
        print("THRESHOLD EXPERIMENT SUMMARY")
        print("=" * 70)

        print(
            summary_df.to_string(
                index=False
            )
        )

        print(
            f"\nSummary saved to:"
            f"\n{summary_path}"
        )