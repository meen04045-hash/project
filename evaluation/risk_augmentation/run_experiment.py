"""Compare the existing Risk training split with and without reviewed rows.

Run from the repository root with:
    python -B evaluation/risk_augmentation/run_experiment.py

This experiment writes reports only. It never serializes or replaces a model.
The training recipe below matches the holdout path in ai_model.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pythainlp
import sklearn
from pythainlp.tokenize import word_tokenize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "evaluation" / "risk_augmentation"
RISK_CSV = ROOT / "dataset" / "risk.csv"
REVIEWED_CSV = ROOT / "dataset" / "risk_augmented_reviewed.csv"
EXISTING_REPORT = ROOT / "evaluation" / "risk_report.txt"
RANDOM_STATE = 42
TEST_SIZE = 0.2
FOCUS_LABELS = (
    "depression",
    "risk_passive",
    "risk_suicidal_ideation",
    "risk_suicide_plan",
    "risk_immediate_danger",
)


# These three functions and their parameters mirror ai_model.py exactly.
def clean_text(text: object) -> str:
    normalized = str(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize(text: object) -> str:
    return " ".join(word_tokenize(clean_text(text), engine="newmm"))


def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 3), min_df=1)


def create_classifier(minimum_class_samples: int = 5) -> CalibratedClassifierCV:
    calibration_folds = max(2, min(5, minimum_class_samples))
    base_model = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    return CalibratedClassifierCV(base_model, cv=calibration_folds)


def load_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    if not {"text", "label"}.issubset(dataset.columns):
        raise ValueError(f"{path} must contain text and label columns")
    return dataset.dropna(subset=["text", "label"]).copy()


def normalized_for_overlap(value: object) -> str:
    """Conservative leakage check; this is not training preprocessing."""
    value = unicodedata.normalize("NFKC", str(value)).lower()
    return "".join(
        character
        for character in value
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(name: str, train_text: pd.Series, train_labels: pd.Series,
             test_text: pd.Series, test_labels: pd.Series,
             labels: list[str]) -> dict:
    vectorizer = create_vectorizer()
    train_features = vectorizer.fit_transform(train_text)
    model = create_classifier(int(train_labels.value_counts().min()))
    model.fit(train_features, train_labels)

    test_features = vectorizer.transform(test_text)
    predictions = model.predict(test_features)
    confidence = model.predict_proba(test_features).max(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, predictions, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        test_labels, predictions, labels=labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(test_labels, predictions, labels=labels)
    wrong = np.asarray(test_labels) != predictions
    return {
        "name": name,
        "train_count": len(train_text),
        "test_count": len(test_text),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "matrix": matrix,
        "predictions": predictions,
        "confidence": confidence,
        "wrong": wrong,
        "error_count": int(wrong.sum()),
        "report": classification_report(
            test_labels, predictions, labels=labels, digits=6, zero_division=0
        ),
        "calibration_folds": model.cv,
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


def render_matrix(matrix: np.ndarray, labels: list[str]) -> str:
    lines = ["Rows = true label; columns = predicted label", "Label order: " + ", ".join(labels)]
    lines.extend(
        f"{label}: " + ", ".join(str(int(value)) for value in matrix[index])
        for index, label in enumerate(labels)
    )
    return "\n".join(lines)


def render_report(result: dict, labels: list[str], test_hash: str,
                  source_hashes: dict[str, str]) -> str:
    return "\n".join([
        f"Risk Model {result['name']} Holdout Report",
        "=" * 50,
        f"Original dataset SHA-256: {source_hashes['risk']}",
        f"Reviewed dataset SHA-256: {source_hashes['reviewed']}",
        f"Shared test set SHA-256: {test_hash}",
        f"Training rows: {result['train_count']}",
        f"Shared test rows: {result['test_count']}",
        f"TF-IDF vocabulary size: {result['vocabulary_size']}",
        f"Calibration folds: {result['calibration_folds']}",
        "Algorithm: LinearSVC(C=1.0, class_weight='balanced', random_state=42) + CalibratedClassifierCV",
        "TF-IDF: PyThaiNLP newmm tokenizer, token_pattern=None, ngram_range=(1, 3), min_df=1",
        "Split: test_size=0.2, random_state=42, shuffle=True, stratify=label",
        f"Accuracy: {result['accuracy']:.6f}",
        f"Macro Precision: {result['macro_precision']:.6f}",
        f"Macro Recall: {result['macro_recall']:.6f}",
        f"Macro F1: {result['macro_f1']:.6f}",
        f"Error count: {result['error_count']}",
        "",
        "Per-class Precision / Recall / F1:",
        result["report"].rstrip(),
        "",
        "Confusion Matrix:",
        render_matrix(result["matrix"], labels),
        "",
    ])


def render_comparison(before: dict, after: dict, labels: list[str],
                      test_hash: str, source_hashes: dict[str, str],
                      legacy_accuracy: float | None) -> str:
    metric_names = (
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
    )
    lines = [
        "Risk Model Augmentation: Before vs After",
        "=" * 50,
        f"Original dataset SHA-256: {source_hashes['risk']}",
        f"Reviewed dataset SHA-256: {source_hashes['reviewed']}",
        f"Shared test set SHA-256: {test_hash}",
        f"Original rows: {before['train_count'] + before['test_count']}",
        f"Baseline training rows: {before['train_count']}",
        f"Augmented training rows: {after['train_count']}",
        f"Reviewed rows added to training only: {after['train_count'] - before['train_count']}",
        f"Shared test rows: {before['test_count']}",
        "Reviewed/test normalized-text overlap: 0 (checked before training)",
        f"Python libraries: pandas={pd.__version__}, scikit-learn={sklearn.__version__}, PyThaiNLP={pythainlp.__version__}",
        "Algorithm, TF-IDF settings, calibration, and random_state match ai_model.py.",
        "No model files were saved.",
        "",
        "Overall metrics (delta = After - Before):",
        "Metric                 Before     After      Delta",
    ]
    lines.extend(
        f"{label:<22} {before[key]:>8.6f}  {after[key]:>8.6f}  {after[key] - before[key]:>+9.6f}"
        for label, key in metric_names
    )
    lines.extend([
        f"Errors                 {before['error_count']:>8}  {after['error_count']:>8}  {after['error_count'] - before['error_count']:>+9}",
        "",
        "Per-class metrics (precision / recall / F1; deltas in parentheses):",
        "Class                         Support  Before P/R/F1                    After P/R/F1                     Delta P/R/F1",
    ])
    for label in labels:
        b = before["per_class"][label]
        a = after["per_class"][label]
        left = "/".join(f"{b[key]:.3f}" for key in ("precision", "recall", "f1"))
        right = "/".join(f"{a[key]:.3f}" for key in ("precision", "recall", "f1"))
        delta = "/".join(f"{a[key] - b[key]:+.3f}" for key in ("precision", "recall", "f1"))
        lines.append(f"{label:<29} {b['support']:>7}  {left:<31} {right:<31} {delta}")

    before_correct = ~before["wrong"]
    after_correct = ~after["wrong"]
    lines.extend([
        "",
        "Paired outcomes on the same test rows:",
        f"Correct in both: {int((before_correct & after_correct).sum())}",
        f"Fixed by augmented model: {int((~before_correct & after_correct).sum())}",
        f"Regressed after augmentation: {int((before_correct & ~after_correct).sum())}",
        f"Wrong in both: {int((~before_correct & ~after_correct).sum())}",
        "",
        "Focus-class error destinations (predicted class=count; only nonzero values shown):",
    ])
    for label in FOCUS_LABELS:
        index = labels.index(label)
        for name, result in (("Before", before), ("After", after)):
            destinations = [
                f"{other}={int(result['matrix'][index, other_index])}"
                for other_index, other in enumerate(labels)
                if other_index != index and result["matrix"][index, other_index]
            ]
            lines.append(f"{label} {name}: " + (", ".join(destinations) if destinations else "none"))

    lines.extend(["", "Focused findings (same test rows):"])
    for label in FOCUS_LABELS:
        index = labels.index(label)
        b = before["per_class"][label]
        a = after["per_class"][label]
        lines.append(
            f"{label}: correct {int(before['matrix'][index, index])}/{b['support']} -> "
            f"{int(after['matrix'][index, index])}/{a['support']}; "
            f"recall {a['recall'] - b['recall']:+.3f}, F1 {a['f1'] - b['f1']:+.3f}"
        )
    ideation = labels.index("risk_suicidal_ideation")
    passive = labels.index("risk_passive")
    lines.append(
        "Ideation predicted as passive: "
        f"{int(before['matrix'][ideation, passive])} -> "
        f"{int(after['matrix'][ideation, passive])}; this safety-relevant error increased."
    )

    lines.extend([
        "",
        "Before confusion matrix:",
        render_matrix(before["matrix"], labels),
        "",
        "After confusion matrix:",
        render_matrix(after["matrix"], labels),
        "",
    ])
    if legacy_accuracy is not None:
        lines.append(
            f"Existing evaluation/risk_report.txt accuracy: {legacy_accuracy:.6f}; "
            f"new baseline difference: {before['accuracy'] - legacy_accuracy:+.6f}"
        )
    lines.extend([
        "Interpretation: this is one fixed holdout comparison, not a clinical validation.",
        "No production model replacement is decided by this experiment.",
        "",
    ])
    return "\n".join(lines)


def write_errors(path: Path, result: dict, test_text: pd.Series,
                 test_labels: pd.Series) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "Test Row Index", "Original Text", "True Label", "Predicted Label", "Confidence Score"
        ])
        for index, text, true_label, prediction, confidence, wrong in zip(
            test_text.index,
            test_text.to_numpy(),
            test_labels.to_numpy(),
            result["predictions"],
            result["confidence"],
            result["wrong"],
            strict=True,
        ):
            if wrong:
                writer.writerow([int(index), text, true_label, prediction, f"{confidence:.6f}"])


def main() -> None:
    source_hashes = {"risk": file_sha256(RISK_CSV), "reviewed": file_sha256(REVIEWED_CSV)}
    original = load_dataset(RISK_CSV)
    reviewed = load_dataset(REVIEWED_CSV)
    if len(reviewed) != 73 or not reviewed["reviewed"].astype(str).str.lower().eq("true").all():
        raise ValueError("Expected exactly 73 reviewed=true rows")
    if not reviewed["label"].isin(original["label"].unique()).all():
        raise ValueError("Reviewed data contains an unknown class")

    train_text, test_text, train_labels, test_labels = train_test_split(
        original["text"], original["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, shuffle=True, stratify=original["label"]
    )
    normalized_test = {normalized_for_overlap(value) for value in test_text}
    overlap = {normalized_for_overlap(value) for value in reviewed["text"]} & normalized_test
    if overlap:
        raise ValueError(f"Reviewed/test text overlap: {len(overlap)}")
    test_identity = [
        [int(index), str(label), str(text)]
        for index, label, text in zip(test_text.index, test_labels, test_text, strict=True)
    ]
    test_hash = hashlib.sha256(
        json.dumps(test_identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    labels = sorted(original["label"].unique())

    print(f"Baseline: {len(train_text)} training rows, {len(test_text)} shared test rows")
    before = evaluate("Baseline", train_text, train_labels, test_text, test_labels, labels)
    print(f"Baseline accuracy={before['accuracy']:.6f}, errors={before['error_count']}")

    augmented_text = pd.concat([train_text.reset_index(drop=True), reviewed["text"].reset_index(drop=True)], ignore_index=True)
    augmented_labels = pd.concat([train_labels.reset_index(drop=True), reviewed["label"].reset_index(drop=True)], ignore_index=True)
    print(f"Augmented: {len(augmented_text)} training rows, {len(test_text)} same test rows")
    after = evaluate("Augmented", augmented_text, augmented_labels, test_text, test_labels, labels)
    print(f"Augmented accuracy={after['accuracy']:.6f}, errors={after['error_count']}")

    legacy_match = re.search(r"^Accuracy:\s*([0-9.]+)", EXISTING_REPORT.read_text(encoding="utf-8"), re.MULTILINE)
    legacy_accuracy = float(legacy_match.group(1)) if legacy_match else None
    if legacy_accuracy is not None:
        print(f"Existing report accuracy={legacy_accuracy:.6f}; baseline delta={before['accuracy'] - legacy_accuracy:+.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "baseline_report.txt").write_text(
        render_report(before, labels, test_hash, source_hashes), encoding="utf-8"
    )
    (OUTPUT_DIR / "augmented_report.txt").write_text(
        render_report(after, labels, test_hash, source_hashes), encoding="utf-8"
    )
    (OUTPUT_DIR / "comparison.txt").write_text(
        render_comparison(before, after, labels, test_hash, source_hashes, legacy_accuracy), encoding="utf-8"
    )
    write_errors(OUTPUT_DIR / "baseline_errors.csv", before, test_text, test_labels)
    write_errors(OUTPUT_DIR / "augmented_errors.csv", after, test_text, test_labels)
    if file_sha256(RISK_CSV) != source_hashes["risk"] or file_sha256(REVIEWED_CSV) != source_hashes["reviewed"]:
        raise RuntimeError("An input dataset changed during the experiment")
    print("Reports and error CSVs saved under evaluation/risk_augmentation/")


if __name__ == "__main__":
    main()
