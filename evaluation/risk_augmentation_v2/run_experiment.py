"""Compare the original Risk split, reviewed v1, and final ideation v2.

Run from the repository root with:
    python -B -u evaluation/risk_augmentation_v2/run_experiment.py

The training recipe is imported from the prior experiment. No model is saved.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pythainlp
import sklearn
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.risk_augmentation import run_experiment as prior  # noqa: E402


OUTPUT_DIR = ROOT / "evaluation" / "risk_augmentation_v2"
RISK_CSV = ROOT / "dataset" / "risk.csv"
V1_CSV = ROOT / "dataset" / "risk_augmented_reviewed.csv"
V2_CSV = ROOT / "dataset" / "risk_suicidal_ideation_final_v2.csv"
PRIOR_DIR = ROOT / "evaluation" / "risk_augmentation"
CONFIGS = ("Baseline", "Augmented v1", "Augmented v2")
FOCUS_LABELS = (
    "risk_passive",
    "risk_suicidal_ideation",
    "risk_suicide_plan",
    "risk_immediate_danger",
    "depression",
)
METRICS = (
    ("Accuracy", "accuracy"),
    ("Macro Precision", "macro_precision"),
    ("Macro Recall", "macro_recall"),
    ("Macro F1", "macro_f1"),
)


def value_from_report(report: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(\S+)$", report, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {name} in previous experiment report")
    return match.group(1)


def verify_previous_result(result: dict, report_path: Path, errors_path: Path,
                           test_text: pd.Series) -> None:
    """Check that A/B reproduce the prior experiment on the same test rows."""
    report = report_path.read_text(encoding="utf-8")
    for title, key in METRICS:
        if abs(result[key] - float(value_from_report(report, title))) > 0.0000005:
            raise RuntimeError(f"{result['name']} {title} differs from previous run")
    if result["error_count"] != int(value_from_report(report, "Error count")):
        raise RuntimeError(f"{result['name']} error count differs from previous run")

    old_errors = pd.read_csv(errors_path, encoding="utf-8-sig")
    old_pairs = list(zip(
        old_errors["Test Row Index"].astype(int),
        old_errors["Predicted Label"].astype(str),
        strict=True,
    ))
    new_pairs = [
        (int(index), str(prediction))
        for index, prediction, wrong in zip(
            test_text.index, result["predictions"], result["wrong"], strict=True
        )
        if wrong
    ]
    if old_pairs != new_pairs:
        raise RuntimeError(f"{result['name']} wrong-row predictions differ from previous run")


def render_report(result: dict, labels: list[str], test_hash: str,
                  hashes: dict[str, str]) -> str:
    return "\n".join([
        f"Risk Model {result['name']} Holdout Report",
        "=" * 54,
        f"risk.csv SHA-256: {hashes['risk']}",
        f"risk_augmented_reviewed.csv SHA-256: {hashes['v1']}",
        f"risk_suicidal_ideation_final_v2.csv SHA-256: {hashes['v2']}",
        f"Shared test set SHA-256: {test_hash}",
        f"Training rows: {result['train_count']}",
        f"Shared test rows: {result['test_count']}",
        f"TF-IDF vocabulary size: {result['vocabulary_size']}",
        f"Calibration folds: {result['calibration_folds']}",
        "Algorithm: LinearSVC(C=1.0, class_weight='balanced', random_state=42) + CalibratedClassifierCV",
        "TF-IDF: PyThaiNLP newmm tokenizer, token_pattern=None, ngram_range=(1, 3), min_df=1",
        "Split: test_size=0.2, random_state=42, shuffle=True, stratify=label",
        *(f"{title}: {result[key]:.6f}" for title, key in METRICS),
        f"Error count: {result['error_count']}",
        "",
        "Per-class Precision / Recall / F1:",
        result["report"].rstrip(),
        "",
        "Confusion Matrix:",
        prior.render_matrix(result["matrix"], labels),
        "",
    ])


def render_comparison(results: list[dict], labels: list[str], test_hash: str,
                      hashes: dict[str, str], test_text: pd.Series,
                      test_labels: pd.Series) -> str:
    baseline, v1, v2 = results
    by_name = dict(zip(CONFIGS, results, strict=True))
    ideation = labels.index("risk_suicidal_ideation")
    passive = labels.index("risk_passive")
    lines = [
        "Risk Model Augmentation v2: Three-Configuration Comparison",
        "=" * 58,
        f"risk.csv SHA-256: {hashes['risk']}",
        f"risk_augmented_reviewed.csv SHA-256: {hashes['v1']}",
        f"risk_suicidal_ideation_final_v2.csv SHA-256: {hashes['v2']}",
        f"Shared test set SHA-256: {test_hash}",
        "Prior experiment test SHA-256 matched exactly before training.",
        "Baseline and v1 metrics and wrong-row predictions matched the prior experiment.",
        f"Original rows: {baseline['train_count'] + baseline['test_count']}",
        f"Training rows A/B/C: {baseline['train_count']} / {v1['train_count']} / {v2['train_count']}",
        f"Shared test rows: {baseline['test_count']}",
        "v1 and v2 normalized-text overlap with the shared test set: 0 / 0",
        f"Python libraries: pandas={pd.__version__}, scikit-learn={sklearn.__version__}, PyThaiNLP={pythainlp.__version__}",
        "Algorithm, TF-IDF, calibration, random_state, and hyperparameters are imported from the prior experiment.",
        "No model files were saved.",
        "",
        "Overall metrics (deltas are v1-A and v2-v1):",
        "Metric                 Baseline      v1          v2        v1-A       v2-v1",
    ]
    for title, key in METRICS:
        lines.append(
            f"{title:<22} {baseline[key]:>8.6f}  {v1[key]:>8.6f}  {v2[key]:>8.6f}  "
            f"{v1[key]-baseline[key]:>+9.6f}  {v2[key]-v1[key]:>+9.6f}"
        )
    lines.extend([
        f"Errors                 {baseline['error_count']:>8}  {v1['error_count']:>8}  "
        f"{v2['error_count']:>8}  {v1['error_count']-baseline['error_count']:>+9}  "
        f"{v2['error_count']-v1['error_count']:>+9}",
        "",
        "Focus classes: precision / recall / F1 (support is from the shared test set)",
    ])
    for label in FOCUS_LABELS:
        scores = [result["per_class"][label] for result in results]
        lines.append(f"{label} (support={scores[0]['support']})")
        for name, score in zip(CONFIGS, scores, strict=True):
            lines.append(
                f"  {name:<13} P={score['precision']:.6f} "
                f"R={score['recall']:.6f} F1={score['f1']:.6f}"
            )
        correct = [int(result["matrix"][labels.index(label), labels.index(label)])
                   for result in results]
        lines.append(f"  Correct cases: {correct[0]} -> {correct[1]} -> {correct[2]}")

    ideation_scores = [result["per_class"]["risk_suicidal_ideation"] for result in results]
    passive_scores = [result["per_class"]["risk_passive"] for result in results]
    ideation_to_passive = [int(result["matrix"][ideation, passive]) for result in results]
    lines.extend([
        "",
        "Target boundary: risk_suicidal_ideation vs risk_passive",
        "---------------------------------------------------------",
        "Ideation recall A -> v1 -> v2: " + " -> ".join(
            f"{score['recall']:.6f}" for score in ideation_scores
        ),
        "Ideation F1 A -> v1 -> v2: " + " -> ".join(
            f"{score['f1']:.6f}" for score in ideation_scores
        ),
        "Ideation predicted as passive A -> v1 -> v2: " + " -> ".join(
            str(value) for value in ideation_to_passive
        ),
        "Passive recall A -> v1 -> v2: " + " -> ".join(
            f"{score['recall']:.6f}" for score in passive_scores
        ),
        "Passive F1 A -> v1 -> v2: " + " -> ".join(
            f"{score['f1']:.6f}" for score in passive_scores
        ),
    ])

    v1_correct = ~v1["wrong"]
    v2_correct = ~v2["wrong"]
    fixed_indices = [
        int(index) for index, was_wrong, now_wrong in zip(
            test_text.index, v1["wrong"], v2["wrong"], strict=True
        ) if was_wrong and not now_wrong
    ]
    fixed_labels = [
        str(label) for label, was_wrong, now_wrong in zip(
            test_labels, v1["wrong"], v2["wrong"], strict=True
        ) if was_wrong and not now_wrong
    ]
    def ideation_as_passive_indices(result: dict) -> list[int]:
        return [
            int(index) for index, label, prediction in zip(
                test_text.index, test_labels, result["predictions"], strict=True
            )
            if label == "risk_suicidal_ideation" and prediction == "risk_passive"
        ]
    v1_ideation_as_passive = ideation_as_passive_indices(v1)
    v2_ideation_as_passive = ideation_as_passive_indices(v2)
    lines.extend([
        "",
        "Paired v1-to-v2 outcomes on the same 243 test rows:",
        f"Correct in both: {int((v1_correct & v2_correct).sum())}",
        f"Fixed by v2: {int((~v1_correct & v2_correct).sum())}",
        f"Regressed under v2: {int((v1_correct & ~v2_correct).sum())}",
        f"Wrong in both: {int((~v1_correct & ~v2_correct).sum())}",
        f"Fixed test indices: {fixed_indices}; true labels: {fixed_labels}",
        "Ideation-to-passive test indices unchanged: " +
        ("yes" if v1_ideation_as_passive == v2_ideation_as_passive else "no"),
        "",
        "Project-relevant changes from v1 to v2:",
    ])
    for label in FOCUS_LABELS:
        i = labels.index(label)
        correct_delta = int(v2["matrix"][i, i] - v1["matrix"][i, i])
        recall_delta = v2["per_class"][label]["recall"] - v1["per_class"][label]["recall"]
        f1_delta = v2["per_class"][label]["f1"] - v1["per_class"][label]["f1"]
        lines.append(
            f"{label}: correct {correct_delta:+d}, recall {recall_delta:+.6f}, F1 {f1_delta:+.6f}"
        )
    if (ideation_scores[2]["recall"] > ideation_scores[1]["recall"]
            and ideation_to_passive[2] < ideation_to_passive[1]):
        lines.append("The target boundary improved on this holdout; consider v2 for further review.")
    else:
        lines.append("The ideation-to-passive target did not improve on both measures.")
        if v2["accuracy"] > v1["accuracy"]:
            lines.append("Overall accuracy improved, but targeted follow-up is needed before any production decision.")
    lines.extend([
        "One fixed holdout cannot establish statistical or clinical significance.",
        "No production model replacement is decided by this experiment.",
        "",
        "Confusion matrices (rows=true, columns=predicted):",
    ])
    for name in CONFIGS:
        lines.extend(["", name, prior.render_matrix(by_name[name]["matrix"], labels)])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    output_names = (
        "baseline_report.txt", "augmented_v1_report.txt", "augmented_v2_report.txt",
        "comparison.txt", "baseline_errors.csv", "augmented_v1_errors.csv",
        "augmented_v2_errors.csv",
    )
    if any((OUTPUT_DIR / name).exists() for name in output_names):
        raise FileExistsError(f"Refusing to overwrite results in {OUTPUT_DIR}")

    hashes = {
        "risk": prior.file_sha256(RISK_CSV),
        "v1": prior.file_sha256(V1_CSV),
        "v2": prior.file_sha256(V2_CSV),
    }
    old_comparison = (PRIOR_DIR / "comparison.txt").read_text(encoding="utf-8")
    if hashes["risk"] != value_from_report(old_comparison, "Original dataset SHA-256"):
        raise RuntimeError("risk.csv differs from the prior experiment")
    if hashes["v1"] != value_from_report(old_comparison, "Reviewed dataset SHA-256"):
        raise RuntimeError("Reviewed v1 data differs from the prior experiment")

    original = prior.load_dataset(RISK_CSV)
    reviewed_v1 = prior.load_dataset(V1_CSV)
    final_v2 = prior.load_dataset(V2_CSV)
    if len(original) != 1211 or len(reviewed_v1) != 73 or len(final_v2) != 28:
        raise ValueError("Unexpected source row count")
    for name, dataset in (("v1", reviewed_v1), ("v2", final_v2)):
        if not dataset["reviewed"].astype(str).str.lower().eq("true").all():
            raise ValueError(f"{name} contains unreviewed data")
        if not dataset["label"].isin(original["label"].unique()).all():
            raise ValueError(f"{name} contains an unknown class")
    if not final_v2["label"].eq("risk_suicidal_ideation").all():
        raise ValueError("Final v2 contains another class")

    train_text, test_text, train_labels, test_labels = train_test_split(
        original["text"], original["label"], test_size=prior.TEST_SIZE,
        random_state=prior.RANDOM_STATE, shuffle=True, stratify=original["label"]
    )
    if len(test_text) != 243 or len(set(train_text.index) & set(test_text.index)):
        raise RuntimeError("Original train/test split differs from expectation")
    test_identity = [
        [int(index), str(label), str(text)]
        for index, label, text in zip(test_text.index, test_labels, test_text, strict=True)
    ]
    test_hash = hashlib.sha256(
        json.dumps(test_identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if test_hash != value_from_report(old_comparison, "Shared test set SHA-256"):
        raise RuntimeError("The test rows differ from the prior experiment")

    normalized_test = {prior.normalized_for_overlap(text) for text in test_text}
    for name, dataset in (("v1", reviewed_v1), ("v2", final_v2)):
        overlap = {prior.normalized_for_overlap(text) for text in dataset["text"]} & normalized_test
        if overlap:
            raise RuntimeError(f"{name} augmentation overlaps the test set: {len(overlap)}")
    print(f"Verified shared test set: {len(test_text)} rows, SHA-256 {test_hash}", flush=True)

    labels = sorted(original["label"].unique())
    baseline = prior.evaluate("Baseline", train_text, train_labels, test_text, test_labels, labels)
    verify_previous_result(
        baseline, PRIOR_DIR / "baseline_report.txt", PRIOR_DIR / "baseline_errors.csv", test_text
    )
    print(f"Baseline reproduced: accuracy={baseline['accuracy']:.6f}", flush=True)

    v1_text = pd.concat([
        train_text.reset_index(drop=True), reviewed_v1["text"].reset_index(drop=True)
    ], ignore_index=True)
    v1_labels = pd.concat([
        train_labels.reset_index(drop=True), reviewed_v1["label"].reset_index(drop=True)
    ], ignore_index=True)
    v1 = prior.evaluate("Augmented v1", v1_text, v1_labels, test_text, test_labels, labels)
    verify_previous_result(v1, PRIOR_DIR / "augmented_report.txt",
                           PRIOR_DIR / "augmented_errors.csv", test_text)
    print(f"Augmented v1 reproduced: accuracy={v1['accuracy']:.6f}", flush=True)

    v2_text = pd.concat([v1_text, final_v2["text"].reset_index(drop=True)], ignore_index=True)
    v2_labels = pd.concat([v1_labels, final_v2["label"].reset_index(drop=True)], ignore_index=True)
    v2 = prior.evaluate("Augmented v2", v2_text, v2_labels, test_text, test_labels, labels)
    print(f"Augmented v2 evaluated: accuracy={v2['accuracy']:.6f}", flush=True)

    if {"risk": prior.file_sha256(RISK_CSV), "v1": prior.file_sha256(V1_CSV),
        "v2": prior.file_sha256(V2_CSV)} != hashes:
        raise RuntimeError("An input dataset changed during the experiment")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = (baseline, v1, v2)
    names = ("baseline", "augmented_v1", "augmented_v2")
    for name, result in zip(names, results, strict=True):
        (OUTPUT_DIR / f"{name}_report.txt").write_text(
            render_report(result, labels, test_hash, hashes), encoding="utf-8"
        )
        prior.write_errors(OUTPUT_DIR / f"{name}_errors.csv", result, test_text, test_labels)
    (OUTPUT_DIR / "comparison.txt").write_text(
        render_comparison(list(results), labels, test_hash, hashes,
                          test_text, test_labels), encoding="utf-8"
    )
    print(f"Saved reports and errors under {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
