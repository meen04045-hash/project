"""Training and inference for the Thai mental-health chatbot classifiers.

Each task intentionally has its own TF-IDF vectorizer and calibrated LinearSVC.
"""

from __future__ import annotations

import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pythainlp
import sklearn
from pythainlp.tokenize import word_tokenize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
EVALUATION_DIR = Path("evaluation")
ERROR_ANALYSIS_DIR = EVALUATION_DIR / "error_analysis"
RANDOM_STATE = 42

# artifact_name keeps compatibility with the models already used by app.py.
MODEL_SPECS = (
    ("risk", "risk", "risk.csv"),
    ("emotion", "emotion", "emotion.csv"),
    ("problem", "problem", "problem.csv"),
    ("support_need", "support", "support_need.csv"),
    ("intent", "intent", "intent.csv"),
    ("conversation_style", "style", "conversation_style.csv"),
)


def clean_text(text: object) -> str:
    """Normalize input without removing word boundaries, including Thai ones."""
    normalized = str(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize(text: object) -> str:
    """Tokenize normalized Thai text for the independent TF-IDF vectorizers."""
    return " ".join(word_tokenize(clean_text(text), engine="newmm"))


def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 3), min_df=1)


def create_classifier(minimum_class_samples: int = 5) -> CalibratedClassifierCV:
    """Create a balanced LinearSVC with calibration for probability prediction."""
    calibration_folds = max(2, min(5, minimum_class_samples))
    base_model = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    return CalibratedClassifierCV(base_model, cv=calibration_folds)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"{csv_path} must contain columns: text, label")
    return dataset.dropna(subset=["text", "label"]).copy()


def save_dataset_statistics(statistics: list[tuple[str, pd.DataFrame]]) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for display_name, dataset in statistics:
        source = dataset
        valid_rows = source.dropna(subset=["text", "label"])
        labels = valid_rows["label"].value_counts()
        sections.extend([
            f"{display_name.replace('_', ' ').title()} Dataset",
            f"Total Samples: {len(source)}",
            f"Number of Labels: {valid_rows['label'].nunique()}",
            f"Missing Values: {int(source.isna().sum().sum())}",
            f"Duplicate Rows: {int(source.duplicated().sum())}",
            "Label Distribution:",
            *[f"{label}: {count}" for label, count in labels.items()],
            "\n" + "-" * 40 + "\n",
        ])
    (EVALUATION_DIR / "dataset_statistics.txt").write_text("\n".join(sections), encoding="utf-8")


def evaluate_model(
    display_name: str,
    artifact_name: str,
    vectorizer: TfidfVectorizer,
    model: CalibratedClassifierCV,
    x_test: pd.Series,
    y_test: pd.Series,
    full_text: pd.Series,
    full_labels: pd.Series,
) -> dict[str, Any]:
    """Print and persist holdout metrics, cross-validation, and bad predictions."""
    transformed_test = vectorizer.transform(x_test)
    predictions = model.predict(transformed_test)
    probabilities = model.predict_proba(transformed_test)
    confidence_scores = probabilities.max(axis=1)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)

    # The pipeline ensures TF-IDF is fitted independently in every CV training fold.
    class_minimum = int(full_labels.value_counts().min())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("tfidf", create_vectorizer()),
        ("classifier", create_classifier(max(2, min(5, class_minimum - 1)))),
    ])
    cv_scores = cross_val_score(pipeline, full_text, full_labels, cv=cv, scoring="accuracy")

    print(f"Accuracy : {accuracy:.3f}")
    print("\n5-Fold Cross Validation")
    for fold_number, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold_number} Accuracy : {score:.3f}")
    print(f"Mean Accuracy : {cv_scores.mean():.3f}")
    print(f"Std : {cv_scores.std():.3f}")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join([
        "=" * 50,
        f"{display_name.replace('_', ' ').title()} Model Evaluation",
        "=" * 50,
        f"Accuracy: {accuracy:.6f}",
        "\nPrecision / Recall / F1-score:\n",
        report,
        "Confusion Matrix:",
        f"Labels: {list(model.classes_)}",
        str(matrix),
        "\n5-Fold Cross Validation:",
        *[f"Fold {index}: {score:.6f}" for index, score in enumerate(cv_scores, 1)],
        f"Mean Accuracy: {cv_scores.mean():.6f}",
        f"Standard Deviation: {cv_scores.std():.6f}",
    ])
    (EVALUATION_DIR / f"{display_name}_report.txt").write_text(report_text, encoding="utf-8")

    errors = pd.DataFrame({
        "Original Text": x_test.to_numpy(),
        "True Label": y_test.to_numpy(),
        "Predicted Label": predictions,
        "Confidence Score": confidence_scores,
    })
    errors = errors[errors["True Label"] != errors["Predicted Label"]]
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(ERROR_ANALYSIS_DIR / f"{display_name}_errors.csv", index=False, encoding="utf-8-sig")
    print("Evaluation Saved")
    return {"accuracy": accuracy, "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std())}


def train_model(csv_path: str | Path, display_name: str = "model", artifact_name: str = "model") -> tuple[TfidfVectorizer, CalibratedClassifierCV, dict[str, Any]]:
    """Train one independent task model with a leakage-free holdout evaluation."""
    dataset = load_dataset(Path(csv_path))
    texts, labels = dataset["text"], dataset["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, shuffle=True, stratify=labels
    )
    print("=" * 36)
    print(f"Training {display_name.replace('_', ' ').title()} Model")
    print("=" * 36)
    print(f"Dataset Size : {len(dataset)}")
    print(f"Training Size : {len(x_train)}")
    print(f"Testing Size : {len(x_test)}")
    print("Training...")

    vectorizer = create_vectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)  # fit only on training text
    model = create_classifier(int(y_train.value_counts().min()))
    model.fit(x_train_vectorized, y_train)
    metrics = evaluate_model(display_name, artifact_name, vectorizer, model, x_test, y_test, texts, labels)
    return vectorizer, model, metrics


def get_model(name: str, csv_path: str, force_train: bool = False, display_name: str | None = None) -> tuple[TfidfVectorizer, CalibratedClassifierCV]:
    """Load an existing pair, or train only when explicitly requested."""
    model_path = MODELS_DIR / f"{name}_model.joblib"
    vectorizer_path = MODELS_DIR / f"{name}_vectorizer.joblib"
    if not force_train:
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing {name} model artifact; train artifacts in advance.")
        if not vectorizer_path.is_file():
            raise FileNotFoundError(f"Missing {name} vectorizer artifact; train artifacts in advance.")
        try:
            return joblib.load(vectorizer_path), joblib.load(model_path)
        except Exception as error:
            raise RuntimeError(f"Unable to load {name} artifacts; train artifacts in advance.") from error

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = display_name or name
    vectorizer, model, _ = train_model(csv_path, model_name, name)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    print("Model Saved")
    return vectorizer, model


def save_model_info(model_details: list[dict[str, Any]]) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Python Version: {platform.python_version()}",
        f"scikit-learn Version: {sklearn.__version__}",
        f"PyThaiNLP Version: {pythainlp.__version__}",
        "LinearSVC Parameters: C=1.0, class_weight='balanced', random_state=42",
        "CalibratedClassifierCV Parameters: cv=min(5, smallest training class size)",
        "TF-IDF Parameters: tokenizer=PyThaiNLP newmm, token_pattern=None",
        f"Training Date: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for item in model_details:
        lines.extend([
            item["name"],
            f"Vocabulary Size: {item['vocabulary_size']}",
            f"Training Samples: {item['training_samples']}",
            f"Testing Samples: {item['testing_samples']}",
            "",
        ])
    (EVALUATION_DIR / "model_info.txt").write_text("\n".join(lines), encoding="utf-8")


def train_all_models(force_train: bool = True) -> None:
    """Train all six independent models and create research evaluation artefacts."""
    # Statistics intentionally inspect raw CSV rows, before training drops missing values.
    datasets = [(display, pd.read_csv(DATASET_DIR / filename)) for display, _, filename in MODEL_SPECS]
    save_dataset_statistics(datasets)
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    global risk_vectorizer, risk_model, emotion_vectorizer, emotion_model
    global problem_vectorizer, problem_model, support_vectorizer, support_model
    global intent_vectorizer, intent_model, style_vectorizer, style_model
    for display_name, artifact_name, filename in MODEL_SPECS:
        dataset = load_dataset(DATASET_DIR / filename)
        vectorizer, model, metrics = train_model(DATASET_DIR / filename, display_name, artifact_name)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, MODELS_DIR / f"{artifact_name}_vectorizer.joblib")
        joblib.dump(model, MODELS_DIR / f"{artifact_name}_model.joblib")
        print("Model Saved")
        summaries.append({"name": display_name, **metrics})
        details.append({"name": display_name.replace('_', ' ').title(), "vocabulary_size": len(vectorizer.vocabulary_), "training_samples": int(len(dataset) * 0.8), "testing_samples": len(dataset) - int(len(dataset) * 0.8)})
        globals()[f"{artifact_name}_vectorizer"] = vectorizer
        globals()[f"{artifact_name}_model"] = model
    summary_lines = ["=" * 48, "MODEL SUMMARY", "=" * 48, ""]
    for item in summaries:
        summary_lines.extend([item["name"].replace('_', ' ').title(), f"Accuracy : {item['accuracy']:.1%}", f"Cross Validation : {item['cv_mean']:.1%}", "-" * 40])
    (EVALUATION_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    save_model_info(details)


# Load once at import so Flask callers can continue using predict_all unchanged.
risk_vectorizer, risk_model = get_model("risk", "dataset/risk.csv")
emotion_vectorizer, emotion_model = get_model("emotion", "dataset/emotion.csv")
problem_vectorizer, problem_model = get_model("problem", "dataset/problem.csv")
support_vectorizer, support_model = get_model("support", "dataset/support_need.csv", display_name="support_need")
intent_vectorizer, intent_model = get_model("intent", "dataset/intent.csv")
style_vectorizer, style_model = get_model("style", "dataset/conversation_style.csv", display_name="conversation_style")


def predict_with_confidence(model: CalibratedClassifierCV, vectorizer: TfidfVectorizer, text: str) -> tuple[str, float]:
    transformed_text = vectorizer.transform([clean_text(text)])
    prediction = model.predict(transformed_text)[0]
    confidence = float(model.predict_proba(transformed_text)[0].max())
    return prediction, confidence


def predict_all(msg: str) -> dict[str, Any]:
    """Return the original Flask-compatible prediction response structure."""
    risk, risk_conf = predict_with_confidence(risk_model, risk_vectorizer, msg)
    emotion, emotion_conf = predict_with_confidence(emotion_model, emotion_vectorizer, msg)
    problem, problem_conf = predict_with_confidence(problem_model, problem_vectorizer, msg)
    support, support_conf = predict_with_confidence(support_model, support_vectorizer, msg)
    intent, intent_conf = predict_with_confidence(intent_model, intent_vectorizer, msg)
    style, style_conf = predict_with_confidence(style_model, style_vectorizer, msg)
    return {"risk": risk, "risk_conf": risk_conf, "emotion": emotion, "emotion_conf": emotion_conf, "problem": problem, "problem_conf": problem_conf, "support_need": support, "support_conf": support_conf, "intent": intent, "intent_conf": intent_conf, "conversation_style": style, "style_conf": style_conf}


def display_prediction(text: str) -> None:
    result = predict_all(text)
    rows = (("Risk", "risk", "risk_conf"), ("Emotion", "emotion", "emotion_conf"), ("Problem", "problem", "problem_conf"), ("Support Need", "support_need", "support_conf"), ("Intent", "intent", "intent_conf"), ("Conversation Style", "conversation_style", "style_conf"))
    print("\n" + "=" * 50 + "\nPrediction Result\n" + "=" * 50)
    print("\nInput received\n" + "-" * 50)
    for label, value_key, confidence_key in rows:
        print(f"{label:<20} : {str(result[value_key]):<20} ({result[confidence_key]:.3f})")
    print("=" * 50)


def run_cli() -> None:
    while True:
        print("\n" + "=" * 36 + "\nAI MODEL\n" + "=" * 36)
        print("1 Train All Models\n2 Predict Text\n3 Exit")
        choice = input("Select: ").strip()
        if choice == "1":
            train_all_models(force_train=True)
        elif choice == "2":
            print("Enter 'exit' to return to the menu.")
            while True:
                text = input("Input: ").strip()
                if text.lower() == "exit":
                    break
                if text:
                    display_prediction(text)
        elif choice == "3":
            break
        else:
            print("Please select 1, 2, or 3.")


if __name__ == "__main__":
    run_cli()
