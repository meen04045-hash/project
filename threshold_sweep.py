import re
import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def clean_text(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def tokenize(text):
    return " ".join(word_tokenize(clean_text(text), engine="newmm"))

def create_vectorizer():
    return TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 3), min_df=1)

def create_classifier(min_class_samples=5):
    folds = max(2, min(5, min_class_samples))
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    return CalibratedClassifierCV(base, cv=folds)

def sweep(name, csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    texts, labels = df["text"], df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, shuffle=True, stratify=labels
    )
    vec = create_vectorizer()
    x_train_v = vec.fit_transform(x_train)
    clf = create_classifier(int(y_train.value_counts().min()))
    clf.fit(x_train_v, y_train)

    x_test_v = vec.transform(x_test)
    probs = clf.predict_proba(x_test_v)
    preds = clf.classes_[np.argmax(probs, axis=1)]
    conf = probs.max(axis=1)
    correct_mask = preds == y_test.to_numpy()

    print(f"\n=== {name} ===")
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        fires = conf >= t
        fire_rate = fires.mean()
        precision = (fires & correct_mask).sum() / fires.sum() if fires.sum() > 0 else float("nan")
        print(f"  threshold={t:.2f}  fires={fire_rate:.1%}  precision_when_fires={precision:.1%}")

sweep("emotion", "dataset/emotion.csv")
sweep("problem", "dataset/problem.csv")
sweep("support_need", "dataset/support_need.csv")
sweep("conversation_style", "dataset/conversation_style.csv")