import re
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTI_SPACE_RE = re.compile(r"\s+")
STOPWORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = MENTION_RE.sub(" ", t)              
    t = HASHTAG_RE.sub(r"\1", t)          
    t = "".join(
        ch for ch in t
        if ch.isalnum() or ch.isspace() or ch == "'"
    )                                          
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t

def preprocess_text(text: str) -> str:
    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def load_and_preprocess_data(
    csv_path: str = "social_media_sentiment.csv",
    text_col: str = "post",
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1").dropna(subset=[text_col, label_col])
    df[label_col].value_counts().plot(
        kind="bar",
        title="Class Distribution"
    )
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.show()
    df["_clean"] = df[text_col].astype(str).apply(preprocess_text)
    X = df["_clean"]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
