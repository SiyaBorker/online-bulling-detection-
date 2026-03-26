"""
Baseline Comparison: TF-IDF + Logistic Regression vs Fine-tuned DistilBERT
Run this BEFORE the main training to establish the baseline F1-score.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import re

# ── Preprocessing (same as main script) ────────────────────────────────────
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("cyberbullying_tweets.csv")
df = df.dropna(subset=["tweet_text", "cyberbullying_type"])
df["tweet_text"] = df["tweet_text"].apply(preprocess_text)
df["label"] = (df["cyberbullying_type"] != "not_cyberbullying").astype(int)
df = df[df["tweet_text"].str.len() > 3].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["tweet_text"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
)

# ── Baseline: TF-IDF + Logistic Regression ──────────────────────────────────
print("=" * 50)
print("BASELINE: TF-IDF + Logistic Regression")
print("=" * 50)

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train_tfidf, y_train)
preds = lr.predict(X_test_tfidf)

print(classification_report(y_test, preds, target_names=["Not Bullying", "Bullying"]))
print(f"Baseline F1-Score (binary): {f1_score(y_test, preds, average='binary'):.4f}")
print("\nNote: Run cyberbullying_detection.py to get fine-tuned DistilBERT results.")
