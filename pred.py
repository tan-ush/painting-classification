import sys
import csv
import random
import json
import re

import numpy as np
import pandas as pd


# =========================
# 1. Load saved artifacts that we used in presplit and gen in json
# =========================
with open("model_artifacts.json", "r") as f:
    ART = json.load(f)

TEXT_COLS = ART["text_cols"]
LIKERT_COLS = ART["likert_cols"]
NUMERIC_COLS = ART["numeric_cols"]
MULTISELECT_COLS = ART["multiselect_cols"]

VOCAB = ART["vocabulary"] # token -> column index
IDF = np.array(ART["idf"], dtype=float)
STRUCTURED_FEATURE_NAMES = ART["structured_feature_names"]
CLASSES = ART["classes"]

# Our weights and bias
W = np.array(ART["coef"], dtype=float)# [num_classes, num_features]
B = np.array(ART["intercept"], dtype=float)    # [num_classes]


# =========================
# 2. Cleaning helpers that we used in presplit
# =========================
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_likert(s):
    if pd.isna(s):
        return 3.0
    s = str(s).strip()
    m = re.search(r"[1-5]", s)
    if m:
        return float(m.group())
    return 3.0


def parse_number(s, default=0.0):
    if pd.isna(s):
        return default
    s = str(s).replace(",", "")
    m = re.search(r"\d+(\.\d+)?", s)
    if m:
        return float(m.group())
    return default


def parse_multiselect(s):
    if pd.isna(s):
        return []
    return [item.strip() for item in str(s).split(",") if item.strip()]


# =========================
# 3. Tokenization for unigram TF-IDF
# =========================
# Tokenization is different without sk learn
def tokenize(text):
    text = clean_text(text)
    if not text:
        return []
    return text.split()


def build_tfidf_matrix(texts):
    """
    Rebuilds sklearn-like unigram TF-IDF with:
    - max_features fixed by saved vocabulary
    - ngram_range=(1,1)
    """
    n = len(texts)
    d = len(VOCAB)
    X = np.zeros((n, d), dtype=float)

    for i, text in enumerate(texts):
        tokens = tokenize(text)
        if not tokens:
            continue

        counts = {}
        for tok in tokens:
            if tok in VOCAB:
                counts[tok] = counts.get(tok, 0) + 1

        total = sum(counts.values())
        if total == 0:
            continue

        # term frequency * idf
        for tok, cnt in counts.items():
            j = VOCAB[tok]
            tf = cnt / total
            X[i, j] = tf * IDF[j]

        # l2 normalize row to match sklearn default norm='l2'
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm

    return X


# =========================
# 4. Structured feature builder
# =========================
def build_structured_features(df):
    rows = []

    for _, row in df.iterrows():
        feat = {}

        # Likert
        for col in LIKERT_COLS:
            feat[col] = parse_likert(row.get(col, np.nan))

        # Numeric
        for col in NUMERIC_COLS[:-1]:
            feat[col] = parse_number(row.get(col, np.nan), default=0.0)

        price_col = NUMERIC_COLS[-1]
        price_val = parse_number(row.get(price_col, np.nan), default=0.0)
        feat[price_col] = np.log1p(price_val)

        # Multi-select
        for col in MULTISELECT_COLS:
            selected = set(parse_multiselect(row.get(col, np.nan)))
            prefix = f"{col}__"

            for feature_name in STRUCTURED_FEATURE_NAMES:
                if feature_name.startswith(prefix):
                    option = feature_name[len(prefix):]
                    feat[feature_name] = 1.0 if option in selected else 0.0

        rows.append(feat)

    struct_df = pd.DataFrame(rows)

    # ensure all training-time structured columns exist and are in the same order
    for col in STRUCTURED_FEATURE_NAMES:
        if col not in struct_df.columns:
            struct_df[col] = 0.0

    struct_df = struct_df[STRUCTURED_FEATURE_NAMES].fillna(0.0)
    return struct_df.to_numpy(dtype=float)


# =========================
# 5. Predict all
# =========================
def predict_all(filename):
    df = pd.read_csv(filename)

    # combined text
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(clean_text)

    df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1)

    # text features
    X_text = build_tfidf_matrix(df["combined_text"].tolist())

    # structured features
    X_struct = build_structured_features(df)

    # combine in same order used during training:
    # [text features | structured features]
    X = np.hstack([X_text, X_struct])

    # linear scores for multiclass logistic regression
    scores = X @ W.T + B
    pred_idx = np.argmax(scores, axis=1)

    predictions = [CLASSES[i] for i in pred_idx]
    return predictions
