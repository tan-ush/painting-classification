import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix
import json


#======================
# Data Loading from CSV
#======================


CSV_Path = "ml_challenge_dataset.csv"
df = pd.read_csv(CSV_Path)

TARGET_COL = "Painting"
TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
]

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy."
]

NUMERIC_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
    "How much (in Canadian dollars) would you be willing to pay for this painting?"
]

MULTISELECT_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?"
]

# Used to essentially normalize text (replace uppercase, whitespaces, special)


def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_likert(s):
    if pd.isna(s):
        return 3
    s = str(s).strip()
    m = re.search(r"[1-5]", s)
    if m:
        return int(m.group())
    return 3

def parse_number(s, default=0.0):
    # handles things like "$5", "300 dollars", "1,000", "maybe $10"
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

# =====================
# 3. Preprocess columns
# =====================
for col in TEXT_COLS:
    df[col] = df[col].apply(clean_text)

df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1)

for col in LIKERT_COLS:
    df[col] = df[col].apply(parse_likert).astype(float)

for col in NUMERIC_COLS[:-1]:
    df[col] = df[col].apply(parse_number).astype(float)

price_col = NUMERIC_COLS[-1]
df[price_col] = df[price_col].apply(parse_number).astype(float)
# This compresses large values (some crazy inputs)
df[price_col] = np.log1p(df[price_col])

for col in MULTISELECT_COLS:
    df[col] = df[col].apply(parse_multiselect)

# =========================
# 4. Multi-select -> binary features
# =========================
def build_multiselect_features(dataframe, columns):
    feature_frames = []

    for col in columns:
        all_options = set()
        for items in dataframe[col]:
            all_options.update(items)

        all_options = sorted(all_options)

        col_features = pd.DataFrame(index=dataframe.index)
        for option in all_options:
            feat_name = f"{col}__{option}"
            col_features[feat_name] = dataframe[col].apply(lambda items: int(option in items))

        feature_frames.append(col_features)

    if feature_frames:
        return pd.concat(feature_frames, axis=1)
    return pd.DataFrame(index=dataframe.index)


multi_df = build_multiselect_features(df, MULTISELECT_COLS)

structured_df = pd.concat([
    df[LIKERT_COLS],
    df[NUMERIC_COLS[:-1]],
    df[[price_col]],
    multi_df
], axis=1)

structured_df = structured_df.fillna(structured_df.median(numeric_only=True))

# =========================
# 5. Train/validation split
# =========================
X_text_full = df["combined_text"]
X_struct_full = structured_df
y_full = df[TARGET_COL]

# =========================
# 6. TF-IDF, basically transforming to numpy
# =========================
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 1),
    min_df=1
)

X_text_full_tfidf = vectorizer.fit_transform(X_text_full)
X_struct_full_np = X_struct_full.to_numpy()

X_full = hstack([X_text_full_tfidf, csr_matrix(X_struct_full_np)])

# =========================
# 7. Logistic Regression
# =========================
logreg = LogisticRegression(C=1, max_iter=3000)
logreg.fit(X_full, y_full)

print("Final model trained on full dataset.")
print("Number of training examples:", len(y_full))
print("Number of features:", X_full.shape[1])

# =========================
#8. Export artifacts for pred.py
# =========================
def to_python(obj):
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


artifacts = {
    "text_cols": TEXT_COLS,
    "likert_cols": LIKERT_COLS,
    "numeric_cols": NUMERIC_COLS,
    "multiselect_cols": MULTISELECT_COLS,

    "vocabulary": vectorizer.vocabulary_,
    "idf": vectorizer.idf_,

    "structured_feature_names": structured_df.columns.tolist(),

    "classes": logreg.classes_,

    "coef": logreg.coef_,
    "intercept": logreg.intercept_
}

# convert everything
artifacts = to_python(artifacts)

with open("model_artifacts.json", "w") as f:
    json.dump(artifacts, f)

print("Saved model_artifacts.json")
