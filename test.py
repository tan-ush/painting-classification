import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

from pred import predict_all


CSV_PATH = "ml_challenge_dataset.csv"

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


# rebuild sklearn pipeline exactly
df = pd.read_csv(CSV_PATH)
raw_df = df.copy()

for col in TEXT_COLS:
    df[col] = df[col].apply(clean_text)

df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1)

for col in LIKERT_COLS:
    df[col] = df[col].apply(parse_likert).astype(float)

for col in NUMERIC_COLS[:-1]:
    df[col] = df[col].apply(parse_number).astype(float)

price_col = NUMERIC_COLS[-1]
df[price_col] = df[price_col].apply(parse_number).astype(float)
df[price_col] = np.log1p(df[price_col])

for col in MULTISELECT_COLS:
    df[col] = df[col].apply(parse_multiselect)

multi_df = build_multiselect_features(df, MULTISELECT_COLS)

structured_df = pd.concat([
    df[LIKERT_COLS],
    df[NUMERIC_COLS[:-1]],
    df[[price_col]],
    multi_df
], axis=1)

structured_df = structured_df.fillna(structured_df.median(numeric_only=True))

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 1),
    min_df=1
)

X_text = vectorizer.fit_transform(df["combined_text"])
X_struct = structured_df.to_numpy()
X = hstack([X_text, csr_matrix(X_struct)])

y = df[TARGET_COL]

model = LogisticRegression(C=1, max_iter=3000)
model.fit(X, y)

sklearn_preds = model.predict(X).tolist()
predpy_preds = predict_all(CSV_PATH)

matches = sum(a == b for a, b in zip(sklearn_preds, predpy_preds))
total = len(sklearn_preds)

print("Total rows:", total)
print("Matches:", matches)
print("Match rate:", matches / total)
print()

print("Sklearn prediction counts:")
print(pd.Series(sklearn_preds).value_counts())
print()

print("pred.py prediction counts:")
print(pd.Series(predpy_preds).value_counts())
print()

for i, (a, b) in enumerate(zip(sklearn_preds, predpy_preds)):
    if a != b:
        print("FIRST MISMATCH")
        print("Row index:", i)
        print("sklearn:", a)
        print("pred.py:", b)
        print()
        print("Original row:")
        print(raw_df.iloc[i].to_string())
        break
else:
    print("No mismatches found.")
