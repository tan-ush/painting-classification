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


#======================
# Data Loading from CSV
#======================


CSV_Path = "ml_challenge_dataset.csv"
df = pd.read_csv(CSV_Path)

TARGET_COL = "Painting"
# Text columns
TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
]
# 1-5 rating columns
LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy."
]
# 1-10 and other rating columns
NUMERIC_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
    "How much (in Canadian dollars) would you be willing to pay for this painting?"
]
# Columns for multiple choice
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

# Used for parsing firs tnumber
def parse_likert(s):
    if pd.isna(s):
        return 3
    s = str(s).strip()
    m = re.search(r"[1-5]", s)
    if m:
        return int(m.group())
    return 3

#Parsing number from numeric questions and cost
def parse_number(s, default=0.0):
    # handles things like "$5", "300 dollars", "1,000", "maybe $10"
    if pd.isna(s):
        return default
    s = str(s).replace(",", "")
    m = re.search(r"\d+(\.\d+)?", s)
    if m:
        return float(m.group())
    return default

# Take the multiselect options and turn to list
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
#Apply preprocessing
for col in LIKERT_COLS:
    df[col] = df[col].apply(parse_likert).astype(float)
#Apply preprocessing
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
    # Track all columns answers in the dataframe
    for col in columns:
        all_options = set()
        for items in dataframe[col]:
            all_options.update(items)
        #have a set of options

        all_options = sorted(all_options)
        # We use a dataframe for easy data transfer and manipulation
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
X_text = df["combined_text"]
X_struct = structured_df
y = df[TARGET_COL]

X_text_train, X_text_val, X_struct_train, X_struct_val, y_train, y_val = train_test_split(
    X_text,
    X_struct,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. TF-IDF, basically transforming to numpy
# =========================
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 1),
    min_df=1
)
# use numpy for future  case
X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_val_tfidf = vectorizer.transform(X_text_val)
#we isolate them into different ones for certain column testing
X_struct_train_np = X_struct_train.to_numpy()
X_struct_val_np = X_struct_val.to_numpy()

X_train = hstack([X_text_train_tfidf, csr_matrix(X_struct_train_np)])
X_val = hstack([X_text_val_tfidf, csr_matrix(X_struct_val_np)])

# =========================
# 7. Logistic Regression
# =========================
#our logistic reg
logreg = LogisticRegression(
    C=1,
    max_iter=3000
)
logreg.fit(X_train, y_train)
# Preidct on training and validation
train_pred_lr = logreg.predict(X_train)
val_pred_lr = logreg.predict(X_val)

print("Logistic Regression")
print("Train acc:", accuracy_score(y_train, train_pred_lr))
print("Val acc:  ", accuracy_score(y_val, val_pred_lr))
print()

# =========================
# 8. Naive Bayes
# =========================
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)
# Preidct on training and validation
train_pred_nb = nb.predict(X_train)
val_pred_nb = nb.predict(X_val)

print("Naive Bayes")
print("Train acc:", accuracy_score(y_train, train_pred_nb))
print("Val acc:  ", accuracy_score(y_val, val_pred_nb))
print()


# =========================
# 9. Decision Tree
# =========================
# using structured only, since trees are not ideal for sparse TF-IDF
tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
tree.fit(X_struct_train, y_train)
# Preidct on training and validation
train_pred_tree = tree.predict(X_struct_train)
val_pred_tree = tree.predict(X_struct_val)

print("Decision Tree (structured only)")
print("Train acc:", accuracy_score(y_train, train_pred_tree))
print("Val acc:  ", accuracy_score(y_val, val_pred_tree))
print()

# =========================
# 10. Feature ablation
# =========================
logreg_text_only = LogisticRegression(C=1.0, max_iter=3000)
logreg_text_only.fit(X_text_train_tfidf, y_train)
val_pred_text_only = logreg_text_only.predict(X_text_val_tfidf)

logreg_struct_only = LogisticRegression(C=1.0, max_iter=3000)
logreg_struct_only.fit(X_struct_train, y_train)
val_pred_struct_only = logreg_struct_only.predict(X_struct_val)

print("LogReg text only val acc:      ", accuracy_score(y_val, val_pred_text_only))
print("LogReg structured only val acc:", accuracy_score(y_val, val_pred_struct_only))
print("LogReg combined val acc:       ", accuracy_score(y_val, val_pred_lr))
