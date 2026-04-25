"""
training.py
===========
- Downloads TMDB 5000 dataset
- Preprocesses features
- Trains 3 models: Cosine Similarity (Content-Based), KNN, Naive Bayes
- Evaluates: Accuracy, R2, Confusion Matrix, ROC-AUC
- Saves model artifacts as .pkl files
"""

import pandas as pd
import numpy as np
import ast
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, r2_score
)

# ── helpers ────────────────────────────────────────────────────────────────
def parse_names(obj, top_n=3):
    try:
        items = ast.literal_eval(obj)
        return [i["name"].replace(" ", "") for i in items[:top_n]]
    except:
        return []

def get_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return [i["name"].replace(" ", "")]
        return []
    except:
        return []


# ── Step 1 : Load data ─────────────────────────────────────────────────────
print("=" * 60)
print("  MOVIE RECOMMENDATION SYSTEM — Model Training")
print("=" * 60)
print("\n[1/6] Loading TMDB 5000 dataset...")

"""
Download the TMDB 5000 dataset from Kaggle:
  https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
Place both CSVs in the same folder as this script:
  - tmdb_5000_movies.csv
  - tmdb_5000_credits.csv   (optional — merged automatically if present)
"""

MOVIES_FILE  = r"C:\Users\Lenovo\Downloads\tmdb_5000_movies.csv"
CREDITS_FILE = r"C:\Users\Lenovo\Downloads\tmdb_5000_credits.csv"

if not os.path.exists(MOVIES_FILE):
    raise FileNotFoundError(
        f"\n❌  '{MOVIES_FILE}' not found!\n"
        "   Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
        "   and place it in the same directory as training.py\n"
    )

movies = pd.read_csv(MOVIES_FILE)

if os.path.exists(CREDITS_FILE):
    credits = pd.read_csv(CREDITS_FILE)
    movies = movies.merge(credits, on="title", how="left")

print(f"   Dataset loaded: {len(movies)} movies, {movies.shape[1]} columns")


# ── Step 2 : Preprocessing ─────────────────────────────────────────────────
print("\n[2/6] Preprocessing features...")
movies = movies[["id","title","overview","genres","keywords","cast","crew",
                 "vote_average","vote_count"]].dropna()

movies["genres_list"]   = movies["genres"].apply(parse_names)
movies["keywords_list"] = movies["keywords"].apply(parse_names)
movies["cast_list"]     = movies["cast"].apply(lambda x: parse_names(x, 3))
movies["director"]      = movies["crew"].apply(get_director)
movies["overview_list"] = movies["overview"].apply(lambda x: x.split())

movies["tags"] = (movies["overview_list"] + movies["genres_list"] +
                  movies["keywords_list"] + movies["cast_list"] + movies["director"])
movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())

# Primary genre label for classification models
movies["primary_genre"] = movies["genres_list"].apply(
    lambda x: x[0] if len(x) > 0 else "Unknown"
)

df = movies[["id","title","tags","vote_average","vote_count","primary_genre"]].reset_index(drop=True)

# Keep only genres with enough samples for classification
genre_counts = df["primary_genre"].value_counts()
valid_genres  = genre_counts[genre_counts >= 20].index.tolist()
df = df[df["primary_genre"].isin(valid_genres)].reset_index(drop=True)
print(f"   After genre filter: {len(df)} movies, {len(valid_genres)} genre classes")


# ── Step 3 : Feature Vectors ───────────────────────────────────────────────
print("\n[3/6] Building feature vectors...")
cv    = CountVectorizer(max_features=5000, stop_words="english")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")

count_matrix = cv.fit_transform(df["tags"]).toarray()
tfidf_matrix = tfidf.fit_transform(df["tags"]).toarray()

le = LabelEncoder()
df["genre_label"] = le.fit_transform(df["primary_genre"])

X = count_matrix
y = df["genre_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")


# ── Step 4 : Cosine Similarity (Recommendation Core) ──────────────────────
print("\n[4/6] Training models...")

# --- Model A: Cosine Similarity ---
cos_sim = cosine_similarity(count_matrix)
print("   [A] Cosine Similarity matrix built ✓")

# Pseudo accuracy: for each movie check if top-5 share the same genre
def cosine_pseudo_accuracy(sim_matrix, labels, top_n=5):
    hits = 0
    for i in range(len(labels)):
        top_idx = np.argsort(sim_matrix[i])[::-1][1:top_n+1]
        if labels[i] in labels[top_idx]:
            hits += 1
    return hits / len(labels)

cos_acc = cosine_pseudo_accuracy(cos_sim, y)
print(f"   [A] Cosine genre-match accuracy (top-5): {cos_acc:.4f}")

# Pseudo R2: compare similarity scores vs same-genre flag
sample_idx   = np.random.choice(len(df), 500, replace=False)
y_sim_true   = []
y_sim_pred   = []
for i in sample_idx:
    for j in sample_idx:
        if i != j:
            y_sim_true.append(1 if y[i] == y[j] else 0)
            y_sim_pred.append(float(cos_sim[i][j]))
cos_r2 = r2_score(y_sim_true, y_sim_pred)
print(f"   [A] Cosine R2 score: {cos_r2:.4f}")


# --- Model B: KNN Classifier ---
knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
knn.fit(X_train, y_train)
y_pred_knn  = knn.predict(X_test)
knn_acc     = accuracy_score(y_test, y_pred_knn)
knn_r2      = r2_score(y_test, y_pred_knn)
knn_cm      = confusion_matrix(y_test, y_pred_knn)
print(f"   [B] KNN Accuracy: {knn_acc:.4f}  |  R2: {knn_r2:.4f}")

# ROC-AUC (macro, one-vs-rest)
n_classes = len(np.unique(y))
y_bin     = label_binarize(y_test, classes=np.unique(y))
try:
    y_prob_knn = knn.predict_proba(X_test)
    knn_auc    = roc_auc_score(y_bin, y_prob_knn, multi_class="ovr", average="macro")
except:
    knn_auc = 0.0
print(f"   [B] KNN ROC-AUC: {knn_auc:.4f}")


# --- Model C: Naive Bayes ---
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
nb_acc    = accuracy_score(y_test, y_pred_nb)
nb_r2     = r2_score(y_test, y_pred_nb)
nb_cm     = confusion_matrix(y_test, y_pred_nb)
print(f"   [C] Naive Bayes Accuracy: {nb_acc:.4f}  |  R2: {nb_r2:.4f}")

try:
    y_prob_nb = nb.predict_proba(X_test)
    nb_auc    = roc_auc_score(y_bin, y_prob_nb, multi_class="ovr", average="macro")
except:
    nb_auc = 0.0
print(f"   [C] Naive Bayes ROC-AUC: {nb_auc:.4f}")


# ── Step 5 : Print Summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Model':<28} {'Accuracy':>10} {'R2 Score':>10} {'ROC-AUC':>10}")
print("-" * 60)
print(f"  {'Cosine Similarity (CB)':<28} {cos_acc:>10.4f} {cos_r2:>10.4f} {'  N/A':>10}")
print(f"  {'KNN (k=10, cosine)':<28} {knn_acc:>10.4f} {knn_r2:>10.4f} {knn_auc:>10.4f}")
print(f"  {'Multinomial Naive Bayes':<28} {nb_acc:>10.4f} {nb_r2:>10.4f} {nb_auc:>10.4f}")
print("=" * 60)

print("\n  KNN Classification Report:")
print(classification_report(y_test, y_pred_knn,
      target_names=le.classes_, zero_division=0))

print("  Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb,
      target_names=le.classes_, zero_division=0))


# ── Step 6 : Save artifacts ────────────────────────────────────────────────
print("[6/6] Saving model artifacts...")
os.makedirs("artifacts", exist_ok=True)

artifacts = {
    "df"           : df,
    "cos_sim"      : cos_sim,
    "knn"          : knn,
    "nb"           : nb,
    "cv"           : cv,
    "tfidf"        : tfidf,
    "le"           : le,
    "y_test"       : y_test,
    "y_pred_knn"   : y_pred_knn,
    "y_pred_nb"    : y_pred_nb,
    "y_prob_knn"   : y_prob_knn if knn_auc > 0 else None,
    "y_prob_nb"    : y_prob_nb  if nb_auc  > 0 else None,
    "knn_cm"       : knn_cm,
    "nb_cm"        : nb_cm,
    "metrics": {
        "cosine": {"accuracy": cos_acc, "r2": cos_r2, "auc": None},
        "knn"   : {"accuracy": knn_acc, "r2": knn_r2, "auc": knn_auc},
        "nb"    : {"accuracy": nb_acc,  "r2": nb_r2,  "auc": nb_auc},
    },
    "genre_counts" : genre_counts,
    "valid_genres" : valid_genres,
}

with open("artifacts/model_data.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("   Saved → artifacts/model_data.pkl")
print("\n✅ Training complete!\n")