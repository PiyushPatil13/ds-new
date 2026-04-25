"""
visualizer.py
=============
Generates all visualizations from trained model artifacts:
  - Confusion Matrix (KNN & Naive Bayes)
  - ROC-AUC Curve (multi-class)
  - Model Accuracy Comparison Bar Chart
  - Genre Distribution
  - Similarity Heatmap (sample)
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import os

# ── Load artifacts ──────────────────────────────────────────────────────────
ARTIFACTS_PATH = "artifacts/model_data.pkl"

if not os.path.exists(ARTIFACTS_PATH):
    raise FileNotFoundError(
        "❌  artifacts/model_data.pkl not found!\n"
        "   Please run training.py first."
    )

with open(ARTIFACTS_PATH, "rb") as f:
    data = pickle.load(f)

df          = data["df"]
cos_sim     = data["cos_sim"]
knn_cm      = data["knn_cm"]
nb_cm       = data["nb_cm"]
y_test      = data["y_test"]
y_pred_knn  = data["y_pred_knn"]
y_pred_nb   = data["y_pred_nb"]
y_prob_knn  = data["y_prob_knn"]
y_prob_nb   = data["y_prob_nb"]
le          = data["le"]
metrics     = data["metrics"]
genre_counts= data["genre_counts"]
valid_genres= data["valid_genres"]

classes     = le.classes_
n_classes   = len(classes)
os.makedirs("artifacts", exist_ok=True)

PALETTE = ["#e50914", "#f5c518", "#00b4d8", "#06d6a0", "#ff6b6b",
           "#c77dff", "#ffb703", "#fb8500", "#8ecae6", "#219ebc"]

print("Generating visualizations...")


# ── 1. Confusion Matrices ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0f0f1a")

for ax, cm, title, color in zip(
    axes,
    [knn_cm, nb_cm],
    ["KNN (k=10, Cosine)", "Multinomial Naive Bayes"],
    ["Blues", "Oranges"]
):
    ax.set_facecolor("#1a1a2e")
    sns.heatmap(cm, annot=True, fmt="d", cmap=color,
                xticklabels=classes, yticklabels=classes,
                ax=ax, cbar=True, linewidths=0.5)
    ax.set_title(f"Confusion Matrix — {title}", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Predicted", color="#aaa")
    ax.set_ylabel("Actual", color="#aaa")
    ax.tick_params(colors="white", rotation=45)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

plt.tight_layout(pad=3)
plt.savefig("artifacts/confusion_matrices.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ confusion_matrices.png")


# ── 2. ROC-AUC Curves ─────────────────────────────────────────────────────
if y_prob_knn is not None and y_prob_nb is not None:
    classes_present = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=classes_present)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("ROC-AUC Curves (One-vs-Rest per Genre)", color="white", fontsize=14)

    for ax, y_prob, model_name, model_color in zip(
        axes,
        [y_prob_knn, y_prob_nb],
        ["KNN (k=10)", "Naive Bayes"],
        ["#00b4d8", "#f5c518"]
    ):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(model_name, color="white", fontsize=12)
        ax.plot([0,1],[0,1], "w--", lw=1, alpha=0.4)

        # per-class curves
        for i, class_idx in enumerate(classes_present):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, class_idx])
            roc_auc = auc(fpr, tpr)
            class_name = le.classes_[class_idx]
            color = PALETTE[i % len(PALETTE)]
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{class_name} (AUC={roc_auc:.2f})")

        ax.set_xlabel("False Positive Rate", color="#aaa")
        ax.set_ylabel("True Positive Rate", color="#aaa")
        ax.tick_params(colors="white")
        ax.legend(fontsize=7, loc="lower right",
                  facecolor="#111", labelcolor="white",
                  framealpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    plt.tight_layout(pad=3)
    plt.savefig("artifacts/roc_auc_curves.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("  ✓ roc_auc_curves.png")


# ── 3. Model Comparison Bar Chart ─────────────────────────────────────────
model_names = ["Cosine Similarity\n(Content-Based)", "KNN\n(k=10, Cosine)", "Naive Bayes\n(Multinomial)"]
accuracy    = [metrics["cosine"]["accuracy"], metrics["knn"]["accuracy"], metrics["nb"]["accuracy"]]
r2_scores   = [max(0, metrics["cosine"]["r2"]), metrics["knn"]["r2"],     metrics["nb"]["r2"]]
auc_scores  = [0, metrics["knn"]["auc"] or 0, metrics["nb"]["auc"] or 0]

x = np.arange(len(model_names))
width = 0.26

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#1a1a2e")

b1 = ax.bar(x - width, accuracy,   width, label="Accuracy",  color="#e50914", alpha=0.9, edgecolor="#111")
b2 = ax.bar(x,          r2_scores,  width, label="R² Score",  color="#f5c518", alpha=0.9, edgecolor="#111")
b3 = ax.bar(x + width,  auc_scores, width, label="ROC-AUC",   color="#00b4d8", alpha=0.9, edgecolor="#111")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", color="white", fontsize=9)

ax.set_title("Model Comparison — Accuracy vs R² vs ROC-AUC",
             color="white", fontsize=14, pad=16)
ax.set_xticks(x)
ax.set_xticklabels(model_names, color="white", fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", color="#aaa")
ax.tick_params(colors="white")
ax.legend(facecolor="#111", labelcolor="white", fontsize=10)
ax.yaxis.grid(True, color="#333", linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

# Highlight winner
best_idx = np.argmax(accuracy)
ax.annotate("🏆 Best", xy=(x[best_idx] - width, accuracy[best_idx] + 0.04),
            ha="center", color="#e50914", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("artifacts/model_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ model_comparison.png")


# ── 4. Genre Distribution ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0f0f1a")

top_genres = genre_counts[genre_counts.index.isin(valid_genres)].sort_values(ascending=False)

# Bar
ax = axes[0]
ax.set_facecolor("#1a1a2e")
colors = PALETTE[:len(top_genres)]
bars = ax.barh(top_genres.index, top_genres.values, color=colors, edgecolor="#111", alpha=0.9)
ax.set_title("Genre Distribution in Dataset", color="white", fontsize=13, pad=12)
ax.set_xlabel("Number of Movies", color="#aaa")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
for bar, val in zip(bars, top_genres.values):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2,
            str(val), va="center", color="white", fontsize=9)

# Pie
ax = axes[1]
ax.set_facecolor("#1a1a2e")
wedge_props = dict(width=0.6, edgecolor="#0f0f1a", linewidth=2)
ax.pie(top_genres.values, labels=top_genres.index,
       autopct="%1.1f%%", colors=PALETTE[:len(top_genres)],
       wedgeprops=wedge_props, textprops={"color": "white", "fontsize": 8},
       pctdistance=0.75, startangle=140)
ax.set_title("Genre Share (%)", color="white", fontsize=13, pad=12)

plt.tight_layout(pad=3)
plt.savefig("artifacts/genre_distribution.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ genre_distribution.png")


# ── 5. Similarity Heatmap (40-movie sample) ───────────────────────────────
sample_n  = min(40, len(df))
sample_idx = np.random.choice(len(df), sample_n, replace=False)
sample_sim = cos_sim[np.ix_(sample_idx, sample_idx)]
sample_titles = [df["title"].iloc[i][:18] for i in sample_idx]

fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#1a1a2e")
sns.heatmap(sample_sim, xticklabels=sample_titles,
            yticklabels=sample_titles, cmap="YlOrRd",
            ax=ax, vmin=0, vmax=1, linewidths=0.2,
            cbar_kws={"shrink": 0.8})
ax.set_title("Cosine Similarity Heatmap (40-Movie Sample)",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="white", labelsize=6)
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

plt.tight_layout()
plt.savefig("artifacts/similarity_heatmap.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ similarity_heatmap.png")


print("\n✅  All visualizations saved to artifacts/")