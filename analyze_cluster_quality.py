import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

# --- Load data ---
with open("jailbreak_data/jailbreakbench.json", "r") as f:
    json_data = json.load(f)
    data = [item["text"] for item in json_data]
    categories = [item["category"] for item in json_data]

print(f"Loaded {len(data)} examples")

# --- Embed ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(data, show_progress_bar=True)

# --- Cluster ---
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

# --- Calculate silhouette scores ---
silhouette_avg = silhouette_score(embeddings, labels, metric="cosine")
sample_silhouette_values = silhouette_samples(embeddings, labels, metric="cosine")

print(f"\nOverall Silhouette Score: {silhouette_avg:.3f}")
print("(Closer to 1.0 = well-separated clusters, closer to 0 = overlapping)")

# --- Calculate per-cluster statistics ---
print("\nPer-Cluster Analysis:")
print("-" * 80)
for i in range(10):
    cluster_silhouette_values = sample_silhouette_values[labels == i]
    print(
        f"Cluster {i}: size={sum(labels == i):2d}, "
        f"avg_silhouette={cluster_silhouette_values.mean():.3f}, "
        f"min={cluster_silhouette_values.min():.3f}, "
        f"max={cluster_silhouette_values.max():.3f}"
    )

# --- Reduce dimensions ---
print("\nReducing dimensions for visualization...")
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d_tsne = tsne.fit_transform(embeddings)

# --- Create comprehensive visualization ---
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. PCA visualization
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(
    embeddings_2d_pca[:, 0],
    embeddings_2d_pca[:, 1],
    c=labels,
    cmap="tab10",
    s=100,
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
ax1.set_title("PCA Projection", fontsize=14, fontweight="bold")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label="Cluster")

# 2. t-SNE visualization
ax2 = fig.add_subplot(gs[0, 1])
scatter = ax2.scatter(
    embeddings_2d_tsne[:, 0],
    embeddings_2d_tsne[:, 1],
    c=labels,
    cmap="tab10",
    s=100,
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
ax2.set_title("t-SNE Projection", fontsize=14, fontweight="bold")
ax2.set_xlabel("t-SNE 1")
ax2.set_ylabel("t-SNE 2")
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label="Cluster")

# 3. Silhouette score visualization (shows overlap)
ax3 = fig.add_subplot(gs[0, 2])
y_lower = 10
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i in range(10):
    cluster_silhouette_values = sample_silhouette_values[labels == i]
    cluster_silhouette_values.sort()

    size_cluster_i = cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    ax3.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        cluster_silhouette_values,
        facecolor=colors[i],
        edgecolor=colors[i],
        alpha=0.7,
    )

    ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax3.set_title("Silhouette Plot (Overlap Analysis)", fontsize=14, fontweight="bold")
ax3.set_xlabel("Silhouette Coefficient")
ax3.set_ylabel("Cluster")
ax3.axvline(
    x=silhouette_avg, color="red", linestyle="--", label=f"Avg: {silhouette_avg:.3f}"
)
ax3.set_yticks([])
ax3.legend()

# 4. PCA with cluster boundaries
ax4 = fig.add_subplot(gs[1, 0])
for i in range(10):
    mask = labels == i
    ax4.scatter(
        embeddings_2d_pca[mask, 0],
        embeddings_2d_pca[mask, 1],
        c=[colors[i]],
        label=f"C{i}",
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
ax4.set_title("PCA with Cluster Separation", fontsize=14, fontweight="bold")
ax4.set_xlabel("PC1")
ax4.set_ylabel("PC2")
ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. t-SNE with cluster boundaries
ax5 = fig.add_subplot(gs[1, 1])
for i in range(10):
    mask = labels == i
    ax5.scatter(
        embeddings_2d_tsne[mask, 0],
        embeddings_2d_tsne[mask, 1],
        c=[colors[i]],
        label=f"C{i}",
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
ax5.set_title("t-SNE with Cluster Separation", fontsize=14, fontweight="bold")
ax5.set_xlabel("t-SNE 1")
ax5.set_ylabel("t-SNE 2")
ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. Cluster size distribution
ax6 = fig.add_subplot(gs[1, 2])
cluster_sizes = [sum(labels == i) for i in range(10)]
bars = ax6.bar(range(10), cluster_sizes, color=colors, edgecolor="black", linewidth=1)
ax6.set_title("Cluster Size Distribution", fontsize=14, fontweight="bold")
ax6.set_xlabel("Cluster")
ax6.set_ylabel("Number of Examples")
ax6.grid(True, alpha=0.3, axis="y")
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax6.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

# 7. Category distribution per cluster (heatmap)
ax7 = fig.add_subplot(gs[2, :])
category_names = sorted(set(categories))
cluster_category_matrix = np.zeros((10, len(category_names)))

for i in range(10):
    cluster_mask = labels == i
    cluster_categories = [
        categories[j] for j in range(len(categories)) if cluster_mask[j]
    ]
    for cat in cluster_categories:
        cat_idx = category_names.index(cat)
        cluster_category_matrix[i, cat_idx] += 1

sns.heatmap(
    cluster_category_matrix,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    xticklabels=category_names,
    yticklabels=[f"Cluster {i}" for i in range(10)],
    ax=ax7,
    cbar_kws={"label": "Count"},
)
ax7.set_title(
    "Category Distribution Across Clusters (Shows Semantic Overlap)",
    fontsize=14,
    fontweight="bold",
)
ax7.set_xlabel("Original Category")
ax7.set_ylabel("Assigned Cluster")
plt.setp(ax7.get_xticklabels(), rotation=45, ha="right")

plt.suptitle(
    f"Cluster Quality Analysis - JailbreakBench (K-Means, k=10)\nOverall Silhouette Score: {silhouette_avg:.3f}",
    fontsize=16,
    fontweight="bold",
)

plt.savefig("pipeline/cluster_quality_analysis.png", dpi=300, bbox_inches="tight")
print(
    f"\n✓ Comprehensive quality analysis saved to: pipeline/cluster_quality_analysis.png"
)

# --- Print interpretation ---
print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("\nSilhouette Score Interpretation:")
print("  0.7 - 1.0: Strong, well-separated clusters")
print("  0.5 - 0.7: Reasonable structure")
print("  0.25 - 0.5: Weak structure, significant overlap")
print("  < 0.25: No substantial structure, heavy overlap")
print(
    f"\nYour score: {silhouette_avg:.3f} - {'Weak structure with significant overlap' if silhouette_avg < 0.25 else 'Some structure but overlapping'}"
)

print("\nSilhouette Plot:")
print("  - Width of bars = quality of assignment")
print("  - Thin/negative bars = points closer to other clusters (overlap)")
print("  - Bars left of avg line = poorly clustered points")

print("\nHeatmap:")
print("  - Dark cells = multiple categories in one cluster (semantic overlap)")
print("  - Diagonal pattern would indicate perfect category alignment")
print("  - Scattered pattern = categories mixed across clusters")
print("=" * 80)
