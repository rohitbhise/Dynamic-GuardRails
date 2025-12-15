import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# --- Load jailbreak data ---
json_path = "jailbreak_data/jailbreak.json"

with open(json_path, "r") as f:
    json_data = json.load(f)
    data = [item["text"] for item in json_data]

print(f"Loaded {len(data)} jailbreak examples\n")

# --- Embed using SBERT ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(data, show_progress_bar=True)

print("\n" + "=" * 80)
print("TESTING DIFFERENT CLUSTERING APPROACHES")
print("=" * 80)

# --- Test different DBSCAN parameters ---
print("\n1. DBSCAN Parameter Tuning:")
print("-" * 80)

dbscan_configs = [
    {"eps": 0.3, "min_samples": 2},
    {"eps": 0.35, "min_samples": 2},
    {"eps": 0.4, "min_samples": 2},
    {"eps": 0.45, "min_samples": 2},
    {"eps": 0.5, "min_samples": 2},
    {"eps": 0.5, "min_samples": 3},
    {"eps": 0.55, "min_samples": 2},
    {"eps": 0.6, "min_samples": 2},
]

best_dbscan_score = -1
best_dbscan_config = None
best_dbscan_labels = None

for config in dbscan_configs:
    dbscan = DBSCAN(
        eps=config["eps"], min_samples=config["min_samples"], metric="cosine"
    )
    labels = dbscan.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Calculate silhouette score (only if we have clusters and not all noise)
    if n_clusters > 1 and n_noise < len(labels):
        # Filter out noise for silhouette calculation
        mask = labels != -1
        if sum(mask) > n_clusters:
            score = silhouette_score(embeddings[mask], labels[mask], metric="cosine")
        else:
            score = -1
    else:
        score = -1

    print(
        f"eps={config['eps']}, min_samples={config['min_samples']}: "
        f"{n_clusters} clusters, {n_noise} noise, silhouette={score:.3f}"
    )

    if score > best_dbscan_score and n_clusters >= 5:
        best_dbscan_score = score
        best_dbscan_config = config
        best_dbscan_labels = labels

# --- Test K-Means (fixed number of clusters) ---
print("\n2. K-Means Clustering:")
print("-" * 80)

best_kmeans_score = -1
best_kmeans_k = None
best_kmeans_labels = None

for k in range(5, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, metric="cosine")

    print(f"k={k}: silhouette={score:.3f}")

    if score > best_kmeans_score:
        best_kmeans_score = score
        best_kmeans_k = k
        best_kmeans_labels = labels

# --- Test Agglomerative Clustering ---
print("\n3. Agglomerative Clustering:")
print("-" * 80)

best_agg_score = -1
best_agg_k = None
best_agg_labels = None

for k in range(5, 11):
    agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    labels = agg.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, metric="cosine")

    print(f"k={k}: silhouette={score:.3f}")

    if score > best_agg_score:
        best_agg_score = score
        best_agg_k = k
        best_agg_labels = labels

# --- Summary ---
print("\n" + "=" * 80)
print("BEST RESULTS SUMMARY")
print("=" * 80)
print(
    f"\nBest DBSCAN: eps={best_dbscan_config['eps']}, min_samples={best_dbscan_config['min_samples']}, "
    f"silhouette={best_dbscan_score:.3f}"
)
print(f"Best K-Means: k={best_kmeans_k}, silhouette={best_kmeans_score:.3f}")
print(f"Best Agglomerative: k={best_agg_k}, silhouette={best_agg_score:.3f}")

# --- Choose the best overall method ---
best_method = "kmeans"
best_labels = best_kmeans_labels
best_score = best_kmeans_score

if best_agg_score > best_score:
    best_method = "agglomerative"
    best_labels = best_agg_labels
    best_score = best_agg_score

if best_dbscan_score > best_score:
    best_method = "dbscan"
    best_labels = best_dbscan_labels
    best_score = best_dbscan_score

print(f"\n*** Winner: {best_method.upper()} with silhouette={best_score:.3f} ***\n")

# --- Visualize best result ---
print("Generating visualizations with best clustering method...")

pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data) - 1))
embeddings_2d_tsne = tsne.fit_transform(embeddings)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

unique_labels = set(best_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# PCA Plot
for label, color in zip(unique_labels, colors):
    if label == -1:
        color = [0, 0, 0, 1]
        marker = "x"
        label_name = "Noise"
    else:
        marker = "o"
        label_name = f"Cluster {label}"

    mask = best_labels == label
    axes[0].scatter(
        embeddings_2d_pca[mask, 0],
        embeddings_2d_pca[mask, 1],
        c=[color],
        label=label_name,
        s=100,
        marker=marker,
        alpha=0.7,
        edgecolors="black" if marker == "o" else None,
        linewidth=0.5 if marker == "o" else 0,
    )

axes[0].set_title(
    f"Jailbreak Clusters - PCA ({best_method.upper()})", fontsize=14, fontweight="bold"
)
axes[0].set_xlabel("First Principal Component")
axes[0].set_ylabel("Second Principal Component")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# t-SNE Plot
for label, color in zip(unique_labels, colors):
    if label == -1:
        color = [0, 0, 0, 1]
        marker = "x"
        label_name = "Noise"
    else:
        marker = "o"
        label_name = f"Cluster {label}"

    mask = best_labels == label
    axes[1].scatter(
        embeddings_2d_tsne[mask, 0],
        embeddings_2d_tsne[mask, 1],
        c=[color],
        label=label_name,
        s=100,
        marker=marker,
        alpha=0.7,
        edgecolors="black" if marker == "o" else None,
        linewidth=0.5 if marker == "o" else 0,
    )

axes[1].set_title(
    f"Jailbreak Clusters - t-SNE ({best_method.upper()})",
    fontsize=14,
    fontweight="bold",
)
axes[1].set_xlabel("t-SNE Component 1")
axes[1].set_ylabel("t-SNE Component 2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
viz_path = "pipeline/improved_clustering.png"
plt.savefig(viz_path, dpi=300, bbox_inches="tight")
print(f"✓ Improved visualization saved to: {viz_path}")
plt.close()

# --- Save best clustering configuration ---
config_output = {
    "best_method": best_method,
    "silhouette_score": float(best_score),
    "n_clusters": len(set(best_labels)) - (1 if -1 in best_labels else 0),
    "n_noise": int(list(best_labels).count(-1)),
}

if best_method == "dbscan":
    config_output["parameters"] = best_dbscan_config
elif best_method == "kmeans":
    config_output["parameters"] = {"n_clusters": best_kmeans_k}
else:
    config_output["parameters"] = {"n_clusters": best_agg_k}

with open("pipeline/best_clustering_config.json", "w") as f:
    json.dump(config_output, f, indent=2)

print(f"✓ Best configuration saved to: pipeline/best_clustering_config.json")
print("\n" + "=" * 80)
