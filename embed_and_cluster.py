import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# --- Load jailbreak data ---
data = []

# Try to load from JSONL first, then fall back to JSON
jsonl_path = "jailbreak_data/jailbreak_attempts.jsonl"
json_path = "jailbreak_data/jailbreak.json"

if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0:
    print(f"Loading data from {jsonl_path}")
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line)["text"])
elif os.path.exists(json_path):
    print(f"Loading data from {json_path}")
    with open(json_path, "r") as f:
        json_data = json.load(f)
        for item in json_data:
            data.append(item["text"])
else:
    raise FileNotFoundError(
        "No jailbreak data file found. Please create either jailbreak_attempts.jsonl or jailbreak.json"
    )

print(f"Loaded {len(data)} jailbreak examples")

# --- Embed using SBERT ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(data, show_progress_bar=True)

# --- Cluster using DBSCAN ---
dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
labels = dbscan.fit_predict(embeddings)

clusters = defaultdict(list)
for text, label in zip(data, labels):
    if label == -1:
        continue  # ignore noise
    clusters[label].append(text)


# --- Extract TF-IDF keywords for each cluster ---
def extract_keywords(texts, top_k=10):
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(texts)
    scores = np.asarray(X.sum(axis=0)).flatten()
    idx = scores.argsort()[::-1][:top_k]
    keywords = [tfidf.get_feature_names_out()[i] for i in idx]
    return keywords


cluster_patterns = {}

for cid, texts in clusters.items():
    keywords = extract_keywords(texts)
    cluster_patterns[int(cid)] = {"examples": texts[:5], "keywords": keywords}

# --- Save results ---
os.makedirs("pipeline", exist_ok=True)
output_path = "pipeline/cluster_patterns.json"

with open(output_path, "w") as f:
    json.dump(cluster_patterns, f, indent=2)

print(f"\n{'=' * 60}")
print(f"✓ Clustering complete!")
print(f"{'=' * 60}")
print(f"Total examples processed: {len(data)}")
print(f"Number of clusters found: {len(cluster_patterns)}")
print(f"Noise points (unclustered): {list(labels).count(-1)}")
print(f"Output saved to: {output_path}")
print(f"{'=' * 60}\n")

# --- Visualize clusters ---
print("Generating visualizations...")

# Reduce dimensions for visualization (384 dims -> 2 dims)
# Method 1: PCA (faster, linear)
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)

# Method 2: t-SNE (slower, non-linear, better for clusters)
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data) - 1))
embeddings_2d_tsne = tsne.fit_transform(embeddings)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: PCA
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points in black
        color = [0, 0, 0, 1]
        marker = "x"
        label_name = "Noise"
    else:
        marker = "o"
        label_name = f"Cluster {label}"

    mask = labels == label
    axes[0].scatter(
        embeddings_2d_pca[mask, 0],
        embeddings_2d_pca[mask, 1],
        c=[color],
        label=label_name,
        s=100,
        marker=marker,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

axes[0].set_title("Jailbreak Clusters (PCA)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("First Principal Component")
axes[0].set_ylabel("Second Principal Component")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: t-SNE
for label, color in zip(unique_labels, colors):
    if label == -1:
        color = [0, 0, 0, 1]
        marker = "x"
        label_name = "Noise"
    else:
        marker = "o"
        label_name = f"Cluster {label}"

    mask = labels == label
    axes[1].scatter(
        embeddings_2d_tsne[mask, 0],
        embeddings_2d_tsne[mask, 1],
        c=[color],
        label=label_name,
        s=100,
        marker=marker,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

axes[1].set_title("Jailbreak Clusters (t-SNE)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("t-SNE Component 1")
axes[1].set_ylabel("t-SNE Component 2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
viz_path = "pipeline/cluster_visualization.png"
plt.savefig(viz_path, dpi=300, bbox_inches="tight")
print(f"✓ Visualization saved to: {viz_path}")
plt.close()

# --- Create a bar chart showing cluster sizes ---
fig, ax = plt.subplots(figsize=(10, 6))

cluster_ids = list(cluster_patterns.keys())
cluster_sizes = [len(clusters[int(cid)]) for cid in cluster_ids]

bars = ax.bar(
    cluster_ids,
    cluster_sizes,
    color=plt.cm.Spectral(np.linspace(0, 1, len(cluster_ids))),
)
ax.set_xlabel("Cluster ID", fontsize=12)
ax.set_ylabel("Number of Examples", fontsize=12)
ax.set_title("Cluster Size Distribution", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Add noise count as text
noise_count = list(labels).count(-1)
ax.text(
    0.98,
    0.98,
    f"Noise points: {noise_count}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    fontsize=10,
)

plt.tight_layout()
size_viz_path = "pipeline/cluster_sizes.png"
plt.savefig(size_viz_path, dpi=300, bbox_inches="tight")
print(f"✓ Cluster size chart saved to: {size_viz_path}")
plt.close()

print(f"\n{'=' * 60}")
print("All visualizations complete!")
print(f"{'=' * 60}\n")
