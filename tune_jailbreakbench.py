import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# --- Load JailbreakBench data ---
jailbreakbench_path = "jailbreak_data/jailbreakbench.json"

print(f"Loading data from {jailbreakbench_path}")
with open(jailbreakbench_path, "r") as f:
    json_data = json.load(f)
    data = [item["text"] for item in json_data]

print(f"Loaded {len(data)} harmful behavior examples\n")

# --- Embed using SBERT ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(data, show_progress_bar=True)

print("\n" + "=" * 80)
print("TESTING DIFFERENT CLUSTERING APPROACHES FOR JAILBREAKBENCH")
print("=" * 80)

# --- Test different DBSCAN parameters ---
print("\n1. DBSCAN Parameter Tuning:")
print("-" * 80)

dbscan_configs = [
    {"eps": 0.4, "min_samples": 2},
    {"eps": 0.45, "min_samples": 2},
    {"eps": 0.5, "min_samples": 2},
    {"eps": 0.55, "min_samples": 2},
    {"eps": 0.6, "min_samples": 2},
    {"eps": 0.65, "min_samples": 2},
    {"eps": 0.7, "min_samples": 2},
    {"eps": 0.75, "min_samples": 2},
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

    # Calculate silhouette score
    if n_clusters > 1 and n_noise < len(labels):
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

# --- Test K-Means ---
print("\n2. K-Means Clustering:")
print("-" * 80)

best_kmeans_score = -1
best_kmeans_k = None
best_kmeans_labels = None

for k in range(8, 13):
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

for k in range(8, 13):
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

if best_dbscan_config:
    print(
        f"\nBest DBSCAN: eps={best_dbscan_config['eps']}, min_samples={best_dbscan_config['min_samples']}, "
        f"silhouette={best_dbscan_score:.3f}"
    )
else:
    print("\nBest DBSCAN: No valid configuration found (less than 5 clusters)")

print(f"Best K-Means: k={best_kmeans_k}, silhouette={best_kmeans_score:.3f}")
print(f"Best Agglomerative: k={best_agg_k}, silhouette={best_agg_score:.3f}")

# Choose the best overall method
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

print(f"\n*** Winner: {best_method.upper()} with silhouette={best_score:.3f} ***")

# Save best config
config_output = {
    "best_method": best_method,
    "silhouette_score": float(best_score),
    "n_clusters": len(set(best_labels)) - (1 if -1 in best_labels else 0),
}

if best_method == "dbscan":
    config_output["parameters"] = best_dbscan_config
elif best_method == "kmeans":
    config_output["parameters"] = {"n_clusters": best_kmeans_k}
else:
    config_output["parameters"] = {"n_clusters": best_agg_k}

with open("pipeline/jailbreakbench_best_config.json", "w") as f:
    json.dump(config_output, f, indent=2)

print(f"\n✓ Best configuration saved to: pipeline/jailbreakbench_best_config.json")
print("=" * 80)
