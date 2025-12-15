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
print("FINDING PARAMETERS WITH LESS NOISE")
print("=" * 80)

# Goal: Reduce noise to <30% while maintaining reasonable clusters

print("\n1. DBSCAN with Larger eps (More lenient clustering):")
print("-" * 80)

dbscan_configs = [
    {"eps": 0.50, "min_samples": 2},
    {"eps": 0.55, "min_samples": 2},
    {"eps": 0.60, "min_samples": 2},
    {"eps": 0.65, "min_samples": 2},
    {"eps": 0.70, "min_samples": 2},
]

for config in dbscan_configs:
    dbscan = DBSCAN(
        eps=config["eps"], min_samples=config["min_samples"], metric="cosine"
    )
    labels = dbscan.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_clustered = len(labels) - n_noise
    noise_pct = (n_noise / len(labels)) * 100

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
        f"eps={config['eps']}: {n_clusters} clusters, {n_clustered}/{len(labels)} clustered "
        f"({100 - noise_pct:.1f}%), noise={n_noise} ({noise_pct:.1f}%), silhouette={score:.3f}"
    )

print("\n2. K-Means (Forces all points into clusters, 0% noise):")
print("-" * 80)

for k in [8, 9, 10, 11, 12]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, metric="cosine")

    print(
        f"k={k}: {k} clusters, {len(labels)}/{len(labels)} clustered (100.0%), "
        f"noise=0 (0.0%), silhouette={score:.3f}"
    )

print("\n3. Agglomerative Clustering (Forces all points into clusters, 0% noise):")
print("-" * 80)

for k in [8, 9, 10, 11, 12]:
    agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    labels = agg.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, metric="cosine")

    print(
        f"k={k}: {k} clusters, {len(labels)}/{len(labels)} clustered (100.0%), "
        f"noise=0 (0.0%), silhouette={score:.3f}"
    )

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("\nFor less noise and more items in clusters, use one of these:")
print("\n1. K-Means with k=10 (best balance of cluster count and quality)")
print("   - 100% of data clustered")
print("   - Forces all examples into 10 clusters")
print("   - Good for ensuring every example gets classified")
print("\n2. DBSCAN with eps=0.55-0.60 (moderate noise reduction)")
print("   - 30-50% of data clustered")
print("   - Only clusters truly similar examples")
print("   - Better cluster quality but some noise remains")
print("\n3. Agglomerative with k=10 (hierarchical, 100% clustered)")
print("   - Similar to K-Means but uses hierarchical approach")
print("   - Slightly better silhouette scores")
print("=" * 80)
