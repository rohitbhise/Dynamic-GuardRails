import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
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
print("OPTIMIZING DBSCAN FOR MAXIMUM PATTERN DISCOVERY")
print("=" * 80)
print(
    "\nGoal: Find tight, meaningful patterns (high silhouette) with reasonable coverage"
)

# Test a range of eps values with min_samples=2 and 3
configs = []
for eps in np.arange(0.30, 0.70, 0.05):
    for min_samples in [2, 3]:
        configs.append({"eps": round(eps, 2), "min_samples": min_samples})

results = []

print("\nTesting DBSCAN configurations:")
print("-" * 80)
print(
    f"{'eps':>5} {'min':>4} {'clusters':>8} {'clustered':>10} {'coverage':>9} {'noise':>6} {'silhouette':>11}"
)
print("-" * 80)

for config in configs:
    dbscan = DBSCAN(
        eps=config["eps"], min_samples=config["min_samples"], metric="cosine"
    )
    labels = dbscan.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_clustered = len(labels) - n_noise
    coverage_pct = (n_clustered / len(labels)) * 100

    # Calculate silhouette score
    if n_clusters > 1 and n_noise < len(labels):
        mask = labels != -1
        if sum(mask) > n_clusters:
            score = silhouette_score(embeddings[mask], labels[mask], metric="cosine")
        else:
            score = -1
    else:
        score = -1

    results.append(
        {
            "config": config,
            "n_clusters": n_clusters,
            "n_clustered": n_clustered,
            "coverage_pct": coverage_pct,
            "n_noise": n_noise,
            "silhouette": score,
        }
    )

    print(
        f"{config['eps']:>5.2f} {config['min_samples']:>4} {n_clusters:>8} "
        f"{n_clustered:>4}/{len(labels):>3} {coverage_pct:>8.1f}% "
        f"{n_noise:>6} {score:>11.3f}"
    )

# --- Find best configurations based on different criteria ---
print("\n" + "=" * 80)
print("TOP CONFIGURATIONS")
print("=" * 80)

# 1. Best silhouette score (tightest patterns)
valid_results = [r for r in results if r["silhouette"] > 0 and r["n_clusters"] >= 3]
if valid_results:
    best_silhouette = max(valid_results, key=lambda x: x["silhouette"])
    print(f"\n1. TIGHTEST PATTERNS (Best Silhouette):")
    print(
        f"   eps={best_silhouette['config']['eps']}, min_samples={best_silhouette['config']['min_samples']}"
    )
    print(
        f"   → {best_silhouette['n_clusters']} clusters, {best_silhouette['n_clustered']}/{len(data)} clustered "
        f"({best_silhouette['coverage_pct']:.1f}%), silhouette={best_silhouette['silhouette']:.3f}"
    )

# 2. Best balance (reasonable silhouette + coverage)
balanced_results = [
    r for r in results if r["silhouette"] > 0.2 and r["coverage_pct"] > 20
]
if balanced_results:
    best_balanced = max(
        balanced_results, key=lambda x: x["silhouette"] * (x["coverage_pct"] / 100)
    )
    print(f"\n2. BEST BALANCE (Quality × Coverage):")
    print(
        f"   eps={best_balanced['config']['eps']}, min_samples={best_balanced['config']['min_samples']}"
    )
    print(
        f"   → {best_balanced['n_clusters']} clusters, {best_balanced['n_clustered']}/{len(data)} clustered "
        f"({best_balanced['coverage_pct']:.1f}%), silhouette={best_balanced['silhouette']:.3f}"
    )

# 3. Most clusters with good quality
good_quality = [r for r in results if r["silhouette"] > 0.25]
if good_quality:
    most_clusters = max(good_quality, key=lambda x: x["n_clusters"])
    print(f"\n3. MOST PATTERNS (High-quality clusters):")
    print(
        f"   eps={most_clusters['config']['eps']}, min_samples={most_clusters['config']['min_samples']}"
    )
    print(
        f"   → {most_clusters['n_clusters']} clusters, {most_clusters['n_clustered']}/{len(data)} clustered "
        f"({most_clusters['coverage_pct']:.1f}%), silhouette={most_clusters['silhouette']:.3f}"
    )

# 4. Maximum coverage with acceptable quality
acceptable = [r for r in results if r["silhouette"] > 0.15 and r["n_clusters"] >= 5]
if acceptable:
    max_coverage = max(acceptable, key=lambda x: x["coverage_pct"])
    print(f"\n4. MAXIMUM COVERAGE (Acceptable quality):")
    print(
        f"   eps={max_coverage['config']['eps']}, min_samples={max_coverage['config']['min_samples']}"
    )
    print(
        f"   → {max_coverage['n_clusters']} clusters, {max_coverage['n_clustered']}/{len(data)} clustered "
        f"({max_coverage['coverage_pct']:.1f}%), silhouette={max_coverage['silhouette']:.3f}"
    )

# --- Recommendation ---
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if valid_results:
    # Choose best silhouette for pattern discovery
    recommended = best_silhouette
    print(f"\nFor pattern discovery, use:")
    print(
        f"  eps={recommended['config']['eps']}, min_samples={recommended['config']['min_samples']}"
    )
    print(f"\nThis will find {recommended['n_clusters']} tight, meaningful patterns")
    print(
        f"covering {recommended['n_clustered']}/{len(data)} examples ({recommended['coverage_pct']:.1f}%)"
    )
    print(f"with high cluster quality (silhouette={recommended['silhouette']:.3f})")
    print(
        f"\nThe remaining {recommended['n_noise']} examples don't fit clear patterns (noise is OK!)"
    )

    # Save recommendation
    with open("pipeline/dbscan_recommendation.json", "w") as f:
        json.dump(
            {
                "recommended_config": recommended["config"],
                "metrics": {
                    "n_clusters": recommended["n_clusters"],
                    "n_clustered": recommended["n_clustered"],
                    "coverage_pct": recommended["coverage_pct"],
                    "n_noise": recommended["n_noise"],
                    "silhouette": recommended["silhouette"],
                },
            },
            f,
            indent=2,
        )

    print(f"\n✓ Recommendation saved to: pipeline/dbscan_recommendation.json")
else:
    print("\nNo valid configurations found. The data may not have strong patterns.")

print("=" * 80)
