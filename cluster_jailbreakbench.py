import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# --- Load JailbreakBench data ---
data = []
categories = []

jailbreakbench_path = "jailbreak_data/jailbreakbench.json"

print(f"Loading data from {jailbreakbench_path}")
with open(jailbreakbench_path, "r") as f:
    json_data = json.load(f)
    for item in json_data:
        data.append(item["text"])
        categories.append(item.get("category", "Unknown"))

print(f"Loaded {len(data)} harmful behavior examples")
print(f"\nCategory distribution:")
category_counts = {}
for cat in categories:
    category_counts[cat] = category_counts.get(cat, 0) + 1
for cat, count in sorted(category_counts.items()):
    print(f"  {cat}: {count} examples")

# --- Embed using SBERT ---
print("\nGenerating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(data, show_progress_bar=True)

# --- Cluster using DBSCAN (optimized for pattern discovery) ---
print(
    "\nClustering with DBSCAN (eps=0.4, min_samples=2) - optimized for tight patterns..."
)
dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
labels = dbscan.fit_predict(embeddings)

clusters = defaultdict(list)
cluster_categories = defaultdict(lambda: defaultdict(int))

for text, label, category in zip(data, labels, categories):
    if label == -1:
        continue  # ignore noise
    clusters[label].append(text)
    cluster_categories[label][category] += 1


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

    # Get the dominant categories for this cluster
    cat_dist = dict(cluster_categories[cid])

    cluster_patterns[int(cid)] = {
        "examples": texts[:5],
        "keywords": keywords,
        "size": len(texts),
        "category_distribution": cat_dist,
    }

# --- Save results ---
os.makedirs("pipeline", exist_ok=True)
output_path = "pipeline/jailbreakbench_clusters.json"

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

# Print cluster summary
print("Cluster Summary:")
print("-" * 80)
for cid in sorted(cluster_patterns.keys()):
    cluster = cluster_patterns[cid]
    print(f"\nCluster {cid} ({cluster['size']} examples):")
    print(f"  Top keywords: {', '.join(cluster['keywords'][:5])}")
    print(f"  Categories: {cluster['category_distribution']}")
    print(f"  Example: {cluster['examples'][0][:80]}...")

# --- Visualize clusters ---
print("\n" + "=" * 60)
print("Generating visualizations...")
print("=" * 60)

# Reduce dimensions for visualization using t-SNE only
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data) - 1))
embeddings_2d_tsne = tsne.fit_transform(embeddings)

# Create single t-SNE plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

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
    ax.scatter(
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

ax.set_title("JailbreakBench Clusters (t-SNE)", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
viz_path = "pipeline/jailbreakbench_visualization.png"
plt.savefig(viz_path, dpi=300, bbox_inches="tight")
print(f"✓ Visualization saved to: {viz_path}")
plt.close()

# --- Create a bar chart showing cluster sizes ---
fig, ax = plt.subplots(figsize=(10, 6))

cluster_ids = sorted(cluster_patterns.keys())
cluster_sizes = [cluster_patterns[cid]["size"] for cid in cluster_ids]

bars = ax.bar(
    cluster_ids,
    cluster_sizes,
    color=plt.cm.Spectral(np.linspace(0, 1, len(cluster_ids))),
)
ax.set_xlabel("Cluster ID", fontsize=12)
ax.set_ylabel("Number of Examples", fontsize=12)
ax.set_title("JailbreakBench Cluster Size Distribution", fontsize=14, fontweight="bold")
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
size_viz_path = "pipeline/jailbreakbench_cluster_sizes.png"
plt.savefig(size_viz_path, dpi=300, bbox_inches="tight")
print(f"✓ Cluster size chart saved to: {size_viz_path}")
plt.close()

print(f"\n{'=' * 60}")
print("All visualizations complete!")
print(f"{'=' * 60}\n")

# --- Generate Colang rules from clusters ---
print("=" * 60)
print("Generating Colang guardrail rules...")
print("=" * 60)

colang_rules = []

# Generate a flow definition for each cluster
for cid in sorted(cluster_patterns.keys()):
    cluster = cluster_patterns[cid]
    keywords = cluster["keywords"][:10]  # Use top 10 keywords

    # Create the user definition for this cluster
    rule = f"define user cluster_{cid}_attack\n"

    # Add keywords as a list
    keywords_str = " ".join([f'"{kw}"' for kw in keywords])
    rule += f"    {keywords_str}\n"

    # Add regex pattern
    regex_pattern = "|".join(keywords)
    rule += f"    (/{regex_pattern}/i)\n"

    colang_rules.append(rule)

# Write to config/flows/jailbreak_auto.co
flows_output_path = "config/flows/jailbreak_auto.co"
with open(flows_output_path, "w") as f:
    f.write("# Auto-generated Colang rules from clustering\n")
    f.write("# Generated by cluster_jailbreakbench.py\n\n")
    f.write("\n".join(colang_rules))

print(f"✓ Generated {len(cluster_patterns)} cluster definitions")
print(f"✓ Colang rules saved to: {flows_output_path}")

# --- Update rails.co to include all cluster flows ---
rails_path = "config/rails.co"

# Read existing rails.co to preserve other flows
with open(rails_path, "r") as f:
    existing_content = f.read()

# Extract non-jailbreak flows (greeting, etc.)
lines = existing_content.split("\n")
preserved_lines = []
skip_jailbreak = False

for line in lines:
    if "jailbreak" in line.lower() and "define" in line.lower():
        skip_jailbreak = True
        continue
    if skip_jailbreak and line.strip() and not line.startswith(" "):
        skip_jailbreak = False
    if not skip_jailbreak:
        preserved_lines.append(line)

# Remove trailing empty lines
while preserved_lines and not preserved_lines[-1].strip():
    preserved_lines.pop()

# Generate new jailbreak flows
jailbreak_flows = []
jailbreak_flows.append("# Jailbreak detection flow")
jailbreak_flows.append("define bot refuse jailbreak")
jailbreak_flows.append(
    '  "I\'m unable to help with that request. I detected an attack. Please ask something else."'
)
jailbreak_flows.append("")

for cid in sorted(cluster_patterns.keys()):
    jailbreak_flows.append(f"define flow jailbreak_detection_cluster_{cid}")
    jailbreak_flows.append(f"  user cluster_{cid}_attack")
    jailbreak_flows.append("  bot refuse jailbreak")
    jailbreak_flows.append("  stop")
    jailbreak_flows.append("")

# Write updated rails.co
with open(rails_path, "w") as f:
    f.write("\n".join(preserved_lines))
    f.write("\n\n")
    f.write("\n".join(jailbreak_flows))

print(f"✓ Updated {len(cluster_patterns)} jailbreak flows in: {rails_path}")

print("\n" + "=" * 60)
print("✓ GUARDRAILS CONFIGURATION UPDATED!")
print("=" * 60)
print(f"Total clusters: {len(cluster_patterns)}")
print(f"Flows updated: config/flows/jailbreak_auto.co")
print(f"Rails updated: config/rails.co")
print("=" * 60 + "\n")
