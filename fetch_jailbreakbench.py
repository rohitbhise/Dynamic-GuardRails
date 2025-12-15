import json
import os

import pandas as pd

print("=" * 80)
print("FETCHING JAILBREAKBENCH DATASET")
print("=" * 80)

# --- Fetch JailbreakBench datasets ---
splits = {
    "harmful": "data/harmful-behaviors.csv",
    "benign": "data/benign-behaviors.csv",
}

print("\nDownloading harmful behaviors dataset...")
df_harmful = pd.read_csv(
    "hf://datasets/JailbreakBench/JBB-Behaviors/" + splits["harmful"]
)

print(f"✓ Loaded {len(df_harmful)} harmful behavior examples")
print(f"\nDataset columns: {list(df_harmful.columns)}")
print(f"\nFirst few examples:")
print(df_harmful.head())

# --- Convert to our JSON format ---
print("\n" + "-" * 80)
print("Converting to JSON format...")

# Create data structure similar to our existing format
jailbreak_data = []

for idx, row in df_harmful.iterrows():
    # Assuming the dataset has a 'Goal' or 'Behavior' column
    # We'll check which columns exist
    if "Goal" in df_harmful.columns:
        text = row["Goal"]
    elif "Behavior" in df_harmful.columns:
        text = row["Behavior"]
    elif "goal" in df_harmful.columns:
        text = row["goal"]
    elif "behavior" in df_harmful.columns:
        text = row["behavior"]
    else:
        # Use the first text column we can find
        text_columns = df_harmful.select_dtypes(include=["object"]).columns
        if len(text_columns) > 0:
            text = row[text_columns[0]]
        else:
            continue

    # Create entry with text and metadata
    entry = {"text": str(text).strip()}

    # Add category if available
    if "Category" in df_harmful.columns:
        entry["category"] = str(row["Category"])
    elif "category" in df_harmful.columns:
        entry["category"] = str(row["category"])

    # Add source metadata
    entry["source"] = "JailbreakBench"

    jailbreak_data.append(entry)

print(f"✓ Converted {len(jailbreak_data)} examples")

# --- Save to JSON file ---
os.makedirs("jailbreak_data", exist_ok=True)
output_path = "jailbreak_data/jailbreakbench.json"

with open(output_path, "w") as f:
    json.dump(jailbreak_data, f, indent=2)

print(f"✓ Saved to: {output_path}")

# --- Download benign behaviors as well (optional) ---
print("\n" + "-" * 80)
print("Downloading benign behaviors dataset...")

try:
    df_benign = pd.read_csv(
        "hf://datasets/JailbreakBench/JBB-Behaviors/" + splits["benign"]
    )
    print(f"✓ Loaded {len(df_benign)} benign behavior examples")

    benign_data = []
    for idx, row in df_benign.iterrows():
        if "Goal" in df_benign.columns:
            text = row["Goal"]
        elif "Behavior" in df_benign.columns:
            text = row["Behavior"]
        elif "goal" in df_benign.columns:
            text = row["goal"]
        elif "behavior" in df_benign.columns:
            text = row["behavior"]
        else:
            text_columns = df_benign.select_dtypes(include=["object"]).columns
            if len(text_columns) > 0:
                text = row[text_columns[0]]
            else:
                continue

        entry = {
            "text": str(text).strip(),
            "source": "JailbreakBench",
            "type": "benign",
        }

        if "Category" in df_benign.columns:
            entry["category"] = str(row["Category"])
        elif "category" in df_benign.columns:
            entry["category"] = str(row["category"])

        benign_data.append(entry)

    benign_output_path = "jailbreak_data/benign_behaviors.json"
    with open(benign_output_path, "w") as f:
        json.dump(benign_data, f, indent=2)

    print(f"✓ Saved benign behaviors to: {benign_output_path}")
except Exception as e:
    print(f"⚠ Could not fetch benign behaviors: {e}")

# --- Summary ---
print("\n" + "=" * 80)
print("DATASET FETCH COMPLETE")
print("=" * 80)
print(f"Harmful behaviors: {len(jailbreak_data)} examples")
print(f"Output: {output_path}")
print("=" * 80)
