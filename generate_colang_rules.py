import json
import os
import re

patterns = json.load(open("pipeline/cluster_patterns.json"))

rule_lines = []

for cid, info in patterns.items():
    kws = info["keywords"]
    regex = "|".join([re.escape(k) for k in kws])
    name = f"cluster_{cid}_attack"

    # CoLang keyword-based rule
    rule_lines.append(f"""
define user {name}
    "{'" "'.join(kws)}"
    (/{regex}/i)
    """)

# Save Colang rules
final = "\n".join(rule_lines)

os.makedirs("config/flows", exist_ok=True)
output_path = "config/flows/jailbreak_auto.co"

with open(output_path, "w") as f:
    f.write(final)

print(f"\n{'=' * 60}")
print(f"✓ CoLang rules generated!")
print(f"{'=' * 60}")
print(f"Rules created for {len(patterns)} cluster(s)")
print(f"Output saved to: {output_path}")
print(f"{'=' * 60}\n")
