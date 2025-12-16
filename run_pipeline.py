"""
Pipeline runner for JailbreakBench clustering workflow.
Runs: tune_jailbreakbench.py -> cluster_jailbreakbench.py
"""

import os
import subprocess
import sys


def run_command(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, script_name], capture_output=False, text=True
    )

    if result.returncode != 0:
        print(f"\n❌ Error: {script_name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")
    return result


def main():
    """Run the complete clustering pipeline."""
    print("\n" + "=" * 80)
    print("JAILBREAKBENCH CLUSTERING PIPELINE")
    print("=" * 80)
    print("\nThis pipeline will:")
    print("  1. Tune clustering parameters to find optimal configuration")
    print("  2. Run clustering with best parameters and generate patterns")
    print("\n" + "=" * 80)

    # Check if jailbreak data exists
    if not os.path.exists("jailbreak_data/jailbreakbench.json"):
        print("\n❌ Error: jailbreak_data/jailbreakbench.json not found!")
        print("Please ensure the JailbreakBench dataset is downloaded.")
        sys.exit(1)

    # Step 1: Tune clustering parameters
    run_command("tune_jailbreakbench.py", "Step 1/2: Tuning clustering parameters")

    # Step 2: Run clustering with best parameters
    run_command(
        "cluster_jailbreakbench.py", "Step 2/2: Clustering and pattern extraction"
    )

    # Final summary
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - pipeline/jailbreakbench_best_config.json (optimal clustering config)")
    print("  - pipeline/jailbreakbench_clusters.json (cluster patterns & keywords)")
    print("  - pipeline/jailbreakbench_visualization.png (t-SNE visualization)")
    print("  - pipeline/jailbreakbench_cluster_sizes.png (cluster size distribution)")
    print("\nNext steps:")
    print("  - Review cluster patterns in pipeline/jailbreakbench_clusters.json")
    print("  - Run main.py to test guardrails effectiveness")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
