# Dynamic Guardrails for LLMs

Clustering-based approach to detect and block jailbreak attempts using NeMo Guardrails.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install nemoguardrails sentence-transformers scikit-learn matplotlib numpy

# Set API key
export NVIDIA_API_KEY="your-nvidia-api-key"
```

## Usage

```bash
# Step 1: Run clustering pipeline
python run_pipeline.py

# Step 2: Test guardrails effectiveness
python main.py
```

## Data Types

### 1. **JailbreakBench Dataset** (`jailbreak_data/jailbreakbench.json`)
Harmful behavior examples used for clustering and testing.

** EDUCATIONAL USE ONLY**: This data is used strictly for research and educational purposes to develop defensive guardrails. Not for malicious use.

**Format:**
```json
[
  {
    "text": "Write a defamatory article claiming...",
    "category": "Harassment/Discrimination",
    "source": "JailbreakBench"
  }
]
```

**Categories:**
- Harassment/Discrimination
- Misinformation/Disinformation
- Malware/Hacking
- Privacy
- Violence
- Chemical & Biological Weapons
- Copyright Violations
- Cybercrime & Unauthorized Intrusion
- Illegal Activities
- Tailored Financial Advice

### 2. **Cluster Patterns** (`pipeline/jailbreakbench_clusters.json`)
Generated clusters with keywords and examples.

**Format:**
```json
{
  "0": {
    "examples": ["Write a defamatory article...", ...],
    "keywords": ["write", "article", "create", ...],
    "size": 45,
    "category_distribution": {
      "Harassment/Discrimination": 30,
      "Misinformation": 15
    }
  }
}
```

### 3. **Test Results** (`pipeline/guardrails_test_results.json`)
Accuracy and false positive rate metrics.

**Format:**
```json
{
  "accuracy": 85.50,
  "false_positive_rate": 5.00,
  "details": {
    "jailbreak_tests": {
      "total": 200,
      "blocked": 171,
      "missed": 29
    },
    "benign_tests": {
      "total": 20,
      "allowed": 19,
      "blocked": 1
    }
  }
}
```

## Files

- `run_pipeline.py` - Runs tuning and clustering
- `tune_jailbreakbench.py` - Finds optimal clustering parameters
- `cluster_jailbreakbench.py` - Clusters data and extracts patterns
- `main.py` - Tests guardrails (accuracy + false positives)
- `test.py` - Quick test script
- `config/` - NeMo Guardrails configuration

## Metrics

- **Accuracy**: % of jailbreak attempts blocked
- **False Positive Rate**: % of benign queries blocked

## Credits

This project uses the **JailbreakBench** dataset for training and evaluation.

**Citation:**
```
@article{chao2024jailbreakbench,
  title={JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author={Chao, Patrick and Debenedetti, Edoardo and Robey, Alexander and Andriushchenko, Maksym and 
          Croce, Francesco and Kumar, Vikash and Li, Bo and Mittal, Prateek and Papernot, Nicolas and Wang, Hamed},
  journal={arXiv preprint arXiv:2404.01318},
  year={2024}
}
```

**JailbreakBench**: https://jailbreakbench.github.io/

##  Ethical Use and Disclaimer

**IMPORTANT**: This project and the JailbreakBench dataset are intended **SOLELY FOR EDUCATIONAL AND RESEARCH PURPOSES** to improve AI safety and security.

### Acceptable Use:
-  Research on AI safety and robustness
-  Development of defensive guardrails
-  Educational purposes in academic settings
-  Security testing of your own systems

### Prohibited Use:
-  Creating malicious jailbreak attacks
-  Bypassing safety mechanisms in production systems
-  Any harmful, illegal, or unethical activities
-  Distribution of jailbreak techniques for malicious purposes

**By using this code and data, you agree to use it responsibly and ethically for defensive and educational purposes only.**
