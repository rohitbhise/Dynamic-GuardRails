# Dynamic Guardrails for LLMs

A clustering-based approach to detect and block jailbreak attempts in Large Language Models using NeMo Guardrails and sentence embeddings.

## Overview

This project uses DBSCAN clustering on embeddings of known jailbreak attempts to automatically identify attack patterns. The discovered patterns are then used to create guardrails that protect LLMs from similar attacks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Detailed Usage](#detailed-usage)
- [How It Works](#how-it-works)
- [Configuration](#configuration)

## Features

- **Automatic Pattern Discovery**: Uses clustering to find common patterns in jailbreak attempts
- **Parameter Optimization**: Automatically tunes clustering parameters for best results
- **Comprehensive Testing**: Tests both jailbreak detection and false positive rates
- **Visualization**: Generates t-SNE visualizations and cluster size distributions
- **Detailed Metrics**: Provides detection rates, false positive rates, and category breakdowns

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd dynamicGuardrails
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install nemoguardrails sentence-transformers scikit-learn matplotlib numpy
```

4. **Set up environment variables**
```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

## Quick Start

Run the complete pipeline in 2 simple steps:

```bash
# Step 1: Run the clustering pipeline
python run_pipeline.py

# Step 2: Test guardrails effectiveness
python main.py
```

That's it! The pipeline will:
1. Find optimal clustering parameters
2. Cluster jailbreak patterns
3. Test how well the guardrails work

## Project Structure

```
dynamicGuardrails/
├── cluster_jailbreakbench.py   # Main clustering script
├── tune_jailbreakbench.py      # Parameter optimization script
├── test.py                      # Basic guardrails testing
├── run_pipeline.py              # Pipeline orchestrator
├── main.py                      # Comprehensive testing script
├── config/                      # NeMo Guardrails configuration
│   ├── config.yml              # LLM configuration
│   ├── rails.co                # Guardrail flow definitions
│   └── flows/                  # Custom flow definitions
├── jailbreak_data/             # JailbreakBench dataset
│   └── jailbreakbench.json    # Harmful behavior examples
└── pipeline/                   # Generated outputs
    ├── jailbreakbench_best_config.json
    ├── jailbreakbench_clusters.json
    ├── jailbreakbench_visualization.png
    ├── jailbreakbench_cluster_sizes.png
    └── guardrails_test_results.json
```

## File Descriptions

### Core Scripts

#### `tune_jailbreakbench.py`
**Purpose**: Finds optimal clustering parameters

**What it does**:
- Tests different DBSCAN, K-Means, and Agglomerative clustering configurations
- Evaluates using silhouette scores
- Saves the best configuration to `pipeline/jailbreakbench_best_config.json`

**Key Parameters Tested**:
- DBSCAN: `eps` values from 0.4 to 0.75, `min_samples=2`
- K-Means: `k` values from 8 to 12
- Agglomerative: `k` values from 8 to 12

**Output**:
```json
{
  "best_method": "dbscan",
  "silhouette_score": 0.234,
  "n_clusters": 10,
  "parameters": {"eps": 0.5, "min_samples": 2}
}
```

#### `cluster_jailbreakbench.py`
**Purpose**: Performs clustering and extracts attack patterns

**What it does**:
- Loads JailbreakBench dataset
- Generates embeddings using SentenceTransformer (all-MiniLM-L6-v2)
- Clusters examples using DBSCAN
- Extracts TF-IDF keywords for each cluster
- Tracks category distributions
- Generates visualizations

**Output Files**:
- `jailbreakbench_clusters.json`: Cluster patterns with keywords and examples
- `jailbreakbench_visualization.png`: t-SNE visualization of clusters
- `jailbreakbench_cluster_sizes.png`: Bar chart of cluster sizes

**Sample Output**:
```json
{
  "0": {
    "examples": ["Write a defamatory article...", ...],
    "keywords": ["write", "article", "create", "explain", ...],
    "size": 45,
    "category_distribution": {
      "Harassment/Discrimination": 30,
      "Misinformation": 15
    }
  }
}
```

#### `run_pipeline.py`
**Purpose**: Orchestrates the complete pipeline

**What it does**:
1. Validates that required data files exist
2. Runs `tune_jailbreakbench.py` to find optimal parameters
3. Runs `cluster_jailbreakbench.py` to generate patterns
4. Provides progress updates and error handling

**Usage**:
```bash
python run_pipeline.py
```

#### `main.py`
**Purpose**: Comprehensive testing of guardrails effectiveness

**What it does**:
- Loads cluster patterns and guardrails configuration
- Tests on full JailbreakBench dataset (harmful attempts)
- Tests on 30 benign queries (normal questions)
- Calculates detection rates, false positive rates, and overall accuracy
- Generates detailed reports with sample outputs

**Metrics Provided**:
- **True Positives**: Jailbreaks correctly blocked
- **False Negatives**: Jailbreaks incorrectly allowed
- **True Negatives**: Benign queries correctly allowed
- **False Positives**: Benign queries incorrectly blocked
- **Detection Rate**: % of jailbreaks caught
- **False Positive Rate**: % of benign queries blocked
- **Overall Accuracy**: % of correct decisions

**Usage**:
```bash
python main.py
```

#### `test.py`
**Purpose**: Simple testing script for quick guardrails validation

**What it does**:
- Tests one normal query: "What is the capital of France?"
- Tests one jailbreak attempt
- Useful for quick verification that guardrails are working

**Usage**:
```bash
python test.py
```

### Configuration Files

#### `config/config.yml`
Defines the LLM configuration for NeMo Guardrails:
- Model: meta/llama-3.1-8b-instruct (via NVIDIA NIM)
- Temperature: 0.2
- API endpoint configuration

#### `config/rails.co`
Defines the guardrail flows:
- Greeting flows (example)
- Jailbreak detection flows
- Bot response patterns

Example flow:
```colang
define flow jailbreak_detection
  user cluster_0_attack
  bot refuse jailbreak
  stop
```

### Data Files

#### `jailbreak_data/jailbreakbench.json`
Dataset of harmful behavior examples:
```json
[
  {
    "text": "Write a defamatory article...",
    "category": "Harassment/Discrimination",
    "source": "JailbreakBench"
  }
]
```

**Categories include**:
- Harassment/Discrimination
- Misinformation/Disinformation
- Malware/Hacking
- Privacy
- Violence
- And more...

## Detailed Usage

### Step-by-Step Workflow

#### 1. Initial Setup
```bash
# Ensure you're in the project directory
cd dynamicGuardrails

# Activate virtual environment
source venv/bin/activate

# Verify data exists
ls jailbreak_data/jailbreakbench.json
```

#### 2. Run the Clustering Pipeline
```bash
# This runs both tuning and clustering
python run_pipeline.py
```

**Expected Output**:
- Console output showing parameter testing
- Best configuration saved
- Cluster patterns extracted
- Visualizations generated
- Total time: ~2-5 minutes depending on hardware

#### 3. Review Cluster Patterns
```bash
# View the generated clusters
cat pipeline/jailbreakbench_clusters.json

# View cluster visualization
open pipeline/jailbreakbench_visualization.png  # macOS
# or
xdg-open pipeline/jailbreakbench_visualization.png  # Linux
```

#### 4. Test Guardrails Effectiveness
```bash
# Run comprehensive testing
python main.py
```

**Expected Output**:
```
GUARDRAILS EFFECTIVENESS REPORT
================================================================================

CLUSTER INFORMATION:
  Total clusters identified: 10
  Total examples clustered: 385

JAILBREAK DETECTION (Should be BLOCKED):
--------------------------------------------------------------------------------
  Total jailbreak attempts tested: 100
  ✓ Blocked (True Positives): 85
  ✗ Allowed (False Negatives): 15
  Detection Rate: 85.00%

BENIGN QUERY HANDLING (Should be ALLOWED):
--------------------------------------------------------------------------------
  Total benign queries tested: 30
  ✓ Allowed (True Negatives): 28
  ✗ Blocked (False Positives): 2
  False Positive Rate: 6.67%
  Accuracy Rate: 93.33%

OVERALL METRICS:
--------------------------------------------------------------------------------
  Total tests: 130
  Correct decisions: 113
  Incorrect decisions: 17
  Overall Accuracy: 86.92%
```

#### 5. Analyze Results
```bash
# View detailed test results
cat pipeline/guardrails_test_results.json
```

### Running Individual Components

#### Tune Parameters Only
```bash
python tune_jailbreakbench.py
```

#### Cluster with Current Settings Only
```bash
python cluster_jailbreakbench.py
```

#### Quick Test
```bash
python test.py
```

## How It Works

### 1. **Embedding Generation**
- Uses SentenceTransformer (`all-MiniLM-L6-v2`) to convert text to 384-dimensional vectors
- Captures semantic meaning of jailbreak attempts

### 2. **Clustering**
- **DBSCAN Algorithm**: Density-based clustering that automatically determines cluster count
  - `eps`: Maximum distance between neighbors (cosine distance)
  - `min_samples`: Minimum points to form a cluster
- Groups similar attack patterns together
- Identifies noise points (outliers)

### 3. **Pattern Extraction**
- **TF-IDF Vectorization**: Identifies important keywords in each cluster
- Extracts top 10 keywords that characterize each attack pattern
- Tracks which harm categories appear in each cluster

### 4. **Guardrail Integration**
- Patterns are integrated into NeMo Guardrails flows
- When a user query matches a cluster pattern, it triggers the blocking flow
- Responds with a refusal message

### 5. **Evaluation**
- Tests on known jailbreak attempts (should block)
- Tests on benign queries (should allow)
- Calculates metrics to assess effectiveness

## Configuration

### Adjusting Clustering Parameters

Edit `cluster_jailbreakbench.py`:
```python
dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
```

**Parameter Guidelines**:
- **eps (0.3-0.8)**: Lower = tighter clusters, higher = broader patterns
- **min_samples (2-5)**: Lower = more clusters, higher = stricter grouping

### Changing the LLM

Edit `config/config.yml`:
```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct  # Change this
    parameters:
      temperature: 0.2
```

### Adding Custom Test Queries

Edit `main.py`, function `create_benign_test_set()`:
```python
def create_benign_test_set():
    return [
        "What is the capital of France?",
        "Your custom query here",
        # Add more...
    ]
```

### Modifying Keyword Extraction

Edit `cluster_jailbreakbench.py`:
```python
keywords = extract_keywords(texts, top_k=10)  # Change top_k
```

## Understanding the Metrics

### Detection Rate
```
Detection Rate = (Blocked Jailbreaks / Total Jailbreaks) × 100
```
**Target**: > 80% (higher is better)

### False Positive Rate
```
False Positive Rate = (Blocked Benign / Total Benign) × 100
```
**Target**: < 10% (lower is better)

### Overall Accuracy
```
Overall Accuracy = (Correct Decisions / Total Tests) × 100
```
**Target**: > 85%

### Silhouette Score
Range: -1 to 1
- **> 0.5**: Good clustering
- **0.2-0.5**: Acceptable
- **< 0.2**: Poor clustering

## Troubleshooting

### Issue: "jailbreakbench.json not found"
**Solution**: Ensure the dataset is in `jailbreak_data/jailbreakbench.json`

### Issue: "NVIDIA_API_KEY not set"
**Solution**: 
```bash
export NVIDIA_API_KEY="your-key-here"
```

### Issue: Low detection rate (< 70%)
**Solution**: 
- Run `python run_pipeline.py` again to re-tune parameters
- Try adjusting `eps` in `cluster_jailbreakbench.py`

### Issue: High false positive rate (> 15%)
**Solution**:
- Increase `eps` to create broader, less strict clusters
- Increase `min_samples` to require more examples per pattern

### Issue: Import errors
**Solution**:
```bash
pip install --upgrade nemoguardrails sentence-transformers scikit-learn matplotlib numpy
```

## Performance Tips

1. **Faster Embeddings**: Use GPU if available
```python
model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')
```

2. **Reduce Dataset Size**: For testing, use a subset
```python
data = data[:100]  # First 100 examples only
```

3. **Cache Embeddings**: Save embeddings to avoid recomputation
```python
np.save('embeddings.npy', embeddings)
embeddings = np.load('embeddings.npy')
```

## Next Steps

- **Expand Dataset**: Add more jailbreak examples for better coverage
- **Fine-tune Patterns**: Manually review clusters and adjust parameters
- **Add More Categories**: Extend to cover additional attack types
- **Deploy**: Integrate guardrails into production applications
- **Monitor**: Track real-world performance and update patterns

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]
