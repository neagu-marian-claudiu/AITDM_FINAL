# PatchCore Federated Learning

Federated learning for visual anomaly detection using PatchCore.

## What It Does

- Trains anomaly detection models on distributed data
- Ensures fairness across product categories
- Supports multiple aggregation methods (FedAvg, FedProx, Category-Aware)
- Provides interpretability and robustness analysis

## Installation

```bash
pip install -r requirements.txt
python test_setup.py
```

## Usage

### Complete Training Pipeline (Recommended)

```bash
python run.py
```

Runs everything: centralized baseline, 5 clients, aggregation, and federated learning (4-6 hours).

### Individual Training

```bash
# Centralized training
python train_standalone.py --data_root data/federated_data

# Federated learning (best method)
python federated_train.py --aggregation category_aware --num_rounds 3

# Evaluation
python evaluate.py --checkpoint <model.joblib> --test_dir data/test_data_centralized
```

### Analysis

```bash
python run_analysis.py  # Complete analysis
python compare_models.py        # Model comparison
```

## Data Structure

```
data/
├── federated_data/
│   ├── client_0/
│   │   └── category/train/good/*.png
│   └── client_1-4/...
└── test_data_centralized/
    └── category/
        ├── test/good|bad/*.png
        └── ground_truth/
```

## Key Files

- `model.py` - PatchCore implementation
- `train_standalone.py` - Centralized/client training
- `federated_train.py` - Federated learning
- `evaluate.py` - Evaluation with fairness metrics
- `run.py` - Complete pipeline

## Results

After training, check `results/` for:
- Performance metrics (AUROC, AP, TPR@TNR)
- Fairness metrics (Disparate Impact, AUROC Range)
- Visualizations (anomaly maps, robustness curves)
- Model comparisons
