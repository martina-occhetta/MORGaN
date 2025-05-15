# MORGaN for Druggable Gene Prediction

This repository contains code for predicting druggable genes using graph neural networks and multi-modal data integration.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .MORGaN
source .MORGaN/bin/activate  # On Unix/macOS
# or
.MORGaN\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

The data should be organized as follows:
```
data/
├── components/
│   ├── features/          # Multi-modal gene features
│   ├── labels/            # Druggable gene labels
│   └── networks/          # Network files (PPI and other networks)
└── paper/
    └── graphs/            # Generated graph files
        ├── edge_ablations/
        ├── feature_ablations/
        ├── ppi_datasets/
        └── predict_druggable_genes/
```

## Running Experiments

The code supports four types of experiments:

### 1. Edge Ablation Experiments
Tests the impact of different network combinations on model performance.

```bash
chmod +x scripts/run_edge_ablations.sh
./scripts/run_edge_ablations.sh
```

This will run experiments:
- With and without PPI edges
- Using different combinations of networks (coexpression, GO, domain similarity, etc.)
- Including random edge versions for each combination

### 2. Feature Ablation Experiments
Tests the impact of different feature combinations on model performance.

```bash
chmod +x scripts/run_feature_ablations.sh
./scripts/run_feature_ablations.sh
```

This will run experiments:
- Using individual features (CNA, GE, METH, MF)
- Using different combinations of features
- Including a random features version

### 3. PPI Comparison Experiments
Compares performance across different PPI networks.

```bash
chmod +x scripts/run_ppi_comparison.sh
./scripts/run_ppi_comparison.sh
```

This will run experiments using different PPI networks:
- CPDB
- IRefIndex_2015
- IRefIndex
- STRINGdb
- PCNet

### 4. Main Druggable Gene Prediction Task
Runs the main prediction task using the best performing configuration.

```bash
chmod +x scripts/run_main_task.sh
./scripts/run_main_task.sh
```

## Configuration Files

Each experiment type has its own configuration file in the `configs/` directory:
- `edge_ablations.yaml`: Edge ablation experiment settings
- `feature_ablation_config.yaml`: Feature ablation experiment settings
- `ppi_comparison_config.yaml`: PPI comparison experiment settings
- `main_task_config.yaml`: Main prediction task settings

## Output

Results are saved in the following locations:
- Model checkpoints: `checkpoints/`
- Generated graphs: `data/paper/graphs/`
- Logs: `logs/`
- Weights & Biases dashboard (if logging is enabled)

## Monitoring Experiments

If logging is enabled in the configuration files, experiments can be monitored through:
1. Console output showing progress and metrics
2. Weights & Biases dashboard (requires wandb account)
3. Log files in the `logs/` directory

## Notes

- Each experiment runs with 2 different random seeds (0, 1), for 3 iterations, for reproducibility
- The code automatically handles the dynamic number of edge types based on the experiment configuration
- Make sure all required data files are in place before running experiments
- For large experiments, consider using a compute cluster or GPU-enabled machine
