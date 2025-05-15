#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints

# Run edge ablation experiments
echo "Running edge ablation experiments..."
./scripts/run_edge_ablations.sh --force > logs/edge_ablations.log 2>&1

# Run feature ablation experiments
echo "Running feature ablation experiments..."
./scripts/run_feature_ablations.sh > logs/feature_ablations.log 2>&1

# Run PPI dataset comparison experiments
echo "Running PPI dataset comparison experiments..."
./scripts/run_ppi_comparison.sh > logs/ppi_datasets.log 2>&1

# Run main druggable gene prediction experiments
echo "Running main druggable gene prediction experiments..."
./scripts/run_druggable_genes.sh > logs/druggable_genes.log 2>&1

# Run no pretrain experiments
echo "Running no pretrain experiments..."
./scripts/run_no_pretrain.sh > logs/no_pretrain.log 2>&1

echo "All experiments completed! Check the logs directory for detailed output." 