#!/bin/bash

# Define PPIs to compare
PPIS=(
    "CPDB"
    "IRefIndex_2015"
    "IRefIndex"
    "STRINGdb"
    "PCNet"
)

# Function to run experiment for a single PPI
run_experiment() {
    local ppi=$1
    local dataset_name
    
    dataset_name="${ppi}"
    
    echo "Running experiment for dataset: $dataset_name"
    python main_transductive.py \
        --dataset "$dataset_name" \
        --experiment_type "ppi_comparison" \
        --use_cfg \
        --seeds 0 1 
}

# Run experiments for each PPI
for ppi in "${PPIS[@]}"; do
    run_experiment "$ppi"
done 