#!/bin/bash

# Define all possible feature types
FEATURE_TYPES=(
    "gastro"
    "respiratory"
    "genitourinary"
    "reproductive"
    "endocrine"
    "headneck"
)

# Define PPIs
PPIS=("CPDB") #"IRefIndex_2015" "IRefIndex" "STRINGdb" "PCNet")

# Function to run experiment for a single dataset
run_experiment() {
    local ppi=$1
    local feature_types=$2
    local dataset_name

    dataset_name="${ppi}_${feature_types}"

    echo "Running experiment for dataset: $dataset_name"
    python main_transductive.py \
        --dataset "$dataset_name" \
        --experiment_type "cancer_ablations" \
        --use_cfg \
        --seeds 0 1
}

# Run experiments for each PPI
for ppi in "${PPIS[@]}"; do
    # Single feature type experiments
    for feature_type in "${FEATURE_TYPES[@]}"; do
        run_experiment "$ppi" "$feature_type"
    done
done
