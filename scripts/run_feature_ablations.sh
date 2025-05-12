#!/bin/bash

# Define all possible feature types
FEATURE_TYPES=(
    "CNA"
    "GE"
    "METH"
    "MF"
)

# Define PPIs
PPIS=("CPDB") #"IRefIndex_2015" "IRefIndex" "STRINGdb" "PCNet")

# Function to run experiment for a single dataset
run_experiment() {
    local ppi=$1
    local feature_types=$2
    local dataset_name
    
    dataset_name="${ppi}_cdgps_${feature_types}"
    
    echo "Running experiment for dataset: $dataset_name"
    python main_transductive.py \
        --dataset "$dataset_name" \
        --experiment_type "feature_ablations" \
        --use_cfg \
        --seeds 0 1 
}

# Run experiments for each PPI
for ppi in "${PPIS[@]}"; do
    # Single feature type experiments
    for feature_type in "${FEATURE_TYPES[@]}"; do
        run_experiment "$ppi" "$feature_type"
    done
    
    # Two feature type combinations
    for ((i=0; i<${#FEATURE_TYPES[@]}; i++)); do
        for ((j=i+1; j<${#FEATURE_TYPES[@]}; j++)); do
            feature_combo="${FEATURE_TYPES[i]}_${FEATURE_TYPES[j]}"
            run_experiment "$ppi" "$feature_combo"
        done
    done
    
    # Three feature type combinations
    for ((i=0; i<${#FEATURE_TYPES[@]}; i++)); do
        for ((j=i+1; j<${#FEATURE_TYPES[@]}; j++)); do
            for ((k=j+1; k<${#FEATURE_TYPES[@]}; k++)); do
                feature_combo="${FEATURE_TYPES[i]}_${FEATURE_TYPES[j]}_${FEATURE_TYPES[k]}"
                run_experiment "$ppi" "$feature_combo"
            done
        done
    done
    
    # All feature types
    all_features=$(IFS="_"; echo "${FEATURE_TYPES[*]}")
    run_experiment "$ppi" "$all_features"
    
    # Random features version
    run_experiment "$ppi" "random_features"
done