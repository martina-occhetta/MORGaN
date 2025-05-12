#!/bin/bash

# Parse command line arguments
FORCE_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Define all possible edge types
EDGE_TYPES=(
    "coexpression"
    "GO"
    "domain"
    "sequence"
    "pathway"
)

# Define PPIs
PPIS=("CPDB") #"IRefIndex_2015" "IRefIndex" "STRINGdb" "PCNet")

# Function to check if experiment has already been run
check_experiment_completed() {
    local dataset_name=$1
    local graph_file="data/paper/graphs/edge_ablations/${dataset_name}.pt"
    
    # Check if graph file exists and is not empty
    if [ -f "$graph_file" ] && [ -s "$graph_file" ]; then
        return 0  # Experiment completed
    else
        return 1  # Experiment not completed
    fi
}

# Function to run experiment for a single dataset
run_experiment() {
    local ppi=$1
    local edge_types=$2
    local no_ppi=$3
    local dataset_name
    
    if [ "$no_ppi" = true ]; then
        dataset_name="${ppi}_noppi_${edge_types}"
    else
        dataset_name="${ppi}_${edge_types}"
    fi
    
    # Check if experiment has already been run (unless force flag is set)
    if [ "$FORCE_RUN" = false ] && check_experiment_completed "$dataset_name"; then
        echo "Skipping experiment for dataset: $dataset_name (graph already exists)"
        return
    fi
    
    echo "Running experiment for dataset: $dataset_name"
    python main_transductive.py \
        --dataset "$dataset_name" \
        --experiment_type "edge_ablations" \
        --use_cfg \
        --seeds 0 1 
}

# Function to create sorted edge combination
create_edge_combo() {
    local -a edges=("$@")
    # Sort the edges alphabetically and join with underscore
    printf "%s" "$(printf "%s\n" "${edges[@]}" | sort | tr '\n' '_')" | sed 's/_$//'
}

# Run experiments for each PPI
for ppi in "${PPIS[@]}"; do
    # Run both with and without PPI edges
    for no_ppi in true false; do
        # Single edge type experiments
        for edge_type in "${EDGE_TYPES[@]}"; do
            run_experiment "$ppi" "$edge_type" "$no_ppi"
            
            # Random edges version
            run_experiment "$ppi" "${edge_type}_random" "$no_ppi"
        done
        
        # Two edge type combinations
        for ((i=0; i<${#EDGE_TYPES[@]}; i++)); do
            for ((j=i+1; j<${#EDGE_TYPES[@]}; j++)); do
                edge_combo=$(create_edge_combo "${EDGE_TYPES[i]}" "${EDGE_TYPES[j]}")
                run_experiment "$ppi" "$edge_combo" "$no_ppi"
                
                # Random edges version
                run_experiment "$ppi" "${edge_combo}_random" "$no_ppi"
            done
        done
        
        # Three edge type combinations
        for ((i=0; i<${#EDGE_TYPES[@]}; i++)); do
            for ((j=i+1; j<${#EDGE_TYPES[@]}; j++)); do
                for ((k=j+1; k<${#EDGE_TYPES[@]}; k++)); do
                    edge_combo=$(create_edge_combo "${EDGE_TYPES[i]}" "${EDGE_TYPES[j]}" "${EDGE_TYPES[k]}")
                    run_experiment "$ppi" "$edge_combo" "$no_ppi"
                    
                    # Random edges version
                    run_experiment "$ppi" "${edge_combo}_random" "$no_ppi"
                done
            done
        done
        
        # Four edge type combinations
        for ((i=0; i<${#EDGE_TYPES[@]}; i++)); do
            for ((j=i+1; j<${#EDGE_TYPES[@]}; j++)); do
                for ((k=j+1; k<${#EDGE_TYPES[@]}; k++)); do
                    for ((l=k+1; l<${#EDGE_TYPES[@]}; l++)); do
                        edge_combo=$(create_edge_combo "${EDGE_TYPES[i]}" "${EDGE_TYPES[j]}" "${EDGE_TYPES[k]}" "${EDGE_TYPES[l]}")
                        run_experiment "$ppi" "$edge_combo" "$no_ppi"
                        
                        # Random edges version
                        run_experiment "$ppi" "${edge_combo}_random" "$no_ppi"
                    done
                done
            done
        done
        
        # All edge types
        all_edges=$(create_edge_combo "${EDGE_TYPES[@]}")
        run_experiment "$ppi" "$all_edges" "$no_ppi"
        
        # Random edges version of all types
        run_experiment "$ppi" "${all_edges}_random" "$no_ppi"
    done
done 