#!/bin/bash

dataset_input=$1
device=$2

[ -z "${device}" ] && device=-1

# List of available datasets
datasets=("CPDB_cdgp" "IRefIndex_2015_cdgp" "IRefIndex_cdgp" "PCNet_cdgp" "STRINGdb_cdgp"
        "CPDB_cdgs" "IRefIndex_2015_cdgs" "IRefIndex_cdgs" "PCNet_cdgs" "STRINGdb_cdgs"
        "CPDB_cdps" "IRefIndex_2015_cdps" "IRefIndex_cdps" "PCNet_cdps" "STRINGdb_cdps"
        "CPDB_cgps" "IRefIndex_2015_cgps" "IRefIndex_cgps" "PCNet_cgps" "STRINGdb_cgps"
        "CPDB_dgps" "IRefIndex_2015_dgps" "IRefIndex_dgps" "PCNet_dgps" "STRINGdb_dgps"
)

if [ -z "${dataset_input}" ]; then
    # No dataset provided; run for all options
    for dataset in "${datasets[@]}"; do
        echo "Running dataset: $dataset"
        python main_transductive.py \
            --device $device \
            --dataset $dataset \
            --mask_rate 0.5 \
            --encoder "rgcn" \
            --decoder "rgcn" \
            --in_drop 0.2 \
            --attn_drop 0.1 \
            --num_layers 2 \
            --num_hidden 32 \
            --num_heads 4 \
            --max_epoch 100 \
            --max_epoch_f 300 \
            --lr 0.001 \
            --weight_decay 0 \
            --lr_f 0.01 \
            --weight_decay_f 1e-4 \
            --activation prelu \
            --optimizer adam \
            --drop_edge_rate 0.0 \
            --loss_fn "sce" \
            --seeds 0 1 2 \
            --replace_rate 0.05 \
            --alpha_l 3 \
            --linear_prob \
            --scheduler \
            --use_cfg \
            --num_edge_types 5 \
            --weight_decomposition "{'type': 'basis', 'num_bases': 2}" \
            --vertical_stacking "True"
    done
else
    # A dataset was provided; run for the given dataset
    dataset=$dataset_input
    python main_transductive.py \
        --device $device \
        --dataset $dataset \
        --mask_rate 0.5 \
        --encoder "rgcn" \
        --decoder "rgcn" \
        --in_drop 0.2 \
        --attn_drop 0.1 \
        --num_layers 2 \
        --num_hidden 32 \
        --num_heads 4 \
        --max_epoch 5 \
        --max_epoch_f 300 \
        --lr 0.001 \
        --weight_decay 0 \
        --lr_f 0.01 \
        --weight_decay_f 1e-4 \
        --activation prelu \
        --optimizer adam \
        --drop_edge_rate 0.0 \
        --loss_fn "sce" \
        --seeds 0 1 2 3 4 \
        --replace_rate 0.05 \
        --alpha_l 3 \
        --linear_prob \
        --scheduler \
        --use_cfg \
        --num_edge_types 5 \
        --weight_decomposition "{'type': 'basis', 'num_bases': 2}" \
        --vertical_stacking "True"
fi