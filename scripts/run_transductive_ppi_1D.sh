dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="CPDB"
[ -z "${device}" ] && device=-1

# Dataset options ['CPDB', 'IRefIndex_2015', 'PCNet', 'STRINGdb'] or 'custom_synthetic' / 'custom_synthetic_multidim'

python main_transductive.py \
	--device $device \
	--dataset $dataset \
	--mask_rate 0.5 \
	--encoder "gcn" \
	--decoder "gcn" \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 4 \
	--num_hidden 256 \
	--num_heads 4 \
	--max_epoch 100 \
	--max_epoch_f 300 \
	--lr 0.001 \
	--weight_decay 0 \
	--lr_f 0.005 \
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
	#--use_cfg \
	#--num_edge_types 5 \