dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="custom_synthetic"
[ -z "${device}" ] && device=-1


python main_transductive.py \
	--device $device \
	--dataset $dataset \
	--mask_rate 0.5 \
	--encoder "gcn" \
	--decoder "gcn" \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 2 \
	--num_hidden 512 \
	--num_heads 4 \
	--max_epoch 1500 \
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