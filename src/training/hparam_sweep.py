#!/usr/bin/env python
import main_transductive
import wandb

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "test_auc",
        "goal": "maximize",
    },
    "parameters": {
        "lr": {"values": [0.001, 0.005, 0.01]},
        "lr_f": {"values": [0.001, 0.005, 0.01]},
        "weight_decay": {"values": [0, 1e-4, 1e-3]},
        "num_hidden": {"values": [16, 32, 64]},
        "num_layers": {"values": [2, 3, 4]},
        "max_epoch": {"values": [20, 50, 100, 200]},
        "max_epoch_f": {"values": [20, 50, 100, 200]},
        "mask_rate": {"values": [0.1, 0.3, 0.5]},
        "activation": {"values": ["relu", "prelu"]},
        "linear_prob": {"values": [True, False]},
        "alpha_l": {"values": [0.5, 1.0, 2.0, 3.0, 5.0]},
        "mask_rate_f": {"values": [0.05, 0.1, 0.3, 0.5]},
    },
}


def sweep_run():
    wandb.init()
    config = wandb.config

    args = main_transductive.build_args()
    args.dataset = "CPDB_cdgps"
    args.lr = config.lr
    args.weight_decay = config.weight_decay
    args.num_hidden = config.num_hidden
    args.num_layers = config.num_layers
    args.max_epoch = config.max_epoch
    args.max_epoch_f = config.max_epoch_f
    args.mask_rate = config.mask_rate
    args.activation = config.activation
    args.linear_prob = config.linear_prob
    args.alpha_l = config.alpha_l
    args.use_scheduler = True
    args.logging = True
    args.encoder = "rgcn"
    args.decoder = "rgcn"
    args.num_edge_types = 6
    args.weight_decomposition = {"type": "basis", "num_bases": 2}
    args.vertical_stacking = True

    print("Running sweep run with configuration:", dict(config))
    main_transductive.main(args)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="graphMAE-DG-sweep")
    wandb.agent(sweep_id, function=sweep_run, count=200)
