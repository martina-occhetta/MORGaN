import logging
import re 
import numpy as np
from tqdm import tqdm
import torch
import time
import os

from torch_geometric.data import Data

from morgan.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    WBLogger,
    get_current_lr,
    load_config,
)
from morgan.datasets.data_util import load_mutag_dataset, map_cancer_types
from morgan.datasets.build_graphs import (
    load_h5_graph,
    load_h5_graph_with_external_edges,
    load_h5_graph_random_features,
    load_h5_graph_with_external_features,
    filter_graph_to_features,
    add_all_edges,
    randomize_edges,
)
from morgan.eval.evaluation import node_classification_eval
from morgan.eval.evaluation_multilabel import multilabel_node_classification_eval
from morgan.models import build_model


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def pretrain(
    model,
    graph,
    feat,
    optimizer,
    max_epoch,
    device,
    scheduler,
    num_classes,
    lr_f,
    weight_decay_f,
    max_epoch_f,
    linear_prob,
    num_edge_types=1,
    logger=None,
):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    # model_dtype = next(model.parameters()).dtype
    # x = x.to(model_dtype)
    # loss, loss_dict = model(x, graph.edge_index)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        # epoch_start = time.time()
        model.train()
        if num_edge_types != 1:
            loss, loss_dict = model(graph, x, num_edge_types)
        else:
            loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            if graph.y.dim() > 1 and graph.y.size(1) > 1:
                (
                    (acc, estp_acc),
                    (auc, estp_auc),
                    (aupr, estp_aupr),
                    (precision, estp_precision),
                    (recall, estp_recall),
                    (f1, estp_f1),
                ) = multilabel_node_classification_eval(
                    model,
                    graph,
                    x,
                    num_classes,
                    lr_f,
                    weight_decay_f,
                    max_epoch_f,
                    device,
                    linear_prob=False,
                )

            else:
                (
                    (acc, estp_acc),
                    (auc, estp_auc),
                    (aupr, estp_aupr),
                    (precision, estp_precision),
                    (recall, estp_recall),
                    (f1, estp_f1),
                ) = node_classification_eval(
                    model,
                    graph,
                    x,
                    num_classes,
                    lr_f,
                    weight_decay_f,
                    max_epoch_f,
                    device,
                    linear_prob=False,
                )
            logger.note(
                {
                    **loss_dict,
                    "accuracy": acc,
                    "estp_accuracy": estp_acc,
                    "auc": auc,
                    "estp_auc": estp_auc,
                    "aupr": aupr,
                    "estp_aupr": estp_aupr,
                    "precision": precision,
                    "estp_precision": estp_precision,
                    "recall": recall,
                    "estp_recall": estp_recall,
                },
                step=epoch,
            )

        # epoch_end = time.time()
        # print(f"Epoch {epoch} runtime: {epoch_end - epoch_start:.2f} seconds")

    # return best_model
    return model


def build_graph(dataset_name: str, experiment_type: str):
    """
    Build a graph based on the dataset name and experiment type.

    Parameters:
        dataset_name (str): Name of the dataset to load
        experiment_type (str): Type of experiment to run. Must be one of:
            - "feature_ablations": For feature ablation experiments
            - "edge_ablations": For edge ablation experiments
            - "ppi_comparison": For PPI dataset comparison
            - "predict_druggable_genes": For main druggable gene prediction task
            - "mutag": For MUTAG chemical compound classification
            - "ogbn_proteins": For OGBN-Proteins dataset

    Returns:
        Data: PyTorch Geometric Data object containing the graph
        tuple: (num_features, num_classes, num_edge_types)
    """
    # Validate experiment type
    valid_experiment_types = [
        "feature_ablations",
        "edge_ablations",
        "cancer_ablations",
        "ppi_comparison",
        "predict_druggable_genes",
        "mutag",
        "ogbn_proteins",
        "no_pretrain",
        "predict_essential_genes",
        "alzheimers",
    ]
    if experiment_type not in valid_experiment_types:
        logging.warning(
            f"Invalid experiment type: {experiment_type}. Must be one of {valid_experiment_types}"
        )

    if dataset_name.upper() == "MUTAG":
        # For MUTAG, we don't need the complex graph building process
        # Just load it directly and ensure it has the right attributes
        graph = load_mutag_dataset()

        # Create save directory if it doesn't exist
        SAVE_DIR = os.path.join("data/paper/graphs/", experiment_type)
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Save the graph
        save_path = os.path.join(SAVE_DIR, f"{dataset_name}.pt")
        torch.save(graph, save_path)
        logging.info(f"Saved graph to {save_path}")

        return graph, (graph.x.shape[1], 2, len(torch.unique(graph.edge_type)))

    # Set up paths for other datasets
    PATH = "data/components/features"
    LABEL_PATH = "data/components/labels/NIHMS80906-small_mol-and-bio-druggable.tsv"
    NETWORK_PATH = "data/components/networks"
    SAVE_DIR = os.path.join("data/paper/graphs/", experiment_type)

    # Extract PPI name from dataset name
    ppi = dataset_name.split("_")[0]

    # Validate PPI name
    valid_ppis = ["CPDB", "IRefIndex_2015", "IRefIndex", "STRINGdb", "PCNet", "ogbn"]
    if ppi not in valid_ppis:
        logging.warning(f"Invalid PPI name: {ppi}. Must be one of {valid_ppis}")

    # Define valid edge types and special keywords
    valid_edge_types = ["coexpression", "GO", "domain", "sequence", "pathway"]
    valid_special_keywords = ["noppi", "random", "preserve_types"]

    # Build graph based on experiment type
    if experiment_type == "feature_ablations":
        # Handle feature ablation experiments
        if "random" in dataset_name:
            graph = load_h5_graph_random_features(
                PATH, LABEL_PATH, ppi, randomize_features=True
            )
        else:
            NETWORKS = ["coexpression", "GO", "domain", "sequence", "pathway"]
            # Extract modalities from dataset name
            modalities = dataset_name.split("_")[
                2:
            ]  # e.g., CPDB_cdgps_CNA_GE_METH -> ['CNA', 'GE', 'METH']
            modalities = [modality.strip() for modality in modalities]
            # Validate modalities
            valid_modalities = ["CNA", "GE", "METH", "MF"]
            for modality in modalities:
                if modality not in valid_modalities:
                    logging.warning(
                        f"Invalid modality: {modality}. Must be one of {valid_modalities}"
                    )
            graph = load_h5_graph(PATH, LABEL_PATH, ppi, modalities=modalities)
            graph = add_all_edges(graph, NETWORK_PATH, NETWORKS)

    elif experiment_type == "edge_ablations":
        # Handle edge ablation experiments
        parts = dataset_name.split("_")
        ppi = parts[0]

        # Validate all parts of the dataset name
        for part in parts[1:]:  # Skip the PPI part
            if part not in valid_edge_types and part not in valid_special_keywords:
                logging.warning(
                    f"Invalid dataset name part: {part}. Must be one of {valid_edge_types + valid_special_keywords}"
                )

        # Extract and sort edge types to ensure consistent ordering
        edge_types = sorted([part for part in parts[1:] if part in valid_edge_types])
        if len(edge_types) != len(set(edge_types)):
            logging.warning(
                f"Dataset name contains repeated edge types: {dataset_name}"
            )

        # Check if this is a noppi experiment
        if "noppi" in parts:
            # Get edge types excluding 'noppi' and 'random'
            edge_types = [et for et in parts[1:] if et not in ["noppi", "random"]]

            # First build the graph with the specified edge types
            graph = load_h5_graph_with_external_edges(
                PATH, LABEL_PATH, ppi, NETWORK_PATH, edge_types
            )

            # Handle randomization if needed
            if "random" in parts:
                preserve_types = "preserve_types" in parts
                graph = randomize_edges(graph, preserve_edge_types=preserve_types)
        else:
            # Experiments WITH PPI edges
            # Get edge types excluding 'random'
            edge_types = [et for et in parts[1:] if et != "random"]

            # First load the base graph with PPI
            graph = load_h5_graph(PATH, LABEL_PATH, ppi)

            # Then add the specified edge types
            if edge_types:  # If there are additional edge types specified
                graph = add_all_edges(graph, NETWORK_PATH, edge_types)

            # Handle randomization if needed
            if "random" in parts:
                preserve_types = "preserve_types" in parts
                graph = randomize_edges(graph, preserve_edge_types=preserve_types)

    elif experiment_type == "ppi_comparison":
        # Handle PPI dataset comparison
        graph = load_h5_graph(PATH, LABEL_PATH, ppi)
        # NETWORKS = ['coexpression', 'GO', 'domain', 'sequence', 'pathway']
        # graph = add_all_edges(graph, NETWORK_PATH, NETWORKS)

    elif experiment_type in "cancer_ablations":
        system_key = re.split(r"_|\.", dataset_name)[1]
        print(system_key)
        cancer_types = map_cancer_types(system_key, "system")
        print(cancer_types)
        valid_cancer_types = [
            "KIRC",
            "BRCA",
            "READ",
            "PRAD",
            "STAD",
            "HNSC",
            "LUAD",
            "THCA",
            "BLCA",
            "ESCA",
            "LIHC",
            "UCEC",
            "COAD",
            "LUSC",
            "CESC",
            "KIRP",
        ]
        for cancer in cancer_types:
            if cancer not in valid_cancer_types:
                logging.warning(
                    f"Invalid cancer type: {cancer}. Must be one of {valid_cancer_types}"
                )
        NETWORKS = ["coexpression", "GO", "domain", "sequence", "pathway"]
        graph = load_h5_graph(PATH, LABEL_PATH, ppi, modalities=cancer_types)
        graph = add_all_edges(graph, NETWORK_PATH, NETWORKS)
        print(graph.x.shape, graph.y.shape)

    elif experiment_type == "predict_druggable_genes":
        # Handle main druggable gene prediction task
        NETWORKS = ["coexpression", "GO", "domain", "sequence", "pathway"]
        graph = load_h5_graph(PATH, LABEL_PATH, ppi)
        graph = add_all_edges(graph, NETWORK_PATH, NETWORKS)

    elif experiment_type == "predict_essential_genes":
        # Handle main druggable gene prediction task
        EG_PATH = "data/components/essential_genes"
        NETWORKS = ["coexpression", "GO", "domain", "sequence", "pathway"]
        EG_LABEL_PATH = "data/components/labels/journal.pcbi.1012076.s003_essential.csv"
        graph = load_h5_graph(EG_PATH, EG_LABEL_PATH, f"{ppi}_essential")
        graph = add_all_edges(graph, NETWORK_PATH, NETWORKS)

    elif experiment_type == "no_pretrain":
        # Handle no pretrain task
        NETWORKS = ["coexpression", "GO", "domain", "sequence", "pathway"]
        graph = load_h5_graph(PATH, LABEL_PATH, ppi)
        graph = add_all_edges(graph, NETWORK_PATH, NETWORKS)

    elif experiment_type == "alzheimers":
        FEAT = "data/components/features/OT_alzheimer_association_scores_zero.tsv"
        # ALZ_LABEL_PATH = "data/components/labels/OT_alzheimer_association_scores_positive.tsv"
        NETWORKS = ["coexpression", "GO", "domain", "sequence", "pathway"]
        graph = load_h5_graph_with_external_features(PATH, LABEL_PATH, ppi, new_features=FEAT)
        graph = add_all_edges(graph, NETWORK_PATH, NETWORKS)
        graph = filter_graph_to_features(graph, FEAT)

    elif experiment_type == "ogbn_proteins":
        graph = torch.load(
            "data/ogbn_proteins/ogbn_proteins_synthetic.pt", weights_only=False
        )
        if not isinstance(graph, Data):
            raise ValueError("Loaded graph is not a PyTorch Geometric Data object")
        return graph, (graph.x.shape[1], graph.y.size(1), graph.edge_attr.shape[1])

    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save the graph
    save_path = os.path.join(SAVE_DIR, f"{dataset_name}.pt")
    torch.save(graph, save_path)
    logging.info(f"Saved graph to {save_path}")

    return graph, (graph.x.shape[1], graph.y.max().item() + 1, graph.num_edge_types)


def main(args):
    start_time = time.time()  # Record the start time
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    experiment_type = args.experiment_type
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    # Initialize lists to store results
    acc_list = []
    estp_acc_list = []
    test_auc_list = []
    estp_test_auc_list = []
    test_aupr_list = []
    estp_test_aupr_list = []
    test_precision_list = []
    estp_precision_f_list = []
    test_recall_list = []
    estp_recall_f_list = []

    # For each seed, build a new graph and run 3 iterations
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        run_start = time.time()

        # Build graph for this seed
        graph, (num_features, num_classes, num_edge_types) = build_graph(
            dataset_name, experiment_type
        )
        args.num_features = num_features
        args.num_edge_types = (
            num_edge_types  # Update args with the correct number of edge types
        )

        # Run 3 iterations with the same graph
        for iter_num in range(2):
            print(f"####### Iteration {iter_num + 1} for seed {seed}")

            # Initialize logger for this iteration
            if logs:
                logger = WBLogger(
                    name=f"{experiment_type}_{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}_seed_{seed}_iter_{iter_num}"
                )
            else:
                logger = None

            model = build_model(args)
            model.to(device)
            optimizer = create_optimizer(optim_type, model, lr, weight_decay)

            if use_scheduler:
                logging.info("Use scheduler")

                def scheduler(epoch):
                    return (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5

                # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=scheduler
                )
            else:
                scheduler = None

            x = graph.x
            model_dtype = next(model.parameters()).dtype
            x = x.to(model_dtype)

            if not load_model:
                model = pretrain(
                    model,
                    graph,
                    x,
                    optimizer,
                    max_epoch,
                    device,
                    scheduler,
                    num_classes,
                    lr_f,
                    weight_decay_f,
                    max_epoch_f,
                    linear_prob,
                    num_edge_types,
                    logger,
                )
                model = model.cpu()

            if load_model:
                logging.info("Loading Model ... ")
                model.load_state_dict(torch.load("checkpoint.pt"))
            if save_model:
                logging.info("Saving Model ...")
                torch.save(model.state_dict(), "checkpoint.pt")

            model = model.to(device)
            model.eval()

            if graph.y.dim() > 1 and graph.y.size(1) > 1:
                (
                    (test_acc, estp_test_acc),
                    (test_auc, estp_test_auc),
                    (test_aupr, estp_test_aupr),
                    (test_precision, estp_test_precision),
                    (test_recall, estp_test_recall),
                    (test_f1, estp_f1),
                ) = multilabel_node_classification_eval(
                    model,
                    graph,
                    x,
                    num_classes,
                    lr_f,
                    weight_decay_f,
                    max_epoch_f,
                    device,
                    linear_prob,
                )
            else:
                (
                    (test_acc, estp_test_acc),
                    (test_auc, estp_test_auc),
                    (test_aupr, estp_test_aupr),
                    (test_precision, estp_test_precision),
                    (test_recall, estp_test_recall),
                    (test_f1, estp_f1),
                ) = node_classification_eval(
                    model,
                    graph,
                    x,
                    num_classes,
                    lr_f,
                    weight_decay_f,
                    max_epoch_f,
                    device,
                    linear_prob,
                    out_dir=f"results/{experiment_type}/{dataset_name}/seed_{seed}/iter_{iter_num}",
                )

            if logger is not None:
                logger.note(
                    {
                        "test_accuracy": test_acc,
                        "test_estp_accuracy": estp_test_acc,
                        "test_auc": test_auc,
                        "test_estp_auc": estp_test_auc,
                        "test_aupr": test_aupr,
                        "test_estp_aupr": estp_test_aupr,
                        "test_precision": test_precision,
                        "test_estp_precision": test_precision,
                        "test_recall": test_recall,
                        "test_estp_recall": test_recall,
                        "test_f1": test_f1,
                        "test_estp_f1": estp_f1,
                    },
                    step=max_epoch,
                )
                logger.finish()  # Finish the logger for this iteration

            acc_list.append(test_acc)
            estp_acc_list.append(estp_test_acc)
            test_auc_list.append(test_auc)
            estp_test_auc_list.append(estp_test_auc)
            test_aupr_list.append(test_aupr)
            estp_test_aupr_list.append(estp_test_aupr)
            test_precision_list.append(test_precision)
            estp_precision_f_list.append(estp_test_precision)
            test_recall_list.append(test_recall)
            estp_recall_f_list.append(estp_test_recall)

        run_end = time.time()
        print(f"Run {i} total runtime: {run_end - run_start:.2f} seconds")

    end_time = time.time()  # Record the end time after everything is done
    total_runtime = end_time - start_time
    print(f"Total runtime: {total_runtime:.2f} seconds")

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    final_auc, final_auc_std = np.mean(test_auc_list), np.std(test_auc_list)
    estp_auc, estp_auc_std = np.mean(estp_test_auc_list), np.std(estp_test_auc_list)
    final_aupr, final_aupr_std = np.mean(test_aupr_list), np.std(test_aupr_list)
    estp_aupr, estp_aupr_std = np.mean(estp_test_aupr_list), np.std(estp_test_aupr_list)
    print(f"# Final Accuracy: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# Early Stopping Accuracy: {estp_acc:.4f}±{estp_acc_std:.4f}")
    print(f"# Final AUC: {final_auc:.4f}±{final_auc_std:.4f}")
    print(f"# Early Stopping AUC: {estp_auc:.4f}±{estp_auc_std:.4f}")
    print(f"# Final AUPR: {final_aupr:.4f}±{final_aupr_std:.4f}")
    print(f"# Early Stopping AUPR: {estp_aupr:.4f}±{estp_aupr_std:.4f}")

    # final_metrics = {
    #     "final_accuracy": final_acc,
    #     "final_accuracy_std": final_acc_std,
    #     "early_stopping_accuracy": estp_acc,
    #     "early_stopping_accuracy_std": estp_acc_std,
    #     "final_auc": final_auc,
    #     "final_auc_std": final_auc_std,
    #     "final_aupr": final_aupr,
    #     "final_aupr_std": final_aupr_std,
    #     }

    # if logger is not None:
    #     logger.note(final_metrics, step=max_epoch)


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        # Determine which config file to use based on experiment type
        if args.experiment_type == "edge_ablations":
            config_path = "configs/edge_ablations.yaml"
        elif args.experiment_type == "feature_ablations":
            config_path = "configs/feature_ablations.yaml"
        elif args.experiment_type == "cancer_ablations":
            config_path = "configs/cancer_ablations.yaml"
        elif args.experiment_type == "ppi_comparison":
            config_path = "configs/ppi_comparison.yaml"
        elif args.experiment_type == "no_pretrain":
            config_path = "configs/no_pretrain.yaml"
        elif args.experiment_type == "predict_essential_genes":
            config_path = "configs/essential_genes.yaml"
        elif args.experiment_type == "alzheimers":
            config_path = "configs/alzheimers.yaml"
        else:
            config_path = "configs/main_task.yaml"

        args = load_config(args, config_path)
    print(args)
    main(args)
