import torch
from torch_geometric.utils import from_networkx
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import networkx as nx
import random
import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


"""Cross-validation"""


def cross_validation(train_path, file_save_path, folds, K=1):
    # Load dataset.
    train_file = pd.read_csv(train_path)
    label_list = train_file["label"].tolist()
    idx_list = train_file.index.tolist()
    print(f"The number of all gene index: {len(idx_list)}")

    train_indices, test_indices = train_test_split(
        idx_list, test_size=0.2, random_state=42, stratify=label_list
    )  # Split into initial train and test sets.

    test_labels = [label_list[i] for i in test_indices]
    positive_test_indices = [i for i, label in enumerate(test_labels) if label == 1]
    negative_test_indices = [
        i for i, label in enumerate(test_labels) if label == 0
    ]  # Ouput some informations of test_sets

    # Placeholder for K-fold sets.
    k_sets = {}
    for i in range(K):
        kf = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=42
        )  # Initialize StratifiedKFold.

        splits = kf.split(
            train_indices, [label_list[idx] for idx in train_indices]
        )  # Obtain stratified splits.

        k_folds = []  # Collecting train and validation indices for each fold.
        for train, val in splits:
            train_mask = torch.LongTensor([train_indices[i] for i in train])
            val_mask = torch.LongTensor([train_indices[i] for i in val])
            train_label = torch.FloatTensor(
                [label_list[train_indices[i]] for i in train]
            ).reshape(-1, 1)
            val_label = torch.FloatTensor(
                [label_list[train_indices[i]] for i in val]
            ).reshape(-1, 1)

            # Append to folds list.
            k_folds.append((train_mask, val_mask, train_label, val_label))

        # Store folds in dict.
        k_sets[i] = k_folds

    # Preparing the test set.
    test_mask = torch.LongTensor([idx_list[i] for i in test_indices])
    test_label = torch.FloatTensor([label_list[i] for i in test_indices]).reshape(-1, 1)

    # torch.save(k_sets, os.path.join(file_save_path, 'k_sets.pkl'))

    return k_sets, test_mask, test_label, idx_list, label_list


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss


def load_featured_graph(network, omicfeature):
    omics_feature_vector = sp.csr_matrix(omicfeature, dtype=np.float32)
    omics_feature_vector = torch.FloatTensor(np.array(omics_feature_vector.todense()))
    # print(f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

    if network.shape[0] == network.shape[1]:
        G = nx.from_pandas_adjacency(network)
    else:
        G = nx.from_pandas_edgelist(network)

    G_adj = nx.convert_node_labels_to_integers(
        G, ordering="sorted", label_attribute="label"
    )
    # print(f'If the graph is connected graph: {nx.is_connected(G_adj)}')
    # print(f'The number of connected components: {nx.number_connected_components(G_adj)}')

    graph = from_networkx(G_adj)
    assert graph.is_undirected() == True

    # print(f'The edge index is {graph.edge_index}')
    graph.x = omics_feature_vector
    graph.network = network
    return graph
