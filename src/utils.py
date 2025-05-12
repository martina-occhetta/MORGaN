import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch import optim as optim
import wandb

from torch_geometric.utils import add_self_loops
from sklearn import metrics

from math import floor, sqrt
import random
import torch

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    # For binary classification, apply a sigmoid and threshold at 0.5
    preds = (torch.sigmoid(y_pred) > 0.5).long()
    correct = preds.eq(y_true).double().sum().item()
    return correct / len(y_true)

def result(pred, true):
    aa = torch.sigmoid(pred)
    # precision, recall, _thresholds = metrics.precision_recall_curve(true, aa)
    # area = metrics.auc(recall, precision)
    # return metrics.roc_auc_score(true, aa), area, precision, recall
    precision, recall, _thresholds = metrics.precision_recall_curve(true, aa)
    aupr = metrics.auc(recall, precision)
    
    # Compute binary predictions using a threshold of 0.5
    pred_labels = (aa >= 0.5)
    f1 = metrics.f1_score(true, pred_labels)
    
    roc_auc = metrics.roc_auc_score(true, aa)
    
    return roc_auc, aupr, precision, recall, f1

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--experiment_type", type=str, default="predict_druggable_genes",
                        help="Type of experiment to run: 'feature_ablations', 'edge_ablations', 'ppi_datasets', or 'predict_druggable_genes'")

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--num_edge_types", type=int, default=None)
    parser.add_argument("--weight_decomposition", type=str, default=None)
    parser.add_argument("--vertical_stacking", type=bool, default=True)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true", default=True)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------

def load_config(args, config_path):
    """Load configuration from a YAML file and update args."""
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    
    logging.info(f"Loading configuration from {config_path}")
    
    # Update args with configuration values
    for section, params in config.items():
        if section == "experiment_type":
            setattr(args, section, params)
            continue
            
        # Handle both dictionary and non-dictionary cases
        if isinstance(params, dict):
            # Handle dictionary case
            for param_name, value in params.items():
                if isinstance(value, dict):
                    # Handle nested parameters
                    for nested_param, nested_value in value.items():
                        full_param_name = f"{param_name}_{nested_param}"
                        # Convert string values to appropriate types
                        if isinstance(nested_value, str):
                            try:
                                if nested_value.lower() == 'true':
                                    nested_value = True
                                elif nested_value.lower() == 'false':
                                    nested_value = False
                                else:
                                    # Try to convert to float first (handles scientific notation)
                                    try:
                                        nested_value = float(nested_value)
                                        # If it's a whole number, convert to int
                                        if nested_value.is_integer():
                                            nested_value = int(nested_value)
                                    except ValueError:
                                        pass  # Keep as string if conversion fails
                            except:
                                pass  # Keep original value if any conversion fails
                        setattr(args, full_param_name, nested_value)
                else:
                    # Handle direct parameters
                    # Convert string values to appropriate types
                    if isinstance(value, str):
                        try:
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            else:
                                # Try to convert to float first (handles scientific notation)
                                try:
                                    value = float(value)
                                    # If it's a whole number, convert to int
                                    if value.is_integer():
                                        value = int(value)
                                except ValueError:
                                    pass  # Keep as string if conversion fails
                        except:
                            pass  # Keep original value if any conversion fails
                    setattr(args, param_name, value)
        else:
            # Handle non-dictionary case
            # Convert string values to appropriate types
            if isinstance(params, str):
                try:
                    if params.lower() == 'true':
                        params = True
                    elif params.lower() == 'false':
                        params = False
                    else:
                        # Try to convert to float first (handles scientific notation)
                        try:
                            params = float(params)
                            # If it's a whole number, convert to int
                            if params.is_integer():
                                params = int(params)
                        except ValueError:
                            pass  # Keep as string if conversion fails
                except:
                    pass  # Keep original value if any conversion fails
            setattr(args, section, params)
    
    return args

def load_best_configs(args, path):
    """Legacy function for backward compatibility."""
    logging.warning("load_best_configs is deprecated. Please use load_config instead.")
    return load_config(args, path)

def drop_edge(data, drop_rate, return_edges = False):

    """
    Drops edges from the input graph with probability drop_rate and adds self-loops.

    Parameters:
      data (Data): PyG Data object containing the graph (must include 'edge_index').
      drop_rate (float): The probability of dropping an edge.
      return_edges (bool): If True, also return the dropped edges as a tensor.

    Returns:
      Data: A new Data object with updated edge_index including self-loops.
      (optional) torch.Tensor: The edge_index tensor of dropped edges.
    """
    if drop_rate <= 0:
        return data

    # Number of nodes in the graph
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # Create a random mask to decide which edges to keep.
    # True means keep the edge; False means drop.
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate

    # Keep only edges that are not dropped.
    kept_edge_index = edge_index[:, mask]

    # Add self-loops to the kept edge_index.
    new_edge_index, _ = add_self_loops(kept_edge_index, num_nodes=num_nodes)

    # Obtain the dropped edges (if needed)
    dropped_edge_index = edge_index[:, ~mask]

    # Clone the original data and update the edge_index.
    new_data = data.clone()
    new_data.edge_index = new_edge_index

    if return_edges:
        return new_data, dropped_edge_index
    return new_data

# ------ logging ------

class WBLogger(object):
    def __init__(self, log_path="./wandb", name="run", project='MORGaN'):
        """
        Initializes a Weights & Biases run. The log_path is ignored in wandb,
        but kept for compatibility.
        """
        wandb.init(project=project, name=name)
        self.last_step = 0

    def note(self, metrics, step=None, accuracy=None, estp_accuracy= None, auc=None, estp_auc = None, aupr = None, estp_aupr = None, precision=None, estp_precision = None, recall=None, estp_recall = None, f1=None):
        if step is None:
            step = self.last_step
        metrics["step"] = step
        if accuracy is not None:
            metrics["accuracy"] = accuracy
        if estp_accuracy is not None:
            metrics["estp_accuracy"] = estp_accuracy
        if auc is not None:
            metrics["auc"] = auc
        if estp_auc is not None:
            metrics["estp_auc"] = estp_auc
        if aupr is not None:
            metrics["aupr"] = aupr
        if estp_aupr is not None:
            metrics["estp_aupr"] = estp_aupr
        if precision is not None:
            metrics["precision"] = precision
        if estp_precision is not None:
            metrics["estp_precision"] = estp_precision
        if recall is not None:
            metrics["recall"] = recall
        if estp_recall is not None:
            metrics["estp_recall"] = estp_recall
        if f1 is not None:
            metrics["f1"] = f1
        wandb.log(metrics)
        self.last_step = step

    def finish(self):
        wandb.finish()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

#----------------------------------------
# RGCN Utils


def block_diag(m):
    """
    Source: https://gist.github.com/yulkang/2e4fc3061b45403f455d7f4c316ab168
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    """

    device = 'cuda' if m.is_cuda else 'cpu'  # Note: Using cuda status of m as proxy to decide device

    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    dim = m.dim()
    n = m.shape[-3]

    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]

    m2 = m.unsqueeze(-2)

    eye = attach_dim(torch.eye(n, device=device).unsqueeze(-2), dim - 3, 1)

    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append))

def stack_matrices(triples, num_nodes, num_rels, vertical_stacking=True, device='cpu'):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    """
    assert triples.dtype == torch.long

    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical_stacking else (n, r * n)

    fr, to = triples[:, 0], triples[:, 2]
    offset = triples[:, 1] * n
    if vertical_stacking:
        fr = offset + fr
    else:
        to = offset + to

    indices = torch.cat([fr[:, None], to[:, None]], dim=1).to(device)

    assert indices.size(0) == triples.size(0)
    assert indices[:, 0].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[:, 1].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices, size

def sum_sparse(indices, values, size, row_normalisation=True, device='cpu'):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    Arguments are interpreted as defining sparse matrix.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/util/util.py#L304
    """

    assert len(indices.size()) == len(values.size()) + 1

    k, r = indices.size()

    if not row_normalisation:
        # Transpose the matrix for column-wise normalisation
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=device)
    if device == 'cuda':
        values = torch.cuda.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    else:
        values = torch.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    sums = torch.spmm(values, ones)
    sums = sums[indices[:, 0], 0]

    return sums.view(k)