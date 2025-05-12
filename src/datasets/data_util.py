from collections import namedtuple, Counter
import numpy as np
import pandas as pd 

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree
from torch_geometric.data import Data, HeteroData

from ogb.nodeproppred import PygNodePropPredDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import h5py

def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def load_processed_graph(filepath: str) -> Data:
    """
    Loads the PyTorch Geometric Data object from disk.
    """
    data = torch.load(filepath, weights_only = False)
    if not isinstance(data, Data):
        raise ValueError("Loaded object is not a torch_geometric.data.Data object.")
    return data

def load_processed_multidim_graph(filepath: str) -> Data:
    """
    Loads the PyTorch Geometric Data object from disk.
    """
    data = torch.load(filepath, weights_only = False)
    if not isinstance(data, HeteroData):
        raise ValueError("Loaded object is not a torch_geometric.data.HeteroDatas object.")
    return data

def load_h5_graph(PATH, LABEL_PATH, ppi):
    f = h5py.File(f'{PATH}/{ppi}_multiomics.h5', 'r')
    # Build edge indices from the network matrix
    network = f['network'][:]
    src, dst = np.nonzero(network)
    edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

    # Load node features and assign a node "name" attribute if desired
    features = f['features'][:]
    x = torch.from_numpy(features)
    num_nodes = x.size(0)
    node_name = f['gene_names'][...,-1].astype(str)

    # Retrieve gene names and create a mapping: gene name -> node index
    gene_name = f['gene_names'][...,-1].astype(str)
    gene_map = {g: i for i, g in enumerate(gene_name)}  # gene name -> node index

    # Originally, the code combined several label arrays but then reads a health.tsv.
    # Here we read the health.tsv file and extract the symbols.
    # Ensure that PATH is defined in your environment.
    label_df = pd.read_csv(LABEL_PATH, sep='\t').astype(str) # TODO fix this for druggable gene prediction
    label_symbols = label_df['symbol'].tolist()

    # Determine positive nodes: indices that appear in both the health.tsv and gene_name list
    mask = [gene_map[g] for g in sorted(list(set(label_symbols) & set(gene_name)))]

    # Randomly select negative samples from those nodes not in the positive mask.
    np.random.seed(42)
    all_indices = set(range(len(gene_name)))
    negative_candidates = sorted(list(all_indices - set(mask)))
    neg_sample_size = min(len(mask), len(gene_name) - len(mask))
    neg_mask = np.random.choice(negative_candidates, size=neg_sample_size, replace=False).tolist()

    print("Negative mask indices:", neg_mask)

    # Create a label vector (1 for positive, 0 for negative)
    y = torch.zeros(len(gene_name), dtype=torch.float)
    y[mask] = 1
    y = y.unsqueeze(1)  # shape: [num_nodes, 1]

    # Combine positive and negative indices for the split
    final_mask = mask + neg_mask
    final_labels = y[final_mask].squeeze(1).numpy()  # converting to numpy for stratification

    # Split indices into train, test, and validation sets using stratification
    train_idx, test_idx, _, _ = train_test_split(final_mask, final_labels, test_size=0.2,
                                                    shuffle=True, stratify=final_labels, random_state=42)
    train_idx, val_idx, _, _ = train_test_split(train_idx, y[train_idx].numpy().squeeze(1),
                                                test_size=0.2, shuffle=True,
                                                stratify=y[train_idx].numpy().squeeze(1), random_state=42)

    # Create boolean masks for all nodes
    train_mask = torch.zeros(len(gene_name), dtype=torch.bool)
    test_mask = torch.zeros(len(gene_name), dtype=torch.bool)
    val_mask = torch.zeros(len(gene_name), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    val_mask[val_idx] = True

    # Add self-loops to the edge_index
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Build the PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask.unsqueeze(1)  # unsqueeze if you want to mimic the original shape
    data.test_mask = test_mask.unsqueeze(1)
    data.val_mask = val_mask.unsqueeze(1)
    data.name = node_name  # optional: storing node names

    return data#, gene_map

def train_val_test_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the nodes of a PyTorch Geometric Data object into train, validation,
    and test sets using stratification based on data.y (labels).

    Args:
        data (torch_geometric.data.Data): The input graph data object.
        train_ratio (float): Proportion of nodes to be used for training.
        val_ratio (float): Proportion of nodes to be used for validation.
        test_ratio (float): Proportion of nodes to be used for testing.

    Returns:
        train_mask (torch.BoolTensor): Boolean mask for training nodes.
        val_mask (torch.BoolTensor): Boolean mask for validation nodes.
        test_mask (torch.BoolTensor): Boolean mask for test nodes.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Determine the number of nodes
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    indices = np.arange(num_nodes)

    # Get labels as a 1D numpy array (squeeze in case data.y has an extra dim)
    if torch.is_tensor(data.y):
        labels = data.y.cpu().numpy().squeeze()
    else:
        labels = np.array(data.y).squeeze()

    # First, split into (train + val) and test sets
    train_val_indices, test_indices, train_val_labels, _ = train_test_split(
        indices, labels, test_size=test_ratio, stratify=labels, random_state=42)

    # Next, split the train_val_indices into training and validation sets.
    # Calculate the relative validation ratio from the (train + val) split.
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, train_val_labels, test_size=relative_val_ratio, stratify=train_val_labels, random_state=42)

    # Create boolean masks for each split
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Optionally, attach the masks to the data object for later use
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

def load_dataset(dataset_name):
    # if dataset_name == "ogbn-arxiv":
    #     dataset = PygNodePropPredDataset(name='ogbn-arxiv', root="./data")
    #     graph = dataset[0]
    #     num_nodes = graph.x.shape[0]
    #     graph.edge_index = to_undirected(graph.edge_index)
    #     graph.edge_index = remove_self_loops(graph.edge_index)[0]
    #     graph.edge_index = add_self_loops(graph.edge_index)[0]
    #     split_idx = dataset.get_idx_split()
    #     train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    #     if not torch.is_tensor(train_idx):
    #         train_idx = torch.as_tensor(train_idx)
    #         val_idx = torch.as_tensor(val_idx)
    #         test_idx = torch.as_tensor(test_idx)
    #     train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    #     val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    #     test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    #     graph.train_mask, graph.val_mask, graph.test_mask = train_mask, val_mask, test_mask
    #     graph.y = graph.y.view(-1)
    #     graph.x = scale_feats(graph.x)
        
    #     num_features = dataset.num_features
    #     num_classes = dataset.num_classes

    if dataset_name == "custom_synthetic":
        graph = load_processed_graph("data/synthetic/synthetic_graph.pt")
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

        graph = train_val_test_split(graph, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


    elif dataset_name == "custom_synthetic_multidim":
        graph = load_processed_graph("data/synthetic/synthetic_graph_multidim.pt")
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

        graph = train_val_test_split(graph, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    elif dataset_name in ['CPDB', 'IRefIndex_2015', 'IRefIndex', 'PCNet', 'STRINGdb']:
        graph = load_h5_graph(PATH='data/real/smg_data', LABEL_PATH='data/real/labels/NIHMS80906-small_mol-and-bio-druggable.tsv', ppi=dataset_name)
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cdgps', 'IRefIndex_2015_cdgps', 'IRefIndex_cdgps', 'PCNet_cdgps', 'STRINGdb_cdgps', 'CPDB_cdgps_shuffled']:
        graph = load_processed_graph(f'data/real/multidim_graph/6d/{dataset_name}_multiomics.pt')
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

    elif dataset_name in ["CPDB_cdgp", "IRefIndex_2015_cdgp", "IRefIndex_cdgp", "PCNet_cdgp", "STRINGdb_cdgp",
        "CPDB_cdgs", "IRefIndex_2015_cdgs", "IRefIndex_cdgs", "PCNet_cdgs", "STRINGdb_cdgs",
        "CPDB_cdps", "IRefIndex_2015_cdps", "IRefIndex_cdps", "PCNet_cdps", "STRINGdb_cdps",
        "CPDB_cgps", "IRefIndex_2015_cgps", "IRefIndex_cgps", "PCNet_cgps", "STRINGdb_cgps",
        "CPDB_dgps", "IRefIndex_2015_dgps", "IRefIndex_dgps", "PCNet_dgps", "STRINGdb_dgps"]:
        graph = load_processed_graph(f'data/real/multidim_graph/5d/{dataset_name}_multiomics.pt')
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cdg', 'CPDB_cdp', 'CPDB_cds', 'CPDB_cgp', 'CPDB_cgs', 'CPDB_cps', 'CPDB_dgp', 'CPDB_dgs', 'CPDB_dps', 'CPDB_gps']:
        graph = load_processed_graph(f'data/real/multidim_graph/4d/{dataset_name}_multiomics.pt')
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cd','CPDB_cg','CPDB_cp','CPDB_cs','CPDB_dg','CPDB_dp','CPDB_ds','CPDB_gp','CPDB_gs','CPDB_ps']:
        graph = load_processed_graph(f'data/real/multidim_graph/3d/{dataset_name}_multiomics.pt')
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

    elif dataset_name in ['CPDB_c', 'CPDB_d', 'CPDB_g', 'CPDB_p', 'CPDB_s']:
        graph = load_processed_graph(f'data/real/multidim_graph/2d/{dataset_name}_multiomics.pt')
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cdgps_CNA_GE_METH', 'CPDB_cdgps_CNA_GE_MF', 'CPDB_cdgps_CNA_METH_MF', 'CPDB_cdgps_CNA', 'CPDB_cdgps_GE_METH_MF', 'CPDB_cdgps_GE', 'CPDB_cdgps_METH', 'CPDB_cdgps_MF', 'CPDB_cdgps_random_features']:
        graph = load_processed_graph(f'data/real/multidim_graph/6d/feature_ablations/{dataset_name}.pt')
        num_features = graph.x.shape[1]
        num_classes = graph.y.max().item() + 1

    else:
        dataset = Planetoid("", dataset_name, transform=T.NormalizeFeatures())
        graph = dataset[0]
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]

        num_features = dataset.num_features
        num_classes = dataset.num_classes

    print('Loaded dataset: ', dataset_name)
    print('Number of features: ', num_features)
    print('Number of classes: ', num_classes)
    return graph, (num_features, num_classes)


def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(root="./data", name=dataset_name)
    dataset = list(dataset)
    graph = dataset[0]


    if graph.x == None:
        if graph.y and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g in dataset:
                feature_dim = max(feature_dim, int(g.y.max().item()))
            
            feature_dim += 1
            for i, g in enumerate(dataset):
                node_label = g.y.view(-1)
                feat = F.one_hot(node_label, num_classes=int(feature_dim)).float()
                dataset[i].x = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g in dataset:
                feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
                degrees.extend(degree(g.edge_index[0]).tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for i, g in enumerate(dataset):
                degrees = degree(g.edge_index[0])
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
                feat = F.one_hot(degrees.to(torch.long), num_classes=int(feature_dim)).float()
                g.x = feat
                dataset[i] = g

    else:
        print("******** Use `attr` as node features ********")
    feature_dim = int(graph.num_features)

    labels = torch.tensor([x.y for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    for i, g in enumerate(dataset):
        dataset[i].edge_index = remove_self_loops(dataset[i].edge_index)[0]
        dataset[i].edge_index = add_self_loops(dataset[i].edge_index)[0]
    #dataset = [(g, g.y) for g in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
    return dataset, (feature_dim, num_classes)
