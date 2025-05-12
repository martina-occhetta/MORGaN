import torch_geometric
from torch_geometric.data import Data
import torch
import pandas as pd
import numpy as np
import h5py
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split

def load_h5_graph(PATH: str, LABEL_PATH: str, ppi: str, modalities: list = None, seed: int = 42):
    '''
    Load a multi-omics graph from an HDF5 file and create a PyTorch Geometric Data object.
    Adds node features, edge indices, labels, and splits for training, testing, and validation.
    
    Parameters:
      PATH: Path to the directory containing the HDF5 file.
      LABEL_PATH: Path to the label file.
      ppi: A string used to construct the filename.
      modalities: List of feature modalities to include. For example, if you want to include
                  only MF and GE, pass ['MF', 'GE']. If None, all features are included.
    
    Returns:
      data: A PyTorch Geometric Data object with the selected node features and splits.
    '''
    # Load the HDF5 file and extract the network and features
    f = h5py.File(f'{PATH}/{ppi}_multiomics.h5', 'r')

    # Build edge indices from the network matrix
    network = f['network'][:]
    src, dst = np.nonzero(network)
    edge_index_ppi = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    edge_type_ppi = torch.full((edge_index_ppi.size(1),), 0, dtype=torch.long)

    # Load node features and assign a node "name" attribute 
    features = f['features'][:]
    # Decode the feature names (they may come as byte strings)
    feature_names = [fn.decode('utf-8') if isinstance(fn, bytes) else str(fn) for fn in f['feature_names'][:]]
    
    # If a list of modalities is provided, filter features accordingly
    if modalities is not None:
        # Determine which columns have prefixes matching one of the modalities
        selected_indices = [
            i for i, name in enumerate(feature_names)
            if any(name.startswith(mod) for mod in modalities)
        ]
        # Slice the features to keep only the selected ones
        features = features[:, selected_indices]
        # Optionally, update the feature_names list too
        feature_names = [feature_names[i] for i in selected_indices]
    x = torch.from_numpy(features)
    num_nodes = x.size(0)
    node_name = f['gene_names'][...,-1].astype(str)

    # Retrieve gene names and create a mapping: gene name -> node index
    gene_name = f['gene_names'][...,-1].astype(str)
    gene_map = {g: i for i, g in enumerate(gene_name)}  # gene name -> node index

    # Load the labels and determine positive and negative nodes
    label_df = pd.read_csv(LABEL_PATH, sep='\t').astype(str) 
    label_symbols = label_df['symbol'].tolist()

    # Determine positive nodes: indices that appear in both the label file and gene_name list
    mask = [gene_map[g] for g in sorted(list(set(label_symbols) & set(gene_name)))]

    # Randomly select negative samples from those nodes not in the positive mask.
    np.random.seed(seed)
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
    #edge_index, _ = add_self_loops(edge_index_ppi, num_nodes=num_nodes)

    # Build the PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index_ppi, edge_type = edge_type_ppi, y=y)
    data.train_mask = train_mask.unsqueeze(1)  # unsqueeze if you want to mimic the original shape
    data.test_mask = test_mask.unsqueeze(1)
    data.val_mask = val_mask.unsqueeze(1)
    data.name = node_name  # optional: storing node names

    return data#, gene_map

def load_h5_graph_random_features(PATH: str, LABEL_PATH: str, ppi: str, modalities: list = None, randomize_features: bool = True, seed: int = 42):
    """
    Load a multi-omics graph from an HDF5 file and create a PyTorch Geometric Data object.
    Adds node features, edge indices, labels, and splits for training, testing, and validation.
    
    Parameters:
      PATH: Path to the directory containing the HDF5 file.
      LABEL_PATH: Path to the label file.
      ppi: A string used to construct the filename.
      modalities: List of feature modalities to include. For example, if you want to include
                  only MF and GE, pass ['MF', 'GE']. If None, all features are included.
      randomize_features: If True, ignores the actual features from the file and generates 
                          random features (using a normal distribution) of the appropriate shape.
    
    Returns:
      data: A PyTorch Geometric Data object with the selected node features (or random features),
            edge indices, labels, and data splits.
    """

    # Load the HDF5 file 
    f = h5py.File(f'{PATH}/{ppi}_multiomics.h5', 'r')

    # Build edge indices from the network matrix
    network = f['network'][:]
    src, dst = np.nonzero(network)
    edge_index_ppi = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    edge_type_ppi = torch.full((edge_index_ppi.size(1),), 0, dtype=torch.long)

    # Read and decode feature names
    feature_names = [fn.decode('utf-8') if isinstance(fn, bytes) else str(fn)
                     for fn in f['feature_names'][:]]
    
    # If modalities are specified, filter feature indices accordingly
    if modalities is not None:
        selected_indices = [
            i for i, name in enumerate(feature_names)
            if any(name.startswith(mod) for mod in modalities)
        ]
        # Update the feature_names to only include those filtered features
        feature_names = [feature_names[i] for i in selected_indices]
        num_features = len(selected_indices)
    else:
        num_features = len(feature_names)
    
    # Determine number of nodes from gene names
    node_name = f['gene_names'][...,-1].astype(str)
    num_nodes = len(node_name)
    
    # --- Create node features ---
    if randomize_features:
        # Generate a random feature matrix with shape (num_nodes, num_features)
        # Using a normal distribution
        x = torch.randn(num_nodes, num_features)
    else:
        # Load the full feature matrix from file
        features = f['features'][:]  # shape: [num_nodes, total_features]
        if modalities is not None:
            features = features[:, selected_indices]
        x = torch.from_numpy(features)
    
    # Create a mapping from gene name to node index
    gene_map = {g: i for i, g in enumerate(node_name)}
    
    # --- Load labels and determine positive/negative nodes ---
    label_df = pd.read_csv(LABEL_PATH, sep='\t').astype(str)
    label_symbols = label_df['symbol'].tolist()
    # Get positive nodes: indices present in both the label file and the gene names list
    mask = [gene_map[g] for g in sorted(list(set(label_symbols) & set(node_name)))]
    
    # Randomly select negative samples from nodes not in the positive mask.
    np.random.seed(seed)
    all_indices = set(range(len(node_name)))
    negative_candidates = sorted(list(all_indices - set(mask)))
    neg_sample_size = min(len(mask), len(node_name) - len(mask))
    neg_mask = np.random.choice(negative_candidates, size=neg_sample_size, replace=False).tolist()

    print("Negative mask indices:", neg_mask)

    # Create label vector: 1 for positive, 0 for negative
    y = torch.zeros(len(node_name), dtype=torch.float)
    y[mask] = 1
    y = y.unsqueeze(1)

    # Combine positive and negative indices for stratified splits
    final_mask = mask + neg_mask
    final_labels = y[final_mask].squeeze(1).numpy()  # convert to numpy for stratification

    # Stratified train-test split
    train_idx, test_idx, _, _ = train_test_split(
        final_mask, final_labels, test_size=0.2, shuffle=True, stratify=final_labels, random_state=42
    )
    train_idx, val_idx, _, _ = train_test_split(
        train_idx, y[train_idx].numpy().squeeze(1), test_size=0.2, shuffle=True,
        stratify=y[train_idx].numpy().squeeze(1), random_state=42
    )

    # Create boolean masks for all nodes
    train_mask = torch.zeros(len(node_name), dtype=torch.bool)
    test_mask = torch.zeros(len(node_name), dtype=torch.bool)
    val_mask = torch.zeros(len(node_name), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    val_mask[val_idx] = True

    # --- Build the PyTorch Geometric Data object ---
    data = Data(x=x, edge_index=edge_index_ppi, edge_type=edge_type_ppi, y=y)
    data.train_mask = train_mask.unsqueeze(1)
    data.test_mask = test_mask.unsqueeze(1)
    data.val_mask = val_mask.unsqueeze(1)
    data.name = node_name  # store node names as an attribute
    # data.feature_names = feature_names  # store the associated feature modality names

    return data

def load_h5_graph_shuffle(PATH: str, LABEL_PATH: str, ppi: str, shuffle_features: bool = False, seed: int = 42):
    '''
    Load a multi-omics graph from an HDF5 file and create a PyTorch Geometric Data object.
    Add the node features, edge indices, and labels to the object.
    Split the nodes into training, testing, and validation sets.
    Add multiple edge types from additional networks.
    
    If shuffle_features is True, both the rows (nodes) and columns (features) of the feature 
    matrix are shuffled. This breaks the association between nodes and their original features,
    serving as an ablation study on the importance of feature organization.
    '''
    # Load the HDF5 file and extract the network and features
    f = h5py.File(f'{PATH}/{ppi}_multiomics.h5', 'r')

    # Build edge indices from the network matrix
    network = f['network'][:]
    src, dst = np.nonzero(network)
    edge_index_ppi = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    edge_type_ppi = torch.full((edge_index_ppi.size(1),), 0, dtype=torch.long)

    # Load node features and assign a node "name" attribute 
    features = f['features'][:]
    if shuffle_features:
        # Set a fixed seed for reproducibility
        np.random.seed(seed)
        # Shuffle rows (nodes)
        row_perm = np.random.permutation(features.shape[0])
        features = features[row_perm, :]
        # Shuffle columns (features)
        col_perm = np.random.permutation(features.shape[1])
        features = features[:, col_perm]
    x = torch.from_numpy(features)
    num_nodes = x.size(0)
    node_name = f['gene_names'][...,-1].astype(str)

    # Retrieve gene names and create a mapping: gene name -> node index
    gene_name = f['gene_names'][...,-1].astype(str)
    gene_map = {g: i for i, g in enumerate(gene_name)}  # gene name -> node index

    # Load the labels and determine positive and negative nodes
    label_df = pd.read_csv(LABEL_PATH, sep='\t').astype(str) 
    label_symbols = label_df['symbol'].tolist()

    # Determine positive nodes: indices that appear in both the label file and gene_name list
    mask = [gene_map[g] for g in sorted(list(set(label_symbols) & set(gene_name)))]

    # Randomly select negative samples from those nodes not in the positive mask.
    np.random.seed(seed)
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

    # Build the PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index_ppi, edge_type=edge_type_ppi, y=y)
    data.train_mask = train_mask.unsqueeze(1)
    data.test_mask = test_mask.unsqueeze(1)
    data.val_mask = val_mask.unsqueeze(1)
    data.name = node_name  # optional: storing node names

    return data

def add_all_edges(data, NETWORK_PATH: str, NETWORKS: list):

    # Add additional edges from the other networks
    for i, network_name in enumerate(NETWORKS):
        adj_matrix = pd.read_csv(f'{NETWORK_PATH}/network_{network_name}.csv', index_col=0)
        data = add_edge_type(data, adj_matrix, i+1)

    return data

def add_edge_type(data, adj_matrix: pd.DataFrame, edge_type_label: int):
    '''
    Adds edges from an additional binary adjacency matrix to the existing PyG data object.
    
    Parameters:
    - data: PyTorch Geometric Data object (already contains data.name with gene names)
    - network: adjacency matrix of the additional edges in the form of a pandas DataFrame
    - edge_type: integer that will label all these additional edges (e.g., 2)
    
    Returns:
    - data: updated Data object with the additional edges and their edge types appended.
    '''
    additional_row, additional_col = np.nonzero(adj_matrix)
    row_names = adj_matrix.index
    col_names = adj_matrix.columns

    # Create mapping from gene name (from the first network) to its node index
    gene_to_index = {gene: idx for idx, gene in enumerate(data.name)}

    # Convert these indices to the indices in the main graph using gene_to_index.
    global_rows = []
    global_cols = []

    # not_found = []

    for r, c in zip(additional_row, additional_col):
        gene_r = row_names[r]
        gene_c = col_names[c]
        # Check that both gene names exist in the main mapping
        if gene_r in gene_to_index and gene_c in gene_to_index:
            global_rows.append(gene_to_index[gene_r])
            global_cols.append(gene_to_index[gene_c])
        else:
            # if gene_r not in gene_to_index and gene_r not in not_found:
            #     not_found.append(gene_r)
            # if gene_c not in gene_to_index and gene_c not in not_found:
            #     not_found.append(gene_c)
            #print(f"Gene {gene_r} or {gene_c} not found in the main graph.")
            continue
    
    if not global_rows:  # No matching edges found
        print("No matching edges found.")
        return data
    
    # Create the edge_index for the additional edges
    additional_edge_index = torch.tensor(np.vstack((global_rows, global_cols)), dtype=torch.long)
    
    # Create an edge attribute for these additional edges, using the provided type value.
    additional_edge_attr = torch.full((additional_edge_index.size(1),), edge_type_label, dtype=torch.long)
    
    # Combine the additional edges with the existing ones.
    # We assume that data.edge_index and data.edge_type already exist.
    data.edge_index = torch.cat([data.edge_index, additional_edge_index], dim=1)
    data.edge_type = torch.cat([data.edge_type, additional_edge_attr], dim=0)
    
    return data

def load_h5_graph_with_external_edges(PATH: str,
                                      LABEL_PATH: str,
                                      ppi: str,
                                      NETWORK_PATH: str,
                                      NETWORKS: list,
                                      modalities: list = None,
                                      randomize_features: bool = False,
                                      seed: int = 42):
    """
    Load a multi-omics graph from an HDF5 file and create a PyTorch Geometric Data object.
    Instead of using the base PPI network from the HDF5 file, this function initializes
    the graph with node features and labels but uses no initial edges. Then, additional
    edges are added from external CSV files located in NETWORK_PATH based on the provided
    list of NETWORKS.
    
    Parameters:
      PATH (str): Directory containing the HDF5 file.
      LABEL_PATH (str): Path to the label file.
      ppi (str): Identifier (part of the filename) for the HDF5 file.
      NETWORK_PATH (str): Directory containing the CSV files for additional networks.
      NETWORKS (list): List of network names to use for external edges (e.g.,
                       ['coexpression', 'GO', 'domain_sim', 'sequence_sim', 'pathway_co_occurrence']).
      modalities (list, optional): List of feature modality prefixes to include (e.g., ['MF', 'GE']).
                                   If None, all features are used.
      randomize_features (bool, optional): If True, the node features will be replaced by random
                                             values (sampled from a normal distribution).
    
    Returns:
      data: A PyTorch Geometric Data object with:
           - x: Node features (filtered or randomized as requested)
           - edge_index: External edge indices (after adding from CSV files)
           - edge_type: Integer edge types (one per external network; base graph has no edges)
           - y: Labels (with 1 for positive and 0 for negative)
           - train_mask, test_mask, val_mask: Train/val/test splits derived via stratification
           - name: The node (gene) names extracted from the HDF5 file.
    """

    # --- Load the HDF5 file and extract node features ---
    f = h5py.File(f'{PATH}/{ppi}_multiomics.h5', 'r')

    # Decode feature names from the HDF5 file
    feature_names = [fn.decode('utf-8') if isinstance(fn, bytes) else str(fn)
                     for fn in f['feature_names'][:]]
    if modalities is not None:
        # Filter columns based on modalities prefixes
        selected_indices = [i for i, name in enumerate(feature_names)
                            if any(name.startswith(mod) for mod in modalities)]
        feature_names = [feature_names[i] for i in selected_indices]
        num_features = len(selected_indices)
    else:
        num_features = len(feature_names)
    
    # Load gene/node names (assumed stored as the last column in the 'gene_names' dataset)
    node_names = f['gene_names'][...,-1].astype(str)
    num_nodes = len(node_names)
    
    # Decide on node feature matrix: randomize or use true features
    if randomize_features:
        x = torch.randn(num_nodes, num_features)
    else:
        features = f['features'][:]  # full feature matrix with shape (num_nodes, total_features)
        if modalities is not None:
            features = features[:, selected_indices]
        x = torch.from_numpy(features)
    
    # --- Create an empty graph structure ---
    # Initialize empty edge_index and edge_type. The dimensions are set such that edge_index is 2 x 0.
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.empty((0,), dtype=torch.long)
    
    # --- Create labels and splits ---
    # Build mapping from gene name (node_names) to node index.
    gene_to_index = {gene: idx for idx, gene in enumerate(node_names)}
    
    # Read labels from the LABEL_PATH file.
    label_df = pd.read_csv(LABEL_PATH, sep='\t').astype(str)
    label_symbols = label_df['symbol'].tolist()
    
    # Identify positive nodes (gene symbols common to both the label file and node_names)
    mask = [gene_to_index[g] for g in sorted(list(set(label_symbols) & set(node_names)))]
    
    # Randomly sample negative nodes (those not in mask)
    np.random.seed(seed)
    all_indices = set(range(len(node_names)))
    negative_candidates = sorted(list(all_indices - set(mask)))
    neg_sample_size = min(len(mask), len(node_names) - len(mask))
    neg_mask = np.random.choice(negative_candidates, size=neg_sample_size, replace=False).tolist()
    
    print("Negative mask indices:", neg_mask)
    
    # Create label vector: 1 for positives, 0 for negatives
    y = torch.zeros(len(node_names), dtype=torch.float)
    y[mask] = 1
    y = y.unsqueeze(1)

    # Combine positive and negative samples for splits
    final_mask = mask + neg_mask
    final_labels = y[final_mask].squeeze(1).numpy()
    
    # Perform stratified splits into train, test, and validation sets
    train_idx, test_idx, _, _ = train_test_split(
        final_mask, final_labels, test_size=0.2, shuffle=True, stratify=final_labels, random_state=42
    )
    train_idx, val_idx, _, _ = train_test_split(
        train_idx, y[train_idx].numpy().squeeze(1), test_size=0.2, shuffle=True,
        stratify=y[train_idx].numpy().squeeze(1), random_state=42
    )
    
    # Create boolean masks
    train_mask = torch.zeros(len(node_names), dtype=torch.bool)
    test_mask  = torch.zeros(len(node_names), dtype=torch.bool)
    val_mask   = torch.zeros(len(node_names), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    val_mask[val_idx] = True

    # --- Build the PyTorch Geometric data object ---
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    data.train_mask = train_mask.unsqueeze(1)
    data.test_mask  = test_mask.unsqueeze(1)
    data.val_mask   = val_mask.unsqueeze(1)
    data.name = node_names  # store the gene (node) names
    # data.feature_names = feature_names

    # --- Add external edges from CSV files ---
    # This step appends all edges from the specified networks to the data object.
    data = add_all_edges(data, NETWORK_PATH, NETWORKS)
    
    return data

def randomize_edges(graph: Data, preserve_edge_types: bool = True) -> Data:
    """
    Randomize the edges in a graph while optionally preserving the edge type distribution.
    
    Parameters:
        graph (Data): PyTorch Geometric Data object containing the graph
        preserve_edge_types (bool): If True, preserves the number of edges of each type
                                   If False, completely randomizes all edges
    
    Returns:
        Data: New graph with randomized edges
    """
    import torch
    import numpy as np
    
    # Create a copy of the graph to avoid modifying the original
    new_graph = Data(
        x=graph.x.clone(),
        y=graph.y.clone(),
        train_mask=graph.train_mask.clone() if hasattr(graph, 'train_mask') else None,
        val_mask=graph.val_mask.clone() if hasattr(graph, 'val_mask') else None,
        test_mask=graph.test_mask.clone() if hasattr(graph, 'test_mask') else None,
        name=graph.name.copy() if hasattr(graph, 'name') else None
    )
    
    if preserve_edge_types:
        # Get unique edge types and their counts
        edge_types = graph.edge_type.unique()
        edge_type_counts = torch.bincount(graph.edge_type)
        
        # Initialize empty lists for new edges
        new_edge_index = []
        new_edge_type = []
        
        # For each edge type, create random edges while preserving count
        for edge_type in edge_types:
            count = edge_type_counts[edge_type].item()
            
            # Generate random source and target nodes
            num_nodes = graph.num_nodes
            src = torch.randint(0, num_nodes, (count,))
            dst = torch.randint(0, num_nodes, (count,))
            
            # Stack source and target nodes
            edge_index = torch.stack([src, dst], dim=0)
            
            # Create edge type tensor
            edge_type_tensor = torch.full((count,), edge_type, dtype=torch.long)
            
            # Append to lists
            new_edge_index.append(edge_index)
            new_edge_type.append(edge_type_tensor)
        
        # Concatenate all edges
        new_graph.edge_index = torch.cat(new_edge_index, dim=1)
        new_graph.edge_type = torch.cat(new_edge_type)
        
    else:
        # Completely randomize all edges
        num_edges = graph.edge_index.size(1)
        num_nodes = graph.num_nodes
        
        # Generate random source and target nodes
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        
        # Stack source and target nodes
        new_graph.edge_index = torch.stack([src, dst], dim=0)
        
        # If original graph had edge types, randomize them too
        if hasattr(graph, 'edge_type'):
            num_edge_types = graph.edge_type.max().item() + 1
            new_graph.edge_type = torch.randint(0, num_edge_types, (num_edges,), dtype=torch.long)
    
    return new_graph