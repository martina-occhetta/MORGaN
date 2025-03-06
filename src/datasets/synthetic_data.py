# File: src/data/synthetic_data.py

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def generate_synthetic_graph(num_nodes=500,
                       num_expr_features=498,
                       num_methylation_features=486,
                       num_cnv_features=491,
                       druggable_ratio=0.40,
                       seed=42):
    """
    Generates a synthetic graph dataset with biologically inspired multiomic features.
    
    Each node represents a gene with a feature vector composed of:
      - Gene Expression: Simulated via a log-normal distribution.
      - Methylation: Simulated via a beta distribution (values between 0 and 1).
      - CNV: Simulated as integer copy number variations between -2 and 2.
      
    The features for each modality are standardized and concatenated.
    
    druggable genes are simulated by embedding a distinct multiomic signature:
      - Increased gene expression (signal added)
      - Decreased methylation (signal subtracted)
      - Increased CNV (signal added)
      
    Extra edges are added among druggable genes to simulate enhanced connectivity.
    
    Args:
        num_nodes (int): Number of genes/nodes.
        num_expr_features (int): Number of gene expression features.
        num_methylation_features (int): Number of methylation features.
        num_cnv_features (int): Number of CNV features.
        druggable_ratio (float): Proportion of nodes labeled as cancer druggable genes.
        seed (int): Random seed for reproducibility.
    
    Returns:
        Data: A PyTorch Geometric Data object containing the graph.
    """
    np.random.seed(seed)

    # 1. Generate a base graph using the Barabási–Albert model.
    m = 3  # Each new node attaches to m existing nodes.
    G = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=seed)
    
    # 2. Simulate multiomic features:
    # Gene Expression: Use a log-normal distribution then standardize.
    expr = np.random.lognormal(mean=1.0, sigma=0.5, size=(num_nodes, num_expr_features)).astype(np.float32)
    expr = (expr - expr.mean(axis=0)) / expr.std(axis=0)
    
    # Methylation: Simulate with a beta distribution then standardize.
    meth = np.random.beta(a=2, b=5, size=(num_nodes, num_methylation_features)).astype(np.float32)
    meth = (meth - meth.mean(axis=0)) / meth.std(axis=0)
    
    # CNV: Simulate as integers between -2 and 2 then standardize.
    cnv = np.random.randint(low=-2, high=3, size=(num_nodes, num_cnv_features)).astype(np.float32)
    cnv = (cnv - cnv.mean(axis=0)) / cnv.std(axis=0)
    
    # Concatenate all features: gene expression, methylation, and CNV.
    raw_features = np.concatenate([expr, meth, cnv], axis=1)
    
    # 3. Identify druggable genes and embed a multiomic signal.
    num_druggables = int(num_nodes * druggable_ratio)
    druggable_indices = np.random.choice(num_nodes, num_druggables, replace=False)
    
    # Define modality-specific signals:
    signal_expr = 2.0   # Increase gene expression.
    signal_meth = -1.0  # Decrease methylation.
    signal_cnv  = 1.0   # Increase CNV.
    
    # Determine feature indices for slicing:
    idx_expr_end = num_expr_features
    idx_meth_end = idx_expr_end + num_methylation_features
    
    # Embed signals into the corresponding modalities for druggable genes.
    raw_features[druggable_indices, :idx_expr_end] += signal_expr
    raw_features[druggable_indices, idx_expr_end:idx_meth_end] += signal_meth
    raw_features[druggable_indices, idx_meth_end:] += signal_cnv
    
    # 4. Create labels: 1 for druggable genes, 0 for non-druggable genes.
    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[druggable_indices] = 1
    
    np.savetxt('data/synthetic/synthetic_labels.tsv', labels, delimiter='\t', fmt='%d')
    np.save('data/synthetic/synthetic_features.npy', raw_features)

    # 5. Enhance connectivity among druggable genes.
    extra_edge_prob = 0.1
    for i in druggable_indices:
        for j in druggable_indices:
            if i < j and not G.has_edge(i, j):
                if np.random.rand() < extra_edge_prob:
                    G.add_edge(i, j)
    
    # 6. Assign node features and labels to the graph.
    for i in range(num_nodes):
        G.nodes[i]['x'] = raw_features[i]
        G.nodes[i]['y'] = int(labels[i])
    
    # 7. Convert the NetworkX graph to a PyTorch Geometric Data object.
    data = from_networkx(G, group_node_attrs=['x', 'y'])
    data.x = torch.tensor(raw_features, dtype=torch.float)
    data.y = torch.tensor(labels, dtype=torch.long)

    torch.save(data, 'data/synthetic/synthetic_graph.pt')
    print("Synthetic data saved to 'data/synthetic/synthetic_graph.pt'")
    
    return data

if __name__ == "__main__":
    # Generate the synthetic multiomic graph dataset and print its summary.
    data = generate_toy_graph()
    print("Synthetic Multiomic Graph Data Summary:")
    print(data)
