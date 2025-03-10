# File: src/data/synthetic_data.py

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx
from itertools import combinations

def generate_synthetic_graph(num_nodes=500,
                             num_expr_features=498,
                             num_methylation_features=486,
                             num_cnv_features=491,
                             druggable_ratio=0.33,
                             seed=42,
                             multidim_edges=False,
                             hetero=False):
    """
    Generates a synthetic graph dataset with biologically inspired multiomic features.
    
    Each node represents a gene with a feature vector composed of:
      - Gene Expression: Simulated via a log-normal distribution.
      - Methylation: Simulated via a beta distribution.
      - CNV: Simulated as integer copy number variations between -2 and 2.
      - SNVs: Simulated via a Poisson distribution.
    
    Druggable genes (label=1) are simulated by embedding a distinct multiomic signature:
      - Increased gene expression
      - Decreased methylation
      - Increased CNV
      (The SNV feature is left unmodified.)
      
    Additionally, node names are assigned as "GENE1", "GENE2", ..., "GENEn".
    
    If multidim_edges is False (default), a single edge type (PPI, type 0) is used.
    If multidim_edges is True, edges are generated for five dimensions:
      0: PPI (base graph from Barabási–Albert + extra connectivity among druggable genes)
      1: Sequence similarity
      2: Semantic similarity
      3: Co-expression
      4: Pathway co-occurrence
    
    Args:
        num_nodes (int): Number of genes/nodes.
        num_expr_features (int): Number of gene expression features.
        num_methylation_features (int): Number of methylation features.
        num_cnv_features (int): Number of CNV features.
        druggable_ratio (float): Proportion of nodes labeled as druggable.
        seed (int): Random seed for reproducibility.
        multidim_edges (bool): If True, generate graph with multiple edge dimensions.
    
    Returns:
        Data: A PyTorch Geometric Data object containing the synthetic graph.
    """
    np.random.seed(seed)

    # 1. Generate a base graph using the Barabási–Albert model.
    m = 3  # Each new node attaches to m existing nodes.
    G = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=seed)
    
    # 2. Simulate multiomic features.
    # Gene Expression: Log-normal distribution then standardize.
    expr = np.random.lognormal(mean=1.0, sigma=0.5, size=(num_nodes, num_expr_features)).astype(np.float32)
    expr = (expr - expr.mean(axis=0)) / expr.std(axis=0)
    
    # Methylation: Beta distribution then standardize.
    meth = np.random.beta(a=2, b=5, size=(num_nodes, num_methylation_features)).astype(np.float32)
    meth = (meth - meth.mean(axis=0)) / meth.std(axis=0)
    
    # CNV: Integers between -2 and 2 then standardize.
    cnv = np.random.randint(low=-2, high=3, size=(num_nodes, num_cnv_features)).astype(np.float32)
    cnv = (cnv - cnv.mean(axis=0)) / cnv.std(axis=0)
    
    # SNVs: Simulated via a Poisson distribution (one extra feature) then standardize.
    snv = np.random.poisson(lam=2, size=(num_nodes, 1)).astype(np.float32)
    snv = (snv - snv.mean(axis=0)) / snv.std(axis=0)
    
    # Concatenate all features.
    raw_features = np.concatenate([expr, meth, cnv, snv], axis=1)
    
    # 3. Identify druggable genes and embed a multiomic signal.
    num_druggables = int(num_nodes * druggable_ratio)
    druggable_indices = np.random.choice(num_nodes, num_druggables, replace=False)
    
    # Signals for each modality (do not alter SNVs)
    signal_expr = 2.0   # Increase gene expression.
    signal_meth = -1.0  # Decrease methylation.
    signal_cnv  = 1.0   # Increase CNV.
    
    idx_expr_end = num_expr_features
    idx_meth_end = idx_expr_end + num_methylation_features
    idx_cnv_end = idx_meth_end + num_cnv_features
    raw_features[druggable_indices, :idx_expr_end] += signal_expr
    raw_features[druggable_indices, idx_expr_end:idx_meth_end] += signal_meth
    raw_features[druggable_indices, idx_meth_end:idx_cnv_end] += signal_cnv
    
    # 4. Create labels: 1 for druggable, 0 otherwise.
    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[druggable_indices] = 1
    
    if multidim_edges:
        np.savetxt('data/synthetic/synthetic_labels_multidim.tsv', labels, delimiter='\t', fmt='%d')
        np.save('data/synthetic/synthetic_features_multidim.npy', raw_features)
    else:
        np.savetxt('data/synthetic/synthetic_labels.tsv', labels, delimiter='\t', fmt='%d')
        np.save('data/synthetic/synthetic_features.npy', raw_features)
    
    # 5. Enhance connectivity among druggable genes (applied to the base graph).
    extra_edge_prob = 0.1
    for i in druggable_indices:
        for j in druggable_indices:
            if i < j and not G.has_edge(i, j):
                if np.random.rand() < extra_edge_prob:
                    G.add_edge(i, j)
    
    # 6. Assign node features, labels, and node names.
    for i in range(num_nodes):
        G.nodes[i]['x'] = raw_features[i]
        G.nodes[i]['y'] = int(labels[i])
        # G.nodes[i]['name'] = f"GENE{i+1}"
    
    if not multidim_edges:
        # Use the base graph edges as PPI edges; assign edge_type = 0 for all.
        for u, v in G.edges():
            G.edges[u, v]['edge_type'] = 0
        # Convert NetworkX graph to a PyTorch Geometric Data object.
        data = from_networkx(G, group_node_attrs=['x', 'y'])
        data.x = torch.tensor(raw_features, dtype=torch.float)
        data.y = torch.tensor(labels, dtype=torch.long)
    else:
        if hetero:
            # multidim_edges == True: generate multi-dimensional edges.
            # Start with base PPI edges from G.
            base_edges = list(G.edges())
            # Prepare a dictionary to store edges per type.
            edge_index_by_type = {0: [], 1: [], 2: [], 3: [], 4: []}

            # Add base PPI edges (edge type 0) in both directions for undirected connectivity.
            for u, v in base_edges:
                edge_index_by_type[0].extend([[u, v], [v, u]])

            # Define additional edge dimensions:
            # 1: Sequence similarity, 2: Semantic similarity, 3: Co-expression, 4: Pathway co-occurrence.
            additional_edge_types = [1, 2, 3, 4]
            additional_edge_prob = 0.05  # probability for generating extra edges in each additional dimension

            # Use combinations to iterate over unique node pairs (i, j) with i < j.
            for etype in additional_edge_types:
                for i, j in combinations(range(num_nodes), 2):
                    if np.random.rand() < additional_edge_prob:
                        # Add edge in both directions.
                        edge_index_by_type[etype].extend([[i, j], [j, i]])

            # Create the HeteroData object and assign node attributes under the "gene" node type.
            data = HeteroData()
            data['gene'].x = torch.tensor(raw_features, dtype=torch.float)
            data['gene'].y = torch.tensor(labels, dtype=torch.long)
            data['gene'].name = [f"GENE{i+1}" for i in range(num_nodes)]

            # Mapping from numeric edge type to a relation name.
            relation_mapping = {
                0: "ppi",
                1: "seq_sim",
                2: "sem_sim",
                3: "coexpr",
                4: "pathway_cooc"
            }
            
            # For each edge type, convert the list to a tensor and assign it under its relation key.
            for etype, relation in relation_mapping.items():
                edges = edge_index_by_type[etype]
                if len(edges) > 0:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                else:
                    # Create an empty edge index if no edges were sampled for this type.
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                data['gene', relation, 'gene'].edge_index = edge_index
        else:
            # multidim_edges == True: generate multi-dimensional edges.
            # if hetero is == False, generate a Data object with multidimensional edges
            
                # Define additional edge dimensions:
            base_edges = list(G.edges())
            edge_index_list = []
            edge_type_list = []
            # Add PPI edges (edge type 0); add both directions for undirected connectivity.
            for u, v in base_edges:
                edge_index_list.append([u, v])
                edge_index_list.append([v, u])
                edge_type_list.append(0)
                edge_type_list.append(0)
                
            # 1: Sequence similarity, 2: Semantic similarity, 3: Co-expression, 4: Pathway co-occurrence.
            additional_edge_types = [1, 2, 3, 4]
            additional_edge_prob = 0.05  # probability for generating extra edges in each additional dimension
            for etype in additional_edge_types:
                # For each pair of nodes (i, j) with i < j, sample an edge.
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        if np.random.rand() < additional_edge_prob:
                            edge_index_list.append([i, j])
                            edge_index_list.append([j, i])
                            edge_type_list.append(etype)
                            edge_type_list.append(etype)
            
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_type_list, dtype=torch.long)
            # Create Data object manually.
            data = Data(x=torch.tensor(raw_features, dtype=torch.float),
                        y=torch.tensor(labels, dtype=torch.long),
                        edge_index=edge_index,
                        edge_type=edge_type)
            # Save node names separately (Data objects do not support string tensors).
            data.name = [f"GENE{i+1}" for i in range(num_nodes)]

    
    if multidim_edges and hetero:
        torch.save(data, 'data/synthetic/synthetic_graph_multidim_hetero.pt')
        print("Synthetic data saved to 'data/synthetic/synthetic_graph_multidim_hetero.pt'")
    elif multidim_edges:
        torch.save(data, 'data/synthetic/synthetic_graph_multidim.pt')
        print("Synthetic data saved to 'data/synthetic/synthetic_graph_multidim.pt'")
    else:
        torch.save(data, 'data/synthetic/synthetic_graph.pt')
        print("Synthetic data saved to 'data/synthetic/synthetic_graph.pt'")

    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    if multidim_edges and hetero:
        print(f"Edge types: {data.edge_types}")
    elif multidim_edges:
        print(f"Number of edge types: {data.num_edge_types}")
    return data

if __name__ == "__main__":
    # Generate the synthetic multiomic graph dataset and print its summary.
    data = generate_synthetic_graph(multidim_edges=False, hetero=False)
    print("Synthetic Multiomic Graph Data Summary:")
    print(data)
    