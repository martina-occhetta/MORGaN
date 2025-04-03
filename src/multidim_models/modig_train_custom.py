import argparse
import os

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard.writer import SummaryWriter

from torch_geometric.data import Data

import sys
sys.path.append('/Users/bty416/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/martina/01_PhD/05_Projects/04_Druggable-genes/SMG-DG')

from src.multidim_models.modig import MODIG
from src.multidim_models.modig_utils import *

from src.datasets.data_util import load_processed_graph, load_h5_graph
from src.utils import set_random_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_relations(data: Data) -> dict:
    """
    Splits a multi-relational graph into separate homogeneous graphs,
    one per relation type.
    
    Args:
        data (Data): A PyTorch Geometric Data object with at least the
            attributes:
                - x: Node features [num_nodes, num_features]
                - edge_index: Edge index tensor [2, num_edges]
                - edge_type: Edge type tensor [num_edges]
                
    Returns:
        dict: A dictionary where each key is a relation type (int) and
              the value is a Data object containing the subgraph for that relation.
    """
    relation_graphs = []
    # Get unique relation types
    unique_relations = torch.unique(data.edge_type)
    
    for rel in unique_relations:
        # Create a mask for the current relation type
        mask = data.edge_type == rel
        # Filter edges based on the mask
        edge_index_rel = data.edge_index[:, mask]
        
        # Create a new graph with the same node features but only these edges
        new_data = Data(x=data.x, edge_index=edge_index_rel, y=data.y)
        
        # If additional edge attributes exist, filter them as well.
        # if hasattr(data, 'edge_attr'):
        #     new_data.edge_attr = data.edge_attr[mask]
            
        relation_graphs.append(new_data)
        
    return relation_graphs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MODIG with cross-validation and save model to file')
    parser.add_argument('-t', '--title', help='the name of running experiment',
                        dest='title',
                        default=None,
                        type=str
                        )
    parser.add_argument('-ppi', '--ppi', help='the chosen type of PPI',
                        dest='ppi',
                        default='CPDB',
                        type=str
                        )
    parser.add_argument('-omic', '--omic', help='the chosen node attribute [multiomic, snv, cnv, mrna, dm]',
                        dest='omic',
                        default='multiomic',
                        type=str
                        )
    parser.add_argument('-cancer', '--cancer', help='the model on pancan or specific cancer type',
                        dest='cancer',
                        default='pancan',
                        type=str
                        )
    parser.add_argument('-e', '--epochs', help='maximum number of epochs (default: 1000)',
                        dest='epochs',
                        default=1000,
                        type=int
                        )
    parser.add_argument('-p', '--patience', help='patience (default: 20)',
                        dest='patience',
                        default=20,
                        type=int
                        )
    parser.add_argument('-dp', '--dropout', help='the dropout rate (default: 0.25)',
                        dest='dp',
                        default=0.25,
                        type=float
                        )
    parser.add_argument('-lr', '--learningrate', help='the learning rate (default: 0.001)',
                        dest='lr',
                        default=0.001,
                        type=float
                        )
    parser.add_argument('-wd', '--weightdecay', help='the weight decay (default: 0.0005)',
                        dest='wd',
                        default=0.0005,
                        type=float
                        )
    parser.add_argument('-hs1', '--hiddensize1', help='the hidden size of first convolution layer (default: 300)',
                        dest='hs1',
                        default=300,
                        type=int
                        )
    parser.add_argument('-hs2', '--hiddensize2', help='the hidden size of second convolution layer (default: 100)',
                        dest='hs2',
                        default=100,
                        type=int
                        )
    parser.add_argument('-thr_go', '--thr_go', help='the threshold for GO semantic similarity (default: 0.8)',
                        dest='thr_go',
                        default=0.8,
                        type=float
                        )
    parser.add_argument('-thr_seq', '--thr_seq', help='the threshold for gene sequence similarity (default: 0.5)',
                        dest='thr_seq',
                        default=0.5,
                        type=float
                        )
    parser.add_argument('-thr_exp', '--thr_exp', help='the threshold for tissue co-expression pattern (default: 0.8)',
                        dest='thr_exp',
                        default=0.8,
                        type=float
                        )
    parser.add_argument('-thr_path', '--thr_path', help='the threshold of gene pathway co-occurrence (default: 0.5)',
                        dest='thr_path',
                        default=0.5,
                        type=float
                        )
    parser.add_argument('-seed', '--seed', help='the random seed (default: 42)',
                        dest='seed',
                        default=42,
                        type=int
                        )
    parser.add_argument('-data', '--dataset', help='the dataset name (default: CPDB_cdgps)',
                        dest='dataset',
                        default='CPDB_cdgps',
                        type=str
                        )
    args = parser.parse_args()
    return args


def main(args):

    dataset_name = 'CPDB_cdgps'

    if dataset_name in ['CPDB', 'IRefIndex_2015', 'IRefIndex', 'PCNet', 'STRINGdb']:
        multidim_graph = load_h5_graph(PATH='data/real/smg_data', LABEL_PATH='data/real/labels/NIHMS80906-small_mol-and-bio-druggable.tsv', ppi=dataset_name)
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cdgps', 'IRefIndex_2015_cdgps', 'IRefIndex_cdgps', 'PCNet_cdgps', 'STRINGdb_cdgps', 'CPDB_cdgps_shuffled']:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/6d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ["CPDB_cdgp", "IRefIndex_2015_cdgp", "IRefIndex_cdgp", "PCNet_cdgp", "STRINGdb_cdgp",
        "CPDB_cdgs", "IRefIndex_2015_cdgs", "IRefIndex_cdgs", "PCNet_cdgs", "STRINGdb_cdgs",
        "CPDB_cdps", "IRefIndex_2015_cdps", "IRefIndex_cdps", "PCNet_cdps", "STRINGdb_cdps",
        "CPDB_cgps", "IRefIndex_2015_cgps", "IRefIndex_cgps", "PCNet_cgps", "STRINGdb_cgps",
        "CPDB_dgps", "IRefIndex_2015_dgps", "IRefIndex_dgps", "PCNet_dgps", "STRINGdb_dgps"]:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/5d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cdg', 'CPDB_cdp', 'CPDB_cds', 'CPDB_cgp', 'CPDB_cgs', 'CPDB_cps', 'CPDB_dgp', 'CPDB_dgs', 'CPDB_dps', 'CPDB_gps']:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/4d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cd','CPDB_cg','CPDB_cp','CPDB_cs','CPDB_dg','CPDB_dp','CPDB_ds','CPDB_gp','CPDB_gs','CPDB_ps']:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/3d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_c', 'CPDB_d', 'CPDB_g', 'CPDB_p', 'CPDB_s']:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/2d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    print(f"Loaded multi-relational graph with {multidim_graph.num_nodes} nodes and {multidim_graph.edge_index.size(1)} edges")

    multidim_graph.x = multidim_graph.x.float()
    
    # Split the multi-relational graph into separate homogeneous graphs (one per relation type)
    relation_graphs = split_relations(multidim_graph)
    print(f"Split multi-relational graph into {len(relation_graphs)} relation-specific graphs.")
    
    final_gene_node = multidim_graph.node_names
    input_dim = multidim_graph.x.shape[1]
    idx_list = np.arange(multidim_graph.num_nodes)
    label_list = multidim_graph.y.cpu().numpy()  # if you need it as a numpy array
    # Convert the dictionary into a list to be used by the model
    graphlist_adj = relation_graphs
    
    def train(mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).to(device))

        acc = metrics.accuracy_score(label.cpu(), np.round(
            torch.sigmoid(output[mask]).cpu().detach().numpy()))
        loss.backward()
        optimizer.step()

        del output
        return loss.item(), acc

    @torch.no_grad()
    def test(mask, label):
        model.eval()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).to(device))

        acc = metrics.accuracy_score(label.cpu(), np.round(
            torch.sigmoid(output[mask]).cpu().detach().numpy()))
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(label.to('cpu'), pred)
        pr, rec, _ = metrics.precision_recall_curve(label.to('cpu'), pred)
        aupr = metrics.auc(rec, pr)

        return pred, loss.item(), acc, auroc, aupr

    file_save_path = os.path.join('modig_results/modig', args['dataset_name'])
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)

    AUC = np.zeros(shape=(10, 5))
    AUPR = np.zeros(shape=(10, 5))
    ACC = np.zeros(shape=(10, 5))

    pred_all = []
    label_all = []

    seeds = [0,1,2]
    results =  []

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        # Extract predefined splits and move to the device
        train_mask = multidim_graph.train_mask.to(device)
        val_mask   = multidim_graph.val_mask.to(device)
        test_mask  = multidim_graph.test_mask.to(device)
        train_label = multidim_graph.y[train_mask].to(device)
        val_label   = multidim_graph.y[val_mask].to(device)
        test_label  = multidim_graph.y[test_mask].to(device)

        print("\nTraining using predefined splits ...")

        model = MODIG(
                nfeat=input_dim, hidden_size1=args['hs1'], hidden_size2=args['hs2'], dropout=args['dp'])
        model.to(device)
        optimizer = optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=args['wd'])
        model_save_file = os.path.join(file_save_path + '_modig.pth')

        early_stopping = EarlyStopping(
        patience=args['patience'], verbose=True)

        for epoch in range(1, args['epochs']+1):
            _, _ = train(train_mask, train_label)
            _, loss_val, _, _, _ = test(val_mask, val_label)

            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print(f"Early stopping at the epoch {epoch}")
                break

            #torch.cuda.empty_cache()

        pred, _, ACC[i], AUC[i], AUPR[i] = test(
                val_mask, val_label)

        pred_all.append(pred)
        label_all.append(val_label.to('cpu'))

    print('Mean AUC', AUC.mean())
    print('Var AUC', AUC.var())
    print('Mean AUPR', AUPR.mean())
    print('Var AUPR', AUPR.var())
    print('Mean ACC', ACC.mean())
    print('Var ACC', ACC.var())

    torch.save(pred_all, os.path.join(file_save_path, 'pred_all.pkl'))
    torch.save(label_all, os.path.join(file_save_path, 'label_all.pkl'))

    ## TODO
    # Use all label to train a final model
    all_mask = torch.LongTensor(idx_list)
    all_label = torch.FloatTensor(label_list).reshape(-1, 1)

    model = MODIG(nfeat=input_dim, hidden_size1=args['hs1'],
                  hidden_size2=args['hs2'], dropout=args['dp'])
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    for epoch in range(1, args['epochs']+1):
        print(epoch)
        _, _ = train(all_mask.to(device), all_label.to(device))
        #torch.cuda.empty_cache()

    output = model(graphlist_adj)

    pred = torch.sigmoid(output).cpu().detach().numpy()
    pred2 = torch.sigmoid(output[~all_mask]).cpu().detach().numpy()
    torch.save(pred, os.path.join(file_save_path, args['ppi'] + '_pred.pkl'))
    torch.save(all_label, os.path.join(
        file_save_path, args['ppi'] + '_label.pkl'))
    torch.save(pred2, os.path.join(file_save_path, args['ppi'] + '_pred2.pkl'))

    pd.Series(final_gene_node).to_csv(os.path.join(file_save_path,
                                                   'final_gene_node.csv'), index=False, header=False)

    # plot_average_PR_curve(pred_all, label_all, file_save_path)
    # plot_average_ROC_curve(pred_all, label_all, file_save_path)

if __name__ == '__main__':

    args = parse_args()
    args_dic = vars(args)
    print('args_dict', args_dic)

    main(args_dic)
    print('The Training is finished!')