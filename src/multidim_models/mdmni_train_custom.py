import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.utils import from_networkx

from torch_geometric.data import Data
import argparse
import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score
import faulthandler  
from datetime import datetime
import time

from mdmni import MDMNI_DGD
from mdmni_network import generate_graph
from mdmni_utils import *

import sys
sys.path.append('/Users/bty416/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/martina/01_PhD/05_Projects/04_Druggable-genes/SMG-DG')
from src.datasets.data_util import load_processed_graph, load_dataset, load_h5_graph
from src.utils import set_random_seed
# cuda = torch.cuda.is_available()
# torch.cuda.empty_cache()
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

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--networks', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6], help='Indices of networks to use')
    parser.add_argument('--thr_go', dest='thr_go', default=0.8, type=float, help='the threshold for GO semantic similarity')
    parser.add_argument('--thr_seq', dest='thr_seq', default=0.8, type=float, help='the threshold for sequence similarity')
    parser.add_argument('--thr_exp', dest='thr_exp', default=0.8, type=float, help='the threshold for gene co-expression pattern')
    parser.add_argument('--thr_path', dest='thr_path', default=0.6, type=float, help='the threshold of pathway co-occurrence')
    parser.add_argument('--thr_cpdb', dest='thr_cpdb', default=0.8, type=float, help='the threshold of CPDB PPI')
    parser.add_argument('--thr_string', dest='thr_string', default=0.8, type=float, help='the threshold of STRING PPI')
    parser.add_argument('--thr_domain', dest='thr_domain', default=0.3, type=float, help='the threshold of Domain similarity')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int, help='maximum number of epochs')
    parser.add_argument('--patience', dest='patience', default=120, type=int, help='waiting iterations when performance no longer improves')
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int, help='the dim of hidden layer')
    parser.add_argument('--output_dim', dest='output_dim', default=32, type=int, help='the dim of output')
    parser.add_argument('--dropout', dest='dropout', default=0.2, type=float, help='the dropout rate')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='the learning rate')
    parser.add_argument('--wd', dest='wd', default=0.0005, type=float, help='the weight decay')
    parser.add_argument('--seed', default=42, type=int, help='the random seed')
    parser.add_argument('--k', default=10, type=int, help='KFold')
    parser.add_argument('--train_name', default='Feature_data', type=str, help='the name of feature file')
    parser.add_argument('--dataset_name', default='CPDB_cdgps', type=str, help='the name of dataset')
    args = parser.parse_args()
    return args


def main(args):
    start_time = time.time()  # Record the start time
    dataset_name = args['dataset_name']

    if dataset_name in ['CPDB', 'IRefIndex_2015', 'IRefIndex', 'PCNet', 'STRINGdb']:
        multidim_graph = load_h5_graph(PATH='data/components/', LABEL_PATH='data/components/labels/NIHMS80906-small_mol-and-bio-druggable.tsv', ppi=dataset_name)
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cdgps', 'IRefIndex_2015_cdgps', 'IRefIndex_cdgps', 'PCNet_cdgps', 'STRINGdb_cdgps', 'CPDB_cdgps_shuffled']:
        multidim_graph = load_processed_graph(f'data/components/multidim_graph/6d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ["CPDB_cdgp", "IRefIndex_2015_cdgp", "IRefIndex_cdgp", "PCNet_cdgp", "STRINGdb_cdgp",
        "CPDB_cdgs", "IRefIndex_2015_cdgs", "IRefIndex_cdgs", "PCNet_cdgs", "STRINGdb_cdgs",
        "CPDB_cdps", "IRefIndex_2015_cdps", "IRefIndex_cdps", "PCNet_cdps", "STRINGdb_cdps",
        "CPDB_cgps", "IRefIndex_2015_cgps", "IRefIndex_cgps", "PCNet_cgps", "STRINGdb_cgps",
        "CPDB_dgps", "IRefIndex_2015_dgps", "IRefIndex_dgps", "PCNet_dgps", "STRINGdb_dgps"]:
        multidim_graph = load_processed_graph(f'data/components/multidim_graph/5d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cdg', 'CPDB_cdp', 'CPDB_cds', 'CPDB_cgp', 'CPDB_cgs', 'CPDB_cps', 'CPDB_dgp', 'CPDB_dgs', 'CPDB_dps', 'CPDB_gps']:
        multidim_graph = load_processed_graph(f'data/components/multidim_graph/4d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cd','CPDB_cg','CPDB_cp','CPDB_cs','CPDB_dg','CPDB_dp','CPDB_ds','CPDB_gp','CPDB_gs','CPDB_ps']:
        multidim_graph = load_processed_graph(f'data/components/multidim_graph/3d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_c', 'CPDB_d', 'CPDB_g', 'CPDB_p', 'CPDB_s']:
        multidim_graph = load_processed_graph(f'data/components/multidim_graph/2d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    print(f"Loaded multi-relational graph with {multidim_graph.num_nodes} nodes and {multidim_graph.edge_index.size(1)} edges")

    multidim_graph.x = multidim_graph.x.float()

    # Split the multi-relational graph into separate homogeneous graphs (one per relation type)
    relation_graphs = split_relations(multidim_graph)
    print(f"Split multi-relational graph into {len(relation_graphs)} relation-specific graphs.")
    
    input_dim = multidim_graph.x.shape[1]
    # Convert the dictionary into a list to be used by the model
    graphlist_adj = relation_graphs

    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

    def train(mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(output[mask], label, pos_weight=torch.Tensor([1]).to(device))
        acc = metrics.accuracy_score(label.cpu(), np.round(torch.sigmoid(output[mask]).cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        del output
        return loss.item(), acc

    @torch.no_grad()
    def test(mask, label):
        model.eval()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(output[mask], label, pos_weight=torch.Tensor([1]).to(device))
        acc = metrics.accuracy_score(label.cpu(), np.round(torch.sigmoid(output[mask]).cpu().detach().numpy()))
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(label.to('cpu'), pred)
        pr, rec, _ = metrics.precision_recall_curve(label.to('cpu'), pred)
        aupr = metrics.auc(rec, pr)
        return pred, loss.item(), acc, auroc, aupr

    file_save_path = os.path.join('mdmni_results/mdmni', args['dataset_name'])
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)

    #seeds = [0,1,2]
    seeds = [1]
    results =  []

    for i, seed in enumerate(seeds):
        run_start = time.time()
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
        model = MDMNI_DGD(input_dim, hidden_dim=args['hidden_dim'],
                        output_dim=args['output_dim'], dropout=args['dropout'])
        

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
        early_stopping = EarlyStopping(patience=args['patience'], verbose=True)

        best_val_acc = 0.0
        best_model_state_dict = None

        for epoch in range(1, args['epochs'] + 1):
            epoch_start = time.time()
            # Train on the training split
            train_loss, train_acc = train(train_mask, train_label)
            # Evaluate on the validation split
            _, loss_val, val_acc, _, _ = test(val_mask, val_label)

            if epoch % 20 == 0:
                train_pred, train_loss, train_acc, train_auroc, train_aupr = test(train_mask, train_label)
                val_pred, val_loss, val_acc, val_auroc, val_aupr = test(val_mask, val_label)
                print(f"Epoch {epoch}：")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
                    f"Train AUC: {train_auroc:.4f}, Train AUPR: {train_aupr:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, "
                    f"Val AUC: {val_auroc:.4f}, Val AUPR: {val_aupr:.4f}")
                print("")

            epoch_end = time.time()
            print(f"Run {i} - Epoch {epoch} runtime: {epoch_end - epoch_start:.2f} seconds")
            
            early_stopping(loss_val, model)
            # Save the best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state_dict = model.state_dict()
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        run_end = time.time()
        print(f"Run {i} total runtime: {run_end - run_start:.2f} seconds")
    
        print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
        print("Testing …")
        # Reload the best model before testing
        model = MDMNI_DGD(input_dim, hidden_dim=args['hidden_dim'],
                        output_dim=args['output_dim'], dropout=args['dropout'])
        model.to(device)
        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)
        test_pred, test_loss, test_acc, test_auc, test_aupr = test(test_mask, test_label)

        test_pred_binary = (test_pred > 0.5).astype(int)
        test_precision = average_precision_score(test_label.cpu(), test_pred_binary)
        test_recall = recall_score(test_label.cpu(), test_pred_binary)
        test_f1 = f1_score(test_label.cpu(), test_pred_binary)
        print("\nTest results:")
        print(f"Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}")
        print(f"Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}\n")
        #     k_sets, test_mask, test_label, idx_list, label_list = cross_validation(train_path, file_save_path, args['k'])

        end_time = time.time()  # Record the end time after everything is done
        total_runtime = end_time - start_time
        print(f"Total runtime: {total_runtime:.2f} seconds")
    

    # print("\nCross validation ……")
    # AUC = np.zeros(shape=(1, args['k']))
    # AUPR = np.zeros(shape=(1, args['k']))
    # ACC = np.zeros(shape=(1, args['k']))
    # best_acc = 0.0
    # best_model_state_dict = None

    # for j in range(len(k_sets)):
    #     for cv_run in range(args['k']):
    #         print("=====================================================")
    #         print(f"Fold {cv_run + 1}：")
    #         train_mask, val_mask, train_label, val_label = [p.to(device) for p in k_sets[j][cv_run] if type(p) == torch.Tensor]

    #         model = MDMNI_DGD(input_dim, hidden_dim=args['hidden_dim'], output_dim=args['output_dim'], dropout=args['dropout'])
    #         model.to(device)
    #         optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    #         early_stopping = EarlyStopping(patience=args['patience'], verbose=True)

    #         for epoch in range(1, args['epochs'] + 1):
    #             _, _ = train(train_mask, train_label)
    #             _, loss_val, _, _, _ = test(val_mask, val_label)

    #             if epoch % 20 == 0:
    #                 train_pred, train_loss, train_acc, train_auroc, train_aupr = test(train_mask, train_label)
    #                 val_pred, val_loss, val_acc, val_auroc, val_aupr = test(val_mask, val_label)

    #                 print(f"Epoch {epoch}：")
    #                 print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train AUC: {train_auroc:.4f}, Train AUPR: {train_aupr:.4f}")
    #                 print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val AUC: {val_auroc:.4f}, Val AUPR: {val_aupr:.4f}")
    #                 print("")

    #             early_stopping(loss_val, model)
    #             if early_stopping.early_stop:
    #                 print(f"Early stopping at epoch {epoch}")
    #                 break

    #             #torch.cuda.empty_cache()

    #         pred, _, ACC[0][cv_run], AUC[0][cv_run], AUPR[0][cv_run] = test(val_mask, val_label)

    #         if ACC[0][cv_run] > best_acc:
    #             best_acc = ACC[0][cv_run]
    #             best_model_state_dict = model.state_dict()

    # #model_save_path = 'best_model.pt'
    # #torch.save(best_model_state_dict, model_save_path)


    # # test
    # print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
    # print("Testing……")
    # model = MDMNI_DGD(input_dim, hidden_dim=args['hidden_dim'], output_dim=args['output_dim'], dropout=args['dropout'])
    # model.to(device)
    # model.load_state_dict(best_model_state_dict)
    # test_pred, test_loss, test_acc, test_auc, test_aupr = test(test_mask.to(device), test_label.to(device))

    # test_pred_binary = (test_pred > 0.5).astype(int)
    # test_precision = average_precision_score(test_label, test_pred_binary)
    # test_recall = recall_score(test_label, test_pred_binary)
    # test_f1 = f1_score(test_label, test_pred_binary)
    # print("\nTest results:")
    # print(f"Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}")
    # print(f"Recall：{test_recall:.4f}, Pre：{test_precision:.4f}, F1：{test_f1:.4f}\n")

    # # current_datetime = datetime.now()
    # # formatted_datetime = current_datetime.strftime("%m-%d-%H:%M")
    # # test_df = pd.DataFrame({'label': test_label.flatten(), 'prediction': test_pred.flatten()})
    # # test_df.to_csv(os.path.join(file_save_path, f"{args['train_name']}_test-result_{formatted_datetime}.csv"), index=False)



if __name__ == '__main__':
    args = parse_args()
    args_dic = vars(args)
    print('\n* args_dict：\n', args_dic)
    main(args_dic)
    print('The Training and Testing are finished!\n')