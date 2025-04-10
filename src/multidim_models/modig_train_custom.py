import argparse
import os

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data

import sys
sys.path.append('/Users/bty416/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/martina/01_PhD/05_Projects/04_Druggable-genes/SMG-DG')

from src.multidim_models.modig import MODIG
from src.multidim_models.modig_utils import *
from src.datasets.data_util import load_processed_graph, load_h5_graph
from src.utils import set_random_seed

# --- wandb import
import wandb

import time 

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
        description='Train MODIG with cross-validation and save model to file'
    )
    parser.add_argument('-t', '--title', help='the name of running experiment',
                        dest='title',
                        default=None,
                        type=str)
    parser.add_argument('-ppi', '--ppi', help='the chosen type of PPI',
                        dest='ppi',
                        default='CPDB',
                        type=str)
    parser.add_argument('-omic', '--omic', help='the chosen node attribute [multiomic, snv, cnv, mrna, dm]',
                        dest='omic',
                        default='multiomic',
                        type=str)
    parser.add_argument('-cancer', '--cancer', help='the model on pancan or specific cancer type',
                        dest='cancer',
                        default='pancan',
                        type=str)
    parser.add_argument('-e', '--epochs', help='maximum number of epochs (default: 100)',
                        dest='epochs',
                        default=100,
                        type=int)
    parser.add_argument('-p', '--patience', help='patience (default: 20)',
                        dest='patience',
                        default=20,
                        type=int)
    parser.add_argument('-dp', '--dropout', help='the dropout rate (default: 0.25)',
                        dest='dp',
                        default=0.25,
                        type=float)
    parser.add_argument('-lr', '--learningrate', help='the learning rate (default: 0.001)',
                        dest='lr',
                        default=0.001,
                        type=float)
    parser.add_argument('-wd', '--weightdecay', help='the weight decay (default: 0.0005)',
                        dest='wd',
                        default=0.0005,
                        type=float)
    parser.add_argument('-hs1', '--hiddensize1', help='the hidden size of first convolution layer (default: 300)',
                        dest='hs1',
                        default=300,
                        type=int)
    parser.add_argument('-hs2', '--hiddensize2', help='the hidden size of second convolution layer (default: 100)',
                        dest='hs2',
                        default=100,
                        type=int)
    parser.add_argument('-thr_go', '--thr_go', help='the threshold for GO semantic similarity (default: 0.8)',
                        dest='thr_go',
                        default=0.8,
                        type=float)
    parser.add_argument('-thr_seq', '--thr_seq', help='the threshold for gene sequence similarity (default: 0.5)',
                        dest='thr_seq',
                        default=0.5,
                        type=float)
    parser.add_argument('-thr_exp', '--thr_exp', help='the threshold for tissue co-expression pattern (default: 0.8)',
                        dest='thr_exp',
                        default=0.8,
                        type=float)
    parser.add_argument('-thr_path', '--thr_path', help='the threshold of gene pathway co-occurrence (default: 0.5)',
                        dest='thr_path',
                        default=0.5,
                        type=float)
    parser.add_argument('-seed', '--seed', help='the random seed (default: 42)',
                        dest='seed',
                        default=42,
                        type=int)
    parser.add_argument('-data', '--dataset', help='the dataset name (default: CPDB_cdgps)',
                        dest='dataset',
                        default='CPDB_cdgps',
                        type=str)
    args = parser.parse_args()
    return args


def main(args):
    start_time = time.time()
    # -- Initialize wandb (adjust project/name/entity as needed)
    wandb.init(
        project="MODIG",  # or whatever your project name is
        name=args['title'] if args['title'] else "MODIG_Run",
        config=args
    )

    dataset_name = args['dataset']  # Or directly: 'CPDB_cdgps'

    # --- Load the dataset
    if dataset_name in ['CPDB', 'IRefIndex_2015', 'IRefIndex', 'PCNet', 'STRINGdb']:
        multidim_graph = load_h5_graph(
            PATH='data/real/smg_data',
            LABEL_PATH='data/real/labels/NIHMS80906-small_mol-and-bio-druggable.tsv',
            ppi=dataset_name
        )
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_cdgps', 'IRefIndex_2015_cdgps', 'IRefIndex_cdgps', 'PCNet_cdgps', 'STRINGdb_cdgps', 'CPDB_cdgps_shuffled']:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/6d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in [
        "CPDB_cdgp", "IRefIndex_2015_cdgp", "IRefIndex_cdgp", "PCNet_cdgp", "STRINGdb_cdgp",
        "CPDB_cdgs", "IRefIndex_2015_cdgs", "IRefIndex_cdgs", "PCNet_cdgs", "STRINGdb_cdgs",
        "CPDB_cdps", "IRefIndex_2015_cdps", "IRefIndex_cdps", "PCNet_cdps", "STRINGdb_cdps",
        "CPDB_cgps", "IRefIndex_2015_cgps", "IRefIndex_cgps", "PCNet_cgps", "STRINGdb_cgps",
        "CPDB_dgps", "IRefIndex_2015_dgps", "IRefIndex_dgps", "PCNet_dgps", "STRINGdb_dgps"
    ]:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/5d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in [
        'CPDB_cdg', 'CPDB_cdp', 'CPDB_cds', 'CPDB_cgp', 'CPDB_cgs', 'CPDB_cps',
        'CPDB_dgp', 'CPDB_dgs', 'CPDB_dps', 'CPDB_gps'
    ]:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/4d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in [
        'CPDB_cd','CPDB_cg','CPDB_cp','CPDB_cs','CPDB_dg','CPDB_dp','CPDB_ds','CPDB_gp','CPDB_gs','CPDB_ps'
    ]:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/3d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    elif dataset_name in ['CPDB_c', 'CPDB_d', 'CPDB_g', 'CPDB_p', 'CPDB_s']:
        multidim_graph = load_processed_graph(f'data/real/multidim_graph/2d/{dataset_name}_multiomics.pt')
        num_features = multidim_graph.x.shape[1]
        num_classes = multidim_graph.y.max().item() + 1

    print(f"Loaded multi-relational graph with {multidim_graph.num_nodes} nodes and {multidim_graph.edge_index.size(1)} edges")
    multidim_graph.x = multidim_graph.x.float()

    # Split the multi-relational graph into separate homogeneous graphs
    relation_graphs = split_relations(multidim_graph)
    print(f"Split multi-relational graph into {len(relation_graphs)} relation-specific graphs.")
    
    final_gene_node = multidim_graph.name
    input_dim = multidim_graph.x.shape[1]
    idx_list = np.arange(multidim_graph.num_nodes)
    label_list = multidim_graph.y.cpu().numpy()  # if you need it as a numpy array
    
    # Convert to list of graphs to be used by the model
    graphlist_adj = relation_graphs

    # ----------------------------
    # Train / Test helper functions
    # ----------------------------
    def train(mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(graphlist_adj)
        
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).to(device)
        )
        acc = metrics.accuracy_score(
            label.cpu(),
            np.round(torch.sigmoid(output[mask]).cpu().detach().numpy())
        )
        # Compute backward pass & update
        loss.backward()
        optimizer.step()

        # Return CPU-based scalars
        return loss.item(), acc

    @torch.no_grad()
    def evaluate(mask, label):
        """
        Evaluate model on the given mask/labels.
        Returns:
           pred, loss, acc, auc, aupr, f1
        """
        model.eval()
        output = model(graphlist_adj)
        
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).to(device)
        )
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()

        acc  = metrics.accuracy_score(label.cpu(), np.round(pred))
        auroc = metrics.roc_auc_score(label.cpu(), pred)
        precision, recall, _ = metrics.precision_recall_curve(label.cpu(), pred)
        aupr = metrics.auc(recall, precision)
        f1   = metrics.f1_score(label.cpu(), np.round(pred))

        return pred, loss.item(), acc, auroc, aupr, f1

    # ----------------------------
    # Directory to save results
    # ----------------------------
    file_save_path = os.path.join('modig_results/modig', args['dataset'])
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)

    # For demonstration, 3 seeds
    seeds = [0, 1, 2]

    # Arrays for tracking final metrics across seeds (if you wish to average them)
    AUC_vals  = []
    AUPR_vals = []
    ACC_vals  = []
    F1_vals   = []

    pred_all  = []
    label_all = []

    # ---------------------------------------------
    # 1) Loop over seeds (like repeated experiments)
    # ---------------------------------------------
    for run_idx, seed in enumerate(seeds):
        run_start = time.time()
        print(f"####### Run {run_idx} for seed {seed}")
        set_random_seed(seed)

        # Extract predefined splits and move to the device
        train_mask = multidim_graph.train_mask.to(device)
        val_mask   = multidim_graph.val_mask.to(device)
        test_mask  = multidim_graph.test_mask.to(device)
        train_label = multidim_graph.y[train_mask].to(device)
        val_label   = multidim_graph.y[val_mask].to(device)
        test_label  = multidim_graph.y[test_mask].to(device)

        print("\nTraining using predefined splits ...")

        # Initialize model/optimizer
        model = MODIG(
            nfeat=input_dim,
            hidden_size1=args['hs1'],
            hidden_size2=args['hs2'],
            dropout=args['dp']
        ).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=args['lr'], weight_decay=args['wd']
        )
        model_save_file = os.path.join(file_save_path + '_modig.pth')

        # Early stopping
        early_stopping = EarlyStopping(
            patience=args['patience'], verbose=True
        )

        # ----------------------------
        # Training loop
        # ----------------------------
        for epoch in range(1, args['epochs']+1):
            epoch_start = time.time()  # Start time for this epoch
            train_loss, train_acc = train(train_mask, train_label)
            _, val_loss, val_acc, val_auroc, val_aupr, val_f1 = evaluate(val_mask, val_label)

            # Log training & validation metrics to wandb
            wandb.log({
                'Epoch': epoch,
                f'Run_{run_idx}_Train_Loss': train_loss,
                f'Run_{run_idx}_Train_Acc': train_acc,
                f'Run_{run_idx}_Val_Loss': val_loss,
                f'Run_{run_idx}_Val_Acc': val_acc,
                f'Run_{run_idx}_Val_AUC': val_auroc,
                f'Run_{run_idx}_Val_AUPR': val_aupr,
                f'Run_{run_idx}_Val_F1': val_f1
            })

            epoch_end = time.time()  # End time for this epoch
            epoch_runtime = epoch_end - epoch_start
            print(f"Epoch {epoch} runtime: {epoch_runtime:.2f} seconds")    
            # Early stopping monitor is val_loss
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        run_end = time.time()
        print(f"Run {run_idx} total runtime: {run_end - run_start:.2f} seconds")
               
        # Load best model
        model.load_state_dict(early_stopping.best_model)

        # ----------------------------
        # Test on test set
        # ----------------------------
        print("Testing on the test set...")
        pred, test_loss, test_acc, test_auc, test_aupr, test_f1 = evaluate(test_mask, test_label)
        print(f"Test  AUC:  {test_auc:.4f}")
        print(f"Test  AUPR: {test_aupr:.4f}")
        print(f"Test  ACC:  {test_acc:.4f}")
        print(f"Test  F1:   {test_f1:.4f}")

        # Log test metrics to wandb
        wandb.log({
            f'Run_{run_idx}_Test_Loss': test_loss,
            f'Run_{run_idx}_Test_Acc': test_acc,
            f'Run_{run_idx}_Test_AUC': test_auc,
            f'Run_{run_idx}_Test_AUPR': test_aupr,
            f'Run_{run_idx}_Test_F1': test_f1
        })

        AUC_vals.append(test_auc)
        AUPR_vals.append(test_aupr)
        ACC_vals.append(test_acc)
        F1_vals.append(test_f1)

        # If you'd like to store predictions for further analysis
        pred_all.append(pred)
        label_all.append(test_label.cpu())

    # ---------------------------------
    # Summaries across runs (if desired)
    # ---------------------------------
    print(f"Mean AUC:  {np.mean(AUC_vals):.4f},  Var AUC:  {np.var(AUC_vals):.4f}")
    print(f"Mean AUPR: {np.mean(AUPR_vals):.4f}, Var AUPR: {np.var(AUPR_vals):.4f}")
    print(f"Mean ACC:  {np.mean(ACC_vals):.4f},  Var ACC:  {np.var(ACC_vals):.4f}")
    print(f"Mean F1:   {np.mean(F1_vals):.4f},   Var F1:   {np.var(F1_vals):.4f}")

    # You can log these averages to wandb as well
    wandb.log({
        'Mean_Test_AUC':  np.mean(AUC_vals),
        'Var_Test_AUC':   np.var(AUC_vals),
        'Mean_Test_AUPR': np.mean(AUPR_vals),
        'Var_Test_AUPR':  np.var(AUPR_vals),
        'Mean_Test_ACC':  np.mean(ACC_vals),
        'Var_Test_ACC':   np.var(ACC_vals),
        'Mean_Test_F1':   np.mean(F1_vals),
        'Var_Test_F1':    np.var(F1_vals)
    })

    # Save predictions if you want
    torch.save(pred_all, os.path.join(file_save_path, 'pred_all.pkl'))
    torch.save(label_all, os.path.join(file_save_path, 'label_all.pkl'))

    # -------------------------------------------------
    # 2) Optional: Retrain using ALL data for final model
    # -------------------------------------------------
    # all_mask = torch.LongTensor(idx_list)
    # all_label = torch.FloatTensor(label_list).reshape(-1, 1)

    # model = MODIG(
    #     nfeat=input_dim,
    #     hidden_size1=args['hs1'],
    #     hidden_size2=args['hs2'],
    #     dropout=args['dp']
    # ).to(device)
    # optimizer = optim.Adam(
    #     model.parameters(), lr=args['lr'], weight_decay=args['wd']
    # )

    # # A short training on entire dataset (no early stopping here by default)
    # for epoch in range(1, args['epochs']+1):
    #     train_loss, train_acc = train(all_mask.to(device), all_label.to(device))
    #     # Optionally log or print:
    #     # print(f"[AllData] Epoch {epoch}, Loss={train_loss:.4f}, Acc={train_acc:.4f}")

    # output = model(graphlist_adj)
    # pred = torch.sigmoid(output).cpu().detach().numpy()
    # pred2 = torch.sigmoid(output[~all_mask]).cpu().detach().numpy()

    # torch.save(pred,      os.path.join(file_save_path, args['ppi'] + '_pred.pkl'))
    # torch.save(all_label, os.path.join(file_save_path, args['ppi'] + '_label.pkl'))
    # torch.save(pred2,     os.path.join(file_save_path, args['ppi'] + '_pred2.pkl'))

    pd.Series(final_gene_node).to_csv(
        os.path.join(file_save_path, 'final_gene_node.csv'),
        index=False,
        header=False
    )

    end_time = time.time()  # Record the end time after everything is done
    total_runtime = end_time - start_time
    print(f"Total runtime: {total_runtime:.2f} seconds")
    

    # If you want to finish the wandb run:
    wandb.finish()

    print('The Training is finished!')


if __name__ == '__main__':
    args = parse_args()
    args_dic = vars(args)
    print('args_dict', args_dic)
    main(args_dic)