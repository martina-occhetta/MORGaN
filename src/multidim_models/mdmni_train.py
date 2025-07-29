import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.utils import from_networkx
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
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
)
import faulthandler
from datetime import datetime

from mdmni import MDMNI_DGD
from mdmni_network import generate_graph
from mdmni_utils import *

# cuda = torch.cuda.is_available()
# torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--networks",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="Indices of networks to use",
    )
    parser.add_argument(
        "--thr_go",
        dest="thr_go",
        default=0.8,
        type=float,
        help="the threshold for GO semantic similarity",
    )
    parser.add_argument(
        "--thr_seq",
        dest="thr_seq",
        default=0.8,
        type=float,
        help="the threshold for sequence similarity",
    )
    parser.add_argument(
        "--thr_exp",
        dest="thr_exp",
        default=0.8,
        type=float,
        help="the threshold for gene co-expression pattern",
    )
    parser.add_argument(
        "--thr_path",
        dest="thr_path",
        default=0.6,
        type=float,
        help="the threshold of pathway co-occurrence",
    )
    parser.add_argument(
        "--thr_cpdb",
        dest="thr_cpdb",
        default=0.8,
        type=float,
        help="the threshold of CPDB PPI",
    )
    parser.add_argument(
        "--thr_string",
        dest="thr_string",
        default=0.8,
        type=float,
        help="the threshold of STRING PPI",
    )
    parser.add_argument(
        "--thr_domain",
        dest="thr_domain",
        default=0.3,
        type=float,
        help="the threshold of Domain similarity",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=1000,
        type=int,
        help="maximum number of epochs",
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        default=120,
        type=int,
        help="waiting iterations when performance no longer improves",
    )
    parser.add_argument(
        "--hidden_dim",
        dest="hidden_dim",
        default=256,
        type=int,
        help="the dim of hidden layer",
    )
    parser.add_argument(
        "--output_dim",
        dest="output_dim",
        default=32,
        type=int,
        help="the dim of output",
    )
    parser.add_argument(
        "--dropout", dest="dropout", default=0.2, type=float, help="the dropout rate"
    )
    parser.add_argument(
        "--lr", dest="lr", default=0.001, type=float, help="the learning rate"
    )
    parser.add_argument(
        "--wd", dest="wd", default=0.0005, type=float, help="the weight decay"
    )
    parser.add_argument("--seed", default=42, type=int, help="the random seed")
    parser.add_argument("--k", default=10, type=int, help="KFold")
    parser.add_argument(
        "--train_name",
        default="Feature_data",
        type=str,
        help="the name of feature file",
    )
    args = parser.parse_args()
    return args


def main(args):
    seed_torch(args["seed"])

    graph_path = os.path.join(
        "/Users/bty416/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/martina/01_PhD/05_Projects/04_Druggable-genes/SMG-DG/data/components/mdmni_data/network"
    )
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    file_save_path = os.path.join("results")
    feature_path = os.path.join(
        "/Users/bty416/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/martina/01_PhD/05_Projects/04_Druggable-genes/SMG-DG/data/components/mdmni_data/feature"
    )

    train_path = os.path.join(feature_path, str(args["train_name"]) + ".csv")
    train_file = pd.read_csv(train_path)
    # final_gene_node = train_file['gene'].tolist()
    train_data = train_file.iloc[:, 1:-1]
    input_dim = train_data.shape[1]
    print("\nThe dim of feature:", input_dim)

    # generate_input = generate_graph(graph_path)

    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
    print("Loading train network datasets……")
    cpdb_path = os.path.join(
        graph_path, str(args["train_name"]) + "_cpdb_" + str(args["thr_cpdb"]) + ".csv"
    )
    string_path = os.path.join(
        graph_path,
        str(args["train_name"]) + "_string_" + str(args["thr_string"]) + ".csv",
    )
    exp_path = os.path.join(
        graph_path, str(args["train_name"]) + "_exp_" + str(args["thr_exp"]) + ".csv"
    )
    go_path = os.path.join(
        graph_path, str(args["train_name"]) + "_go_" + str(args["thr_go"]) + ".csv"
    )
    path_path = os.path.join(
        graph_path, str(args["train_name"]) + "_path_" + str(args["thr_path"]) + ".csv"
    )
    seq_path = os.path.join(
        graph_path, str(args["train_name"]) + "_seq_" + str(args["thr_seq"]) + ".csv"
    )
    dom_path = os.path.join(
        graph_path,
        str(args["train_name"]) + "_domain_" + str(args["thr_domain"]) + ".csv",
    )

    # if os.path.exists(cpdb_path) & os.path.exists(string_path) & os.path.exists(dom_path) & os.path.exists(go_path) & os.path.exists(exp_path) & os.path.exists(seq_path) & os.path.exists(path_path):
    print("The train profiles already exist!\n")
    cpdb_file = pd.read_csv(cpdb_path, index_col=0)
    string_file = pd.read_csv(string_path, index_col=0)
    exp_file = pd.read_csv(exp_path, index_col=0)
    go_file = pd.read_csv(go_path, index_col=0)
    path_file = pd.read_csv(path_path, index_col=0)
    seq_file = pd.read_csv(seq_path, index_col=0)
    dom_file = pd.read_csv(dom_path, index_col=0)
    # else:
    #     cpdb_file, string_file, dom_file, exp_file, go_file, path_file, seq_file = generate_input.generate_graph(
    #            train_path, args['train_name'], args['thr_cpdb'], args['thr_string'], args['thr_domain'], args['thr_exp'], args['thr_go'], args['thr_path'], args['thr_seq'])

    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

    print("Selected Network INFO：")
    name_of_network = ["CPDB", "STRING", "DOMAIN", "EXP", "GO", "PATH", "SEQ"]
    graphlist = []
    for i, network in enumerate(
        [cpdb_file, string_file, dom_file, exp_file, go_file, path_file, seq_file]
    ):
        featured_graph = load_featured_graph(network, train_data)
        graphlist.append(featured_graph)

    selected_networks = [graphlist[i] for i in args["networks"]]
    for sel_i in args["networks"]:
        print(f"    The {name_of_network[sel_i]} network.")
    graphlist_adj = [graph for graph in selected_networks]

    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

    def train(mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([1]).to(device)
        )
        acc = metrics.accuracy_score(
            label.cpu(), np.round(torch.sigmoid(output[mask]).cpu().detach().numpy())
        )
        loss.backward()
        optimizer.step()
        del output
        return loss.item(), acc

    @torch.no_grad()
    def test(mask, label):
        model.eval()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([1]).to(device)
        )
        acc = metrics.accuracy_score(
            label.cpu(), np.round(torch.sigmoid(output[mask]).cpu().detach().numpy())
        )
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(label.to("cpu"), pred)
        pr, rec, _ = metrics.precision_recall_curve(label.to("cpu"), pred)
        aupr = metrics.auc(rec, pr)
        return pred, loss.item(), acc, auroc, aupr

    k_sets, test_mask, test_label, idx_list, label_list = cross_validation(
        train_path, file_save_path, args["k"]
    )

    print("\nCross validation ……")
    AUC = np.zeros(shape=(1, args["k"]))
    AUPR = np.zeros(shape=(1, args["k"]))
    ACC = np.zeros(shape=(1, args["k"]))
    best_acc = 0.0
    best_model_state_dict = None

    for j in range(len(k_sets)):
        for cv_run in range(args["k"]):
            print("=====================================================")
            print(f"Fold {cv_run + 1}：")
            train_mask, val_mask, train_label, val_label = [
                p.to(device) for p in k_sets[j][cv_run] if type(p) == torch.Tensor
            ]

            model = MDMNI_DGD(
                input_dim,
                hidden_dim=args["hidden_dim"],
                output_dim=args["output_dim"],
                dropout=args["dropout"],
            )
            model.to(device)
            optimizer = optim.Adam(
                model.parameters(), lr=args["lr"], weight_decay=args["wd"]
            )

            early_stopping = EarlyStopping(patience=args["patience"], verbose=True)

            for epoch in range(1, args["epochs"] + 1):
                _, _ = train(train_mask, train_label)
                _, loss_val, _, _, _ = test(val_mask, val_label)

                if epoch % 20 == 0:
                    train_pred, train_loss, train_acc, train_auroc, train_aupr = test(
                        train_mask, train_label
                    )
                    val_pred, val_loss, val_acc, val_auroc, val_aupr = test(
                        val_mask, val_label
                    )

                    print(f"Epoch {epoch}：")
                    print(
                        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train AUC: {train_auroc:.4f}, Train AUPR: {train_aupr:.4f}"
                    )
                    print(
                        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val AUC: {val_auroc:.4f}, Val AUPR: {val_aupr:.4f}"
                    )
                    print("")

                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # torch.cuda.empty_cache()

            pred, _, ACC[0][cv_run], AUC[0][cv_run], AUPR[0][cv_run] = test(
                val_mask, val_label
            )

            if ACC[0][cv_run] > best_acc:
                best_acc = ACC[0][cv_run]
                best_model_state_dict = model.state_dict()

    # model_save_path = 'best_model.pt'
    # torch.save(best_model_state_dict, model_save_path)

    # test
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
    print("Testing……")
    model = MDMNI_DGD(
        input_dim,
        hidden_dim=args["hidden_dim"],
        output_dim=args["output_dim"],
        dropout=args["dropout"],
    )
    model.to(device)
    model.load_state_dict(best_model_state_dict)
    test_pred, test_loss, test_acc, test_auc, test_aupr = test(
        test_mask.to(device), test_label.to(device)
    )

    test_pred_binary = (test_pred > 0.5).astype(int)
    test_precision = average_precision_score(test_label, test_pred_binary)
    test_recall = recall_score(test_label, test_pred_binary)
    test_f1 = f1_score(test_label, test_pred_binary)
    print("\nTest results:")
    print(f"Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}")
    print(f"Recall：{test_recall:.4f}, Pre：{test_precision:.4f}, F1：{test_f1:.4f}\n")

    # current_datetime = datetime.now()
    # formatted_datetime = current_datetime.strftime("%m-%d-%H:%M")
    # test_df = pd.DataFrame({'label': test_label.flatten(), 'prediction': test_pred.flatten()})
    # test_df.to_csv(os.path.join(file_save_path, f"{args['train_name']}_test-result_{formatted_datetime}.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    args_dic = vars(args)
    print("\n* args_dict：\n", args_dic)
    main(args_dic)
    print("The Training and Testing are finished!\n")
