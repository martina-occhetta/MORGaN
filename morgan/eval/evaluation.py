import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm

from morgan.utils import accuracy, create_optimizer, result


def node_classification_eval(
    model,
    graph,
    x,
    num_classes,
    lr_f,
    weight_decay_f,
    max_epoch_f,
    device,
    linear_prob=True,
    mute=False,
    out_dir: str | Path = None,
):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            if graph.num_edge_types != 1:
                x = model.embed(
                    x.to(device),
                    graph.edge_index.to(device),
                    graph.edge_type.to(device),
                )
                in_feat = x.shape[1]
            else:
                x = model.embed(graph.to(device), x.to(device))
                in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes, concat=True, datas_dim=0)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    (
        (test_acc, estp_test_acc),
        (test_auc, estp_test_auc),
        (test_aupr, estp_test_aupr),
        (test_precision, estp_precision_f),
        (test_recall, estp_recall_f),
        (test_f1, estp_f1),
    ) = linear_probing_for_transductive_node_classifcation(
        encoder, graph, x, optimizer_f, max_epoch_f, device, mute, out_dir=out_dir
    )
    return (
        (test_acc, estp_test_acc),
        (test_auc, estp_test_auc),
        (test_aupr, estp_test_aupr),
        (test_precision, estp_precision_f),
        (test_recall, estp_recall_f),
        (test_f1, estp_f1),
    )


def linear_probing_for_transductive_node_classifcation(
    model, graph, feat, optimizer, max_epoch, device, mute=False, out_dir: str | Path = None
):
    graph = graph.to(device)
    x = feat.to(device)

    # Ensure masks and labels have correct shapes
    train_mask = (
        graph.train_mask.squeeze()
        if hasattr(graph, "train_mask") and len(graph.train_mask.shape) > 1
        else graph.train_mask
    )
    val_mask = (
        graph.val_mask.squeeze()
        if hasattr(graph, "val_mask") and len(graph.val_mask.shape) > 1
        else graph.val_mask
    )
    test_mask = (
        graph.test_mask.squeeze()
        if hasattr(graph, "test_mask") and len(graph.test_mask.shape) > 1
        else graph.test_mask
    )
    labels = (
        graph.y.squeeze() if hasattr(graph, "y") and len(graph.y.shape) > 1 else graph.y
    )

    # Ensure all tensors exist
    if not all(
        hasattr(graph, attr) for attr in ["train_mask", "val_mask", "test_mask", "y"]
    ):
        raise ValueError(
            "Graph missing required attributes (train_mask, val_mask, test_mask, or y)"
        )

    # Convert to float for loss calculation
    labels = labels.float()

    # Calculate class balance for weighting
    if train_mask.sum() > 0:  # Only if we have training samples
        pos_labels = labels[train_mask].sum()
        neg_labels = train_mask.sum() - pos_labels

        print("Number of positive labels:", pos_labels.item())
        print("Number of negative labels:", neg_labels.item())

    # For MUTAG dataset, we might not need pos_weight since it's balanced
    if pos_labels > 0 and neg_labels > 0:
        weight = torch.tensor([neg_labels / pos_labels])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
    else:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)

    best_val_acc = 0
    best_val_epoch = 0
    best_val_aupr = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        if hasattr(graph, "edge_type") and graph.edge_type is not None:
            out = model(graph, x, len(torch.unique(graph.edge_type)))
        elif graph.num_edge_types != 1:
            out = model(graph, x, graph.num_edge_types)
        else:
            out = model(graph, x)

        # Ensure out and labels match in shape
        out = out.squeeze() if len(out.shape) > 1 and out.shape[1] == 1 else out
        labels = labels.float()

        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            if hasattr(graph, "edge_type") and graph.edge_type is not None:
                pred = model(graph, x, len(torch.unique(graph.edge_type)))
            elif graph.num_edge_types != 1:
                pred = model(graph, x, graph.num_edge_types)
            else:
                pred = model(graph, x)

            # Ensure pred has correct shape
            pred = (
                pred.squeeze() if len(pred.shape) > 1 and pred.shape[1] == 1 else pred
            )

            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            val_auc, val_aupr, precision, recall, f1 = result(
                pred[val_mask].cpu(), labels[val_mask].cpu()
            )
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_auc, test_aupr, test_precision, test_recall, test_f1 = result(
                pred[test_mask].cpu(), labels[test_mask].cpu()
            )
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_aupr >= best_val_aupr:
            best_val_aupr = val_aupr
            best_val_epoch_aupr = epoch

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, val_aupr:{val_aupr: .4f}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f},test_auc:{test_auc: .4f}, test_aupr:{test_aupr: .4f}"
            )

    best_model.eval()
    with torch.no_grad():
        if hasattr(graph, "edge_type") and graph.edge_type is not None:
            pred = best_model(graph, x, len(torch.unique(graph.edge_type)))
        elif graph.num_edge_types != 1:
            pred = best_model(graph, x, graph.num_edge_types)
        else:
            pred = best_model(graph, x)

        # Ensure pred has correct shape
        pred = pred.squeeze() if len(pred.shape) > 1 and pred.shape[1] == 1 else pred

        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        estp_test_auc, estp_test_aupr, estp_precision_f, estp_recall_f, estp_f1 = (
            result(pred[test_mask].cpu(), labels[test_mask].cpu())
        )

    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- "
        )
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- "
        )
        print(
            f"--- TestAUPR: {test_aupr:.4f}, early-stopping-TestAUPR: {estp_test_aupr:.4f}, Best ValAUPR: {best_val_aupr:.4f} in epoch {best_val_epoch_aupr} --- "
        )

    y_proba = torch.sigmoid(pred[test_mask]).cpu().numpy()
    y_true  = labels[test_mask].cpu().numpy()
    y_pred = pred[test_mask].cpu().numpy()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(out_dir, "y_true.npy"), y_true)
        np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(out_dir, "y_proba.npy"), y_proba)

    if hasattr(graph, "name"):
        names_arr = np.asarray(graph.name)
        mask_np   = test_mask.cpu().numpy().astype(bool)
        test_gene_names = names_arr[mask_np]
        np.save(out_dir / "test_gene_names.npy", test_gene_names)

    return (
        (test_acc, estp_test_acc),
        (test_auc, estp_test_auc),
        (test_aupr, estp_test_aupr),
        (test_precision, estp_precision_f),
        (test_recall, estp_recall_f),
        (test_f1, estp_f1),
    )


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        # x = x.float()  # Ensure the input is of type float
        logits = self.linear(x)
        return logits
