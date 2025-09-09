import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from morgan.utils import create_optimizer
from morgan.eval.evaluation import LogisticRegression


def multilabel_eval(pred_logits, labels, mask, threshold=0.5):
    """
    Compute multilabel evaluation metrics on a masked subset.

    Args:
        pred_logits (Tensor): Raw logits of shape [N, L].
        labels (Tensor): Ground-truth one-hot labels of shape [N, L].
        mask (BoolTensor): Boolean mask of shape [N] indicating which nodes to evaluate.
        threshold (float): Classification threshold for converting probabilities to binary predictions.

    Returns:
        subset_acc (float): Fraction of samples with all labels correct.
        rocauc (float): Micro-averaged ROC-AUC over all labels.
        aupr (float): Micro-averaged AUPR over all labels.
        precision (float): Micro-averaged precision.
        recall (float): Micro-averaged recall.
        f1 (float): Micro-averaged F1 score.
    """
    # Select masked entries
    logits = pred_logits[mask]  # [n_mask, L]
    y_true = labels[mask].float().cpu().numpy()  # (n_mask, L)

    # Convert logits to probabilities
    y_prob = torch.sigmoid(logits).cpu().numpy()

    # Compute ROC-AUC and AUPR
    rocauc = roc_auc_score(y_true, y_prob, average="micro")
    aupr = average_precision_score(y_true, y_prob, average="micro")

    # Binarize predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Compute precision, recall, F1 (micro)
    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Subset accuracy: all labels must match exactly
    subset_acc = (y_pred == y_true).all(axis=1).mean()

    return subset_acc, rocauc, aupr, precision, recall, f1


def multilabel_node_classification_eval(
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
):
    """
    Multilabel node classification evaluation via linear probing or end-to-end.

    Returns the same tuple structure as the original function:
      ((test_acc, estp_test_acc), (test_auc, estp_test_auc),
       (test_aupr, estp_test_aupr), (test_precision, estp_precision),
       (test_recall, estp_recall), (test_f1, estp_f1))
    """
    model.eval()
    # Prepare encoder for linear probing, if requested
    if linear_prob:
        with torch.no_grad():
            if graph.num_edge_types != 1:
                x_emb = model.embed(
                    x.to(device),
                    graph.edge_index.to(device),
                    graph.edge_type.to(device),
                )
            else:
                x_emb = model.embed(graph.to(device), x.to(device))
        in_feat = x_emb.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
        probe_input = x_emb
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes, concat=True, datas_dim=0)
        probe_input = x.to(device)

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)

    return multilabel_linear_probing_for_transductive_node_classification(
        encoder, graph, probe_input, optimizer_f, max_epoch_f, device, mute
    )


def multilabel_linear_probing_for_transductive_node_classification(
    model, graph, feat, optimizer, max_epoch, device, mute=False
):
    """
    Core training loop for multilabel transductive node classification.
    Follows the original API but uses multilabel metrics.
    """
    graph = graph.to(device)
    x = feat.to(device)

    # Masks and labels (keep 2D labels)
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    labels = graph.y

    if not all(
        hasattr(graph, attr) for attr in ["train_mask", "val_mask", "test_mask", "y"]
    ):
        raise ValueError(
            "Graph missing required attributes (train_mask, val_mask, test_mask, or y)"
        )

    # Loss: BCEWithLogits for multilabel
    criterion = nn.BCEWithLogitsLoss().to(device)

    best_val_aupr = 0.0
    best_val_epoch = 0
    best_model = None

    # Iterate epochs
    epoch_iter = tqdm(range(max_epoch), disable=mute)
    for epoch in epoch_iter:
        model.train()
        # Forward pass
        if hasattr(graph, "edge_type") and graph.edge_type is not None:
            logits = model(graph, x, len(torch.unique(graph.edge_type)))
        elif graph.num_edge_types != 1:
            logits = model(graph, x, graph.num_edge_types)
        else:
            logits = model(graph, x)

        # Compute loss
        loss = criterion(logits[train_mask], labels[train_mask].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval on val and test
        model.eval()
        with torch.no_grad():
            if hasattr(graph, "edge_type") and graph.edge_type is not None:
                logits = model(graph, x, len(torch.unique(graph.edge_type)))
            elif graph.num_edge_types != 1:
                logits = model(graph, x, graph.num_edge_types)
            else:
                logits = model(graph, x)

            # Last-epoch metrics
            test_acc, test_auc, test_aupr, test_prec, test_rec, test_f1 = (
                multilabel_eval(logits, labels, test_mask)
            )

            # Validation metrics for early stopping
            val_acc, val_auc, val_aupr, val_prec, val_rec, val_f1 = multilabel_eval(
                logits, labels, val_mask
            )

        # Track best model by validation AUPR
        if val_aupr >= best_val_aupr:
            best_val_aupr = val_aupr
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"Epoch {epoch:d} | Val AUPR: {val_aupr:.4f} | Test AUPR: {test_aupr:.4f}"
            )

    # Final evaluation of best model
    best_model.eval()
    with torch.no_grad():
        if hasattr(graph, "edge_type") and graph.edge_type is not None:
            best_logits = best_model(graph, x, len(torch.unique(graph.edge_type)))
        elif graph.num_edge_types != 1:
            best_logits = best_model(graph, x, graph.num_edge_types)
        else:
            best_logits = best_model(graph, x)

        estp_test_acc, estp_test_auc, estp_test_aupr, estp_prec, estp_rec, estp_f1 = (
            multilabel_eval(best_logits, labels, test_mask)
        )

    # Return same structure as original
    return (
        (test_acc, estp_test_acc),
        (test_auc, estp_test_auc),
        (test_aupr, estp_test_aupr),
        (test_prec, estp_prec),
        (test_rec, estp_rec),
        (test_f1, estp_f1),
    )
