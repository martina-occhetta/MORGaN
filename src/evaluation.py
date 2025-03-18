import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
from src.utils import create_optimizer, accuracy

def result(pred, true):
    aa = torch.sigmoid(pred)
    precision, recall, _thresholds = metrics.precision_recall_curve(true, aa)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(true, aa), area, precision, recall

def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            if graph.num_edge_types != 1:
                x = model.embed(x.to(device), graph.edge_index.to(device), graph.edge_type.to(device))
                in_feat = x.shape[1]
            else:
                x = model.embed(graph.to(device), x.to(device))
                in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes, concat=True, datas_dim = 0)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classifcation(encoder, graph, x, optimizer_f, max_epoch_f, device, mute)
    return final_acc, estp_acc


def linear_probing_for_transductive_node_classifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    
    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    labels = graph.y

    pos_labels = labels[train_mask].sum()
    neg_labels = len(labels[train_mask]) - pos_labels

    print('Number of positive labels:', pos_labels)
    print('Number of negative labels:', neg_labels)

    weight = torch.tensor([neg_labels/pos_labels])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)

    best_val_acc = 0
    best_val_epoch = 0
    best_val_aupr = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train().to(device)
        # if graph.num_edge_types != 1:
        #     out = model(x, graph.edge_index, graph.edge_type)
        # else:
        #     out = model(graph, x)
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            # if graph.num_edge_types != 1:
            #     pred = model(x, graph.edge_index, graph.edge_type)
            # else:
            #     pred = model(graph, x)
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            val_auc, val_aupr, precision, recall = result(pred[val_mask].cpu(), labels[val_mask].cpu())
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_auc, test_aupr, precision, recall = result(pred[test_mask].cpu(), labels[test_mask].cpu())
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_aupr >= best_val_aupr:
            best_val_aupr = val_aupr
            #best_val_acc = val_acc
            best_val_epoch_aupr = epoch
            #best_model = copy.deepcopy(model)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, val_aupr:{val_aupr: .4f}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f},test_auc:{test_auc: .4f}, test_aupr:{test_aupr: .4f}")

    best_model.eval()
    with torch.no_grad():
        # if graph.num_edge_types != 1:
        #     pred = best_model(graph, x)
        # else:
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        estp_test_auc, estp_test_aupr, precision_f, recall_f = result(pred[test_mask].cpu(), labels[test_mask].cpu())
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
        print(f"--- TestAUPR: {test_aupr:.4f}, early-stopping-TestAUPR: {estp_test_aupr:.4f}, Best ValAUPR: {best_val_aupr:.4f} in epoch {best_val_epoch_aupr} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        #x = x.float()  # Ensure the input is of type float
        logits = self.linear(x)
        return logits
