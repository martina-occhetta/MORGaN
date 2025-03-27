import argparse
import os

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard.writer import SummaryWriter

from src.multidim_models.modig import MODIG
from src.multidim_models.modig_utils import *
from src.multidim_models.mdmni import MDMNI_DGD
from src.datasets.data_util import load_processed_graph

cuda = torch.cuda.is_available()

def main(args):
    graph = load_processed_graph(PATH='data/real/smg_data', LABEL_PATH='data/real/labels/NIHMS80906-small_mol-and-bio-druggable.tsv', ppi='CPDB')
    num_features = graph.x.shape[1]
    num_classes = graph.y.max().item() + 1
    
    def train(mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(graph)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).cuda())

        acc = metrics.accuracy_score(label.cpu(), np.round(
            torch.sigmoid(output[mask]).cpu().detach().numpy()))
        loss.backward()
        optimizer.step()

        del output
        return loss.item(), acc

    @torch.no_grad()
    def test(mask, label):
        model.eval()
        output = model(graph)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).cuda())

        acc = metrics.accuracy_score(label.cpu(), np.round(
            torch.sigmoid(output[mask]).cpu().detach().numpy()))
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(label.to('cpu'), pred)
        pr, rec, _ = metrics.precision_recall_curve(label.to('cpu'), pred)
        aupr = metrics.auc(rec, pr)

        return pred, loss.item(), acc, auroc, aupr

    AUC = np.zeros(shape=(10, 5))
    AUPR = np.zeros(shape=(10, 5))
    ACC = np.zeros(shape=(10, 5))

    pred_all = []
    label_all = []

    for j in range(len(k_sets)):
        print(j)
        for cv_run in range(5):
            train_mask, val_mask, train_label, val_label = [
                p.cuda() for p in k_sets[j][cv_run] if type(p) == torch.Tensor]

            model = MODIG(
                nfeat=n_fdim, hidden_size1=args['hs1'], hidden_size2=args['hs2'], dropout=args['dp'])
            model.cuda()
            optimizer = optim.Adam(
                model.parameters(), lr=args['lr'], weight_decay=args['wd'])
            # model_save_file = os.path.join(log_dir, str(cv_run) + '_modig.pth')

            early_stopping = EarlyStopping(
                patience=args['patience'], verbose=True)

            for epoch in range(1, args['epochs']+1):
                _, _ = train(train_mask, train_label)
                _, loss_val, _, _, _ = test(val_mask, val_label)

                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at the epoch {epoch}")
                    break

                torch.cuda.empty_cache()

            pred, _, ACC[j][cv_run], AUC[j][cv_run], AUPR[j][cv_run] = test(
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

    # Use all label to train a final model
    all_mask = torch.LongTensor(idx_list)
    all_label = torch.FloatTensor(label_list).reshape(-1, 1)

    model = MODIG(nfeat=n_fdim, hidden_size1=args['hs1'],
                  hidden_size2=args['hs2'], dropout=args['dp'])
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    for epoch in range(1, args['epochs']+1):
        print(epoch)
        _, _ = train(all_mask.cuda(), all_label.cuda())
        torch.cuda.empty_cache()

    output = model(graphlist_adj)

    pred = torch.sigmoid(output).cpu().detach().numpy()
    pred2 = torch.sigmoid(output[~all_mask]).cpu().detach().numpy()
    torch.save(pred, os.path.join(file_save_path, args['ppi'] + '_pred.pkl'))
    torch.save(all_label, os.path.join(
        file_save_path, args['ppi'] + '_label.pkl'))
    torch.save(pred2, os.path.join(file_save_path, args['ppi'] + '_pred2.pkl'))

    pd.Series(final_gene_node).to_csv(os.path.join(file_save_path,
                                                   'final_gene_node.csv'), index=False, header=False)

    plot_average_PR_curve(pred_all, label_all, file_save_path)
    plot_average_ROC_curve(pred_all, label_all, file_save_path)