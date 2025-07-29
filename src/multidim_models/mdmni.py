import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATConv, SAGEConv, GraphSAGE
from torch.nn.parameter import Parameter
import numpy as np
import math

# from torch_sparse import spmm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAT_SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_heads=2):
        super(GAT_SAGE, self).__init__()
        # print("num_heads:", num_heads)
        assert hidden_dim % num_heads == 0

        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
        )

        self.sage1 = SAGEConv(hidden_dim * num_heads, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, output_dim)

        self.dropout = dropout
        self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, data, x=None, return_hidden=False):
        x, edge_index = data.x, data.edge_index

        # GAT layers
        x_gat = F.dropout(x, self.dropout, training=self.training)
        x_gat = F.relu(self.gat1(x_gat, edge_index))
        x_gat = F.dropout(x_gat, self.dropout, training=self.training)
        x_gat = self.gat2(x_gat, edge_index)

        # SAGE layers
        x_sage = F.dropout(x_gat, self.dropout, training=self.training)
        x_sage = F.relu(self.sage1(x_sage, edge_index))
        x_sage = F.dropout(x_sage, self.dropout, training=self.training)
        x_sage = F.relu(self.sage2(x_sage, edge_index))

        # BatchNorm layer
        x = self.batchnorm(x_sage)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=8):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class MDMNI_DGD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MDMNI_DGD, self).__init__()

        self.gat_sage = GAT_SAGE(input_dim, hidden_dim, output_dim, dropout)
        self.attn = Attention(output_dim)
        self.MLP = nn.Linear(output_dim, 1)
        self.dropout = dropout

    def forward(self, graphs):
        old_emb2 = []
        for i in range(len(graphs)):
            emb = self.gat_sage(graphs[i])
            old_emb2.append(emb)
        att_emb, atts = self.attn(torch.stack(old_emb2, dim=1))

        predict_class = self.MLP(att_emb)

        return predict_class
