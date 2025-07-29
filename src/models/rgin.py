from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, dtype
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

from src.utils import block_diag, create_activation


class RGIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_edge_types: int,
        num_layers: int,
        dropout: float,
        activation: str,
        residual: bool = True,
        norm=None,
        encoding: bool = False,
        decomposition: Optional[dict] = None,
        diagonal_weight_matrix: bool = False,
        vertical_stacking: bool = False,
    ):
        """
        A multi‐layer Relational GIN (RGIN).

        Each layer uses an MLP after relational message passing.
        """
        super().__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.dropout = dropout

        # Build a ModuleList of RGINConv layers
        self.layers = nn.ModuleList()
        self.activation = create_activation(activation)
        last_activation = create_activation(activation) if encoding else None
        last_norm = norm if encoding else None
        last_res = residual if encoding else False

        # single‐layer case
        if num_layers == 1:
            self.layers.append(
                RGINConv(
                    in_channels,
                    out_channels,
                    num_edge_types,
                    dropout=dropout,
                    activation=last_activation,
                    residual=last_res,
                    norm=last_norm,
                    decomposition=decomposition,
                    diagonal_weight_matrix=diagonal_weight_matrix,
                    vertical_stacking=vertical_stacking,
                )
            )
        else:
            # input → hidden
            self.layers.append(
                RGINConv(
                    in_channels,
                    hidden_channels,
                    num_edge_types,
                    dropout=dropout,
                    activation=create_activation(activation),
                    residual=residual,
                    norm=norm,
                    decomposition=decomposition,
                    diagonal_weight_matrix=diagonal_weight_matrix,
                    vertical_stacking=vertical_stacking,
                )
            )
            # hidden → hidden
            for _ in range(1, num_layers - 1):
                self.layers.append(
                    RGINConv(
                        hidden_channels,
                        hidden_channels,
                        num_edge_types,
                        dropout=dropout,
                        activation=create_activation(activation),
                        residual=residual,
                        norm=norm,
                        decomposition=decomposition,
                        diagonal_weight_matrix=diagonal_weight_matrix,
                        vertical_stacking=vertical_stacking,
                    )
                )
            # hidden → out
            self.layers.append(
                RGINConv(
                    hidden_channels,
                    out_channels,
                    num_edge_types,
                    dropout=dropout,
                    activation=last_activation,
                    residual=last_res,
                    norm=last_norm,
                    decomposition=decomposition,
                    diagonal_weight_matrix=diagonal_weight_matrix,
                    vertical_stacking=vertical_stacking,
                )
            )

        self.head = nn.Identity()

    def forward(
        self, graph, x: Tensor, num_edge_types: int, return_hidden: bool = False
    ):
        edge_index = graph.edge_index
        edge_type = graph.edge_type
        num_nodes = graph.num_nodes
        triples = torch.stack([edge_index[0], edge_type, edge_index[1]], dim=1)

        h = x
        hiddens = []
        for conv in self.layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(graph, h, edge_index, num_edge_types, num_nodes, triples)
            hiddens.append(h)

        out = self.head(h)
        return (out, hiddens) if return_hidden else out

    def reset_classifier(self, num_classes, concat=False, datas_dim=0):
        param_dtype = next(self.parameters()).dtype  # Get the current dtype.
        if concat:
            self.head = nn.Sequential(
                nn.Linear(self.out_channels + datas_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            ).to(param_dtype)
        else:
            self.head = nn.Sequential(nn.Linear(self.out_channels, 1)).to(param_dtype)
        self.link_head = nn.Sequential(nn.Linear(self.out_channels, 128)).to(
            param_dtype
        )


class RGINConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        activation=None,
        residual: bool = False,
        norm=None,
        decomposition: Optional[dict] = None,
        diagonal_weight_matrix: bool = False,
        vertical_stacking: bool = False,
        **kwargs,
    ):
        """
        A relational GIN conv layer: relation‐specific weighting + 2‐layer MLP.
        """
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        self.activation = activation
        self.residual = residual
        self.norm = norm(out_channels) if norm is not None else None
        self.eps = nn.Parameter(torch.zeros(1))
        self.vertical_stacking = vertical_stacking

        # decomposition flags
        self.decomp_type = decomposition.get("type") if decomposition else None
        self.num_bases = decomposition.get("num_bases") if decomposition else None
        self.num_blocks = decomposition.get("num_blocks") if decomposition else None
        self.diag_weights = diagonal_weight_matrix

        # Relation‐specific weights
        if self.diag_weights:
            # each relation has a diagonal transform
            self.weights = nn.Parameter(torch.Tensor(self.num_edge_types, in_channels))
            # force out_channels == in_channels
            self.out_channels = in_channels
        elif self.decomp_type is None:
            self.weights = nn.Parameter(
                torch.Tensor(self.num_edge_types, in_channels, out_channels)
            )
        elif self.decomp_type == "basis":
            self.bases = nn.Parameter(
                torch.Tensor(self.num_bases, in_channels, out_channels)
            )
            self.comps = nn.Parameter(torch.Tensor(self.num_edge_types, self.num_bases))
        elif self.decomp_type == "block":
            assert in_channels % self.num_blocks == 0
            assert out_channels % self.num_blocks == 0
            self.blocks = nn.Parameter(
                torch.Tensor(
                    self.num_edge_types,
                    self.num_blocks,
                    in_channels // self.num_blocks,
                    out_channels // self.num_blocks,
                )
            )
        else:
            raise ValueError(f"Unknown decomposition: {self.decomp_type}")

        # Optional bias after sum
        if bias and not self.diag_weights:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        act = self.activation if self.activation is not None else create_activation("relu")

        # 2‐layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            act,
            nn.Linear(self.out_channels, self.out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.eps)
        gain = nn.init.calculate_gain("relu")
        if self.decomp_type == "block":
            nn.init.xavier_uniform_(self.blocks, gain=gain)
        elif self.decomp_type == "basis":
            nn.init.xavier_uniform_(self.bases, gain=gain)
            nn.init.xavier_uniform_(self.comps, gain=gain)
        else:
            nn.init.xavier_uniform_(
                self.weights.unsqueeze(-1) if self.diag_weights else self.weights,
                gain=gain,
            )

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        graph,
        x: Tensor,
        edge_index: Adj,
        num_edge_types: int,
        num_nodes: int,
        triples: Tensor,
    ) -> Tensor:
        # ensure dtype match
        param_dtype = next(self.parameters()).dtype
        x = x.to(param_dtype)

        # assemble full relation‐weight tensor
        if self.decomp_type is None and not self.diag_weights:
            W = self.weights
        elif self.decomp_type == "basis":
            W = torch.einsum("rb,bio->rio", self.comps, self.bases)
        elif self.decomp_type == "block":
            W = block_diag(self.blocks)
        else:  # diagonal
            W = self.weights

        if self.vertical_stacking:
            from src.utils import stack_matrices, sum_sparse

            adj_idx, adj_size = stack_matrices(
                triples,
                num_nodes,
                num_edge_types,
                vertical_stacking=True,
                device=x.device,
            )
            vals = torch.ones(adj_idx.size(0), dtype=x.dtype, device=x.device)
            deg = sum_sparse(
                adj_idx, vals, adj_size, row_normalisation=self.vertical_stacking, device=x.device
            )
            vals = vals / deg
            adj = torch.sparse.FloatTensor(adj_idx.t(), vals, adj_size)

            #    adj: (R*N, N),   x: (N, in) --> ag: (R*N, in)
            ag = torch.spmm(adj, x)

            ag = ag.view(num_edge_types, num_nodes, self.in_channels)

            out_msgs = torch.einsum("rio,rni->no", W, ag)

        else:
            # fallback to standard MessagePassing per‐edge
            out_msgs = self.propagate(
                edge_index,
                x=x,
                weights=W,
                edge_type=triples[:, 1],
                size=(num_nodes, num_nodes),
            )

        # injective update with learnable eps:
        #   (1 + eps) * x  +  sum_{u in N(v)} W_{r(u,v)} x_u
        h_total = (1.0 + self.eps) * x + out_msgs

        out = self.mlp(h_total)

        if self.bias is not None:
            out = out + self.bias

        if self.norm is not None:
            out = self.norm(out)

        return out

    def message(self, x_j: Tensor, weights: Tensor, edge_type: Tensor) -> Tensor:
        """
        x_j: [E, in_channels]
        weights: [R, in_channels, out_channels]  or [R, in_channels] for diag
        edge_type: [E]
        """
        if self.diag_weights:
            # element–wise scale
            w = weights[edge_type]  # [E, in_channels]
            return x_j * w  # [E, in_channels]
        else:
            # full mat‐mul per edge
            w = weights[edge_type]  # [E, in_channels, out_channels]
            return torch.einsum("ni,nio->no", x_j, w)  # [E, out_channels]
