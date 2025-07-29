from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size, SparseTensor
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn.conv import MessagePassing
from src.utils import create_activation, block_diag, stack_matrices, sum_sparse
import math


class RGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_edge_types: int,
        num_layers: int,
        dropout: float,
        activation: str,
        residual=True,
        norm=None,
        encoding=False,
        decomposition=None,
        diagonal_weight_matrix: bool = False,
        vertical_stacking=True,
    ):
        """
        A multi-layer RGCN model that uses the custom RGCNConv layer.

        Args:
            in_channels (int): Input node feature dimension.
            hidden_channels (int): Hidden layer dimension.
            out_channels (int): Output node feature dimension.
            num_edge_types (int): Number of relation types.
            num_layers (int): Number of RGCN layers.
            dropout (float): Dropout probability.
            activation (callable): Activation function.
            decomposition (dict or None): Decomposition config (e.g., {'type': 'basis', 'num_bases': 3}).
            diagonal_weight_matrix (bool): If True, use a diagonal weight matrix.
            vertical_stacking (bool): Whether to use vertical stacking for message aggregation.
        """
        super(RGCN, self).__init__()

        self.out_channels = out_channels
        self.num_layers = num_layers
        # stack of RGCNConv layers
        self.rgcn_layers = nn.ModuleList()
        self.num_edge_types = num_edge_types

        self.dropout = dropout

        self.activation = create_activation(activation)
        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.rgcn_layers.append(
                RGCNConv(
                    in_channels,
                    out_channels,
                    num_edge_types,
                    dropout=dropout,
                    activation=last_activation,
                    residual=last_residual,
                    norm=last_norm,
                    decomposition=decomposition,
                    diagonal_weight_matrix=diagonal_weight_matrix,
                    vertical_stacking=vertical_stacking,
                )
            )
        else:
            # input projection
            self.rgcn_layers.append(
                RGCNConv(
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

            # hidden layers
            for _ in range(1, num_layers - 1):
                self.rgcn_layers.append(
                    RGCNConv(
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
            # output projection
            self.rgcn_layers.append(
                RGCNConv(
                    hidden_channels,
                    out_channels,
                    num_edge_types,
                    dropout=dropout,
                    activation=last_activation,
                    residual=last_residual,
                    norm=last_norm,
                    decomposition=decomposition,
                    diagonal_weight_matrix=diagonal_weight_matrix,
                    vertical_stacking=vertical_stacking,
                )
            )

        self.head = nn.Identity()

    def forward(self, graph, x, num_edge_types, return_hidden=False):
        """
        Args:
            x (Tensor): Node feature tensor of shape [N, in_channels].
            edge_index (LongTensor): Edge indices of shape [2, E].
            edge_type (LongTensor): Edge type tensor of shape [E].
            return_hidden (bool): If True, also return hidden representations from each layer.

        Returns:
            Tensor: Final node representations.
            (optional) List[Tensor]: List of hidden representations per layer.
        """
        # Extract edge_index and edge_type from the graph.
        edge_index = graph.edge_index  # shape: [2, E]
        edge_type = graph.edge_type  # shape: [E]
        num_nodes = graph.num_nodes
        h = x

        # Compute triples: stack [source, edge_type, destination].
        triples = torch.stack([edge_index[0], edge_type, edge_index[1]], dim=1)

        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.rgcn_layers[l](
                graph, h, edge_index, num_edge_types, num_nodes, triples
            )
            if self.activation is not None:
                h = self.activation(h)
            hidden_list.append(h)
        out = self.head(h)
        if return_hidden:
            return out, hidden_list
        else:
            return out

    def reset_classifier(self, num_classes, concat=False, datas_dim=0):
        dtype = next(self.parameters()).dtype  # Get the current dtype.
        if concat:
            self.head = nn.Sequential(
                nn.Linear(self.out_channels + datas_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            ).to(dtype)
        else:
            self.head = nn.Sequential(nn.Linear(self.out_channels, 1)).to(dtype)
        self.link_head = nn.Sequential(nn.Linear(self.out_channels, 128)).to(dtype)


class RGCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_edge_types: Optional[int] = 1,
        dropout: float = 0.0,
        bias: bool = True,
        activation=None,
        residual=False,
        norm=None,
        decomposition: Optional[dict] = None,
        diagonal_weight_matrix: bool = False,
        vertical_stacking: bool = False,
        **kwargs,
    ):
        """
        A relational graph convolution layer for homogeneous graphs with integer-encoded edge types.

        Args:
            in_channels (int): Dimension of input node features.
            out_channels (int): Dimension of output node features.
            num_edge_types (int): Number of distinct relation types.
            residual (bool): Whether to use residual connection.
            norm (callable or None): Normalization layer constructor.
            bias (bool): Whether to include a bias term.
        """
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)
        # super(RGCNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types

        self.dropout = dropout

        self.vertical_stacking = vertical_stacking

        self.activation = activation

        self.weight_decomposition = (
            decomposition["type"]
            if decomposition is not None and "type" in decomposition
            else None
        )
        self.num_bases = (
            decomposition["num_bases"]
            if decomposition is not None and "num_bases" in decomposition
            else None
        )
        self.num_blocks = (
            decomposition["num_blocks"]
            if decomposition is not None and "num_blocks" in decomposition
            else None
        )
        self.diagonal_weight_matrix = diagonal_weight_matrix

        if self.diagonal_weight_matrix:
            self.weights = torch.nn.Parameter(
                torch.empty(num_edge_types, in_channels, out_channels)
            )
            self.out_channels = self.in_channels
            self.weight_decomposition = None
            bias = None
        elif self.weight_decomposition is None:
            self.weights = nn.Parameter(
                torch.Tensor(num_edge_types, in_channels, out_channels)
            )
        elif self.weight_decomposition == "basis":
            # Weight regularisation using basis decomposition.
            assert self.num_bases > 0, "Number of bases must be greater than 0."
            self.bases = nn.Parameter(
                torch.Tensor(self.num_bases, in_channels, out_channels)
            )
            self.comps = nn.Parameter(torch.Tensor(num_edge_types, self.num_bases))
        elif self.weight_decomposition == "block":
            # Weight regularisation using block decomposition.
            assert self.num_blocks > 0, "Number of blocks must be greater than 0."
            assert in_channels % self.num_blocks == 0, (
                f"Number of input channels {in_channels} must be divisible by the number of blocks {self.num_blocks}."
            )
            assert out_channels % self.num_blocks == 0, (
                f"Number of output channels {out_channels} must be divisible by the number of blocks {self.num_blocks}."
            )
            self.blocks = nn.Parameter(
                torch.Tensor(
                    num_edge_types,
                    self.num_blocks,
                    in_channels // self.num_blocks,
                    out_channels // self.num_blocks,
                )
            )
        else:
            raise NotImplementedError(
                f'Unknown weight decomposition type: {self.weight_decomposition}. Must be one of "basis", "block", or None.'
            )
        self.norm = norm(out_channels) if norm is not None else None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # self.fc = nn.Linear(in_channels, out_channels) # In case extra linear transformation required
        self.reset_parameters()

    def reset_parameters(self, reset_mode="glorot_uniform"):
        if reset_mode == "glorot_uniform":
            if self.weight_decomposition == "block":
                xavier_uniform_(self.blocks, gain=nn.init.calculate_gain("relu"))
            elif self.weight_decomposition == "basis":
                xavier_uniform_(self.bases, gain=nn.init.calculate_gain("relu"))
                xavier_uniform_(self.comps, gain=nn.init.calculate_gain("relu"))
            else:
                xavier_uniform_(self.weights, gain=nn.init.calculate_gain("relu"))

            if self.bias is not None:
                zeros_(self.bias)
        elif reset_mode == "uniform":
            stdv = 1.0 / math.sqrt(self.weights.size(1))
            if self.weight_decomposition == "block":
                self.blocks.data.uniform_(-stdv, stdv)
            elif self.weight_decomposition == "basis":
                self.bases.data.uniform_(-stdv, stdv)
                self.comps.data.uniform_(-stdv, stdv)
            else:
                self.weights.data.uniform_(-stdv, stdv)

            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise NotImplementedError(
                f"{reset_mode} parameter initialisation method has not been implemented"
            )

    def forward(
        self,
        graph,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        num_edge_types,
        num_nodes,
        triples,
    ):
        """
        Perform a single pass of message propagation.

        Args:
            graph (Data): The graph object.
            x (Tensor): Node features of shape [N, in_channels].
            edge_index (LongTensor): Edge indices with shape [2, E].
            edge_type (LongTensor): Edge type tensor with shape [E] and integer values in [0, num_edge_types-1].
            triples (Tensor): Precomputed triples (e.g., torch.stack([edge_index[0], edge_type, edge_index[1]], dim=1)) computed externally.

        Returns:
            Tensor: Updated node features of shape [N, out_channels].
        """
        x = x.to(
            next(self.parameters()).dtype
        )  # Ensure the input features have the same dtype as the model parameters.

        general_edge_count = int((triples.size(0) - num_nodes) / 2)

        # Choose weights
        if self.weight_decomposition is None:
            weights = self.weights
        elif self.weight_decomposition == "basis":
            weights = torch.einsum("rb, bio -> rio", self.comps, self.bases)
        elif self.weight_decomposition == "block":
            weights = block_diag(self.blocks)
        else:
            raise NotImplementedError(
                f'Unknown weight decomposition type: {self.weight_decomposition}. Must be one of "basis", "block", or None.'
            )

        # Stack adjancency matrices
        adj_indices, adj_size = stack_matrices(
            triples,
            num_nodes,
            num_edge_types,
            vertical_stacking=self.vertical_stacking,
            device=x.device,
        )

        num_triples = adj_indices.size(0)

        vals = torch.ones(
            num_triples, dtype=next(self.parameters()).dtype, device=x.device
        )

        # Normalisation
        sums = sum_sparse(
            adj_indices,
            vals,
            adj_size,
            row_normalisation=self.vertical_stacking,
            device=x.device,
        )
        if not self.vertical_stacking:
            n = general_edge_count
            i = num_nodes
            sums = torch.cat([sums[n : 2 * n], sums[:n], sums[-i:]], dim=0)

        vals = vals / sums

        # Construct adj matrix
        if x.device.type == "cuda":
            adj = torch.cuda.sparse.FloatTensor(adj_indices.t(), vals, adj_size)
        else:
            adj = torch.sparse.FloatTensor(adj_indices.t(), vals, adj_size)

        # Perform message passing
        if self.diagonal_weight_matrix:
            fw = torch.einsum("ij,kj -> kij", x, weights)
            fw = torch.reshape(fw, (num_edge_types * num_nodes, self.in_channels))
            out = torch.mm(adj, fw)
        elif self.vertical_stacking:
            af = torch.spmm(adj, x)
            af = af.view(num_edge_types, num_nodes, self.in_channels)
            out = torch.einsum("rio, rni -> no", weights, af)
        else:
            fw = torch.einsum("ni, rio -> rno", x, weights).contiguous()
            out = torch.mm(adj, fw.view(num_edge_types * num_nodes, self.out_channels))

        assert out.size() == (num_nodes, self.out_channels)

        # Add bias if provided
        if self.bias is not None:
            out = torch.add(out, self.bias)

        return out
