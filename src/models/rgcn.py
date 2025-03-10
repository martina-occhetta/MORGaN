
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn.conv import MessagePassing

class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, residual=False, norm=None, bias=True):
        """
        A relational graph convolution layer for homogeneous graphs with integer-encoded edge types.
        
        Args:
            in_channels (int): Dimension of input node features.
            out_channels (int): Dimension of output node features.
            num_relations (int): Number of distinct relation types.
            residual (bool): Whether to use residual connection.
            norm (callable or None): Normalization layer constructor.
            bias (bool): Whether to include a bias term.
        """
        super(RGCNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        
        # Each relation type has its own weight matrix.
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        # Self-loop (or root) transformation.
        self.self_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        # Set up residual connection.
        self.residual = residual
        if residual:
            if in_channels != out_channels:
                self.res_fc = nn.Linear(in_channels, out_channels, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None
        
        # Set up normalization layer.
        self.norm_layer = norm(out_channels) if norm is not None else None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        xavier_uniform_(self.weight)
        xavier_uniform_(self.self_weight)
        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            xavier_uniform_(self.res_fc.weight)
        if self.bias is not None:
            zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x (Tensor): Node features of shape [N, in_channels].
            edge_index (LongTensor): Edge indices with shape [2, E].
            edge_type (LongTensor): Edge type tensor with shape [E] and integer values in [0, num_relations-1].
            
        Returns:
            Tensor: Updated node features of shape [N, out_channels].
        """
        # Self-loop transformation.
        out = torch.matmul(x, self.self_weight)
        
        # Propagate messages from neighbors using relation-specific weight matrices.
        out = out + self.propagate(edge_index, x=x, edge_type=edge_type)
        
        # Add residual connection if enabled
        if self.res_fc is not None:
            out = out + self.res_fc(x)
        
        # Apply normalization if provided
        if self.norm_layer is not None:
            out = self.norm_layer(out)
        
        # Add bias if provided
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_j, edge_type):
        # For each edge, select its corresponding weight matrix.
        # x_j: shape [E, in_channels]
        # edge_type: shape [E]
        weight = self.weight[edge_type]  # shape [E, in_channels, out_channels]
        # Multiply each source node feature by the corresponding weight.
        # Unsqueeze x_j to shape [E, 1, in_channels] then batch multiply.
        msg = torch.bmm(x_j.unsqueeze(1), weight)  # shape [E, 1, out_channels]
        return msg.squeeze(1)

class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_layers, dropout, activation, residual=True, norm=None, encoding=False):
        """
        A multi-layer RGCN model for homogeneous graphs with relation types encoded as integers.
        
        Args:
            in_channels (int): Input node feature dimension.
            hidden_channels (int): Hidden layer dimension.
            out_channels (int): Output node feature dimension.
            num_relations (int): Number of distinct relation types.
            num_layers (int): Number of RGCN layers.
            dropout (float): Dropout probability.
            activation (callable): Activation function (e.g., torch.relu).
            residual (bool): Whether to use residual connections.
            norm (callable or None): Normalization layer constructor.
            encoding (bool): If True, apply activation, residual, and norm in the last layer.
        """
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.encoding = encoding
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            last_activation = activation if encoding else None
            last_residual = residual if encoding else False
            last_norm = norm if encoding else None
            self.layers.append(RGCNConv(in_channels, out_channels, num_relations, residual=last_residual, norm=last_norm))
        else:
            self.layers.append(RGCNConv(in_channels, hidden_channels, num_relations, residual=residual, norm=norm))          
            for _ in range(num_layers - 2):
                self.layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations, residual=residual, norm=norm))
            last_activation = activation if encoding else None
            last_residual = residual if encoding else False
            last_norm = norm if encoding else None
            self.layers.append(RGCNConv(hidden_channels, out_channels, num_relations, residual=last_residual, norm=last_norm))       
        self.head = nn.Identity()
    
    def forward(self, x, edge_index, edge_type, return_hidden=False):
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
        hidden_list = []
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index, edge_type)
            if self.activation is not None:
                x = self.activation(x)
            hidden_list.append(x)
        out = self.head(x)
        if return_hidden:
            return out, hidden_list
        else:
            return out