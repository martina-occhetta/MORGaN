from functools import partial
from itertools import chain
from typing import Optional

import torch
import torch.nn as nn

from src.utils import create_norm, drop_edge

from .gat import GAT
from .gcn import GCN
from .gin import GIN
from .rgcn import RGCN
from .rgin import RGIN
from .loss_func import sce_loss


def setup_module(
    m_type,
    enc_dec,
    in_dim,
    num_hidden,
    out_dim,
    num_layers,
    dropout,
    activation,
    residual,
    norm,
    nhead,
    nhead_out,
    attn_drop,
    weight_decomposition=None,
    vertical_stacking=True,
    num_edge_types=1,
    negative_slope=0.2,
    concat_out=True,
) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "rgcn":
        mod = RGCN(
            in_channels=int(in_dim),
            hidden_channels=int(num_hidden),
            out_channels=int(out_dim),
            num_edge_types=int(num_edge_types),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            decomposition=weight_decomposition,  # {'type': 'basis', 'num_bases': 2},
            vertical_stacking=vertical_stacking,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "rgin":
        mod = RGIN(
            in_channels=int(in_dim),
            hidden_channels=int(num_hidden),
            out_channels=int(out_dim),
            num_edge_types=int(num_edge_types),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            decomposition=weight_decomposition,  # {'type': 'basis', 'num_bases': 2},
            vertical_stacking=vertical_stacking,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim),
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        nhead: int,
        nhead_out: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
        norm: Optional[str],
        mask_rate: float = 0.3,
        encoder_type: str = "gat",
        decoder_type: str = "gat",
        loss_fn: str = "sce",
        drop_edge_rate: float = 0.0,
        replace_rate: float = 0.1,
        alpha_l: float = 2,
        concat_hidden: bool = False,
        num_edge_types: int = 1,
        weight_decomposition=None,
        vertical_stacking=True,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = (
            num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden
        )

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            num_edge_types=num_edge_types,
            weight_decomposition=weight_decomposition,
            vertical_stacking=vertical_stacking,
            # return_hidden=False,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
            num_edge_types=num_edge_types,
            weight_decomposition=weight_decomposition,
            vertical_stacking=vertical_stacking,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(
                dec_in_dim * num_layers, dec_in_dim, bias=False
            )
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        """
        Mask node features for encoding.

        Args:
            g: Graph
            x: Input features
            mask_rate: Masking rate

        Returns:
            g: Graph with masked features
            x: Masked features
            (mask_nodes, keep_nodes): Tuple of masked and kept node indices
        """
        num_nodes = g.num_nodes if hasattr(g, "num_nodes") else x.size(0)
        perm = torch.randperm(num_nodes, device=x.device)

        # Ensure we mask at least one node
        num_mask = max(1, int(mask_rate * num_nodes))
        mask_nodes = perm[:num_mask]
        keep_nodes = perm[num_mask:]

        out_x = x.clone()

        # For categorical features (like in MUTAG), we randomly change the atom type
        if (
            x.dtype == torch.float32 and x.sum(dim=1).mean() == 1.0
        ):  # One-hot encoded features
            # For each masked node, randomly select a different atom type
            num_atom_types = x.size(1)
            for node_idx in mask_nodes:
                current_type = x[node_idx].argmax()
                # Get all possible types except current
                possible_types = list(range(num_atom_types))
                possible_types.remove(current_type.item())
                # Randomly select a new type
                new_type = torch.tensor(possible_types)[
                    torch.randint(len(possible_types), (1,))
                ].item()
                # Create new one-hot vector
                new_feature = torch.zeros_like(x[node_idx])
                new_feature[new_type] = 1
                out_x[node_idx] = new_feature
        else:
            # For continuous features, use standard noise masking
            noise_nodes = mask_nodes
            num_noise = len(noise_nodes)

            if num_noise > 0:
                noise_to_be_chosen = torch.randperm(num_noise)
                out_x[noise_nodes] = x[noise_nodes[noise_to_be_chosen]]

        return g, out_x, (mask_nodes, keep_nodes)

    def forward(self, graph, x, num_edge_types=None):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(graph, x, num_edge_types)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, graph, x, num_edge_types=None):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            graph, x, self._mask_rate
        )

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(
                pre_use_g, self._drop_edge_rate, return_edges=True
            )
            # use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_g = pre_use_g

        if self._encoder_type == "rgcn" or self._encoder_type == "rgin":
            enc_rep, all_hidden = self.encoder(
                use_g, use_x, num_edge_types, return_hidden=True
            )
        else:
            enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        elif self._decoder_type == "rgcn" or self._decoder_type == "rgin":
            recon = self.decoder(pre_use_g, rep, num_edge_types)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, graph, x, num_edge_types=None):
        if self._encoder_type == "rgcn" or self._encoder_type == "rgin":
            rep, hidden_list = self.encoder(
                graph, x, num_edge_types, return_hidden=True
            )
            return rep, hidden_list
        else:
            rep = self.encoder(graph, x)
            return rep, None

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
