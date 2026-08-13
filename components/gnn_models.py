"""
Contains all GNN model classes used in testing.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from components.external.relgnn.examples.atomic_routes import get_atomic_routes
from components.external.relgnn.examples.relgnn_conv import RelGNNConv
from components.external.relgnn.examples.relgnn_hetero_conv import RelGNN_HeteroConv
from redelex.nn.layers import CrossAttentionConv, SelfAttention
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.nn import (
    HeteroConv,
    HeteroDictLinear,
    LayerNorm,
    SAGEConv,
)
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.dropout import dropout_edge


class GNNModel(torch.nn.Module, ABC):
    """
    The base class for all GNN models.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset_parameters(self) -> None:
        """
        Resets all learnable parameters.
        """
        ...

    @abstractmethod
    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[NodeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
    ) -> dict[NodeType, Tensor]:
        """
        The forward pass. Updates all node embeddings using the specific GNN.

        Args:
            x_dict: The input node embeddings and their types
            edge_index_dict: The input edges and their types

        Returns:
            The updated node embeddings
        """
        ...

    def apply_edge_dropout(
        self, edge_index_dict: dict[NodeType, Tensor], p: float
    ) -> dict[NodeType, Tensor]:
        """
        Utility function for applying edge dropout to a given dict of edges.

        Args:
            edge_index_dict: The input edges and their types

        Returns:
            The edges after applying dropout
        """
        if p == 0.0 or not self.training:
            return edge_index_dict

        else:
            return {
                key: dropout_edge(edge_index, p=p, training=True)[0]
                for key, edge_index in edge_index_dict.items()
            }


# Adapted from RelBench under MIT License
# Original source: https://github.com/snap-stanford/relbench/blob/main/relbench/modeling/nn.py
# License: https://opensource.org/license/mit
class HeteroGraphSAGE(GNNModel):
    """
    A heterogeneous implementation of the GraphSAGE model.

    Adapted from RelBench under the MIT License, modified to add edge and feature dropout.
    """

    def __init__(
        self,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            node_types: The node types of the heterogeneous graph
            edge_types: The edge types of the heterogeneous graph
            channels: The number of channels
            aggr: The aggregation method
            num_layers: The number of conv layers
            dropout: The feature dropout chance (added to original)
        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.dropout = torch.nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        """
        Overrides `GNNModel.reset_parameters`.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[NodeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
    ) -> dict[NodeType, Tensor]:
        """
        Overrides `GNNModel.forward`.

        Adds feature and edge dropout to the original implementation.
        """
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, self.apply_edge_dropout(edge_index_dict, p=0.2))
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return x_dict


# Adapted from RelGNN under MIT License
# Original source: https://github.com/snap-stanford/RelGNN/blob/main/examples/relgnn_nn.py
# License: https://opensource.org/license/mit
class RelGNN(GNNModel):
    """
    The RelGNN model.

    Adapted from RelGNN under MIT License, but with added feature, attention and edge dropout.
    """

    def __init__(
        self,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        channels: int,
        aggr: str = "sum",
        num_layers: int = 2,
        num_heads: int = 1,
        simplified_MP=False,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        """
        Args:
            node_types: The node types of the heterogeneous graph
            edge_types: The edge types of the heterogeneous graph
            channels: The number of channels
            aggr: The aggregation method
            num_layers: The number of conv layers
            num_heads: The number of attention heads
            ff_dropout: The feature dropout chance (added to original)
            attn_dropout: The attention dropout chance (added to original)
        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = RelGNN_HeteroConv(
                {
                    edge_type: RelGNNConv(
                        edge_type[0],
                        (channels, channels),
                        channels,
                        num_heads,
                        aggr=aggr,
                        simplified_MP=simplified_MP,
                        dropout=attn_dropout,
                    )
                    for edge_type in get_atomic_routes(edge_types)
                },
                aggr=aggr,
                simplified_MP=simplified_MP,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.dropout = torch.nn.Dropout(ff_dropout)

    def reset_parameters(self) -> None:
        """
        Overrides `GNNModel.reset_parameters`.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[NodeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
    ) -> dict[NodeType, Tensor]:
        """
        Overrides `GNNModel.forward`.

        Added feature and edge dropout.
        """
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, self.apply_edge_dropout(edge_index_dict, p=0.2))
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return x_dict


# Adapted from ReDeLEx under MIT License
# Original source: https://github.com/jakubpeleska/redelex/blob/develop/redelex/nn/models/dbformer.py
# License: https://opensource.org/license/mit
class DBFormer(GNNModel):
    """
    The DBFormer model.

    Adapted from ReDeLEx under MIT License, but added feature dropout next to the existing attention dropout.
    """

    def __init__(
        self,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        channels: int,
        col_stats_dict: dict[str, dict[str, dict[StatType, Any]]],
        aggr: str = "mean",
        num_layers: int = 2,
        num_heads: int = 2,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        with_norm: bool = True,
        with_residuals: bool = True,
        with_output_transform: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.with_norm = with_norm
        self.with_residuals = with_residuals

        self.attn = torch.nn.ModuleList()
        for _ in range(num_layers):
            attn_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                attn_dict[node_type] = SelfAttention(
                    channels, num_heads=num_heads, dropout=attn_dropout
                )
            self.attn.append(attn_dict)

        self.attn_norm = None
        if with_norm:
            self.attn_norm = torch.nn.ModuleList()
            for _ in range(num_layers):
                norm_dict = torch.nn.ModuleDict()
                for node_type in node_types:
                    num_cols = len(col_stats_dict[node_type].keys())
                    norm_dict[node_type] = torch.nn.LayerNorm([num_cols, channels])
                self.attn_norm.append(norm_dict)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: CrossAttentionConv(
                        channels, num_heads=num_heads, dropout=attn_dropout, aggr=aggr
                    )
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.conv_norm = None
        if with_norm:
            self.conv_norm = torch.nn.ModuleList()
            for _ in range(num_layers):
                norm_dict = torch.nn.ModuleDict()
                for node_type in node_types:
                    num_cols = len(col_stats_dict[node_type].keys())
                    norm_dict[node_type] = torch.nn.LayerNorm([num_cols, channels])
                self.conv_norm.append(norm_dict)

        self.with_output_transform = with_output_transform
        if with_output_transform:
            self._preout_channels = {
                node_type: channels * len(col_stats_dict[node_type].keys())
                for node_type in node_types
            }
            self.output_transform = HeteroDictLinear(
                in_channels=self._preout_channels,
                out_channels=channels,
                types=node_types,
                bias=True,
            )

        self.dropout = torch.nn.Dropout(ff_dropout)

    def reset_parameters(self):
        for attn_dict in self.attn:
            for attn in attn_dict.values():
                attn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

        if self.with_norm:
            for norm_dict in self.attn_norm + self.conv_norm:
                for norm in norm_dict.values():
                    norm.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[NodeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
    ) -> dict[NodeType, Tensor]:
        for i in range(self.num_layers):
            x_dict_next = {}
            # Apply self-attention
            for key in x_dict:
                x_dict_next[key] = self.attn[i][key](x_dict[key])
                if self.with_norm:
                    x = x_dict_next[key]
                    if self.with_residuals:
                        # Optionally apply residuals
                        x += x_dict[key]
                    # Apply normalization
                    x_dict_next[key] = self.dropout(self.attn_norm[i][key](x))
            # Update x_dict
            x_dict = x_dict_next
            # Apply cross-attention
            x_dict_next: dict[str, Tensor] = self.convs[i](
                x_dict, self.apply_edge_dropout(edge_index_dict, p=0.2)
            )
            if self.with_norm:
                for key in x_dict.keys():
                    x = x_dict_next[key]
                    if self.with_residuals:
                        # Optionally apply residuals
                        x += x_dict[key]
                    # Apply normalization
                    x_dict_next[key] = self.dropout(self.conv_norm[i][key](x))
            # Update x_dict
            x_dict = x_dict_next

        if self.with_output_transform:
            x_dict = {
                node_type: x.view(x.size(0), self._preout_channels[node_type])
                for node_type, x in x_dict.items()
            }
            x_dict = self.output_transform(x_dict)

        return x_dict
