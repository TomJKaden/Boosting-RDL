"""
Contains all feature and time encoders.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch_frame
from redelex.nn.encoders import LinearRowEncoder, PerFeatureRowEncoder
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.nn import PositionalEncoding
from torch_geometric.typing import NodeType


class DataEncoder(torch.nn.Module, ABC):
    """
    Base class for the data encoders that create the initial node embeddings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.extra_dimension = False

    @abstractmethod
    def reset_parameters(self) -> None:
        """
        Reset all learnable parameters.
        """
        ...

    @abstractmethod
    def forward(
        self, tf_dict: dict[NodeType, torch_frame.TensorFrame]
    ) -> dict[NodeType, Tensor]:
        """
        The forward pass. Transforms TensorFrames into Tensor embeddings.

        Args:
            tf_dict: Mapping of node types to their TensorFrames.

        Returns:
            Mapping of node types to Tensor embeddings.
        """
        ...


class TimeEncoder(torch.nn.Module, ABC):
    """
    Base class for temporal encoders that create nodes' time embeddings.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset_parameters(self) -> None:
        """
        Reset all learnable parameters.
        """
        ...

    @abstractmethod
    def forward(
        self,
        seed_time: Tensor,
        time_dict: dict[NodeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        """
        The forward pass. Transforms nodes and their times to time embeddings given a seed time.
        """
        ...


# Adapted from RelBench under MIT License
# Original source: https://github.com/snap-stanford/relbench/blob/main/relbench/modeling/nn.py
# License: https://opensource.org/license/mit
class HeteroEncoder(DataEncoder):
    """HeteroEncoder based on PyTorch Frame.

    Adapted from RelBench under MIT License to inherit from :class:`DataEncoder` to enable easy replacement with other approaches.
    """

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: dict[NodeType, dict[torch_frame.stype, list[str]]],
        node_to_col_stats: dict[NodeType, dict[str, dict[StatType, Any]]],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ) -> None:
        """
        Args:
            channels: The output channels for each node type.
            node_to_col_names_dict:
                A dictionary mapping from node type to column names dictionary
                compatible to PyTorch Frame.
            torch_frame_model_cls: Model class for PyTorch Frame. The class object
                takes :class:`TensorFrame` object as input and outputs
                :obj:`channels`-dimensional embeddings. Default to
                :class:`torch_frame.nn.ResNet`.
            torch_frame_model_kwargs: Keyword arguments for
                :class:`torch_frame_model_cls` class. Default keyword argument is
                set specific for :class:`torch_frame.nn.ResNet`. Expect it to
                be changed for different :class:`torch_frame_model_cls`.
            default_stype_encoder_cls_kwargs:
                A dictionary mapping from :obj:`torch_frame.stype` object into a
                tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
                keyword arguments :obj:`kwargs`.
        """
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict:
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self) -> None:
        """
        Overrides `DataEncoder.reset_parameters`.
        """
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: dict[NodeType, torch_frame.TensorFrame],
    ) -> dict[NodeType, Tensor]:
        """
        Overrides `DataEncoder.forward`.
        """
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict


class ResNetEncoder(HeteroEncoder):
    """
    A preconfigured :class:`HeteroEncoder` using a 4-layer ResNet for creating embeddings.
    """

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: dict[NodeType, dict[torch_frame.stype, list[str]]],
        node_to_col_stats: dict[NodeType, dict[str, dict[StatType, Any]]],
    ) -> None:
        super().__init__(
            channels,
            node_to_col_names_dict,
            node_to_col_stats,
            torch_frame_model_cls=ResNet,
            torch_frame_model_kwargs={
                "channels": channels,
                "num_layers": 4,
            },
        )


class LinearEncoder(HeteroEncoder):
    """
    A preconfigured :class:`HeteroEncoder` using the ReDeLex :class:`LinearRowEncoder` for creating embeddings.
    """

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: dict[NodeType, dict[torch_frame.stype, list[str]]],
        node_to_col_stats: dict[NodeType, dict[str, dict[StatType, Any]]],
    ) -> None:
        super().__init__(
            channels,
            node_to_col_names_dict,
            node_to_col_stats,
            torch_frame_model_cls=LinearRowEncoder,
            torch_frame_model_kwargs={
                "channels": channels,
            },
        )


class PerFeatureEncoder(HeteroEncoder):
    """
    A preconfigured :class:`HeteroEncoder` using the ReDeLex :class:`PerFeatureRowEncoder` for creating embeddings.
    Adds an extra dimension to the embeddings. Used e.g. for the DBFormer model.
    """

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: dict[NodeType, dict[torch_frame.stype, list[str]]],
        node_to_col_stats: dict[NodeType, dict[str, dict[StatType, Any]]],
    ) -> None:
        super().__init__(
            channels,
            node_to_col_names_dict,
            node_to_col_stats,
            torch_frame_model_cls=PerFeatureRowEncoder,
            torch_frame_model_kwargs={
                "channels": channels,
            },
        )
        self.extra_dimension = True


# Adapted from RelBench under MIT License
# Original source: https://github.com/snap-stanford/relbench/blob/main/relbench/modeling/nn.py
# License: https://opensource.org/license/mit
class HeteroTemporalEncoder(TimeEncoder):
    """
    Adapted from RelBench under MIT License to inherit from :class:`TimeEncoder` class to allow for easy replacements for other methods.
    """

    def __init__(self, node_types: list[NodeType], channels: int) -> None:
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict(
            {node_type: PositionalEncoding(channels) for node_type in node_types}
        )
        self.lin_dict = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(channels, channels) for node_type in node_types}
        )

    def reset_parameters(self) -> None:
        """
        Overrides `TimeEncoder.reset_parameters`.
        """
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: dict[NodeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        """
        Overrides `TimeEncoder.forward`.
        """
        out_dict: dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict
