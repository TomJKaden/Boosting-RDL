"""
Contains all full models and their configuration.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch_frame
from components.data import FrameData
from components.encoders import DataEncoder, TimeEncoder
from components.external.relgt.model import RelGT
from components.gnn_models import GNNModel
from torch import Tensor
from torch_frame.gbdt import GBDT
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


class ModelConfig:
    """
    Class holding all customizable parameters for most models (except RelGT).
    """

    def __init__(
        self,
        channels: int = 128,
        layers: int = 2,
        aggr: str = "sum",
        params: dict() = None,
        adaptable: bool = True,
    ) -> None:
        """
        Args:
            channels: The number of channels
            layers: The number of model layers
            aggr: The type of aggregation
            params: A dict of additional model parameters
            adaptable: True if the adapt method may be called for this model
        """
        self.channels = channels
        self.layers = layers
        self.aggr = aggr

        self.out_channels = 1
        self.node_to_col_names_dict = None
        self.node_to_col_stats = None
        self.time_node_types = None
        self.gnn_node_types = None
        self.gnn_edge_types = None
        self.num_nodes_dict = None
        self.norm = "batch_norm"
        self.entity_table = None
        self.params = params
        self.adaptable = adaptable

    def adapt(self, data: HeteroData, frame_data: FrameData, trainer) -> None:
        """
        Adapts many model settings from the given trainer and data classes.

        Args:
            data: The heterogeneous graph data
            frame_data: The full :class:`FrameData` instance
            trainer: The initialized model trainer
        """
        self.out_channels = trainer.out_channels
        self.node_to_col_names_dict = {
            node_type: data[node_type].tf.col_names_dict
            for node_type in data.node_types
        }
        self.node_to_col_stats = frame_data.col_stats_dict
        self.time_node_types = [
            node_type for node_type in data.node_types if "time" in data[node_type]
        ]
        self.gnn_node_types = data.node_types
        self.gnn_edge_types = data.edge_types
        self.num_nodes_dict = data.num_nodes_dict
        self.entity_table = frame_data.task.entity_table


class RelGTConfig(ModelConfig):
    """
    Config specifically for the RelGT model as it requires many additional settings.
    """

    def __init__(
        self,
        channels: int,
        layers: int,
        aggr: str,
        heads: int = 4,
        ff_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_type: str = "full",
        ablate: str = "none",
        gnn_pe_dim: int = 0,
        centroids: int = 4096,
    ) -> None:
        self.channels = channels
        self.layers = layers
        self.aggr = aggr
        self.heads = heads
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout
        self.conv_type = conv_type
        self.ablate = ablate
        self.gnn_pe_dim = gnn_pe_dim
        self.centroids = centroids
        self.params = dict()
        self.adaptable = True

    def adapt(self, data: HeteroData, frame_data: FrameData, trainer) -> None:
        self.out_channels = trainer.out_channels
        self.node_to_col_names_dict = {
            node_type: data[node_type].tf.col_names_dict
            for node_type in data.node_types
        }
        self.node_type_to_index = {
            nt: idx for idx, nt in enumerate(frame_data.data.node_types)
        }
        self.node_to_col_stats = frame_data.col_stats_dict
        self.gnn_node_types = data.node_types
        self.gnn_edge_types = data.edge_types
        self.num_nodes = data.num_nodes
        self.max_neighbor_hop = 2 + 1
        self.num_neighbors = trainer.config.num_neighbors


class Model(ABC):
    """
    Base class for all full models.
    """

    def __init__(self) -> None:
        super().__init__()


class TabularModel(Model):
    """
    The tabular model using Gradient Boosted Decision Trees.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        gbdt: GBDT,
    ) -> None:
        """
        Args:
            model_config: The configuration of the model.
            gbdt: The :class:`torch_frame.gbdt.GBDT` class used.
        """
        self.gbdt = gbdt
        self.initialized = False

    def init_model(
        self,
        task_type: torch_frame.TaskType,
        num_classes: int | None = None,
        metric: torch_frame.Metric | None = None,
    ) -> None:
        """
        Initializes the underlying GBDT model.

        Args:
            task_type: The type of ML task
            num_classes: The number of classes if applicable
            metric: The tuning metric used during training
        """
        if task_type == torch_frame.TaskType.MULTILABEL_CLASSIFICATION:
            task_type = torch_frame.TaskType.BINARY_CLASSIFICATION
        self.model = self.gbdt(
            task_type=task_type, num_classes=num_classes, metric=metric
        )
        self.initialized = True

    def train(
        self,
        tf_train: torch_frame.TensorFrame,
        tf_val: torch_frame.TensorFrame,
        num_trials: int = 10,
    ) -> None:
        """
        Tunes the underlying GBDT model.

        Args:
            tf_train: The :class:`TensorFrame` containing the training data.
            tf_val: The :class:`TensorFrame` containing the validation data.
            num_trials: The number of Optuna trials.
        """
        self.model.tune(tf_train, tf_val, num_trials=num_trials)

    def test(self, tf_test: torch_frame.TensorFrame) -> Tensor:
        """
        Uses the trained GBDT model to generate predictions.

        Args:
            tf_test: The :class:`TensorFrame` containing the test data.

        Returns:
            The model predictions.
        """
        return self.model.predict(tf_test)

    def save(self, path: str) -> None:
        """
        Saves the model weights to the given path.

        Args:
            path: The file path
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Load the model weights from the given path.

        Args:
            path: The file path
        """
        self.model.load(path)


class TorchModel(torch.nn.Module, Model, ABC):
    """
    Base class for models that are a PyTorch module.
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
    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        """
        The forward pass. Given a batch of graph data it generates predictions.

        Args:
            batch: The batch of graph data
            entity_table: The type of the nodes belonging to the entity table

        Returns:
            The tensor containing the predictions for the batch.
        """
        ...


class RelGTModel(torch.nn.Module, Model):
    """
    The class for the RelGT model. Mainly a wrapper class.
    Does not inherit :class:`TorchModel` due to the different config and inputs required.
    """

    def __init__(self, config: RelGTConfig) -> None:
        """
        Args:
            config: The model configuration
        """
        super().__init__()
        self.config = config
        self.model = RelGT(
            num_nodes=config.num_nodes,
            max_neighbor_hop=config.max_neighbor_hop,
            node_type_map=config.node_type_to_index,
            col_names_dict=config.node_to_col_names_dict,
            col_stats_dict=config.node_to_col_stats,
            local_num_layers=config.layers,
            channels=config.channels,
            out_channels=config.out_channels,
            global_dim=int(config.channels / 2),
            heads=config.heads,
            ff_dropout=config.ff_dropout,
            attn_dropout=config.attn_dropout,
            conv_type=config.conv_type,
            ablate=config.ablate,
            gnn_pe_dim=config.gnn_pe_dim,
            num_centroids=config.centroids,
            sample_node_len=config.num_neighbors,
        )

    def reset_parameters(self) -> None:
        """
        Resets all trainable parameters.
        """
        self.model.reset_parameters()

    def forward(
        self,
        neighbor_types,
        node_indices,
        neighbor_hops,
        neighbor_times,
        grouped_tf_dict,
        edge_index=None,
        batch=None,
    ):
        return self.model(
            neighbor_types,
            node_indices,
            neighbor_hops,
            neighbor_times,
            grouped_tf_dict,
            edge_index,
            batch,
        )


# Partially adapted from RelBench under MIT License
# Original source: https://github.com/snap-stanford/relbench/blob/main/examples/model.py
# License: https://opensource.org/license/mit
class RelbenchModel(TorchModel):
    """
    The class for models following the main RDL structure.

    Partly adapted from RelBench under MIT License, but modified to inherit from :class:`TorchModel`,
    and have modular components and configurations. Also removed link prediction functionality and
    added support for an extra encoder dimension.
    """

    def __init__(
        self,
        config: ModelConfig,
        encoder: type[DataEncoder],
        time_encoder: type[TimeEncoder],
        gnn: type[GNNModel],
        head: type[Any],
        encoder_params: dict[str, Any] | None = None,
        time_encoder_params: dict[str, Any] | None = None,
        gnn_params: dict[str, Any] | None = None,
        head_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            config: The model configuration
            encoder: The class used for the data encoder
            time_encoder: The class used for the temporal encoder
            gnn: The class used for the GNN
            head: The class used for the task head
            encoder_params: Extra parameters for the data encoder
            time_encoder_params: Extra parameters for the temporal encoder
            gnn_params: Extra parameters for the GNN
            head_params: Extra parameters for the head
        """
        super().__init__()

        self.encoder = encoder(
            channels=config.channels,
            node_to_col_names_dict=config.node_to_col_names_dict,
            node_to_col_stats=config.node_to_col_stats,
            **(encoder_params or {}),
        )
        self.time_encoder = time_encoder(
            channels=config.channels,
            node_types=config.time_node_types,
            **(time_encoder_params or {}),
        )
        self.gnn = gnn(
            num_layers=config.layers,
            channels=config.channels,
            node_types=config.gnn_node_types,
            edge_types=config.gnn_edge_types,
            aggr=config.aggr,
            **(gnn_params or {}),
        )
        head_channels = config.channels
        if self.encoder.extra_dimension:
            head_channels = config.channels * len(
                config.node_to_col_stats[config.entity_table].keys()
            )
        self.head = head(
            head_channels,
            out_channels=config.out_channels,
            norm=config.norm,
            num_layers=1,
            # hidden_channels=head_channels,
            **(head_params or {}),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Overrides `TorchModel.reset_parameters`.
        """
        self.encoder.reset_parameters()
        self.time_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        """
        Overrides `TorchModel.forward`.
        Uses all RDL pipeline components to produce the final prediction.
        """
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.time_encoder(seed_time, batch.time_dict, batch.batch_dict)

        if self.encoder.extra_dimension:
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time[:, None, :]
        else:
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        entity_x = x_dict[entity_table][: seed_time.size(0)]
        if self.encoder.extra_dimension:
            entity_x = entity_x.view(entity_x.size(0), -1)
        return self.head(entity_x)
