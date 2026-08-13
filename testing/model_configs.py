"""
Contains the base TestModelConfig class and its implementations.
"""

from components.data import FrameData
from components.encoders import HeteroTemporalEncoder, PerFeatureEncoder, ResNetEncoder
from components.full_models import (
    ModelConfig,
    RelbenchModel,
    RelGTConfig,
    RelGTModel,
    TabularModel,
)
from components.gnn_models import DBFormer, HeteroGraphSAGE, RelGNN
from components.trainer import (
    GNNTrainer,
    RelGTTrainer,
    TabularTrainer,
    TrainerConfig,
)
from torch_frame.gbdt import LightGBM
from torch_geometric.nn import MLP


def get_model_configs(frame_data: FrameData):
    """
    Returns the list of available TestModelConfigs for the given FrameData.
    """
    return [
        TabularConfig(frame_data),
        HeteroGraphSAGEConfig(frame_data),
        RelGNNConfig(frame_data),
        DBFormerConfig(frame_data),
        RelGTModelConfig(frame_data),
    ]


class TestModelConfig:
    def __init__(self, data: dict):
        self.configs: list[tuple] = []
        base_dicts = (data["base"]["model"], data["base"]["trainer"])
        if "variants" not in data:
            self.configs.append(
                dict(
                    name=data["name"],
                    model_config=data["config"](**base_dicts[0]),
                    trainer_config=TrainerConfig(**base_dicts[1]),
                    model_class=data["model"],
                    trainer_class=data["trainer"],
                )
            )
        else:
            for variant in data["variants"]:
                self.configs.append(
                    dict(
                        name=data["name"] + "_" + variant["name"],
                        model_config=data["config"](
                            **(base_dicts[0] | variant["model"])
                        ),
                        trainer_config=TrainerConfig(
                            **(base_dicts[1] | variant["trainer"])
                        ),
                        model_class=data["model"],
                        trainer_class=data["trainer"],
                    )
                )


class TabularConfig(TestModelConfig):
    def __init__(self, frame_data: FrameData):
        super().__init__(
            data=dict(
                name="Tabular",
                base=dict(
                    model=dict(adaptable=False, params=dict(gbdt=LightGBM)),
                    trainer=dict(
                        num_trials=20,
                        model_path="cache/models/testing/",
                        model_name="tabular.pt",
                    ),
                ),
                model=TabularModel,
                trainer=TabularTrainer,
                config=ModelConfig,
            )
        )


class HeteroGraphSAGEConfig(TestModelConfig):
    def __init__(self, frame_data: FrameData):
        super().__init__(
            data=dict(
                name="GraphSAGE",
                base=dict(
                    model=dict(
                        aggr="sum",
                        params=dict(
                            encoder=ResNetEncoder,
                            time_encoder=HeteroTemporalEncoder,
                            gnn=HeteroGraphSAGE,
                            head=MLP,
                            gnn_params=dict(dropout=0.3),
                        ),
                    ),
                    trainer=dict(
                        epochs=20,
                        lr=0.002,
                        model_path="cache/models/testing/",
                        batch_size=1024,
                    ),
                ),
                variants=[
                    dict(
                        name="shallow",
                        model=dict(channels=128, layers=2),
                        trainer=dict(
                            num_neighbors=128, num_hops=2, model_name="sage_shallow.pt"
                        ),
                    ),
                    dict(
                        name="medium",
                        model=dict(channels=64, layers=3),
                        trainer=dict(
                            num_neighbors=64, num_hops=3, model_name="sage_medium.pt"
                        ),
                    ),
                    dict(
                        name="deep",
                        model=dict(channels=32, layers=4),
                        trainer=dict(
                            num_neighbors=64, num_hops=4, model_name="sage_deep.pt"
                        ),
                    ),
                ],
                model=RelbenchModel,
                trainer=GNNTrainer,
                config=ModelConfig,
            )
        )


class RelGNNConfig(TestModelConfig):
    def __init__(self, frame_data: FrameData):
        super().__init__(
            data=dict(
                name="RelGNN",
                base=dict(
                    model=dict(
                        aggr="sum",
                        params=dict(
                            encoder=ResNetEncoder,
                            time_encoder=HeteroTemporalEncoder,
                            gnn=RelGNN,
                            head=MLP,
                            gnn_params=dict(
                                num_heads=4, ff_dropout=0.3, attn_dropout=0.3
                            ),
                        ),
                    ),
                    trainer=dict(
                        epochs=20,
                        lr=0.002,
                        model_path="cache/models/testing/",
                        batch_size=1024,
                    ),
                ),
                variants=[
                    dict(
                        name="shallow",
                        model=dict(channels=128, layers=2),
                        trainer=dict(
                            num_neighbors=128,
                            num_hops=2,
                            model_name="relgnn_shallow.pt",
                        ),
                    ),
                    dict(
                        name="medium",
                        model=dict(channels=64, layers=3),
                        trainer=dict(
                            num_neighbors=64, num_hops=3, model_name="relgnn_medium.pt"
                        ),
                    ),
                    dict(
                        name="deep",
                        model=dict(channels=32, layers=4),
                        trainer=dict(
                            num_neighbors=64, num_hops=4, model_name="relgnn_deep.pt"
                        ),
                    ),
                ],
                model=RelbenchModel,
                trainer=GNNTrainer,
                config=ModelConfig,
            )
        )


class DBFormerConfig(TestModelConfig):
    def __init__(self, frame_data: FrameData):
        super().__init__(
            data=dict(
                name="DBFormer",
                base=dict(
                    model=dict(
                        aggr="sum",
                        params=dict(
                            encoder=PerFeatureEncoder,
                            time_encoder=HeteroTemporalEncoder,
                            gnn=DBFormer,
                            head=MLP,
                            gnn_params=dict(
                                num_heads=4,
                                col_stats_dict=frame_data.col_stats_dict,
                                ff_dropout=0.3,
                                attn_dropout=0.3,
                            ),
                        ),
                    ),
                    trainer=dict(
                        epochs=20,
                        lr=0.002,
                        model_path="cache/models/testing/",
                        batch_size=64,
                    ),
                ),
                variants=[
                    dict(
                        name="shallow",
                        model=dict(channels=128, layers=2),
                        trainer=dict(
                            num_neighbors=128,
                            num_hops=2,
                            model_name="dbformer_shallow.pt",
                            batch_size=128,
                        ),
                    ),
                    dict(
                        name="medium",
                        model=dict(channels=64, layers=3),
                        trainer=dict(
                            num_neighbors=64,
                            num_hops=3,
                            model_name="dbformer_medium.pt",
                        ),
                    ),
                    dict(
                        name="deep",
                        model=dict(channels=32, layers=4),
                        trainer=dict(
                            num_neighbors=64,
                            num_hops=4,
                            model_name="dbformer_deep.pt",
                        ),
                    ),
                ],
                model=RelbenchModel,
                trainer=GNNTrainer,
                config=ModelConfig,
            )
        )


class RelGTModelConfig(TestModelConfig):
    def __init__(self, frame_data: FrameData):
        super().__init__(
            data=dict(
                name="RelGT",
                base=dict(
                    model=dict(
                        aggr="sum",
                        attn_dropout=0.3,
                        ff_dropout=0.3,
                        channels=512,
                    ),
                    trainer=dict(
                        epochs=20,
                        lr=0.0005,
                        batch_size=1024,
                        model_path="cache/models/testing/",
                        num_workers=8,
                    ),
                ),
                variants=[
                    dict(
                        name="shallow",
                        model=dict(layers=2),
                        trainer=dict(
                            num_neighbors=400,
                            model_name="relgt_shallow.pt",
                        ),
                    ),
                    dict(
                        name="medium",
                        model=dict(layers=4),
                        trainer=dict(
                            num_neighbors=300,
                            model_name="relgt_medium.pt",
                        ),
                    ),
                    dict(
                        name="deep",
                        model=dict(layers=8),
                        trainer=dict(
                            num_neighbors=200,
                            model_name="relgt_deep.pt",
                        ),
                    ),
                ],
                model=RelGTModel,
                trainer=RelGTTrainer,
                config=RelGTConfig,
            )
        )
