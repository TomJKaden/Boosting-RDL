"""
Contains all trainers for the different trainable full models.
"""

import math
import os
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout

import components.external.relgt_utils as rgu
import numpy as np
import torch
import torch_frame
from components.data import FrameData
from components.external.relgt_utils import RelGTTokens
from components.full_models import Model, ModelConfig
from relbench.base import TaskType
from relbench.modeling.graph import (
    get_node_train_table_input,
)
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from utils import log, weighted_mae, weighted_micro_f1, weighted_roc_auc


class TrainerConfig:
    """
    The trainer configuration class containing all training parameters.
    """

    def __init__(
        self,
        batch_size=512,
        epochs=10,
        num_neighbors=128,
        num_hops=2,
        lr=0.005,
        weight_decay=0.0,
        temporal_strategy="uniform",
        num_workers=0,
        num_trials=10,
        model_path="cache/models",
        model_name="model.pt",
    ) -> None:
        """
        Args:
            batch_size: The batch size used for training
            epochs: The number of training epochs
            num_neighbors: The number of neighbors used for neighbor sampling
            num_hops: The number of hops used for neighbor sampling. Should usually match model layers.
            lr: The learning rate
            weight_decay: The weight decay value used by the optimizer
            temporal_strategy: The temporal strategy used by the neighbor sampling
            num_workers: Number of data loading workers. Prone to issues.
            num_trials: Number of trials used for GBDT models
            model_path: The folder to save the model weights to
            model_name: The name of the model file
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_neighbors = num_neighbors
        self.lr = lr
        self.weight_decay = weight_decay
        self.temporal_strategy = temporal_strategy
        self.num_workers = num_workers
        self.num_trials = num_trials
        self.model_path = model_path
        self.model_name = model_name
        self.num_hops = num_hops


class Trainer(ABC):
    """
    Base class for all model trainers.
    """

    @abstractmethod
    def __init__(
        self,
        frame_data: FrameData,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        boost: bool,
    ) -> None:
        """
        Args:
            frame_data: The dataset and task data
            model_config: The model configuration
            trainer_config: The trainer configuration
            boost: True if this trainer is used in a boosting pipeline
        """
        self.frame_data = frame_data
        self.task = frame_data.task
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.container = os.getenv("RDL_CONTAINER") == "1"

    @abstractmethod
    def init_data_loaders(self) -> None:
        """
        Initializes the data loaders for train, val and test sets.
        """
        ...

    @abstractmethod
    def save_model(self, model: Model) -> None:
        """
        Saves the model weights.

        Args:
            model: The model belonging to this trainer.
        """
        ...

    @abstractmethod
    def load_model(
        self, model: Model, device: torch.device, load_zero: bool = False
    ) -> None:
        """
        Loads the model weights.

        Args:
            model: The model belonging to this trainer.
            device: The torch device
            load_zero
        """
        ...

    @abstractmethod
    def train_model(self, model: Model, device, weights, val_weights):
        """
        Trains the given model.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, model: Model, loader, device, prob=True) -> np.ndarray:
        """
        Returns the model predictions for the given data loader.
        """
        raise NotImplementedError

    def calculate_boosting_weights(
        self,
        model: Model,
        weights: np.ndarray,
        device,
        update_alpha=True,
        loader="train_sorted",
        loss_type="linear",
        boost_lr=0.5,
    ) -> Tensor:
        """
        Updates the boosting weights with the AdaBoost algorithm.
        """
        if not update_alpha:
            if self.alpha == 0.0:
                return weights
        log(f"Calculating boosting weights for {loader}...")
        targets = (
            self.task.get_table(loader.split("_")[0]).df[self.task.target_col].values
        )
        if targets.dtype == bool:
            targets = targets.astype(int)

        if self.task.task_type == TaskType.REGRESSION:
            pred = self.test(model, self.loader_dict[loader], device)
            sample_error = np.abs(pred - targets)
            max_error = sample_error.max()
            sample_error = sample_error / max_error

            if loss_type == "square":
                sample_error = np.pow(sample_error, 2)
            elif loss_type == "exp":
                sample_error = 1.0 - np.exp(-sample_error)

            error = ((weights / weights.sum()) * sample_error).sum()
            log(f"Error of this classifiert: {error}")
            if update_alpha:
                if error >= 0.5:
                    log("Discarding classifier.", "red")
                    self.alpha = 0.0
                    return weights
                self.alpha = boost_lr * np.log(1.0 / (error / (1.0 - error)))
                log(f"Weight of this classifier: {self.alpha}")

            new_weights = np.zeros_like(weights)
            for i in range(len(targets)):
                new_weights[i] = weights[i] * np.pow(
                    error / (1.0 - error), (1.0 - sample_error[i]) * boost_lr
                )
            new_weights = new_weights / (new_weights.sum() / len(new_weights))
            log(
                f"Weights updated to {new_weights[:10]}, min: {new_weights.min()}, max: {new_weights.max()}"
            )

            return new_weights

        class_prob = self.test(model, self.loader_dict[loader], device)
        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            num_classes = 2
            class_prob = np.transpose(np.array([1.0 - class_prob, class_prob]))
        elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            num_classes = self.task.num_classes
        print(class_prob[0])

        pred = class_prob.argmax(axis=1)
        one_hot_targets = np.eye(num_classes)[targets.astype(int)]

        false_predictions = np.where(targets != pred, 1, 0)
        log(
            f"False predictions of this classifier: {np.sum(false_predictions)}/{len(false_predictions)}"
        )
        error = ((false_predictions * weights) / len(targets)).sum()
        log(f"Error of this classifiert: {error}")
        if update_alpha:
            if error >= 1.0 - (1.0 / num_classes):
                log("Discarding classifier.", "red")
                self.alpha = 0.0
                return weights
            self.alpha = boost_lr * (
                np.log((1 - error) / error) + np.log(num_classes - 1)
            )
            log(f"Weight of this classifier: {self.alpha}")

        new_weights = np.zeros_like(weights)
        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            for i in range(len(targets)):
                new_weights[i] = weights[i] * np.exp(
                    -self.alpha
                    * ((num_classes - 1) / float(num_classes))
                    * np.dot(
                        one_hot_targets[i],
                        np.log(np.clip(class_prob[i], 0.0001, 0.9999)),
                    )
                )
        elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            for i in range(len(targets)):
                new_weights[i] = weights[i] * np.exp(self.alpha * false_predictions[i])
        new_weights = new_weights / (new_weights.sum() / len(new_weights))
        log(
            f"Weights updated to {new_weights[:10]}, min: {new_weights.min()}, max: {new_weights.max()}"
        )
        return new_weights

    def test_model(self, model: Model, device, load=True):
        """
        Evaluates the model against the test data split and returns a DataFrame containing the results.
        """
        model = model.to(device)
        if load:
            self.load_model(model, device)
        test_table = self.task.get_table("test", mask_input_cols=False).df.copy(
            deep=True
        )

        test_pred = self.test(model, self.loader_dict["test"], device)

        test_metrics = self.task.evaluate(test_pred)
        print(f"Test metrics: {test_metrics}")
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            test_pred = test_pred.argmax(axis=1)
        test_table["prediction"] = test_pred
        print(test_table)
        return test_table


# Some functionality adapted from RelGT under MIT License
# Original source: https://github.com/snap-stanford/relgt/blob/main/main_node_ddp.py
# License: https://opensource.org/license/mit
class RelGTTrainer(Trainer):
    """
    The trainer for RelGTModels.
    """
    def __init__(
        self,
        frame_data: FrameData,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        boost: bool = False,
    ):
        super().__init__(frame_data, model_config, trainer_config, boost)
        task = self.task
        self.config = trainer_config
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            self.out_channels = 1
            self.tune_metric = "roc_auc"
            self.tune_fn = weighted_roc_auc
            self.higher_is_better = True
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.out_channels = task.num_classes
            self.tune_metric = "micro_f1" # Reduces to Accuracy
            self.tune_fn = weighted_micro_f1
            self.higher_is_better = True
        elif task.task_type == TaskType.REGRESSION:
            self.out_channels = 1
            self.tune_metric = "mae"
            self.tune_fn = weighted_mae
            self.higher_is_better = False

        self.save_name = None
        self.boost = boost
        if not boost:
            self.save_name = self.trainer_config.model_name

    def set_iteration(self, iteration: int = 0):
        self.save_name = str(iteration) + "_" + self.trainer_config.model_name

    def init_data_loaders(self, second_attempt=False):
        rgu.GLOBAL_DATA = None
        rgu.GLOBAL_ADJ = None
        rgu.GLOBAL_ALL_NODES = None
        try:
            if not os.path.exists(
                os.path.join(
                    self.frame_data.cache_path,
                    "precomputed",
                    str(self.trainer_config.num_neighbors),
                )
            ):
                log("Precomputing RelGT Tokens...")
                data = {
                    split: RelGTTokens(
                        data=self.frame_data.data,
                        task=self.task,
                        K=self.trainer_config.num_neighbors,
                        split=split,
                        undirected=True,
                        precompute=True,
                        precomputed_dir=os.path.join(
                            self.frame_data.cache_path, "precomputed"
                        ),
                        num_workers=self.trainer_config.num_workers,
                        train_stage="finetune",
                    )
                    for split in ["train", "val", "test"]
                }
            log("Initializing RelGT Tokens...")
            data = {
                split: RelGTTokens(
                    data=self.frame_data.data,
                    task=self.task,
                    K=self.trainer_config.num_neighbors,
                    split=split,
                    undirected=True,
                    precompute=False,
                    precomputed_dir=os.path.join(
                        self.frame_data.cache_path, "precomputed"
                    ),
                    num_workers=self.trainer_config.num_workers,
                    train_stage="finetune",
                )
                for split in ["train", "val", "test"]
            }
            self.loader_dict = {}
            for split in ["train", "train_sorted", "val", "test"]:
                self.loader_dict[split] = DataLoader(
                    data[split.split("_")[0]],
                    batch_size=self.trainer_config.batch_size,
                    shuffle=split == "train",
                    collate_fn=data[split.split("_")[0]].collate,
                    num_workers=self.trainer_config.num_workers,
                    persistent_workers=self.trainer_config.num_workers > 0,
                    pin_memory=True,
                )
            for d in self.loader_dict["test"]:
                pass
        except Exception:
            log("Error during RelGT Token creation!", "red")
            if not second_attempt:
                dir = os.path.join(
                    self.frame_data.cache_path,
                    "precomputed",
                    str(self.trainer_config.num_neighbors),
                )
                if os.path.exists(dir):
                    for file in os.listdir(dir):
                        os.remove(os.path.join(dir, file))
                    os.rmdir(dir)
                self.init_data_loaders(second_attempt=True)
            else:
                raise Exception

    def save_model(self, model: Model):
        if self.trainer_config.model_path is not None and self.save_name is not None:
            log(
                f"Saving model to {os.path.join(self.trainer_config.model_path, self.save_name)}"
            )
            os.makedirs(self.trainer_config.model_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    self.trainer_config.model_path,
                    self.save_name,
                ),
            )

    def load_model(self, model: Model, device, load_zero=False):
        if self.trainer_config.model_path is not None and self.save_name is not None:
            name = self.save_name
            if load_zero:
                name = "_".join(["0"] + name.split("_")[1:])
            model.load_state_dict(
                torch.load(
                    os.path.join(self.trainer_config.model_path, name),
                    map_location=device,
                )
            )

    def train_model(
        self, model: Model, device, weights: Tensor = None, val_weights=None
    ):
        model = model.to(device)
        if self.trainer_config.weight_decay > 0.0:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.trainer_config.lr,
                weight_decay=self.trainer_config.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.trainer_config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer_config.epochs,
            eta_min=self.trainer_config.lr / 5,
        )
        best_val_metric = -math.inf if self.higher_is_better else math.inf
        for epoch in range(1, self.trainer_config.epochs + 1):
            train_loss = self.train(optimizer, model, device, weights)
            val_pred = self.test(model, self.loader_dict["val"], device)

            if not self.boost:
                val_metrics = self.task.evaluate(val_pred, self.task.get_table("val"))
            else:
                val_metrics = self.tune_fn(
                    self.task.get_table("val").df[self.task.target_col].to_numpy(),
                    val_pred,
                    val_weights,
                )
            print(
                f"Epoch: {epoch:02d}, LR {scheduler.get_last_lr()[-1]}, Train loss: {train_loss}, Val metrics: {val_metrics}"
            )
            if (
                self.higher_is_better
                and val_metrics[self.tune_metric] > best_val_metric
            ) or (
                not self.higher_is_better
                and val_metrics[self.tune_metric] < best_val_metric
            ):
                best_val_metric = val_metrics[self.tune_metric]
                self.save_model(model)
            scheduler.step()

    def train(self, optimizer: Optimizer, model: Model, device, weights) -> float:
        model.train()
        loss_accum = count_accum = 0

        if not self.container:
            pbar = tqdm(self.loader_dict["train"])
        else:
            pbar = tqdm(self.loader_dict["train"], file=open(os.devnull, "w"))
        if weights is not None:
            weights = weights.to(device)
        for step, batch in enumerate(pbar):
            neighbor_types = batch["neighbor_types"].to(device)
            node_indices = batch["node_indices"].to(device)
            neighbor_hops = batch["neighbor_hops"].to(device)
            neighbor_times = batch["neighbor_times"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_vec = batch["batch"].to(device)

            grouped_tf_dict = {
                "grouped_tfs": batch["grouped_tfs"],
                "grouped_indices": batch["grouped_indices"],
                "flat_batch_idx": batch["flat_batch_idx"],
                "flat_nbr_idx": batch["flat_nbr_idx"],
            }
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            pred = model(
                neighbor_types,
                node_indices,
                neighbor_hops,
                neighbor_times,
                grouped_tf_dict,
                edge_index=edge_index,
                batch=batch_vec,
            )
            if weights is not None:
                pred_weights = weights.index_select(0, batch["global_idx"].to(device))
            else:
                pred_weights = torch.ones_like(labels)

            if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = pred.view(-1).float()
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        pred, labels, reduction="none"
                    )
                    * pred_weights
                )
            elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                loss = (
                    torch.nn.functional.cross_entropy(
                        pred.float(),
                        labels,
                        reduction="none",
                    )
                    * pred_weights
                )
            elif self.task.task_type == TaskType.REGRESSION:
                pred = pred.view(-1).float()
                loss = (
                    torch.nn.functional.l1_loss(pred, labels, reduction="none")
                    * pred_weights
                )
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = loss.detach().item()
            loss_accum += loss_value * pred.size(0)
            count_accum += pred.size(0)
            pbar.set_description(f"Loss: {loss_accum / count_accum:.4f}")

        if self.container:
            print(pbar)

        return loss_accum / count_accum if count_accum > 0 else float("inf")

    @torch.no_grad()
    def test(self, model: Model, loader, device, prob=True) -> np.ndarray:
        model.eval()

        pred_list = []

        if not self.container:
            pbar = tqdm(loader)
        else:
            pbar = tqdm(loader, file=open(os.devnull, "w"))
        for batch in pbar:
            neighbor_types = batch["neighbor_types"].to(device)
            node_indices = batch["node_indices"].to(device)
            neighbor_hops = batch["neighbor_hops"].to(device)
            neighbor_times = batch["neighbor_times"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_vec = batch["batch"].to(device)

            grouped_tf_dict = {
                "grouped_tfs": batch["grouped_tfs"],
                "grouped_indices": batch["grouped_indices"],
                "flat_batch_idx": batch["flat_batch_idx"],
                "flat_nbr_idx": batch["flat_nbr_idx"],
            }
            pred = model(
                neighbor_types,
                node_indices,
                neighbor_hops,
                neighbor_times,
                grouped_tf_dict,
                edge_index=edge_index,
                batch=batch_vec,
            )
            if prob:
                if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                    pred = torch.sigmoid(pred)

                elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                    pred = torch.softmax(pred, dim=1)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())

        if self.container:
            print(pbar)

        return torch.cat(pred_list, dim=0).numpy()


class TabularTrainer(Trainer):
    """
    The trainer for TabularModels.
    """
    def __init__(
        self,
        frame_data: FrameData,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        boost: bool = False,
    ):
        super().__init__(frame_data, model_config, trainer_config, boost)
        task = self.task

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            self.tune_fn = torch_frame.Metric.ROCAUC
            self.tune_metric = "roc_auc"
        elif task.task_type == TaskType.REGRESSION:
            self.tune_fn = torch_frame.Metric.MAE
            self.tune_metric = "mae"
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.tune_fn = torch_frame.Metric.ACCURACY
            self.tune_metric = "micro_f1" # Reduces to Accuracy
        self.boost = boost

        self.save_name = None
        if not boost:
            self.save_name = self.trainer_config.model_name

    def init_data_loaders(self):
        df_train = self.frame_data.tabular_dfs["train"]
        df_val = self.frame_data.tabular_dfs["val"]
        df_test = self.frame_data.tabular_dfs["test"]
        print(df_train)
        if self.container:
            with open(os.devnull, "w") as f:
                with redirect_stdout(f), redirect_stderr(f):
                    self.dataset = torch_frame.data.Dataset(
                        df=df_train,
                        col_to_stype=self.frame_data.tabular_col_to_stype,
                        target_col=self.task.target_col,
                        col_to_text_embedder_cfg=self.frame_data.text_embedder_cfg,
                    ).materialize(
                        path=os.path.join(
                            self.frame_data.cache_path, "tabular_dataset.pt"
                        )
                    )
        else:
            self.dataset = torch_frame.data.Dataset(
                df=df_train,
                col_to_stype=self.frame_data.tabular_col_to_stype,
                target_col=self.task.target_col,
                col_to_text_embedder_cfg=self.frame_data.text_embedder_cfg,
            ).materialize(
                path=os.path.join(self.frame_data.cache_path, "tabular_dataset.pt")
            )

        if len(df_train) > 100000:
            sampled_idx = np.random.permutation(len(df_train))[:100000]
            df_train_small = df_train.iloc[sampled_idx].copy()
            self.loader_dict = {
                "train": self.dataset.convert_to_tensor_frame(df_train_small),
                "train_sorted": self.dataset.tensor_frame,
                "val": self.dataset.convert_to_tensor_frame(df_val),
                "test": self.dataset.convert_to_tensor_frame(df_test),
            }
        else:
            self.loader_dict = {
                "train": self.dataset.tensor_frame,
                "train_sorted": self.dataset.tensor_frame,
                "val": self.dataset.convert_to_tensor_frame(df_val),
                "test": self.dataset.convert_to_tensor_frame(df_test),
            }

    def init_model(self, model: Model):
        if not model.initialized:
            num_classes = (
                self.task.num_classes
                if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION
                else None
            )
            model.init_model(
                self.dataset.task_type,
                num_classes,
                self.tune_fn,
            )

    def set_iteration(self, iteration: int = 0):
        self.save_name = str(iteration) + "_" + self.trainer_config.model_name

    def save_model(self, model: Model):
        model.save(os.path.join(self.trainer_config.model_path, self.save_name))

    def load_model(self, model: Model, device, load_zero=False):
        self.init_model(model)
        name = self.save_name
        if load_zero:
            name = "_".join(["0"] + name.split("_")[1:])
        model.load(os.path.join(self.trainer_config.model_path, name))

    def train_model(self, model: Model, device, weights=None, val_weights=None):
        self.init_model(model)
        model.train(
            self.loader_dict["train"],
            self.loader_dict["val"],
            num_trials=self.trainer_config.num_trials,
        )
        pred = self.test(model, self.loader_dict["train_sorted"], device)
        train_metrics = self.task.evaluate(pred, self.task.get_table("train"))

        pred = self.test(model, self.loader_dict["val"], device)
        val_metrics = self.task.evaluate(pred, self.task.get_table("val"))

        pred = self.test(model, self.loader_dict["test"], device)
        test_metrics = self.task.evaluate(
            pred, self.task.get_table("test", mask_input_cols=False)
        )

        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_metrics}")
        print(f"Test metrics: {test_metrics}")
        self.save_model(model)

    def test(self, model: Model, loader, device, prob=True) -> np.ndarray:
        pred = model.test(loader).numpy()
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = np.eye(self.task.num_classes)[pred]
        elif self.task.task_type == TaskType.BINARY_CLASSIFICATION and not prob:
            pred = pred * 2.0 - 1.0
        return pred


class GNNTrainer(Trainer):
    """
    The trainer for RelbenchModels.
    """
    def __init__(
        self,
        frame_data: FrameData,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        boost: bool = False,
    ):
        super().__init__(frame_data, model_config, trainer_config, boost)
        task = self.task

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            self.out_channels = 1
            self.tune_metric = "roc_auc"
            self.tune_fn = weighted_roc_auc
            self.higher_is_better = True
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.out_channels = task.num_classes
            self.tune_metric = "micro_f1" # Reduces to Accuracy
            self.tune_fn = weighted_micro_f1
            self.higher_is_better = True
        elif task.task_type == TaskType.REGRESSION:
            self.out_channels = 1
            self.tune_metric = "mae"
            self.tune_fn = weighted_mae
            self.higher_is_better = False

        self.save_name = None
        self.boost = boost
        if not boost:
            self.save_name = self.trainer_config.model_name

    def set_iteration(self, iteration: int = 0):
        self.save_name = str(iteration) + "_" + self.trainer_config.model_name

    def init_data_loaders(self):
        trainer_config = self.trainer_config
        task = self.task
        frame_data = self.frame_data
        data = self.frame_data.data

        self.loader_dict = {}
        self.entity_table = task.entity_table
        for split in ["train", "train_sorted", "val", "test"]:
            table_input = get_node_train_table_input(
                table=frame_data.task.get_table(split.split("_")[0]),
                task=frame_data.task,
            )
            self.loader_dict[split] = NeighborLoader(
                data,
                num_neighbors=[
                    int(trainer_config.num_neighbors / 2**i)
                    for i in range(trainer_config.num_hops)
                ],
                time_attr="time",
                input_nodes=table_input.nodes,
                input_time=table_input.time,
                transform=table_input.transform,
                batch_size=trainer_config.batch_size,
                temporal_strategy=trainer_config.temporal_strategy,
                shuffle=split == "train",
                num_workers=trainer_config.num_workers,
                persistent_workers=trainer_config.num_workers > 0,
                pin_memory=True,
            )

    def save_model(self, model: Model):
        if self.trainer_config.model_path is not None and self.save_name is not None:
            log(
                f"Saving model to {os.path.join(self.trainer_config.model_path, self.save_name)}"
            )
            os.makedirs(self.trainer_config.model_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    self.trainer_config.model_path,
                    self.save_name,
                ),
            )

    def load_model(self, model: Model, device, load_zero=False):
        model = model.to(device)
        if not self.container:
            log("Compiling model...")
            model = torch.compile(model, dynamic=True)
        if self.trainer_config.model_path is not None and self.save_name is not None:
            name = self.save_name
            if load_zero:
                name = "_".join(["0"] + name.split("_")[1:])
            model.load_state_dict(
                torch.load(
                    os.path.join(self.trainer_config.model_path, name),
                    map_location=device,
                )
            )

    def train_model(
        self, model: Model, device, weights: Tensor = None, val_weights=None
    ):
        model = model.to(device)
        if not self.container:
            log("Compiling model...")
            model = torch.compile(model, dynamic=True)

        if self.trainer_config.weight_decay > 0.0:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.trainer_config.lr,
                weight_decay=self.trainer_config.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.trainer_config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer_config.epochs,
            eta_min=self.trainer_config.lr / 5,
        )
        best_val_metric = -math.inf if self.higher_is_better else math.inf
        for epoch in range(1, self.trainer_config.epochs + 1):
            train_loss = self.train(optimizer, model, device, weights)
            val_pred = self.test(model, self.loader_dict["val"], device)

            if not self.boost:
                val_metrics = self.task.evaluate(val_pred, self.task.get_table("val"))
            else:
                val_metrics = self.tune_fn(
                    self.task.get_table("val").df[self.task.target_col].to_numpy(),
                    val_pred,
                    val_weights,
                )
            print(
                f"Epoch: {epoch:02d}, LR {scheduler.get_last_lr()[-1]}, Train loss: {train_loss}, Val metrics: {val_metrics}"
            )
            if (
                self.higher_is_better
                and val_metrics[self.tune_metric] > best_val_metric
            ) or (
                not self.higher_is_better
                and val_metrics[self.tune_metric] < best_val_metric
            ):
                best_val_metric = val_metrics[self.tune_metric]
                self.save_model(model)
            scheduler.step()

    def train(
        self, optimizer: Optimizer, model: Model, device, weights: Tensor = None
    ) -> float:
        model.train()

        loss_accum = count_accum = 0
        if not self.container:
            pbar = tqdm(self.loader_dict["train"])
        else:
            pbar = tqdm(self.loader_dict["train"], file=open(os.devnull, "w"))
        if weights is not None:
            weights = weights.to(device)
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch, self.task.entity_table)
            if weights is not None:
                pred_weights = weights.index_select(
                    0, batch[self.task.entity_table].input_id
                )
            else:
                pred_weights = torch.ones_like(batch[self.task.entity_table].y)

            if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = pred.view(-1).float()
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        pred, batch[self.task.entity_table].y.float(), reduction="none"
                    )
                    * pred_weights
                )
            elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                loss = (
                    torch.nn.functional.cross_entropy(
                        pred.float(),
                        batch[self.task.entity_table].y.long(),
                        reduction="none",
                    )
                    * pred_weights
                )
            elif self.task.task_type == TaskType.REGRESSION:
                pred = pred.view(-1).float()
                loss = (
                    torch.nn.functional.l1_loss(
                        pred, batch[self.task.entity_table].y.float(), reduction="none"
                    )
                    * pred_weights
                )
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)
            pbar.set_description(f"Loss: {loss_accum / count_accum:.4f}")

        if self.container:
            print(pbar)

        return loss_accum / count_accum

    @torch.no_grad()
    def test(
        self, model: Model, loader: NeighborLoader, device, prob=True
    ) -> np.ndarray:
        model.eval()

        pred_list = []
        if not self.container:
            pbar = tqdm(loader)
        else:
            pbar = tqdm(loader, file=open(os.devnull, "w"))
        for batch in pbar:
            batch = batch.to(device)
            pred = model(batch, self.task.entity_table)
            if prob:
                if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                    pred = torch.sigmoid(pred)

                elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                    pred = torch.softmax(pred, dim=1)

            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())

        if self.container:
            print(pbar)

        return torch.cat(pred_list, dim=0).numpy()
