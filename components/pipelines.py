"""
Contains the main boosting pipeline.
"""

import json

import numpy as np
import pandas as pd
import torch
from components.data import FrameData
from relbench.base import TaskType
from utils import ModelBuilder, log


class AdaBoostPipeline:
    """
    The class for the main AdaBoost pipeline.
    """

    def __init__(
        self,
        frame_data: FrameData,
        model_builders: list[ModelBuilder],
        device: torch.device,
        boost_lr: int = 0.5,
        load_first: bool = False,
    ) -> None:
        """
        Args:
            frame_data: The dataset and task data
            model_builders: The list of :class:`ModelBuilder`s to be boosted.
            device: The torch device for training
            boost_lr: The boost learning rate. Affects weight updates.
            load_first: Set to true to load the weights of the first model from disk
        """
        self.frame_data = frame_data
        self.model_builders = model_builders
        self.device = device
        self.boost_lr = boost_lr
        self.load_first = load_first
        print(len(self.model_builders))

    def save(self, path: str) -> None:
        """
        Saves the learner weights to the specified json file.

        Args:
            path: The file path
        """
        alphas = []
        for mb in self.model_builders:
            alphas.append(mb.trainer.alpha)
        with open(path, "w") as f:
            json.dump(alphas, f, indent=2)

    def load(self, path: str) -> None:
        """
        Loads the learner weights from the specified json file.

        Args:
            path: The file path
        """
        with open(path, "r") as f:
            alphas = json.load(f)
        for i, a in enumerate(alphas):
            mb = self.model_builders[i]
            mb.create_model(self.frame_data, self.device, trainer_only=True, boost=True)
            mb.trainer.alpha = a

    def train(self) -> None:
        """
        Trains and boosts all models in order.
        """
        num_samples = len(self.frame_data.task.get_table("train").df)
        num_val_samples = len(self.frame_data.task.get_table("val").df)
        weights = np.ones(num_samples)
        val_weights = np.ones(num_val_samples)
        for iteration, mb in enumerate(self.model_builders):
            mb.create_model(self.frame_data, self.device, boost=True)
            mb.trainer.set_iteration(iteration)
            mb.trainer.init_data_loaders()
            if not (iteration == 0 and self.load_first):
                mb.trainer.train_model(
                    mb.model, self.device, torch.Tensor(weights), val_weights
                )
            else:
                log("Loading first model from disk...", "green")
            self.tune_metric = mb.trainer.tune_metric
            mb.trainer.load_model(mb.model, self.device)
            weights = mb.trainer.calculate_boosting_weights(
                mb.model,
                weights,
                self.device,
                update_alpha=True,
                boost_lr=self.boost_lr,
                loss_type="exp",
            )
            val_weights = mb.trainer.calculate_boosting_weights(
                mb.model,
                val_weights,
                self.device,
                loader="val",
                update_alpha=False,
                boost_lr=self.boost_lr,
                loss_type="exp",
            )
            mb.unload_model()

    def test(self, loader: str, iteration: int = -1) -> torch.Tensor:
        """
        Generate predictions using the boosting pipeline.

        Args:
            loader: Which :class:`DataLoader` to load the input data from
            interation: The number of models to use for prediction. Set to -1 to use all models (default).

        Returns:
            The prediction tensor
        """
        if iteration == -1:
            iteration = len(self.model_builders)
        assert iteration > 0

        accum = None
        alpha_accum = 0.0
        for i in range(iteration):
            mb = self.model_builders[i]
            mb.create_model(self.frame_data, self.device, load=True, boost=True)
            mb.trainer.set_iteration(i)
            mb.trainer.init_data_loaders()
            pred = mb.trainer.alpha * mb.trainer.test(
                mb.model, mb.trainer.loader_dict[loader], self.device, prob=False
            )
            alpha_accum += mb.trainer.alpha
            mb.unload_model()
            if accum is None:
                accum = pred
            else:
                accum += pred

        if self.frame_data.task.task_type == TaskType.BINARY_CLASSIFICATION:
            return torch.from_numpy(accum / alpha_accum).sigmoid().numpy()
        elif self.frame_data.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return torch.from_numpy(accum / alpha_accum).softmax(dim=1).numpy()
        else:
            return torch.from_numpy(accum / alpha_accum).numpy()

    def test_model(self, device: torch.device) -> pd.DataFrame:
        """
        Creates a table with test inputs, lables and test predictions.

        Args:
            device: The torch device to use

        Returns:
            The dataframe containing the test results
        """
        test_table = self.frame_data.task.get_table(
            "test", mask_input_cols=False
        ).df.copy(deep=True)

        test_pred = self.test("test")
        test_metrics = self.frame_data.task.evaluate(test_pred)
        print(f"Test metrics: {test_metrics}")

        if self.frame_data.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            test_pred = test_pred.argmax(axis=1)
        test_table["prediction"] = test_pred
        print(test_table)
        return test_table
