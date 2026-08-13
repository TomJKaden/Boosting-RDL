"""
Contains the log function, weighted metrics and the ModelBuilder class.
"""

import gc

import sklearn.metrics as skm
import torch
from termcolor import colored


def log(message, color="blue"):
    print(colored(message, color))


def weighted_roc_auc(true, pred, weights):
    return {"roc_auc": skm.roc_auc_score(true, pred, sample_weight=weights)}


def weighted_mae(true, pred, weights):
    return {"mae": skm.mean_absolute_error(true, pred, sample_weight=weights)}


def weighted_micro_f1(true, pred, weights): # Reduces to Accuracy metric
    label = pred.argmax(axis=1)
    return {
        "micro_f1": skm.f1_score(true, label, average="micro", sample_weight=weights)
    }


class ModelBuilder:
    """
    This class handles Model and Trainer loading/unloading to manage memory.
    """
    def __init__(
        self,
        model_class,
        model_config,
        trainer_class,
        trainer_config,
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.trainer_class = trainer_class
        self.trainer_config = trainer_config
        self.trainer = None

    def create_model(
        self,
        frame_data,
        device,
        load=False,
        trainer_only=False,
        prev_mb=None,
        boost=False,
    ):
        if self.trainer is None:
            self.trainer = self.trainer_class(
                frame_data, self.model_config, self.trainer_config, boost
            )
        if trainer_only:
            return

        if self.model_config.adaptable:
            self.model_config.adapt(frame_data.data, frame_data, self.trainer)
        self.model = self.model_class(self.model_config, **self.model_config.params)
        to = getattr(self.model, "to", None)
        if callable(to):
            self.model = self.model.to(device)
        if load:
            self.trainer.load_model(self.model, device)

    def unload_model(self):
        del self.trainer.loader_dict
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
