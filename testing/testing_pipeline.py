"""
Contains the code for running the testing pipeline.
"""

import gc
import itertools
import os
import shutil
import traceback

import numpy as np
import pandas as pd
import torch
from components.data import FrameData
from components.full_models import TabularModel
from components.graph_models import RelationalEntityGraphModel
from components.pipelines import AdaBoostPipeline
from relbench.base import TaskType
from testing.model_configs import TestModelConfig, get_model_configs
from utils import ModelBuilder, log


def test_task(dataset, task, device):
    """
    Performs tests for the given dataset and task.
    """
    try:
        frame_data = FrameData(
            dataset,
            task,
            torch.device("cpu"),
            RelationalEntityGraphModel,
            include_tabular=True,
        )
    except Exception:
        log(traceback.format_exc(), "red")
        torch.cuda.empty_cache()
        gc.collect()
        return

    model_path = os.path.join("cache", "models", "testing", dataset, task)
    # if os.path.exists(model_path):
    #    for file in os.listdir(model_path):
    #        os.remove(os.path.join(model_path, file))

    n_train = len(frame_data.task.get_table("train").df)
    max_batch_size = 1024
    while n_train / max_batch_size < 200 and max_batch_size > 128:
        max_batch_size = int(max_batch_size / 2)
    log(f"Choosing maximum batch size of {max_batch_size} for n_train {n_train}.")

    type_str = "binary_classification"
    if frame_data.task.task_type == TaskType.REGRESSION:
        type_str = "regression"
    elif frame_data.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        type_str = "multiclass_classification"

    result_path = os.path.join("results", type_str, dataset + "-" + task + ".csv")
    prev_results = None
    if os.path.exists(result_path):
        prev_results = pd.read_csv(result_path, index_col=0)
        os.makedirs(os.path.join("backup", type_str), exist_ok=True)
        backup_path = os.path.join("backup", type_str, dataset + "-" + task + ".csv")
        shutil.copy(result_path, backup_path)
        log("Continuing with previous results...")
        print(prev_results)

    training_configs = get_all_training_configs(frame_data)
    results = []
    for i, (name, configs) in enumerate(training_configs.items()):
        if prev_results is not None and name in list(prev_results["name"]):
            log(f"Skipping configuration {name}, already tested.")
            results.append(list(prev_results.loc[prev_results["name"] == name].iloc[0]))
            continue
        log(f"Testing configuration {name}")
        mbs = []
        for config in configs:
            mbs.append(
                ModelBuilder(
                    config["model_class"],
                    config["model_config"],
                    config["trainer_class"],
                    config["trainer_config"],
                )
            )

        for mb in mbs:
            mb.trainer_config.model_path = model_path

        if len(mbs) == 1 and mbs[0].model_class != TabularModel:
            batch_size = min(max_batch_size, mbs[0].trainer_config.batch_size)
            log(f"Trying with batch size {batch_size}...")
            while batch_size > 32:
                mbs[0].trainer_config.batch_size = batch_size
                n_steps = min(1200.0, max(200.0, n_train / batch_size))
                epochs = 20 - int(15.0 * ((n_steps - 200) / 1000))
                log(f"Setting epoch count to {epochs}...")
                mbs[0].trainer_config.epochs = epochs
                run = TrainingRun(frame_data, mbs, 1, device)
                metrics = run.run()
                if metrics is not None:
                    break
                batch_size = int(batch_size / 2)
                log(
                    f"Run failed. Reducing batch size to {batch_size} and retrying.",
                    "red",
                )
                del run
                torch.cuda.empty_cache()
                gc.collect()

        else:
            run = TrainingRun(frame_data, mbs, 1, device)
            metrics = run.run()
            del run

        if len(mbs) == 1 and metrics is not None:
            shutil.copy(
                os.path.join(
                    mbs[0].trainer_config.model_path, mbs[0].trainer_config.model_name
                ),
                os.path.join(
                    mbs[0].trainer_config.model_path,
                    "0_" + mbs[0].trainer_config.model_name,
                ),
            )
        del mbs
        torch.cuda.empty_cache()
        gc.collect()
        if metrics is None:
            continue
        metric_names = list(metrics.keys())
        results.append([name] + list(metrics.values()))

        columns = ["name"] + metric_names
        df = pd.DataFrame(results, columns=columns)

        path = os.path.join("results", type_str)
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, dataset + "-" + task + ".csv"))

    del frame_data
    del training_configs
    torch.cuda.empty_cache()
    gc.collect()


def get_all_training_configs(frame_data: FrameData):
    """
    Compiles and returns all model configurations for the given FrameData.
    """
    model_configs: list[TestModelConfig] = get_model_configs(frame_data)
    training_configs: dict[str, list] = {}

    for model_config in model_configs:
        for config in model_config.configs:
            training_configs[config["name"]] = [config]

    for model_config in model_configs:
        if len(model_config.configs) > 1:
            training_configs[
                model_config.configs[0]["name"].split("_")[0] + "-boost"
            ] = [config for config in model_config.configs] * 2
            training_configs[
                model_config.configs[0]["name"].split("_")[0] + "-rev_boost"
            ] = [config for config in reversed(model_config.configs)] * 2

    training_configs = get_combined_configs(
        training_configs,
        "All",
        model_configs[1:],
        model_configs[0],
        2,
        include_tabular=frame_data.feat,
    )

    for comb in itertools.combinations(range(len(model_configs) - 1), 2):
        names = []
        for i in comb:
            names.append(model_configs[i + 1].configs[0]["name"].split("_")[0])
        name = "+".join(names)
        training_configs = get_combined_configs(
            training_configs,
            name,
            [model_configs[i + 1] for i in comb],
            model_configs[0],
            3,
            use_categories=False,
            include_tabular=frame_data.feat,
        )

    return training_configs


def get_combined_configs(
    training_configs,
    name,
    regular_configs,
    tabular_config,
    n,
    use_categories=True,
    include_tabular=False,
):
    all_boost_list = []
    rev_boost_list = []
    for i in range(3):
        category_boost_list = []
        category_name = ""
        for model_config in regular_configs:
            all_boost_list.append(model_config.configs[i])
            rev_boost_list.append(model_config.configs[2 - i])
            category_boost_list.append(model_config.configs[i])
            category_name = model_config.configs[i]["name"].split("_")[1]
        if use_categories:
            training_configs[f"{name}-{category_name}"] = category_boost_list * n
            if include_tabular:
                training_configs[f"Tabular+{name}-{category_name}"] = (
                    tabular_config.configs + category_boost_list * n
                )
    training_configs[f"{name}-boost"] = all_boost_list
    # training_configs[f"{name}-rev_boost"] = rev_boost_list
    if include_tabular:
        training_configs[f"Tabular+{name}-boost"] = (
            tabular_config.configs + all_boost_list
        )
    # training_configs[f"Tabular+{name}-rev_boost"] = (
    #    tabular_config.configs + rev_boost_list
    # )
    return training_configs


class TrainingRun:
    """
    Class to perform a training run for a list of ModelBuilders.
    Dynamically runs a boosting pipeline for lists with more than one element.
    """
    def __init__(
        self,
        frame_data: FrameData,
        model_builders: list[ModelBuilder],
        num_runs: int,
        device,
    ):
        self.frame_data = frame_data
        self.model_builders = model_builders
        self.num_runs = num_runs
        self.boost = len(model_builders) > 1
        self.device = device

        if self.boost:
            boost_lr = 0.5
            if self.frame_data.task.task_type == TaskType.REGRESSION:
                boost_lr = 0.25
            elif self.frame_data.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                boost_lr = 0.2
            self.pipeline = AdaBoostPipeline(
                frame_data, model_builders, device, boost_lr=boost_lr, load_first=True
            )
        else:
            self.mb = model_builders[0]

    def get_metrics(self, test_pred, train_pred, tune_metric):
        test_metrics = self.frame_data.task.evaluate(test_pred)
        train_metrics = self.frame_data.task.evaluate(
            train_pred, self.frame_data.task.get_table("train")
        )
        overfit = test_metrics[tune_metric] / train_metrics[tune_metric]
        test_metrics["overfit"] = overfit
        return test_metrics

    def run(self):
        try:
            metrics = []
            for i in range(self.num_runs):
                if self.boost:
                    self.pipeline.train()
                    pred = self.pipeline.test("test")
                    train_pred = self.pipeline.test("train_sorted")
                    metrics.append(
                        self.get_metrics(pred, train_pred, self.pipeline.tune_metric)
                    )
                else:
                    self.mb.create_model(self.frame_data, self.device)
                    self.mb.trainer.init_data_loaders()
                    self.mb.trainer.train_model(self.mb.model, self.device)
                    self.mb.trainer.load_model(self.mb.model, self.device)
                    pred = self.mb.trainer.test(
                        self.mb.model, self.mb.trainer.loader_dict["test"], self.device
                    )
                    train_pred = self.mb.trainer.test(
                        self.mb.model,
                        self.mb.trainer.loader_dict["train_sorted"],
                        self.device,
                    )
                    metrics.append(
                        self.get_metrics(pred, train_pred, self.mb.trainer.tune_metric)
                    )
                    self.mb.unload_model()
            aggregated_metrics = {}
            for key in metrics[0].keys():
                aggr = []
                for metric in metrics:
                    aggr.append(metric[key])
                aggregated_metrics[key] = np.mean(np.array(aggr))
            print(aggregated_metrics)
            return aggregated_metrics
        except Exception:
            del self.model_builders
            if self.boost:
                del self.pipeline
            else:
                del self.mb
            torch.cuda.empty_cache()
            gc.collect()
            log("EXCEPTION OCCURED! CANCELLING RUN.", "red")
            log(traceback.format_exc(), "red")
            return None
