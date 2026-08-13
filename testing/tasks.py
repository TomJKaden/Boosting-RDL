"""
Contains the logic responsible for calculating the task statistics.
"""

import gc

import numpy as np
import pandas as pd
from graph_properties import create_schema_graph, get_task_metrics
from relbench.base import AutoCompleteTask, TaskType
from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task, get_task_names
from scipy import stats


def main():
    common_cols = [
        "dataset",
        "task",
        "type",
        "n_train",
        "unique",
        "overlap",
        "eccentricity",
        "avg_shortest_path",
        "density",
        "clustering",
    ]
    binary_df = pd.DataFrame(columns=common_cols + ["balance"])
    regression_df = pd.DataFrame(
        columns=common_cols + ["min", "max", "median", "mean", "skewness", "kurtosis"]
    )
    multiclass_df = pd.DataFrame(
        columns=common_cols + ["n_classes", "freq_ratio", "hhi"]
    )

    for dataset_name in get_dataset_names():
        if "mimic" in dataset_name or "rel" not in dataset_name:
            continue
        dataset = get_dataset(dataset_name, download=True)
        for task_name in get_task_names(dataset_name):
            if "event_interest" in task_name:
                continue
            task = get_task(dataset_name, task_name, download=True)
            if task.task_type == TaskType.LINK_PREDICTION:
                continue
            print(dataset_name, task_name)
            stats = [dataset_name, task_name] + get_task_stats(dataset, task)
            print(stats)
            if task.task_type == TaskType.BINARY_CLASSIFICATION:
                binary_df.loc[len(binary_df)] = stats
            elif task.task_type == TaskType.REGRESSION:
                regression_df.loc[len(regression_df)] = stats
            elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                multiclass_df.loc[len(multiclass_df)] = stats
            del task
            gc.collect()
        del dataset
        gc.collect()

    binary_df.to_csv("binary_stats.csv")
    regression_df.to_csv("regression_stats.csv")
    multiclass_df.to_csv("multiclass_stats.csv")


def get_task_stats(dataset, task):
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)
    type_str = "auto" if isinstance(task, AutoCompleteTask) else "entity"
    # entity_df = pd.concat([train_table.df, val_table.df, test_table.df])

    n_train = len(train_table.df)
    n_val = len(val_table.df)
    n_test = len(test_table.df)

    train_unique = train_table.df[task.entity_col].unique()
    test_unique = test_table.df[task.entity_col].unique()
    unique_percent = len(train_unique) / n_train
    overlap = len(
        np.intersect1d(
            train_unique,
            test_unique,
        )
    ) / len(test_unique)
    del train_unique
    del test_unique
    gc.collect()

    schema_graph = create_schema_graph(dataset.get_db())
    eccentricity, avg_shortest_path, density, clustering = get_task_metrics(
        schema_graph, dataset.get_db(), task
    )
    del schema_graph
    gc.collect()

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        df: pd.DataFrame = train_table.df
        if "f" in df[task.target_col].values:
            df[task.target_col] = df[task.target_col].replace({"t": 1, "f": 0})
        df[task.target_col] = df[task.target_col].astype(int)
        n_pos = len(df[df[task.target_col] == 1])
        n_neg = len(df[df[task.target_col] == 0])
        assert n_pos + n_neg == n_train
        balance = min(n_pos, n_neg) / n_train

        return [
            type_str,
            n_train,
            unique_percent,
            overlap,
            eccentricity,
            avg_shortest_path,
            density,
            clustering,
            balance,
        ]

    elif task.task_type == TaskType.REGRESSION:
        data = train_table.df[task.target_col].to_numpy()
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        return [
            type_str,
            n_train,
            unique_percent,
            overlap,
            eccentricity,
            avg_shortest_path,
            density,
            clustering,
            data.min(),
            data.max(),
            np.median(data),
            data.mean(),
            skewness,
            kurtosis,
        ]

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        num_classes = task.num_classes
        data = train_table.df[task.target_col].to_numpy()
        class_count = np.bincount(data)
        imbalance = class_count.max() / class_count.min()
        hhi = np.sum((class_count / class_count.sum()) ** 2)

        return [
            type_str,
            n_train,
            unique_percent,
            overlap,
            eccentricity,
            avg_shortest_path,
            density,
            clustering,
            num_classes,
            imbalance,
            hhi,
        ]


if __name__ == "__main__":
    main()
