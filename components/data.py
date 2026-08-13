"""
Everything needed for loading and preprocessing a RelBench dataset and task.
"""

import gc
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import relbench
import torch
import torch_frame
from components.graph_models import GraphModel
from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from relbench.tasks import get_task
from sentence_transformers import SentenceTransformer
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.utils import infer_df_stype
from torch_geometric.data import HeteroData
from utils import log


class FrameData:
    """
    This class initializes and holds all data related to a RelBench dataset and task.
    This includes downloading dataset and task, creating the stype cache, creating the dataset for tabular models,
    creating TensorFrames from the database tables and creating the graph used for training.
    """

    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        device: torch.device,
        graph_model: GraphModel,
        cache_dir: str = "./cache/data",
        include_tabular: bool = False,
        text_embedder_batch_size: int = 256,
    ) -> None:
        """
        Args:
            dataset_name: The name of the RelBench dataset
            task_name: The name of the RelBench task
            device: The torch device for the TensorFrames
            graph_model: The model to use for graph creation
            cache_dir: The directory to store the materialized datasets
            include_tabular: Whether to create a dataset for tabular models
            text_embedder_batch_size: The batch size used by the Glove text embedding
        """
        self.text_embedder_cfg = TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=torch.device("cuda")),
            batch_size=text_embedder_batch_size,
        )
        self.col_stats_dict = dict()
        self.graph_model = graph_model
        self.dataset_name = dataset_name
        self.task_name = task_name

        log(f"(Down)Loading {dataset_name} dataset with task {task_name}...")
        dataset = get_dataset(dataset_name, download=True)
        task = get_task(dataset_name, task_name, download=True)
        self.dataset = dataset
        self.task = task

        if isinstance(task, relbench.base.AutoCompleteTask):
            log("Detected Autocomplete task!", "yellow")
            db = dataset.get_db(upto_test_timestamp=False)
        else:
            db = dataset.get_db()

        stype_path = os.path.join(cache_dir, dataset_name, task_name, "stypes.json")
        if os.path.exists(stype_path):
            log("Loading stypes cache...")
            with open(stype_path, "r") as f:
                self.col_to_stype_dict = json.load(f)
            for table, col_to_stype in self.col_to_stype_dict.items():
                for col, stype_str in col_to_stype.items():
                    col_to_stype[col] = stype(stype_str)
        else:
            log("Creating new stypes cache...")
            self.col_to_stype_dict = get_stype_proposal(db)
            replacement_dict = {}

            # Transforming text columns with only a few unique values to categorical
            for table_name, table in db.table_dict.items():
                df = table.df
                replacement_dict[table_name] = {}
                for col in self.col_to_stype_dict[table_name].keys():
                    if str(self.col_to_stype_dict[table_name][col]) == "text_embedded":
                        count = len(pd.unique(df[col]))
                        log(f"Distinct values for col {col}: {count}/{len(df)}")
                        if count < min(50, int(len(df) / 10000)):
                            self.col_to_stype_dict[table_name][col] = stype(
                                "categorical"
                            )
                            unique = pd.unique(df[col])
                            index_map = {}
                            for i in range(len(unique)):
                                if unique[i] == "":
                                    continue
                                index_map[unique[i]] = i
                            replacement_dict[table_name][col] = index_map
            self.replacement_dict = replacement_dict
            Path(stype_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stype_path, "w") as f:
                json.dump(self.col_to_stype_dict, f, indent=2, default=str)

        # Removing target and correlated columns from autocomplete tasks
        if isinstance(task, relbench.base.AutoCompleteTask):
            if task.target_col in self.col_to_stype_dict[task.entity_table]:
                log(f"Deleting {task.target_col}...")
                del self.col_to_stype_dict[task.entity_table][task.target_col]
            for col in dataset.remove_columns:
                if col in self.col_to_stype_dict[task.entity_table]:
                    log(f"Deleting {col}...")
                    del self.col_to_stype_dict[task.entity_table][col]

        if include_tabular:
            feat_path = os.path.join("cache", "feat", dataset_name, task_name)

            # Load feature engineered dataset if it exists
            if os.path.exists(feat_path):
                log("Found feature-engineered table data!", "green")
                self.feat = True
                self.tabular_dfs = dict()
                for split in ["train", "val", "test"]:
                    df = pd.read_csv(os.path.join(feat_path, split + ".csv"))
                    if split == "train":
                        col_to_stype = infer_df_stype(df)
                        if task.task_type == TaskType.REGRESSION:
                            col_to_stype[task.target_col] = torch_frame.numerical
                        elif (
                            task.task_type == TaskType.BINARY_CLASSIFICATION
                            or task.task_type == TaskType.MULTICLASS_CLASSIFICATION
                        ):
                            col_to_stype[task.target_col] = torch_frame.categorical
                        elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                            col_to_stype[task.target_col] = torch_frame.embedding
                        self.tabular_col_to_stype = col_to_stype
                    self.tabular_dfs[split] = df
            else:
                log("Creating tabular dataset...")
                self.feat = False
                entity_table = db.table_dict[task.entity_table]
                entity_df = entity_table.df
                col_to_stype = self.col_to_stype_dict[task.entity_table].copy()
                remove_pkey_fkey(col_to_stype, entity_table)

                if task.task_type == TaskType.REGRESSION:
                    col_to_stype[task.target_col] = torch_frame.numerical
                elif (
                    task.task_type == TaskType.BINARY_CLASSIFICATION
                    or task.task_type == TaskType.MULTICLASS_CLASSIFICATION
                ):
                    col_to_stype[task.target_col] = torch_frame.categorical
                elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                    col_to_stype[task.target_col] = torch_frame.embedding

                self.tabular_col_to_stype = col_to_stype
                self.tabular_dfs = dict()

                # Merge the entity table to the training table to create dataset
                for split, table in [
                    ("train", task.get_table("train")),
                    ("val", task.get_table("val")),
                    ("test", task.get_table("test")),
                ]:
                    left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
                    entity_df = entity_df.astype(
                        {entity_table.pkey_col: table.df[left_entity].dtype}
                    )
                    if task.time_col in entity_df.columns:
                        new_table_df = table.df.copy().rename(
                            columns={task.time_col: "timestamp"}
                        )
                    else:
                        new_table_df = table.df
                    self.tabular_dfs[split] = new_table_df.merge(
                        entity_df,
                        how="left",
                        left_on=left_entity,
                        right_on=entity_table.pkey_col,
                    )
                self.tabular_col_to_stype = col_to_stype

        log("Materializing Datasets...")
        os.makedirs(os.path.join(cache_dir, dataset_name, task_name), exist_ok=True)
        self.cache_path = os.path.join(cache_dir, dataset_name, task_name)
        self.data = HeteroData()

        while db.table_dict:
            table_name, table = db.table_dict.popitem()
            df = table.df
            if table.pkey_col is not None:
                assert (df[table.pkey_col].values == np.arange(len(df))).all()
            col_to_stype = self.col_to_stype_dict[table_name]
            remove_pkey_fkey(col_to_stype, table)

            if len(col_to_stype) == 0:
                col_to_stype = {"__const__": stype.numerical}
                fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
                df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

            path = os.path.join(self.cache_path, f"{table_name}.pt")

            if os.getenv("RDL_CONTAINER") == "1":
                with open(os.devnull, "w") as f:
                    with redirect_stdout(f), redirect_stderr(f):
                        dataset = Dataset(
                            df=df,
                            col_to_stype=col_to_stype,
                            col_to_text_embedder_cfg=self.text_embedder_cfg,
                        ).materialize(device=device, path=path)
            else:
                dataset = Dataset(
                    df=df,
                    col_to_stype=col_to_stype,
                    col_to_text_embedder_cfg=self.text_embedder_cfg,
                ).materialize(device=device, path=path)
            self.col_stats_dict[table_name] = dataset.col_stats
            self.graph_model().make_graph(self.data, table_name, dataset, table)
            del table_name, table
            gc.collect()


class DummyData(FrameData):
    """
    Minimal empty FrameData object.
    """

    def __init__(self):
        self.col_stats_dict = {}
        self.col_to_stype_dict = {}
        self.feat = True


class GloveTextEmbedding:
    """
    The model used for text embedding by Pytorch Frame.
    """

    def __init__(self, device: torch.device | None = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: list[str]) -> torch.Tensor:
        return torch.from_numpy(
            self.model.encode(list(map(str, sentences)), batch_size=256)
        )
