"""
Contains all graph creation models.
"""

from abc import ABC, abstractmethod

import torch
from relbench.base import Table
from relbench.modeling.utils import to_unix_time
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index


class GraphModel(ABC):
    """
    Base class for all graph creation models.
    """

    @abstractmethod
    def make_graph(
        self, data: HeteroData, table_name: str, dataset: Dataset, table: Table
    ) -> None:
        """
        Processes the given table to add its entities to the given graph.

        Args:
            data: The graph data
            table_name: The name of the table to process
            dataset: The TensorFrame dataset
            table: The table
        """
        ...


# Adapted from RelBench under MIT License
# Original source: https://github.com/snap-stanford/relbench/blob/main/relbench/modeling/graph.py
# License: https://opensource.org/license/mit
class RelationalEntityGraphModel(GraphModel):
    """
    The relational entity graph model as implemented in RelBench.

    Adapted from RelBench under the MIT License.
    """

    def make_graph(
        self, data: HeteroData, table_name: str, dataset: Dataset, table: Table
    ) -> None:
        """
        Overrides `GraphModel.make_graph`.
        """
        data[table_name].tf = dataset.tensor_frame
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = table.df[fkey_name]
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # assert (pkey_index < len(frame_data.datasets[pkey_table_name][1])).all()

            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)
