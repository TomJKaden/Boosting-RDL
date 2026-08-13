"""
Contains utility functions to calculate graph-related task properties.
"""

import matplotlib.pyplot as plt
import networkx as nx
from relbench.base import Database


def create_schema_graph(db: Database) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(list(db.table_dict.keys()))
    for table_name, table in db.table_dict.items():
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            G.add_edge(table_name, pkey_table_name)
    return G


def get_task_metrics(G: nx.DiGraph, db: Database, task):
    source = task.entity_table
    eccentricity = nx.eccentricity(G.to_undirected(), source)
    avg_shortest_path = nx.average_shortest_path_length(G.to_undirected())
    density = nx.density(G.to_undirected())
    clustering = nx.clustering(G.to_undirected())[source]
    return eccentricity, avg_shortest_path, density, clustering


def draw_graph(G: nx.DiGraph):
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, verticalalignment="bottom")
    plt.show()
