"""
Purpose:
*.
"""

import array

import networkx as nx
import pytest

from ndleafy.models import IndependentCascadeModel


def networkx_to_csr(graph: nx.Graph | nx.DiGraph) -> tuple[array.array, array.array]:
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    starts = array.array("I")  # []
    edges = array.array("I")  # []
    curr_start = 0
    for node in graph.nodes():
        starts.append(curr_start)
        for neighbor in graph.neighbors(node):
            curr_start += 1
            edges.append(node_mapping[neighbor])

    return starts, edges


def test_basic():
    n = 100
    p = 0.1
    test_graph = nx.fast_gnp_random_graph(n, p)

    starts, edges = networkx_to_csr(test_graph)
    thing = IndependentCascadeModel(starts, edges)
