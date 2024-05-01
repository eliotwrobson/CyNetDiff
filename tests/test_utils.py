import array

import networkx as nx
import pytest
from cynetdiff.utils import check_csr_arrays


def test_check_csr_array_no_exception() -> None:
    graph = nx.fast_gnp_random_graph(100, 0.01, directed=True)

    node_list = list(enumerate(graph.nodes()))
    node_mapping = {node: i for i, node in node_list}

    starts = array.array("I")
    edges = array.array("I")

    curr_start = 0
    for _, node in node_list:
        starts.append(curr_start)
        for neighbor in graph.neighbors(node):
            curr_start += 1
            edges.append(node_mapping[neighbor])

    check_csr_arrays(starts, edges)

    check_csr_arrays(array.array("I", [0, 0, 1]), array.array("I", [2]))


@pytest.mark.parametrize(
    "starts,edges",
    [
        ([1, 0], [0, 0, 0]),  # Out of order node array
        ([0, 1], [2]),  # Out of bounds edge array
        ([0, 5], [1]),  # Out of bounds node array
        ([0, 0, 0], [5]),  # Out of bounds edge
        ([4, 0, 0], [1]),  # Out of bounds node array
        ([0], [0]),  # No self loops
        ([0, 0], [0, 0]),  # No multi-edges
    ],
)
def test_check_csr_array(starts: list[int], edges: list[int]) -> None:
    typecode = "I"
    with pytest.raises(ValueError):
        check_csr_arrays(array.array(typecode, starts), array.array(typecode, edges))
