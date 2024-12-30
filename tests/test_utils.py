import array

import networkx as nx
import pytest

from cynetdiff.utils import check_csr_arrays, edgelist_to_csr_arrays


def test_edge_list_to_csr_array() -> None:
    n = 10_000
    m = 100_000
    seed = 42

    # Generate random graph with relabeled nodes
    graph = nx.gnm_random_graph(n, m, directed=True, seed=seed)
    graph = nx.convert_node_labels_to_integers(graph, first_label=n + 1)

    # Convert to CSR arrays and manually create model
    starts, edges, rename_dict = edgelist_to_csr_arrays(graph.edges())
    check_csr_arrays(starts, edges)

    new_graph = nx.DiGraph()

    # Iterate over the rows in the CSR matrix
    for i in range(len(starts)):
        row_start = starts[i]  # The start index of the row
        if i + 1 == len(starts):
            row_end = len(edges)
        else:
            row_end = starts[i + 1]  # The end index of the row

        # Iterate over the non-zero entries in the row
        for j in range(row_start, row_end):
            # Add an edge between node i and node j
            new_graph.add_edge(i, edges[j])

    assert graph.number_of_edges() == new_graph.number_of_edges()

    # Check if the edges are preserved under the node mapping
    for u, v in graph.edges:
        # Map nodes u, v from G1 to corresponding nodes in G2
        u_mapped, v_mapped = rename_dict.get(u), rename_dict.get(v)
        assert new_graph.has_edge(u_mapped, v_mapped)


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
