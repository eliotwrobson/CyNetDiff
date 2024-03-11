"""
Functions used to convert NetworkX graphs to usable models.
"""

import array
import random
import typing as t

import networkx as nx

from cynetdiff.models import IndependentCascadeModel, LinearThresholdModel

# TODO speed up these model creation functions if needed

Graph = t.Union[nx.Graph, nx.DiGraph]


def set_activation_uniformly_random(
    graph: Graph, *, range_start: float = 0.0, range_end: float = 1.0
) -> None:
    """
    Set activation probability on each edge uniformly at random in the range
    [range_start, range_end]. Must have that
    0.0 <= range_start < range_end <= 1.0
    """

    assert 0.0 <= range_start < range_end <= 1.0

    for _, _, edge_data in graph.edges(data=True):
        edge_data["activation_prob"] = random.uniform(range_start, range_end)


def set_activation_weighted_cascade(graph: Graph) -> None:
    """
    Set activation probability on each edge to 1/in_degree[u].
    """
    deg_fun = graph.in_degree if graph.is_directed() else graph.degree

    for _, to_node, edge_data in graph.edges(data=True):
        edge_data["activation_prob"] = 1 / (deg_fun(to_node))


def set_activation_random_sample(
    graph: Graph, weight_set: t.AbstractSet[float]
) -> None:
    """
    Set activation probability on each edge uniformly at random from the
    given input set.
    """
    weights = tuple(weight_set)

    for _, _, edge_data in graph.edges(data=True):
        edge_data["activation_prob"] = random.choice(weights)


def networkx_to_ic_model(
    graph: nx.Graph | nx.DiGraph,
    *,
    threshold: t.Optional[float] = None,
    _include_succcess_prob: bool = False,
) -> IndependentCascadeModel:
    """
    Converts a NetworkX graph into an Independent Cascade model.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        A NetworkX graph or directed graph.
    threshold : float, optional
        Threshold for the Independent Cascade model, by default None.
        If not set, and "activation_prob" key not found on edges, set to 0.1.
    _include_succcess_prob : bool, optional
        If True, includes success probabilities for each edge. These probabilities
        are then stored in the edge data dictionary with the key "success_prob",
        by default False.

    Returns
    -------
    IndependentCascadeModel
        An instance of IndependentCascadeModel representing the given graph.
    """

    node_list = list(enumerate(graph.nodes()))
    node_mapping = {node: i for i, node in node_list}

    starts = array.array("I")
    edges = array.array("I")
    success_prob = None
    activation_prob = None

    if _include_succcess_prob:
        success_prob = array.array("f")

    if next(iter(graph.edges.data("activation_prob", None)))[2] is not None:
        assert threshold is None  # Don't have both things set.
        activation_prob = array.array("f")

    curr_start = 0
    for _, node in node_list:
        starts.append(curr_start)
        for neighbor in graph.neighbors(node):
            other = node_mapping[neighbor]
            curr_start += 1
            edges.append(other)

            if success_prob is not None:
                success_prob.append(graph.get_edge_data(node, other)["success_prob"])

            if activation_prob is not None:
                act_prob = graph[node][neighbor]["activation_prob"]
                assert 0.0 <= act_prob <= 1.0
                activation_prob.append(act_prob)

    # Can always set threshold, as it gets ignored if edge probabilities are set.
    if threshold is None:
        threshold = 0.1

    return IndependentCascadeModel(
        starts,
        edges,
        threshold=threshold,
        edge_probabilities=success_prob,
        edge_thresholds=activation_prob,
    )


def networkx_to_lt_model(
    graph: nx.Graph | nx.DiGraph, *, include_influence: bool = False
) -> LinearThresholdModel:
    """
    Converts a NetworkX graph into a Linear Threshold model.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        A NetworkX graph or directed graph.
    include_influence : bool, optional
        If True, includes influence values for each edge. These influence values
        are then stored in the edge data dictionary with the key "influence",
        by default False.

    Returns
    -------
    LinearThresholdModel
        An instance of LinearThresholdModel representing the given graph.
    """

    node_list = list(enumerate(graph.nodes(data=True)))
    node_mapping = {node: i for i, (node, _) in node_list}

    successors = array.array("I")
    successor_starts = array.array("I")

    predecessors = array.array("I")
    predecessor_starts = array.array("I")

    # TODO make setting these optional
    # At the very least the influence can be defaulted
    threshold = array.array("f")
    influence = None

    if include_influence:
        influence = array.array("f")

    curr_successor = 0
    curr_predecessor = 0
    for _, (node, data) in node_list:
        # First, add to out neighbors
        successor_starts.append(curr_successor)
        for successor in graph.successors(node):
            other = node_mapping[successor]
            curr_successor += 1
            successors.append(other)

        # Next, do the same but for in-neighbors,
        # logic is largely the same
        predecessor_starts.append(curr_predecessor)
        for predecessor in graph.predecessors(node):
            other = node_mapping[predecessor]
            curr_predecessor += 1
            predecessors.append(other)

            if influence is not None:
                influence.append(graph[other][node]["influence"])

        threshold.append(data["threshold"])

    return LinearThresholdModel(
        successors,
        successor_starts,
        predecessors,
        predecessor_starts,
        threshold=threshold,
        influence=influence,
    )
