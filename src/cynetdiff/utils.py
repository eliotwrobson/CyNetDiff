"""
Functions used to convert NetworkX graphs to usable models.
"""

import array
import random
import typing as t

import networkx as nx

from cynetdiff.models import IndependentCascadeModel, LinearThresholdModel

Graph = t.Union[nx.Graph, nx.DiGraph]


def set_activation_uniformly_random(
    graph: Graph, *, range_start: float = 0.0, range_end: float = 1.0
) -> None:
    """
    Set activation probability on each edge uniformly at random in the range
    [`range_start`, `range_end`]. Must have that
    `0.0` <= `range_start` < `range_end` <= `1.0`. Should be used on graphs before
    creating the independent cascade model.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        A NetworkX graph or directed graph.
    range_start : float, optional
        The start of the range to sample activation probabilities from. If not set,
        defaults to `0.0`.
    range_end : float, optional
        The end of the range to sample activation probabilities from. If not set,
        defaults to `1.0`.
    """

    assert 0.0 <= range_start < range_end <= 1.0

    for _, _, edge_data in graph.edges(data=True):
        edge_data["activation_prob"] = random.uniform(range_start, range_end)


def set_activation_weighted_cascade(graph: nx.DiGraph) -> None:
    """
    Set activation probability on each edge (u,v) to 1/in_degree(v). Graph
    must be directed. Should be used on graphs before creating the independent
    cascade model.

    Parameters
    ----------
    graph : nx.DiGraph
        A NetworkX directed graph.
    """
    if not graph.is_directed():
        raise ValueError("Graph must be directed or weighting will not be correct.")

    for _, to_node, edge_data in graph.edges(data=True):
        edge_data["activation_prob"] = 1 / (graph.in_degree(to_node))


def set_activation_random_sample(
    graph: Graph, weight_set: t.AbstractSet[float]
) -> None:
    """
    Set activation probability on each edge uniformly at random from the given weight set.
    Should be used on graphs before creating the independent cascade model.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        A NetworkX graph or directed graph.
    weight_set : AbstractSet[float]
        The set of weights to sample from. Assigns each edge in the input graph
        a weight uniformly at random from this set.
    """
    weights = tuple(weight_set)

    for _, _, edge_data in graph.edges(data=True):
        edge_data["activation_prob"] = random.choice(weights)


def networkx_to_ic_model(
    graph: Graph,
    *,
    activation_prob: t.Optional[float] = None,
    _include_succcess_prob: bool = False,
) -> IndependentCascadeModel:
    """
    Converts a NetworkX graph into an Independent Cascade model. Includes activation
    probability values if they are defined on each edge under the key `"activation_prob"`.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        A NetworkX graph or directed graph.
    activation_prob : float, optional
        Activation probability for the Independent Cascade model, by default `None`.
        If not set, and `"activation_prob"` key not found on edges, set to 0.1.
    _include_succcess_prob : bool, optional
        If True, includes success probabilities for each edge. These probabilities
        are then stored in the edge data dictionary with the key "success_prob",
        by default False. Used mainly for testing.

    Returns
    -------
    IndependentCascadeModel
        An instance of IndependentCascadeModel using the given graph.
    """

    node_list = list(enumerate(graph.nodes()))
    node_mapping = {node: i for i, node in node_list}

    starts = array.array("I")
    edges = array.array("I")
    success_prob = None
    activation_probs = None

    if _include_succcess_prob:
        success_prob = array.array("f")

    if next(iter(graph.edges.data("activation_prob", None)))[2] is not None:
        assert activation_prob is None  # Don't have both things set.
        activation_probs = array.array("f")

    curr_start = 0
    for _, node in node_list:
        starts.append(curr_start)
        for neighbor in graph.neighbors(node):
            other = node_mapping[neighbor]
            curr_start += 1
            edges.append(other)

            if success_prob is not None:
                success_prob.append(graph.get_edge_data(node, other)["success_prob"])

            if activation_probs is not None:
                act_prob = graph[node][neighbor]["activation_prob"]
                assert 0.0 <= act_prob <= 1.0
                activation_probs.append(act_prob)

    # Can always set threshold, as it gets ignored if edge probabilities are set.
    if activation_prob is not None:
        return IndependentCascadeModel(
            starts,
            edges,
            activation_prob=activation_prob,
            _edge_probabilities=success_prob,
        )

    elif activation_probs is not None:
        return IndependentCascadeModel(
            starts,
            edges,
            activation_probs=activation_probs,
            _edge_probabilities=success_prob,
        )
    else:
        return IndependentCascadeModel(
            starts,
            edges,
            _edge_probabilities=success_prob,
        )


def networkx_to_lt_model(graph: Graph) -> LinearThresholdModel:
    """
    Converts a NetworkX graph into a Linear Threshold model. Includes influence
    values if they are defined on each edge under the key `"influence"`. Includes
    threshold values if they are defined on each node under the key `"threshold"`.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        A NetworkX graph or directed graph.

    Returns
    -------
    LinearThresholdModel
        An instance of LinearThresholdModel using the given graph.
    """

    node_list = list(enumerate(graph.nodes(data=True)))
    node_mapping = {node: i for i, (node, _) in node_list}

    starts = array.array("I")
    edges = array.array("I")

    thresholds = None
    influence = None

    if next(iter(graph.edges.data("influence", None)))[2] is not None:
        influence = array.array("f")

    if next(iter(graph.nodes.data("threshold", None)))[1] is not None:
        thresholds = array.array("f")

    curr_successor = 0
    for _, (node, data) in node_list:
        # First, add to out neighbors
        starts.append(curr_successor)
        for successor in graph.successors(node):
            other = node_mapping[successor]
            curr_successor += 1
            edges.append(other)

            if influence is not None:
                edge_influence = graph[node][successor]["influence"]
                influence.append(edge_influence)

        if influence is not None:
            # Check that in-sum is not too high.
            pred_sum = 0.0

            for predecessor in graph.predecessors(node):
                pred_sum += graph[predecessor][node]["influence"]

            # 1.0001 instead of 1.0 to avoid floating point issues.
            # TODO will this annoy people?
            if pred_sum > 1.0001:
                raise ValueError(
                    f"Node {node} has inward influence {pred_sum}, "
                    "must be less than 1.0."
                )

        if thresholds is not None:
            threshold = data["threshold"]
            assert 0.0 <= threshold <= 1.0
            thresholds.append(threshold)

    return LinearThresholdModel(
        starts,
        edges,
        thresholds=thresholds,
        influence=influence,
    )
