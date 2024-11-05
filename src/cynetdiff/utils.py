"""
Functions used to convert NetworkX graphs to usable models.
"""

import array
import random
import typing as t

import networkx as nx

from cynetdiff.models import IndependentCascadeModel, LinearThresholdModel

Graph = t.Union[nx.Graph, nx.DiGraph]
NodeMappingDict = t.Dict[t.Any, int]


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
) -> t.Tuple[IndependentCascadeModel, NodeMappingDict]:
    """
    Converts a NetworkX graph into an Independent Cascade model. Includes activation
    probability values if they are defined on each edge under the key `"activation_prob"`.
    Activation probability should only be set on either edges or through the function
    argument, not both.

    Payoffs for each node are included if they are defined on each node under the key
    `"payoff"`.

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
    tuple[IndependentCascadeModel, dict[Any, int]]
        A tuple with the instance of IndependentCascadeModel using the given graph
        and a dictionary mapping the nodes of the graph to their integer labels in
        the model.
    """

    node_list = list(enumerate(graph.nodes(data=True)))
    node_mapping = {node: i for i, (node, _) in node_list}

    starts = array.array("I")
    edges = array.array("I")
    payoffs = None
    success_prob = None
    activation_probs = None

    if _include_succcess_prob:
        success_prob = array.array("f")

    if next(iter(graph.nodes.data("payoff", None)))[1] is not None:
        payoffs = array.array("f")

    if next(iter(graph.edges.data("activation_prob", None)))[2] is not None:
        # Don't have both things set.
        if activation_prob is not None:
            raise ValueError(
                "Activation probability set on both graph data and function argument, "
                "only one should be set."
            )

        activation_probs = array.array("f")

    curr_start = 0
    for _, (node, data_dict) in node_list:
        starts.append(curr_start)

        if payoffs is not None:
            payoffs.append(data_dict["payoff"])

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

    check_csr_arrays(starts, edges)

    # Can always set threshold, as it gets ignored if edge probabilities are set.
    if activation_prob is not None:
        return IndependentCascadeModel(
            starts,
            edges,
            payoffs=payoffs,
            activation_prob=activation_prob,
            _edge_probabilities=success_prob,
        ), node_mapping

    elif activation_probs is not None:
        return IndependentCascadeModel(
            starts,
            edges,
            payoffs=payoffs,
            activation_probs=activation_probs,
            _edge_probabilities=success_prob,
        ), node_mapping
    else:
        return IndependentCascadeModel(
            starts,
            edges,
            payoffs=payoffs,
            _edge_probabilities=success_prob,
        ), node_mapping


def networkx_to_lt_model(
    graph: Graph,
) -> t.Tuple[LinearThresholdModel, NodeMappingDict]:
    """
    Converts a NetworkX graph into a Linear Threshold model. Includes influence
    values if they are defined on each edge under the key `"influence"`.

    Payoffs for each node are included if they are defined on each node under the key
    `"payoff"`.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        A NetworkX graph or directed graph.

    Returns
    -------
    tuple[LinearThresholdModel, dict[Any, int]]
        A tuple with the instance of LinearThresholdModel using the given graph
        and a dictionary mapping the nodes of the graph to their integer labels in
        the model.
    """

    node_list = list(enumerate(graph.nodes(data=True)))
    node_mapping = {node: i for i, (node, _) in node_list}

    starts = array.array("I")
    edges = array.array("I")

    influence = None
    payoffs = None

    if next(iter(graph.nodes.data("payoff", None)))[1] is not None:
        payoffs = array.array("f")

    if next(iter(graph.edges.data("influence", None)))[2] is not None:
        influence = array.array("f")

    curr_successor = 0
    for _, (node, data_dict) in node_list:
        # First, add to out neighbors
        starts.append(curr_successor)

        if payoffs is not None:
            payoffs.append(data_dict["payoff"])

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

    check_csr_arrays(starts, edges)

    model = LinearThresholdModel(
        starts,
        edges,
        payoffs=payoffs,
        influence=influence,
    )

    return model, node_mapping


def check_csr_arrays(starts: array.array, edges: array.array) -> None:
    """
    Asserts that the graph represented by `starts` and `edges` is in
    valid compressed sparse row (CSR) format. Useful for debugging,
    before the manual construction of a model.

    Parameters
    ----------
    starts : array.array
        An array of start indices for each node's edges in the edge array. Type
        of array elements must be `unsigned int`.
    edges : array.array
        An array of edges represented as integer indices of nodes. Type
        of array elements must be `unsigned int`.

    Raises
    ------
    ValueError
        If the input parameters are not in valid CSR format.
    """

    # Check typecodes
    if starts.typecode != "I":
        raise ValueError(
            f'starts array must have typecode "I" not "{starts.typecode}".'
        )

    if edges.typecode != "I":
        raise ValueError(f'edges array must have typecode "I" not "{edges.typecode}".')

    n = len(starts)
    m = len(edges)

    # Boundscheck edges
    for edge_link in edges:
        if not (0 <= edge_link < n):
            raise ValueError(
                f'Value in edges "{edge_link}" must ben in the range [0,{n-1}].'
            )

    # Boundscheck nodes
    prev_node_start = 0

    if starts and starts[0] != 0:
        raise ValueError("Indices in starts must start at 0.")

    for node_start in starts:
        if not (0 <= node_start <= m):
            raise ValueError(
                f"Value in starts '{node_start}' must be in the range [0,{m}]."
            )
        if node_start < prev_node_start:
            raise ValueError("Values stored in starts must be in increasing order.")

        prev_node_start = node_start

    # Check for self-loops or multi-edges
    for node, node_start in enumerate(starts):
        node_end = m if node == n - 1 else starts[node + 1]
        neighbors = set()

        for neighbor_idx in range(node_start, node_end):
            neighbor = edges[neighbor_idx]
            if neighbor == node:
                raise ValueError(f'Node "{node}" has a self loop at edge "{neighbor}".')
            elif neighbor in neighbors:
                raise ValueError(f'Node "{node}" has multi-edges to node "{neighbor}".')

            neighbors.add(neighbor)
