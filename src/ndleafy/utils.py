import array

import networkx as nx

from ndleafy.models import IndependentCascadeModel, LinearThresholdModel

# TODO speed up these model creation functions if needed


def networkx_to_ic_model(
    graph: nx.Graph | nx.DiGraph,
    *,
    threshold=0.1,
    include_succcess_prob: bool = False,
) -> IndependentCascadeModel:
    node_list = list(enumerate(graph.nodes()))
    node_mapping = {node: i for i, node in node_list}

    starts = array.array("I")
    edges = array.array("I")
    success_prob = None

    if include_succcess_prob:
        success_prob = array.array("f")

    curr_start = 0
    for _, node in node_list:
        starts.append(curr_start)
        for neighbor in graph.neighbors(node):
            other = node_mapping[neighbor]
            curr_start += 1
            edges.append(other)

            if success_prob is not None:
                success_prob.append(graph.get_edge_data(node, other)["success_prob"])

    return IndependentCascadeModel(
        starts, edges, threshold=threshold, edge_probabilities=success_prob
    )


def networkx_to_lt_model(
    graph: nx.Graph | nx.DiGraph, *, include_influence: bool = False
) -> LinearThresholdModel:
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
