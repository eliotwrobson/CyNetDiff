import copy
import random
import typing as t

import networkx as nx
import pytest
from cynetdiff.utils import networkx_to_lt_model

#    Code below adapted from code by
#    Hung-Hsuan Chen <hhchen@psu.edu>
#    All rights reserved.
#    BSD license.
#    NetworkX:http://networkx.lanl.gov/.


def linear_threshold(
    G: t.Union[nx.Graph, nx.DiGraph], seeds: t.Iterable[int], steps: int = 0
) -> t.List[t.List[int]]:
    """Return the active nodes of each diffusion step by linear threshold model

  Parameters
  ----------
  G : networkx graph
      The number of nodes.

  seeds: list of nodes
      The seed nodes of the graph

  steps: int
      The number of steps to diffuse
      When steps <= 0, the model diffuses until no more nodes
      can be activated

  Return
  ------
  layer_i_nodes : list of list of activated nodes
    layer_i_nodes[0]: the seeds
    layer_i_nodes[k]: the nodes activated at the kth diffusion step

  Notes
  -----
  1. Each edge is supposed to have an attribute "influence".  If not, the
     default value is given (1/in_degree)

  References
  ----------
  [1] GranovetterMark. Threshold models of collective behavior.
      The American journal of sociology, 1978.

  Examples
  --------
  >>> DG = nx.DiGraph()
  >>> DG.add_edges_from([(1,2), (1,3), (1,5), (2,1), (3,2), (4,2), (4,3), \
  >>>   (4,6), (5,3), (5,4), (5,6), (6,4), (6,5)])
  >>> layers = networkx_addon.information_propagation.linear_threshold(DG, [1])

  """
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise Exception("linear_threshold() is not defined for graphs with multiedges.")

    # make sure the seeds are in the graph
    for s in seeds:
        if s not in G.nodes():
            raise Exception("seed", s, "is not in graph")

    # change to directed graph
    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)

    # init thresholds
    for n in DG.nodes():
        if "threshold" not in DG._node[n]:
            DG._node[n]["threshold"] = random.random()
        elif DG._node[n]["threshold"] > 1:
            raise Exception(
                "node threshold:", DG._node[n]["threshold"], "cannot be larger than 1"
            )

    # init influences
    in_deg = DG.in_degree()
    for e in DG.edges():
        if "influence" not in DG[e[0]][e[1]]:
            DG[e[0]][e[1]]["influence"] = 1.0 / in_deg[e[1]]
        elif DG[e[0]][e[1]]["influence"] > 1:
            raise Exception(
                "edge influence:",
                DG[e[0]][e[1]]["influence"],
                "cannot be larger than 1",
            )

    # perform diffusion
    A = copy.deepcopy(seeds)
    if steps <= 0:
        # perform diffusion until no more nodes can be activated
        return _diffuse_all(DG, A)
    # perform diffusion for at most "steps" rounds only
    return _diffuse_k_rounds(DG, A, steps)


def _diffuse_all(G, A):
    layer_i_nodes = []
    layer_i_nodes.append([i for i in A])
    while True:
        len_old = len(A)
        A, activated_nodes_of_this_round = _diffuse_one_round(G, A)
        layer_i_nodes.append(activated_nodes_of_this_round)
        if len(A) == len_old:
            break
    return layer_i_nodes


def _diffuse_k_rounds(G, A, steps):
    layer_i_nodes = []
    layer_i_nodes.append([i for i in A])
    while steps > 0 and len(A) < len(G):
        len_old = len(A)
        A, activated_nodes_of_this_round = _diffuse_one_round(G, A)
        layer_i_nodes.append(activated_nodes_of_this_round)
        if len(A) == len_old:
            break
        steps -= 1
    return layer_i_nodes


def _diffuse_one_round(G, A):
    activated_nodes_of_this_round = set()
    for s in A:
        nbs = G.successors(s)

        for nb in nbs:
            if nb in A:
                continue
            active_nb = list(set(G.predecessors(nb)).intersection(set(A)))

            if _influence_sum(G, active_nb, nb) >= G._node[nb]["threshold"]:
                activated_nodes_of_this_round.add(nb)
    A.extend(list(activated_nodes_of_this_round))
    return A, list(activated_nodes_of_this_round)


def _influence_sum(G, froms, to):
    influence_sum = 0.0
    for f in froms:
        influence_sum += G[f][to]["influence"]

    return influence_sum


def generate_random_graph_from_seed(
    n: int, p: float, directed: bool, include_influence: bool, seed: int = 12345
) -> nx.DiGraph:
    graph = nx.fast_gnp_random_graph(n, p, seed=seed, directed=directed).to_directed()

    random.seed(seed)
    for _, data in graph.nodes(data=True):
        data["threshold"] = random.random()

    if include_influence:
        for _, _, data in graph.edges(data=True):
            data["influence"] = random.random()

        for node in graph.nodes():
            pred_dicts = tuple(
                graph[predecessor][node] for predecessor in graph.predecessors(node)
            )

            # Get sum of influences and then normalize.
            inf_sum = 0.0
            for pred_dict in pred_dicts:
                inf_sum += pred_dict["influence"]
            for pred_dict in pred_dicts:
                pred_dict["influence"] /= inf_sum

    return graph


# Start of actual test code


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("nondefault_influence", [True, False])
def test_specific_model(directed: bool, nondefault_influence: bool) -> None:
    n = 1_000
    p = 0.05
    k = 10
    test_graph = generate_random_graph_from_seed(
        n, p, directed=directed, include_influence=nondefault_influence
    )

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    activated_nodes_levels = linear_threshold(test_graph, seeds)

    model = networkx_to_lt_model(test_graph)

    model.set_seeds(seeds)

    # Run twice to check that the reset works
    for _ in range(2):
        assert model.get_num_activated_nodes() == len(seeds)
        assert len(sorted(model.get_newly_activated_nodes())) == len(seeds)

        for node_level in activated_nodes_levels:
            model_set = sorted(model.get_newly_activated_nodes())
            node_set = sorted(node_level)

            assert model_set == node_set
            model.advance_model()

        model.reset_model()

    total_num = sum(map(len, activated_nodes_levels))

    # Reassigning the threshold should change the total number of activated nodes
    model.reassign_thresholds()
    model.advance_until_completion()

    # TODO this test passes with high-enough probability. Refactor to avoid a possible
    # random failure,
    assert total_num != model.get_num_activated_nodes()
