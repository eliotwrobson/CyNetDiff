"""
Purpose:
*.
"""

import array
import copy
import math
import random
import typing as t

import networkx as nx

from ndleafy.models import IndependentCascadeModel

#    Code below adapted from code by
#    Hung-Hsuan Chen <hhchen@psu.edu>
#    All rights reserved.
#    BSD license.
#    NetworkX:http://networkx.lanl.gov/.


def independent_cascade(G, seeds, *, steps=0, random_seed=None) -> list[list[int]]:
    """Return the active nodes of each diffusion step by the independent cascade
  model

  Parameters
  -----------
  G : graph
    A NetworkX graph
  seeds : list of nodes
    The seed nodes for diffusion
  steps: integer
    The number of steps to diffuse.  If steps <= 0, the diffusion runs until
    no more nodes can be activated.  If steps > 0, the diffusion runs for at
    most "steps" rounds

  Returns
  -------
  layer_i_nodes : list of list of activated nodes
    layer_i_nodes[0]: the seeds
    layer_i_nodes[k]: the nodes activated at the kth diffusion step

  Notes
  -----
  When node v in G becomes active, it has a *single* chance of activating
  each currently inactive neighbor w with probability p_{vw}

  Examples
  --------
  >>> DG = nx.DiGraph()
  >>> DG.add_edges_from([(1,2), (1,3), (1,5), (2,1), (3,2), (4,2), (4,3), \
  >>>   (4,6), (5,3), (5,4), (5,6), (6,4), (6,5)], act_prob=0.2)
  >>> layers = networkx_addon.information_propagation.independent_cascade(DG, [6])

  References
  ----------
  [1] David Kempe, Jon Kleinberg, and Eva Tardos.
      Influential nodes in a diffusion model for social networks.
      In Automata, Languages and Programming, 2005.
  """
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise Exception(
            "independent_cascade() is not defined for graphs with multiedges."
        )

    rand_gen = random.Random(random_seed)

    # change to directed graph
    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)

    # init activation probabilities
    for u, v, data in DG.edges(data=True):
        act_prob = data.setdefault("act_prob", 0.1)
        # if "act_prob" not in data:
        #    data["act_prob"] = 0.1
        if act_prob > 1.0:
            raise Exception(
                f"edge activation probability: {act_prob} cannot be larger than 1."
            )

        data.setdefault("success_prob", rand_gen.random())

    # perform diffusion
    A = copy.deepcopy(seeds)  # prevent side effect
    if steps <= 0:
        # perform diffusion until no more nodes can be activated
        return _diffuse_all(DG, A, rand_gen)
    # perform diffusion for at most "steps" rounds
    return _diffuse_k_rounds(DG, A, steps, rand_gen)


def _diffuse_all(G, A, rand_gen):
    tried_edges = set()
    layer_i_nodes = []
    layer_i_nodes.append([i for i in A])  # prevent side effect
    while True:
        len_old = len(A)
        (A, activated_nodes_of_this_round, cur_tried_edges) = _diffuse_one_round(
            G, A, tried_edges, rand_gen
        )
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(A) == len_old:
            break
    return layer_i_nodes


def _diffuse_k_rounds(G, A, steps, rand_gen):
    tried_edges = set()
    layer_i_nodes = []
    layer_i_nodes.append([i for i in A])
    while steps > 0 and len(A) < len(G):
        len_old = len(A)
        (A, activated_nodes_of_this_round, cur_tried_edges) = _diffuse_one_round(
            G, A, tried_edges, rand_gen
        )
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(A) == len_old:
            break
        steps -= 1
    return layer_i_nodes


def _diffuse_one_round(G, A, tried_edges, rand_gen):
    activated_nodes_of_this_round = set()
    cur_tried_edges = set()
    for s in A:
        for nb in G.successors(s):
            if nb in A or (s, nb) in tried_edges or (s, nb) in cur_tried_edges:
                continue
            if _prop_success(G, s, nb, rand_gen):
                activated_nodes_of_this_round.add(nb)
            cur_tried_edges.add((s, nb))
    activated_nodes_of_this_round = list(activated_nodes_of_this_round)
    A.extend(activated_nodes_of_this_round)
    return A, activated_nodes_of_this_round, cur_tried_edges


def _prop_success(G, src, dest, rand_gen):
    return G[src][dest]["success_prob"] <= G[src][dest]["act_prob"]


# Start of actual test code


# TODO speed up this CSR function
def networkx_to_csr(
    graph: nx.Graph | nx.DiGraph, include_succcess_prob: bool = True
) -> tuple[array.array, array.array, t.Optional[array.array]]:
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    starts = array.array("I")
    edges = array.array("I")
    # TODO switch to optional return type
    success_prob = None

    if include_succcess_prob:
        success_prob = array.array("f")

    curr_start = 0
    for node in graph.nodes():
        starts.append(curr_start)
        for neighbor in graph.neighbors(node):
            other = node_mapping[neighbor]
            curr_start += 1
            edges.append(other)

            if success_prob is not None:
                success_prob.append(graph.get_edge_data(node, other)["success_prob"])

    return starts, edges, success_prob


def generate_random_graph_from_seed(n: int, p: float, seed: int = 12345) -> nx.Graph:
    graph = nx.fast_gnp_random_graph(n, p, seed=seed)

    random.seed(12345)
    for _, _, data in graph.edges(data=True):
        data["success_prob"] = random.random()

    return graph


# The start of the actual test cases


def test_model_basic() -> None:
    n = 10000
    k = 10
    p = 0.01
    # Just trying the main functions with no set thresholds
    graph = generate_random_graph_from_seed(n, p)
    starts, edges, _ = networkx_to_csr(graph, include_succcess_prob=False)
    model = IndependentCascadeModel(starts, edges, threshold=0.1)

    # Didn't set the seeds
    model.advance_until_completion()
    assert model.get_num_activated_nodes() == 0

    model.set_seeds(random.sample(list(graph.nodes), k))
    assert len(set(model.get_newly_activated_nodes())) == k

    model.advance_until_completion()
    assert k <= model.get_num_activated_nodes() <= n


def test_specific_model() -> None:
    n = 1000
    p = 0.01
    k = 10
    test_graph = generate_random_graph_from_seed(n, p)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    activated_nodes_levels = independent_cascade(test_graph, seeds)

    # Set up the model
    starts, edges, success_probs = networkx_to_csr(test_graph)
    thing = IndependentCascadeModel(
        starts, edges, threshold=0.1, edge_probabilities=success_probs
    )
    thing.initialize_model(seeds)

    for node_level in activated_nodes_levels:
        assert set(node_level) == set(thing.get_newly_activated_nodes())
        thing.advance_model()


def test_basic_2() -> None:
    n = 1000
    p = 0.01
    k = 10
    test_graph = generate_random_graph_from_seed(n, p)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    activated_nodes_levels = independent_cascade(test_graph, seeds)
    num_seen = sum(len(level) for level in activated_nodes_levels)

    # Set up the model
    starts, edges, success_probs = networkx_to_csr(test_graph)
    thing = IndependentCascadeModel(
        starts, edges, threshold=0.1, edge_probabilities=success_probs
    )
    thing.initialize_model(seeds)
    thing.advance_until_completion()

    assert num_seen == thing.get_num_activated_nodes()


def test_parallel() -> None:
    n = 1000
    p = 0.01
    k = 10
    test_graph = generate_random_graph_from_seed(n, p)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    activated_nodes_levels = independent_cascade(test_graph, seeds)
    total_seen = sum(len(level) for level in activated_nodes_levels)
    print(total_seen)

    # Set up the model
    starts, edges, success_probs = networkx_to_csr(test_graph)
    thing = IndependentCascadeModel(
        starts, edges, threshold=0.1, edge_probabilities=success_probs
    )
    thing.initialize_model(seeds)
    res = thing.run_in_parallel(100)

    assert math.isclose(res, float(total_seen))
