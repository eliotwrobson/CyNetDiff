import copy
import random
import typing as t

import networkx as nx
import pytest
from cynetdiff.utils import (
    networkx_to_ic_model,
    set_activation_random_sample,
    set_activation_uniformly_random,
    set_activation_weighted_cascade,
)


def independent_cascade(
    G: nx.Graph | nx.DiGraph,
    seeds: t.Iterable[int],
    *,
    steps: int = 0,
    random_seed: t.Any = None,
    activation_prob: t.Optional[float] = None,
) -> t.List[t.List[int]]:
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
  >>>   (4,6), (5,3), (5,4), (5,6), (6,4), (6,5)], activation_prob=0.2)
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

    act_prob_default = 0.1 if activation_prob is None else activation_prob

    # init activation probabilities
    for u, v, data in DG.edges(data=True):
        if "activation_prob" not in data:
            data["activation_prob"] = act_prob_default
        elif activation_prob is not None:
            raise Exception("Activation prob set even though graph has probabilities")
        # if "act_prob" not in data:
        #    data["act_prob"] = 0.1
        if act_prob_default > 1.0:
            raise Exception(
                f"edge activation probability: {act_prob_default} cannot be larger than 1."
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
    return G[src][dest]["success_prob"] <= G[src][dest]["activation_prob"]


# Start of actual test code


def generate_random_graph_from_seed(
    n: int, p: float, directed: bool, seed: int = 12345
) -> t.Union[nx.Graph, nx.DiGraph]:
    graph = nx.fast_gnp_random_graph(n, p, directed=directed, seed=seed)

    random.seed(12345)
    for _, _, data in graph.edges(data=True):
        data["success_prob"] = random.random()

    return graph


# The start of the actual test cases


@pytest.mark.parametrize("directed", [True, False])
def test_model_basic(directed: bool) -> None:
    n = 10000
    k = 10
    p = 0.01
    # Just trying the main functions with no set thresholds
    graph = generate_random_graph_from_seed(n, p, directed)
    model = networkx_to_ic_model(graph)

    # Didn't set the seeds
    model.advance_until_completion()
    assert model.get_num_activated_nodes() == 0

    seeds = set(random.sample(list(graph.nodes), k))
    model.set_seeds(seeds)
    assert set(model.get_newly_activated_nodes()) == seeds

    model.advance_until_completion()
    assert k <= model.get_num_activated_nodes() <= n
    model.reset_model()
    assert set(model.get_newly_activated_nodes()) == seeds


@pytest.mark.parametrize(
    "set_act_prob_fn, directed",
    [
        (None, True),
        (None, False),
        (set_activation_uniformly_random, True),
        (set_activation_uniformly_random, False),
        (set_activation_weighted_cascade, True),
        (lambda graph: set_activation_random_sample(graph, {0.1, 0.01, 0.001}), True),
        (lambda graph: set_activation_random_sample(graph, {0.1, 0.01, 0.001}), False),
    ],
)
def test_specific_model(
    set_act_prob_fn: t.Optional[t.Callable[[nx.Graph | nx.DiGraph], None]],
    directed: bool,
) -> None:
    n = 1000
    p = 0.01
    k = 10

    test_graph = generate_random_graph_from_seed(n, p, directed)
    indep_cascade_prob = 0.2

    if set_act_prob_fn is not None:
        set_act_prob_fn(test_graph)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    # If this function is None, we use the uniform activation threshold.
    if set_act_prob_fn is None:
        activated_nodes_levels = independent_cascade(
            test_graph, seeds, activation_prob=indep_cascade_prob
        )
    else:
        activated_nodes_levels = independent_cascade(test_graph, seeds)

    # Set up the model
    if set_act_prob_fn is None:
        model = networkx_to_ic_model(
            test_graph, activation_prob=indep_cascade_prob, _include_succcess_prob=True
        )
    else:
        model = networkx_to_ic_model(test_graph, _include_succcess_prob=True)

    model.set_seeds(seeds)

    # Run twice to check that the reset works
    for _ in range(2):
        assert model.get_num_activated_nodes() == len(seeds)
        assert len(set(model.get_newly_activated_nodes())) == len(seeds)

        for node_level in activated_nodes_levels:
            assert set(node_level) == set(model.get_newly_activated_nodes())
            model.advance_model()

        model.reset_model()


@pytest.mark.parametrize("directed", [True, False])
def test_basic_2(directed: bool) -> None:
    n = 1000
    p = 0.01
    k = 10
    test_graph = generate_random_graph_from_seed(n, p, directed)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    activated_nodes_levels = independent_cascade(test_graph, seeds)
    num_seen = sum(len(level) for level in activated_nodes_levels)

    # Set up the model
    model = networkx_to_ic_model(test_graph, _include_succcess_prob=True)
    model.set_seeds(seeds)
    model.advance_until_completion()

    assert num_seen == model.get_num_activated_nodes()
