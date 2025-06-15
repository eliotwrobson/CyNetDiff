import copy
import math
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

Graph = t.Union[nx.Graph, nx.DiGraph]


#    Code below adapted from code by
#    Hung-Hsuan Chen <hhchen@psu.edu>
#    All rights reserved.
#    BSD license.


def independent_cascade(
    graph: Graph,
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

    References
    ----------
    [1] David Kempe, Jon Kleinberg, and Eva Tardos.
        Influential nodes in a diffusion model for social networks.
        In Automata, Languages and Programming, 2005.
    """
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raise Exception("independent_cascade() is not defined for graphs with multiedges.")

    rand_gen = random.Random(random_seed)

    # change to directed graph
    if not graph.is_directed():
        diffusion_graph = graph.to_directed()
    else:
        diffusion_graph = copy.deepcopy(graph)

    act_prob_default = 0.1 if activation_prob is None else activation_prob

    # init activation probabilities
    for _, _, data in diffusion_graph.edges(data=True):
        if "activation_prob" not in data:
            data["activation_prob"] = act_prob_default
        elif activation_prob is not None:
            raise Exception("Activation prob set even though graph has probabilities")
        # if "act_prob" not in data:
        #    data["act_prob"] = 0.1
        if act_prob_default > 1.0:
            raise Exception(f"edge activation probability: {act_prob_default} cannot be larger than 1.")

        data.setdefault("success_prob", rand_gen.random())

    # perform diffusion
    activated = copy.deepcopy(seeds)  # prevent side effect
    if steps <= 0:
        # perform diffusion until no more nodes can be activated
        return _diffuse_all(diffusion_graph, activated, rand_gen)
    # perform diffusion for at most "steps" rounds
    return _diffuse_k_rounds(diffusion_graph, activated, steps, rand_gen)


def _diffuse_all(graph, activated, rand_gen):
    tried_edges = set()
    layer_i_nodes = []
    layer_i_nodes.append([i for i in activated])  # prevent side effect
    while True:
        len_old = len(activated)
        (activated, activated_nodes_of_this_round, cur_tried_edges) = _diffuse_one_round(
            graph, activated, tried_edges, rand_gen
        )
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(activated) == len_old:
            break
    return layer_i_nodes


def _diffuse_k_rounds(graph, activated, steps, rand_gen):
    tried_edges = set()
    layer_i_nodes = []
    layer_i_nodes.append([i for i in activated])
    while steps > 0 and len(activated) < len(graph):
        len_old = len(activated)
        (activated, activated_nodes_of_this_round, cur_tried_edges) = _diffuse_one_round(
            graph, activated, tried_edges, rand_gen
        )
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(activated) == len_old:
            break
        steps -= 1
    return layer_i_nodes


def _diffuse_one_round(graph, activated, tried_edges, rand_gen):
    activated_nodes_of_this_round = set()
    cur_tried_edges = set()
    for s in activated:
        for nb in graph.successors(s):
            if nb in activated or (s, nb) in tried_edges or (s, nb) in cur_tried_edges:
                continue
            if _prop_success(graph, s, nb, rand_gen):
                activated_nodes_of_this_round.add(nb)
            cur_tried_edges.add((s, nb))
    activated_nodes_of_this_round = list(activated_nodes_of_this_round)
    activated.extend(activated_nodes_of_this_round)
    return activated, activated_nodes_of_this_round, cur_tried_edges


def _prop_success(graph, src, dest, rand_gen):
    return graph[src][dest]["success_prob"] <= graph[src][dest]["activation_prob"]


# Start of actual test code


def generate_random_graph_from_seed(
    n: int,
    p: float,
    directed: bool,
    include_payoff: bool,
    *,
    include_success_prob: bool = True,
    seed: int = 12345,
) -> t.Union[nx.Graph, nx.DiGraph]:
    graph = nx.fast_gnp_random_graph(n, p, directed=directed, seed=seed)

    random.seed(seed)
    if include_success_prob:
        for _, _, data in graph.edges(data=True):
            data["success_prob"] = random.random()

    if include_payoff:
        for _, data in graph.nodes(data=True):
            data["payoff"] = random.uniform(1.0, 10.0)

    return graph


# The start of the actual test cases


@pytest.mark.parametrize("seed", [12345, 505050, 2024])
def test_randomized_activation(seed: int) -> None:
    n = 100
    k = 10
    p = 0.01

    # Create random graph and initialize model
    graph = generate_random_graph_from_seed(n, p, True, False, seed=seed)
    model, _ = networkx_to_ic_model(graph, rng=seed)

    random.seed(seed)
    seeds = random.sample(list(graph.nodes), k)
    seed_probs = random.sample([1.0 for _ in range(k // 2)] + [0.0 for _ in range(k // 2)], k)

    model.set_seeds(seeds, seed_probs)

    # Check that the seeds are set correctly
    assert model.get_num_activated_nodes() == k // 2

    activated_nodes = set(model.get_activated_nodes())

    for seed_node, prob in zip(seeds, seed_probs, strict=True):
        assert (seed_node in activated_nodes) == (prob == 1.0)


@pytest.mark.parametrize("seed", [12345, 505050, 2024])
def test_model_payoffs(seed: int) -> None:
    n = 10_000
    k = 10
    p = 0.01

    # Just trying the main functions with no set thresholds
    graph = generate_random_graph_from_seed(n, p, True, True, seed=seed)
    model, _ = networkx_to_ic_model(graph, rng=seed)
    random.seed(seed)
    seeds = set(random.sample(list(graph.nodes), k))

    # Run the model
    model.set_seeds(seeds)
    model.advance_until_completion()
    payoff_score = model.compute_payoffs()

    # Compute score manually and compare
    manual_score = 0.0
    for node in model.get_activated_nodes():
        manual_score += graph.nodes[node]["payoff"]

    assert math.isclose(payoff_score, manual_score, abs_tol=0.3)


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("seed", [12345, 505050, 2024])
def test_model_rng_seed(directed: bool, seed: int) -> None:
    n = 10000
    k = 10
    p = 0.005
    num_runs = 10
    # Just trying the main functions with no set thresholds
    graph = generate_random_graph_from_seed(n, p, directed, False, seed=seed, include_success_prob=False)
    model, _ = networkx_to_ic_model(graph, rng=seed)

    random.seed(seed)
    seeds = set(random.sample(list(graph.nodes), k))

    model.set_seeds(seeds)
    activated_nodes_sets = []

    # Get activated nodes sets for 10 runs
    for _ in range(num_runs):
        model.reset_model()
        model.advance_until_completion()

        activated_nodes = set(model.get_activated_nodes())
        activated_nodes_sets.append(activated_nodes)

    # Assert that sets are different
    model.reset_model()
    model.advance_until_completion()
    assert activated_nodes != set(model.get_activated_nodes())

    # Reseed model and run again
    model.set_rng(seed)

    for activated_nodes in activated_nodes_sets:
        model.reset_model()
        model.advance_until_completion()
        assert activated_nodes == set(model.get_activated_nodes())


@pytest.mark.parametrize("directed", [True, False])
def test_model_basic(directed: bool) -> None:
    n = 10000
    k = 10
    p = 0.01
    # Just trying the main functions with no set thresholds
    graph = generate_random_graph_from_seed(n, p, directed, False)
    model, _ = networkx_to_ic_model(graph)

    # Didn't set the seeds
    model.advance_until_completion()
    assert model.get_num_activated_nodes() == 0

    seeds = set(random.sample(list(graph.nodes), k))
    model.set_seeds(seeds)
    assert set(model.get_newly_activated_nodes()) == seeds
    assert set(model.get_activated_nodes()) == seeds

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
    set_act_prob_fn: t.Optional[t.Callable[[Graph], None]],
    directed: bool,
) -> None:
    n = 1000
    p = 0.01
    k = 10

    test_graph = generate_random_graph_from_seed(n, p, directed, False)
    indep_cascade_prob = 0.2

    if set_act_prob_fn is not None:
        set_act_prob_fn(test_graph)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    # If this function is None, we use the uniform activation threshold.
    if set_act_prob_fn is None:
        activated_nodes_levels = independent_cascade(test_graph, seeds, activation_prob=indep_cascade_prob)
    else:
        activated_nodes_levels = independent_cascade(test_graph, seeds)

    # Set up the model
    if set_act_prob_fn is None:
        model, _ = networkx_to_ic_model(test_graph, activation_prob=indep_cascade_prob, _include_succcess_prob=True)
    else:
        model, _ = networkx_to_ic_model(test_graph, _include_succcess_prob=True)

    model.set_seeds(seeds)
    seen_set = set()

    # Run twice to check that the reset works
    for _ in range(2):
        assert model.get_num_activated_nodes() == len(seeds)
        assert len(set(model.get_newly_activated_nodes())) == len(seeds)

        for node_level in activated_nodes_levels:
            assert set(node_level) == set(model.get_newly_activated_nodes())
            seen_set |= set(node_level)

            model.advance_model()

        assert seen_set == set(model.get_activated_nodes())
        model.reset_model()


@pytest.mark.parametrize("directed", [True, False])
def test_basic_2(directed: bool) -> None:
    n = 1000
    p = 0.01
    k = 10
    test_graph = generate_random_graph_from_seed(n, p, directed, False)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    activated_nodes_levels = independent_cascade(test_graph, seeds)
    num_seen = sum(len(level) for level in activated_nodes_levels)

    # Set up the model
    model, _ = networkx_to_ic_model(test_graph, _include_succcess_prob=True)
    model.set_seeds(seeds)
    model.advance_until_completion()

    assert num_seen == model.get_num_activated_nodes()


def test_invalid_seed_error() -> None:
    n = 1000
    p = 0.01
    test_graph = generate_random_graph_from_seed(n, p, False, False)

    model, _ = networkx_to_ic_model(test_graph)

    with pytest.raises(ValueError):
        model.set_seeds({-1})  # Insert a seed not in the graph

    with pytest.raises(ValueError):
        model.set_seeds({0, 1, 4, n})  # Insert a seed not in the graph

    with pytest.raises(ValueError):
        model.set_seeds({0.1})  # type: ignore[arg-type]
        model.advance_until_completion()


def test_duplicate_arguments() -> None:
    n = 1000
    p = 0.01
    test_graph = generate_random_graph_from_seed(n, p, False, False)

    for _, _, data in test_graph.edges(data=True):
        data["activation_prob"] = random.random()

    with pytest.raises(ValueError):
        networkx_to_ic_model(test_graph, activation_prob=0.1)


def compute_graph_marginal_gain(
    graph: Graph,
    new_set_lists: t.List[t.List[int]],
) -> float:
    result = 0.0

    new_set = set(item for sublist in new_set_lists for item in sublist)

    for elem in new_set:
        result += graph.nodes[elem].get("payoff", 1.0)

    return result


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("include_payoffs", [True, False])
@pytest.mark.parametrize("seed", [12345, 505050, 2024])
def test_marginal_gain(directed: bool, include_payoffs: bool, seed: int) -> None:
    """
    Test the marginal gain function under a couple of parameter settings.
    """

    n = 1000
    p = 0.01
    k = 10
    test_graph = generate_random_graph_from_seed(n, p, directed, include_payoffs, seed=seed)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    # Set up the model
    model, _ = networkx_to_ic_model(test_graph, _include_succcess_prob=True)

    result = model.compute_marginal_gains(seeds, [], 1000)[0]
    total_activated = compute_graph_marginal_gain(test_graph, independent_cascade(test_graph, seeds))

    assert math.isclose(result, total_activated, abs_tol=0.05)

    results: t.List[float] = model.compute_marginal_gains([], seeds, 1000)

    assert math.isclose(results[0], 0.0)
    assert math.isclose(sum(results), result, abs_tol=0.05)

    set_so_far: t.List[int] = []

    for seed, result in zip(seeds, results[1:], strict=False):
        without_new_seed_total = compute_graph_marginal_gain(test_graph, independent_cascade(test_graph, set_so_far))

        with_new_seed_total = compute_graph_marginal_gain(
            test_graph, independent_cascade(test_graph, set_so_far + [seed])
        )

        marg_gain = with_new_seed_total - without_new_seed_total

        assert math.isclose(result, marg_gain, abs_tol=0.05)
        set_so_far.append(seed)
