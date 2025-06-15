import array
import copy
import math
import random
import typing as t

import networkx as nx
import pytest

from cynetdiff.utils import networkx_to_lt_model

#    Code below adapted from code by
#    Hung-Hsuan Chen <hhchen@psu.edu>
#    All rights reserved.
#    BSD license.


def linear_threshold(
    graph: t.Union[nx.Graph, nx.DiGraph], seeds: t.Iterable[int], steps: int = 0
) -> t.List[t.List[int]]:
    """Return the active nodes of each diffusion step by linear threshold model

    Parameters
    ----------
    graph : networkx graph
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
    """
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raise Exception("linear_threshold() is not defined for graphs with multiedges.")

    # make sure the seeds are in the graph
    for s in seeds:
        if s not in graph.nodes():
            raise Exception("seed", s, "is not in graph")

    # change to directed graph
    if not graph.is_directed():
        diffusion_graph = graph.to_directed()
    else:
        diffusion_graph = copy.deepcopy(graph)

    # init thresholds
    for n in diffusion_graph.nodes():
        if "threshold" not in diffusion_graph._node[n]:
            diffusion_graph._node[n]["threshold"] = random.random()
        elif diffusion_graph._node[n]["threshold"] > 1:
            raise Exception(
                "node threshold:",
                diffusion_graph._node[n]["threshold"],
                "cannot be larger than 1",
            )

    # init influences
    in_deg = diffusion_graph.in_degree()
    for e in diffusion_graph.edges():
        if "influence" not in diffusion_graph[e[0]][e[1]]:
            diffusion_graph[e[0]][e[1]]["influence"] = 1.0 / in_deg[e[1]]
        elif diffusion_graph[e[0]][e[1]]["influence"] > 1:
            raise Exception(
                "edge influence:",
                diffusion_graph[e[0]][e[1]]["influence"],
                "cannot be larger than 1",
            )

    # perform diffusion
    activated = copy.deepcopy(seeds)
    if steps <= 0:
        # perform diffusion until no more nodes can be activated
        return _diffuse_all(diffusion_graph, activated)
    # perform diffusion for at most "steps" rounds only
    return _diffuse_k_rounds(diffusion_graph, activated, steps)


def _diffuse_all(graph, activated):
    layer_i_nodes = []
    layer_i_nodes.append([i for i in activated])
    while True:
        len_old = len(activated)
        activated, activated_nodes_of_this_round = _diffuse_one_round(graph, activated)
        layer_i_nodes.append(activated_nodes_of_this_round)
        if len(activated) == len_old:
            break
    return layer_i_nodes


def _diffuse_k_rounds(graph, activated, steps):
    layer_i_nodes = []
    layer_i_nodes.append([i for i in activated])
    while steps > 0 and len(activated) < len(graph):
        len_old = len(activated)
        activated, activated_nodes_of_this_round = _diffuse_one_round(graph, activated)
        layer_i_nodes.append(activated_nodes_of_this_round)
        if len(activated) == len_old:
            break
        steps -= 1
    return layer_i_nodes


def _diffuse_one_round(graph, activated):
    activated_nodes_of_this_round = set()
    for s in activated:
        nbs = graph.successors(s)

        for nb in nbs:
            if nb in activated:
                continue
            active_nb = list(set(graph.predecessors(nb)).intersection(set(activated)))

            if _influence_sum(graph, active_nb, nb) >= graph._node[nb]["threshold"]:
                activated_nodes_of_this_round.add(nb)
    activated.extend(list(activated_nodes_of_this_round))
    return activated, list(activated_nodes_of_this_round)


def _influence_sum(graph, froms, to):
    influence_sum = 0.0
    for f in froms:
        influence_sum += graph[f][to]["influence"]

    return influence_sum


def generate_random_graph_from_seed(
    n: int,
    p: float,
    directed: bool,
    include_influence: bool,
    include_payoff: bool,
    *,
    seed: int = 12345,
) -> nx.DiGraph:
    graph = nx.fast_gnp_random_graph(n, p, seed=seed, directed=directed).to_directed()

    random.seed(seed)
    for _, data in graph.nodes(data=True):
        data["threshold"] = random.random()

        if include_payoff:
            data["payoff"] = random.uniform(1.0, 10.0)

    if include_influence:
        for _, _, data in graph.edges(data=True):
            data["influence"] = random.random()

        for node in graph.nodes():
            pred_dicts = tuple(graph[predecessor][node] for predecessor in graph.predecessors(node))

            # Get sum of influences and then normalize.
            inf_sum = 0.0
            for pred_dict in pred_dicts:
                inf_sum += pred_dict["influence"]
            for pred_dict in pred_dicts:
                pred_dict["influence"] /= inf_sum

    return graph


def get_thresholds(graph: nx.DiGraph) -> array.array:
    thresholds = array.array("f")

    for _, data in sorted(graph.nodes(data=True)):
        threshold = data["threshold"]
        assert 0.0 <= threshold <= 1.0
        thresholds.append(threshold)

    return thresholds


# Start of actual test code


@pytest.mark.parametrize("seed", [12345, 505050, 2024])
def test_randomized_activation(seed: int) -> None:
    n = 100
    k = 10
    p = 0.01

    # Create random graph and initialize model
    graph = generate_random_graph_from_seed(n, p, True, False, False, seed=seed)
    model, _ = networkx_to_lt_model(graph, rng=seed)

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
    graph = generate_random_graph_from_seed(n, p, True, False, True, seed=seed)
    model, _ = networkx_to_lt_model(graph, rng=seed)
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

    assert math.isclose(payoff_score, manual_score, abs_tol=0.05)


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("seed", [12345, 505050, 2024])
def test_model_rng_seed(directed: bool, seed: int) -> None:
    n = 10000
    k = 10
    p = 0.01
    num_runs = 10
    # Just trying the main functions with no set thresholds
    graph = generate_random_graph_from_seed(n, p, directed, False, False, seed=seed)
    model, _ = networkx_to_lt_model(graph, rng=seed)

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
@pytest.mark.parametrize("nondefault_influence", [True, False])
def test_specific_model(directed: bool, nondefault_influence: bool) -> None:
    n = 500
    p = 0.05
    k = 10
    test_graph = generate_random_graph_from_seed(
        n,
        p,
        directed=directed,
        include_influence=nondefault_influence,
        include_payoff=False,
    )

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    activated_nodes_levels = linear_threshold(test_graph, seeds)

    model, _ = networkx_to_lt_model(test_graph)

    model.set_seeds(seeds)
    seen_set = set()

    assert model.get_num_activated_nodes() == len(seeds)
    assert len(sorted(model.get_newly_activated_nodes())) == len(seeds)

    model._assign_thresholds(get_thresholds(test_graph))

    for node_level in activated_nodes_levels:
        model_set = sorted(model.get_newly_activated_nodes())
        node_set = sorted(node_level)

        assert model_set == node_set
        seen_set |= set(node_level)
        model.advance_model()

    assert seen_set == set(model.get_activated_nodes())

    # Resetting should change the total number of activated nodes
    model.reset_model()
    model.advance_until_completion()

    assert seen_set != set(model.get_activated_nodes())


def test_invalid_seed_error() -> None:
    n = 1000
    p = 0.01
    test_graph = generate_random_graph_from_seed(
        n,
        p,
        False,
        True,
        False,
    )

    model, _ = networkx_to_lt_model(test_graph)

    with pytest.raises(ValueError):
        model.set_seeds({-1})  # Insert a seed not in the graph

    with pytest.raises(ValueError):
        model.set_seeds({0, 1, 4, n})  # Insert a seed not in the graph

    with pytest.raises(ValueError):
        model.set_seeds({0.1})  # type: ignore[arg-type]
        model.advance_until_completion()


def compute_graph_marginal_gain(
    graph: nx.DiGraph,
    new_set_lists: t.List[t.List[int]],
) -> float:
    result = 0.0

    new_set = set(item for sublist in new_set_lists for item in sublist)

    for elem in new_set:
        result += graph.nodes[elem].get("payoff", 1.0)

    return result


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("nondefault_influence", [True, False])
@pytest.mark.parametrize("include_payoffs", [True, False])
@pytest.mark.parametrize("seed", [12345, 505050])
def test_marginal_gain(directed: bool, nondefault_influence: bool, include_payoffs: bool, seed: int) -> None:
    """
    Test the marginal gain function under a couple of parameter settings.
    """

    n = 500
    p = 0.01
    k = 10
    test_graph = generate_random_graph_from_seed(n, p, directed, nondefault_influence, include_payoffs, seed=seed)

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    # Set up the model
    model, _ = networkx_to_lt_model(test_graph)
    node_thresholds = get_thresholds(test_graph)

    result = model.compute_marginal_gains(seeds, [], 1, _node_thresholds=node_thresholds)[0]
    total_activated = compute_graph_marginal_gain(test_graph, linear_threshold(test_graph, seeds))

    assert math.isclose(result, total_activated, abs_tol=0.05)

    results: t.List[float] = model.compute_marginal_gains([], seeds, 1, _node_thresholds=node_thresholds)

    assert math.isclose(results[0], 0.0)
    assert math.isclose(sum(results), result, abs_tol=0.05)

    set_so_far: t.List[int] = []

    for seed, result in zip(seeds, results[1:], strict=False):
        without_new_seed_total = compute_graph_marginal_gain(test_graph, linear_threshold(test_graph, set_so_far))

        with_new_seed_total = compute_graph_marginal_gain(test_graph, linear_threshold(test_graph, set_so_far + [seed]))

        marg_gain = with_new_seed_total - without_new_seed_total

        assert math.isclose(result, marg_gain, abs_tol=0.05)
        set_so_far.append(seed)
