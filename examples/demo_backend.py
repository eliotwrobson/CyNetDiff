import random
import typing as t
import warnings

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
from cynetdiff.utils import networkx_to_ic_model
from tqdm.notebook import trange

DiffusionGraphT = t.Union[nx.Graph, nx.DiGraph]
SeedSetT = set[int]


def independent_cascade(G: DiffusionGraphT, seeds: SeedSetT) -> list[list[int]]:
    """
    A basic pure-python implementation of independent cascade. Optimized so we can get
    a baseline reading on the best possible speed for this algorithm in pure-Python.
    """
    if not G.is_directed():
        G = G.to_directed()

    visited = set(seeds)

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    res = []
    current_layer = list(seeds)

    while current_layer:
        res.append(current_layer)
        current_layer = []

        for next_node in res[-1]:
            for child, data in G[next_node].items():
                if child not in visited:
                    # Lazy getter to deal with not having this set but still being
                    # efficient
                    succ_prob = data.get("success_prob")
                    if succ_prob is None:
                        succ_prob = random.random()

                    if succ_prob <= data.get("activation_prob", 0.1):
                        visited.add(child)
                        current_layer.append(child)

    return res


def diffuse_python(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    if not graph.is_directed():
        graph = graph.to_directed()

    res = 0.0
    seeds = seeds
    for _ in trange(num_samples):
        res += float(sum(len(level) for level in independent_cascade(graph, seeds)))

    return res / num_samples


def diffuse_ndlib(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    model = ep.IndependentCascadesModel(graph)

    config = mc.Configuration()

    # Assume that thresholds were already set.
    for u, v, data in graph.edges(data=True):
        config.add_edge_configuration("threshold", (u, v), data["activation_prob"])

    # Don't randomly infect anyone to start, just use given seeds.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config.add_model_parameter("fraction_infected", 0.0)
        model.set_initial_status(config)

    total_infected = 0.0

    for _ in trange(num_samples):
        model.reset(seeds)
        prev_iter_count = model.iteration()["node_count"]
        curr_iter_count = model.iteration()["node_count"]

        while prev_iter_count != curr_iter_count:
            prev_iter_count = curr_iter_count
            curr_iter_count = model.iteration()["node_count"]

        total_infected += curr_iter_count[2]

    return total_infected / num_samples


def diffuse_CyNetDiff(
    graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int
) -> float:
    model = networkx_to_ic_model(graph)
    model.set_seeds(seeds)

    total_activated = 0.0

    for _ in trange(num_samples):
        model.reset_model()
        model.advance_until_completion()
        total_activated += model.get_num_activated_nodes()

    return total_activated / num_samples
