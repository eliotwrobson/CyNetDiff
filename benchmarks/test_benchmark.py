import random
import typing as t

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as Mc
import networkx as nx
import pytest

from cynetdiff.utils import networkx_to_ic_model, set_activation_random_sample

DiffusionGraphT = t.Union[nx.Graph, nx.DiGraph]
DiffusionFunctionT = t.Callable[[DiffusionGraphT, set[int], int], float]

# Start of helper functions for diffusion algorithms


def diffuse_cynetdiff(graph: DiffusionGraphT, seeds: set[int], num_samples: int) -> float:
    model, _ = networkx_to_ic_model(graph)
    model.set_seeds(seeds)

    return model.compute_marginal_gains(seeds, [], num_samples)[0]


def diffuse_ndlib(graph: DiffusionGraphT, seeds: set[int], num_samples: int) -> float:
    model = ep.IndependentCascadesModel(graph)

    config = Mc.Configuration()

    # Assume that thresholds were already set.
    for u, v, data in graph.edges(data=True):
        config.add_edge_configuration("threshold", (u, v), data["activation_prob"])

    # Don't randomly infect anyone to start, just use given seeds.
    config.add_model_parameter("fraction_infected", 0.0)

    model.set_initial_status(config)
    total_infected = 0.0

    for _ in range(num_samples):
        model.reset(seeds)
        prev_iter_count = model.iteration()["node_count"]
        curr_iter_count = model.iteration()["node_count"]

        while prev_iter_count != curr_iter_count:
            prev_iter_count = curr_iter_count
            curr_iter_count = model.iteration()["node_count"]

        total_infected += curr_iter_count[2]

    return total_infected / num_samples


def make_diffusion_graph(n: int, p: float, seed: int) -> DiffusionGraphT:
    graph = nx.erdos_renyi_graph(n, p, seed=seed)
    set_activation_random_sample(graph, {0.1})
    return graph


# Benchmark function for the three different diffusion algorithms


@pytest.mark.parametrize(
    "diffusion_function",
    [diffuse_cynetdiff, diffuse_ndlib],
    ids=["CyNetDiff", "ndlib"],
)
@pytest.mark.parametrize(
    "n, p",
    [(1_000, 0.01), (10_000, 0.01)],
)
@pytest.mark.parametrize(
    "num_samples",
    [10, 100],
)
def test_speed(
    benchmark,
    diffusion_function: DiffusionFunctionT,
    n: int,
    p: float,
    num_samples: int,
) -> None:
    random_seed = 12345
    num_seeds = 10

    diffusion_graph = make_diffusion_graph(n, p, random_seed)

    random.seed(random_seed)
    seeds = set(random.sample(list(diffusion_graph.nodes()), num_seeds))

    benchmark.group = f"n={n}, p={p}, num_samples={num_samples}"
    benchmark(diffusion_function, diffusion_graph, seeds, num_samples)
