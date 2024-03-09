import itertools as it
import random
import time
import typing as t

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
import pandas as pd
from cynetdiff.utils import networkx_to_ic_model
from tqdm import trange

DiffusionGraphT = nx.Graph | nx.DiGraph
SeedSetT = set[int]
DiffusionFuncT = t.Callable[[DiffusionGraphT, SeedSetT, int], float]


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

                    if succ_prob <= data.get("act_prob", 0.1):
                        visited.add(child)
                        current_layer.append(child)

    return res


def diffuse_python(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    if not graph.is_directed():
        graph = graph.to_directed()

    res = 0.0
    seeds_list = list(seeds)
    for _ in trange(num_samples):
        res += sum(len(level) for level in independent_cascade(graph, seeds_list))

    return res / num_samples


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


def diffuse_ndlib(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    model = ep.IndependentCascadesModel(graph)

    config = mc.Configuration()

    # Assume that thresholds were already set.
    for u, v, data in graph.edges(data=True):
        config.add_edge_configuration("threshold", (u, v), data["threshold"])

    # Don't randomly infect anyone to start, just use given seeds.
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
        # print(curr_iter_count[2])
        total_infected += curr_iter_count[2]

    return total_infected / num_samples


def time_diffusion(
    func: DiffusionFuncT, graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int
):
    start = time.perf_counter()
    diffused = func(graph, seeds, num_samples)
    end = time.perf_counter()
    return func.__name__, diffused, end - start


def get_graphs() -> list[tuple[str, DiffusionGraphT]]:
    """Get graphs with accompanying names for diffusion benchmarks."""
    n_values = [20, 10_000]
    frac_values = [0.002, 0.007]
    threshold_values = [0.1]

    res = []

    for n, frac, threshold in it.product(n_values, frac_values, threshold_values):
        graph = nx.fast_gnp_random_graph(n, frac)

        nx.set_edge_attributes(graph, threshold, "threshold")

        name = f"Gnp, n = {n}, p = {frac}"

        res.append((name, graph))

    return res


def main() -> None:
    # TODO parameterize these benchmarks with
    # https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.random_graphs

    seed_values = [1, 2, 5, 10, 20]
    num_samples_values = [10, 1000]
    diffusion_functions = [diffuse_CyNetDiff, diffuse_python, diffuse_ndlib]

    underlying_graphs = get_graphs()

    results = []

    for (graph_name, graph), num_seeds, num_samples in it.product(
        underlying_graphs, seed_values, num_samples_values
    ):
        seeds = set(random.sample(list(graph.nodes()), num_seeds))

        for func in diffusion_functions:
            # Print the parameters before running
            print(
                f"Running {func.__name__} on {graph_name} with {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges, {num_seeds} seeds, {num_samples} samples."
            )

            model_name, diffused, time_taken = time_diffusion(
                func, graph, seeds, num_samples
            )

            # Print the results after running
            print(
                f"Completed {model_name}: Diffused = {diffused}, Time = {time_taken:.4f} seconds\n"
            )

            results.append(
                {
                    "graph name": graph_name,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "num seeds": num_seeds,
                    "num samples": num_samples,
                    "model": model_name,
                    "diffused": diffused,
                    "time": time_taken,
                }
            )

    df = pd.DataFrame(results)
    print(df)


if __name__ == "__main__":
    main()
