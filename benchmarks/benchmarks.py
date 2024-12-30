import gzip
import itertools as it
import random
import time
import typing as t

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as Mc
import networkx as nx
import pandas as pd
import pooch
from tqdm import trange

from cynetdiff.utils import (
    networkx_to_ic_model,
    set_activation_random_sample,
    set_activation_weighted_cascade,
)

DiffusionGraphT = t.Union[nx.Graph, nx.DiGraph]
SeedSetT = set[int]
DiffusionFuncT = t.Callable[[DiffusionGraphT, SeedSetT, int], float]


def get_facebook_graph() -> nx.Graph:
    # Data from: https://snap.stanford.edu/data/index.html#socnets

    graph_data_file = pooch.retrieve(
        progressbar=True,
        url="https://snap.stanford.edu/data/facebook_combined.txt.gz",
        known_hash="125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7",
    )

    with gzip.open(graph_data_file, "r") as file:
        fb_network = nx.read_edgelist(
            file,
            create_using=nx.DiGraph(),
            nodetype=int,
        )

    return fb_network


def independent_cascade(graph: DiffusionGraphT, seeds: SeedSetT) -> list[list[int]]:
    """
    A basic pure-python implementation of independent cascade. Optimized so we can get
    a baseline reading on the best possible speed for this algorithm in pure-Python.
    """
    if not graph.is_directed():
        graph = graph.to_directed()

    visited = set(seeds)

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    res = []
    current_layer = list(seeds)

    while current_layer:
        res.append(current_layer)
        current_layer = []

        for next_node in res[-1]:
            for child, data in graph[next_node].items():
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


def diffuse_cynetdiff(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    model, _ = networkx_to_ic_model(graph)
    model.set_seeds(seeds)

    total_activated = 0.0

    for _ in trange(num_samples):
        model.reset_model()
        model.advance_until_completion()
        total_activated += model.get_num_activated_nodes()

    return total_activated / num_samples


def diffuse_ndlib(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    model = ep.IndependentCascadesModel(graph)

    config = Mc.Configuration()

    # Assume that thresholds were already set.
    for u, v, data in graph.edges(data=True):
        config.add_edge_configuration("threshold", (u, v), data["activation_prob"])

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

        total_infected += curr_iter_count[2]

    return total_infected / num_samples


def time_diffusion(func: DiffusionFuncT, graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int):
    start = time.perf_counter()
    diffused = func(graph, seeds, num_samples)
    end = time.perf_counter()
    return func.__name__, diffused, end - start


def get_graphs() -> list[tuple[str, DiffusionGraphT]]:
    """Get graphs with accompanying names for diffusion benchmarks."""

    # TODO parameterize these benchmarks with
    # https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.random_graphs

    n_values = [2_000, 10_000]
    frac_values = [0.002, 0.007]

    res = []

    for n, frac in it.product(n_values, frac_values):
        graph = nx.fast_gnp_random_graph(n, frac)

        name = f"Gnp, n = {n}, p = {frac}"

        res.append((name, graph))

    k_vals = [4, 10]

    for n, k, frac in it.product(n_values, k_vals, frac_values):
        graph = nx.watts_strogatz_graph(n, k, frac)

        name = f"Watts–Strogatz small-world graph, n = {n}, k = {k}, p = {frac}"
        res.append((name, graph))

    res.append(("Facebook graph", get_facebook_graph()))

    return res


def main() -> None:
    seed_values = [1, 5, 10, 20, 50, 100]
    num_samples_values = [1_000]
    diffusion_functions = [diffuse_cynetdiff, diffuse_python, diffuse_ndlib]
    weighting_schemes = ["Trivalency", "Uniform", "Weighted Cascade"]

    underlying_graphs = get_graphs()

    results = []

    workloads = list(it.product(underlying_graphs, seed_values, num_samples_values, weighting_schemes))

    print(f"Running {len(workloads)} benchmarks.")
    i = 0

    for (graph_name, graph), num_seeds, num_samples, scheme in workloads:
        i += 1
        print(f"Starting benchmark {i}/{len(workloads)}")
        if scheme == "Trivalency":
            graph = graph.copy()
            set_activation_random_sample(graph, {0.1, 0.01, 0.001})
        elif scheme == "Uniform":
            graph = graph.copy()
            set_activation_random_sample(graph, {0.1})
        elif scheme == "Weighted Cascade":
            graph = graph.to_directed()
            set_activation_weighted_cascade(graph)

        seeds = set(random.sample(list(graph.nodes()), num_seeds))
        res_dict = {
            "graph name": graph_name,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "num seeds": num_seeds,
            "num samples": num_samples,
            "weighting scheme": scheme,
        }
        min_time_taken = float("inf")
        for func in diffusion_functions:
            # Print the parameters before running
            print(
                f"Running {func.__name__} on {graph_name} with {graph.number_of_nodes():,d} nodes, "
                f"{graph.number_of_edges():,d} edges, {num_seeds} seeds, {num_samples} samples."
            )

            model_name, diffused, time_taken = time_diffusion(func, graph, seeds, num_samples)

            # Print the results after running
            print(f"Completed {model_name}: Diffused = {diffused}, Time = {time_taken:.4f} seconds\n")
            min_time_taken = min(min_time_taken, time_taken)
            res_dict[func.__name__] = time_taken

        for func in diffusion_functions:
            name = func.__name__
            time_taken = res_dict[name]
            res_dict[f"{name} normalized"] = time_taken / min_time_taken

        results.append(res_dict)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("benchmark_results.csv")


if __name__ == "__main__":
    main()
