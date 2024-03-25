import random
import time
import typing as t
import warnings
from collections import defaultdict
from functools import wraps

import matplotlib.pyplot as plt
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
import numpy as np
from coloraide import Color
from cynetdiff.utils import networkx_to_ic_model
from tqdm.notebook import trange

DiffusionGraphT = t.Union[nx.Graph, nx.DiGraph]
SeedSetT = set[int]

# Timing decorator
# from https://stackoverflow.com/a/27737385


def timing(f: t.Callable) -> t.Callable:
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.perf_counter()
        result = f(*args, **kw)
        te = time.perf_counter()
        # print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))

        return result

    return wrap


# Pure Python diffusion code


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


@timing
def diffuse_python(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    if not graph.is_directed():
        graph = graph.to_directed()

    res = 0.0
    seeds = seeds
    for _ in trange(num_samples):
        res += float(sum(len(level) for level in independent_cascade(graph, seeds)))

    return res / num_samples


# NDlib diffusion wrapper


@timing
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


# CyNetDiff diffusion wrapper


@timing
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


# Mini benchmark function


def simple_benchmark(
    graph: DiffusionGraphT,
    *,
    num_seeds: int = 10,
    num_trials: int = 1_000,
    backends_to_run: t.List[t.Literal["cynetdiff", "ndlib", "python"]] = [
        "cynetdiff",
        "ndlib",
        "python",
    ],
) -> None:
    """
    A simple benchmark function for three different diffusion
    implementations.
    """

    # Number of seed nodes
    print(f"Number of randomly chosen seed nodes: {num_seeds:_}")
    print(f"Graph nodes: {graph.number_of_nodes():_}")
    print(f"Graph edges: {graph.number_of_edges():_}")
    print(f"Number of trials: {num_trials:_}")
    seeds = random.sample(list(graph.nodes()), num_seeds)

    if "ndlib" in backends_to_run:
        print("Starting diffusion with NDlib.")
        avg_ndlib = diffuse_ndlib(graph, seeds, num_trials)

    if "python" in backends_to_run:
        print("Starting diffusion with pure Python.")
        avg_python = diffuse_python(graph, seeds, num_trials)

    if "cynetdiff" in backends_to_run:
        print("Starting diffusion with CyNetDiff.")
        avg_cynetdiff = diffuse_CyNetDiff(graph, seeds, num_trials)

    if "ndlib" in backends_to_run:
        print("NDlib avg activated:", avg_ndlib)

    if "python" in backends_to_run:
        print("Pure Python avg activated:", avg_python)

    if "cynetdiff" in backends_to_run:
        print("CyNetDiff avg activated:", avg_cynetdiff)


@timing
def diffusion_get_frequencies(
    graph: DiffusionGraphT, seeds: t.Set[int], num_trials: int = 1_000
) -> t.Dict[int, int]:
    """
    Get the frequency dict from the network diffusion.
    """

    model = networkx_to_ic_model(graph)
    model.set_seeds(seeds)

    activated_nodes_dict: t.DefaultDict[int, int] = defaultdict(lambda: 0)

    for _ in trange(num_trials):
        model.reset_model()
        model.advance_until_completion()
        activated_nodes = model.get_activated_nodes()

        for node in activated_nodes:
            activated_nodes_dict[node] += 1

    return dict(activated_nodes_dict)


def plot_num_nodes_activated(
    model_data_tuples: t.List[t.Tuple[str, DiffusionGraphT, t.Set[int]]],
    *,
    num_trials: int = 1_000,
    plot_iqr: bool = False,
) -> None:
    """
    Create a plot of number of nodes infected for CyNetDiff models across multiple trials.
    If plot_iqr is True, plots the interquartile range and horizontal lines.
    """

    # Generate distinct colors for each graph using Coloraide
    colors = [Color.random("srgb") for _ in model_data_tuples]

    # Generate distinct lighter-colors for each graph for IQR plotting
    light_colors = []
    for color_str in colors:
        color = Color(color_str, "srgb")
        # I think this sets the transparency.
        color[3] = 0.5
        light_colors.append(color)

    max_length = 0
    all_graphs_data: t.List[t.Tuple[str, t.List[t.List[int]]]] = []

    # Get all the data for graphs
    for model_name, graph, seeds in model_data_tuples:
        model = networkx_to_ic_model(graph)
        model.set_seeds(seeds)

        all_trials_infected_nodes: t.List[t.List[int]] = []

        for _ in range(num_trials):
            model.reset_model()

            infected_nodes_over_time: t.List[int] = []
            previous_activated = -1
            current_activated = 0

            while previous_activated != current_activated:
                previous_activated = current_activated
                model.advance_model()
                current_activated = model.get_num_activated_nodes()
                number_infected = current_activated
                infected_nodes_over_time.append(number_infected)

            all_trials_infected_nodes.append(infected_nodes_over_time)
            max_length = max(max_length, len(infected_nodes_over_time))

        all_graphs_data.append((model_name, all_trials_infected_nodes))

    # Pad all data
    padded_all_graphs_data: t.List[t.Tuple[str, t.List[t.List[int]]]] = []
    for model_name, graph_data in all_graphs_data:
        padded_trials = [
            trial + [trial[-1]] * (max_length - len(trial)) for trial in graph_data
        ]
        padded_all_graphs_data.append((model_name, padded_trials))

    # Graph Data
    for idx, (model_name, graph_data) in enumerate(padded_all_graphs_data):
        mean_infected = np.mean(graph_data, axis=0)

        if plot_iqr:
            iqr_values = [
                np.percentile(
                    [trial[i] if i < len(trial) else trial[-1] for trial in graph_data],
                    [25, 75],
                )
                for i in range(max_length)
            ]
            lower_quartile, upper_quartile = zip(*iqr_values)

            for y in mean_infected:
                plt.axhline(y=y, color=colors[idx], linestyle="--", alpha=0.2)

            plt.fill_between(
                range(max_length),
                lower_quartile,
                upper_quartile,
                color=light_colors[idx],
                alpha=0.3,
            )

        plt.plot(mean_infected, label=f"{model_name} Mean Infected", color=colors[idx])

    plt.xlabel("Iteration")
    plt.ylabel("Number of Activated Nodes")
    plt.title("Diffusion Process Over Time")
    plt.legend()
    plt.show()


"""
Other plotting function that we implemented but didn't wind up using.

def create_plot_for_delta_nodes_infected(
    graph1: nx.Graph,
    graph2: nx.Graph = None,
    graph3: nx.Graph = None,
    graph4: nx.Graph = None,
    plot_iqr: bool = False,
):

    Create a plot of the delta of nodes infected for a CyNetDiff model across multiple trials for up to four graphs.
    If plot_iqr is True, plots the interquartile range and horizontal lines.

    graphs = [g for g in [graph1, graph2, graph3, graph4] if g is not None]
    for graph in graphs:
        model = networkx_to_ic_model(graph)
        seeds = set(random.sample(list(graph.nodes()), 100))
        model.set_seeds(seeds)
        num_trials = 1

        all_trials_delta_nodes = []
        max_length = 0

        for _ in range(num_trials):
            model.reset_model()

            delta_nodes_over_time = []
            previous_activated = -1
            current_activated = 0

            while previous_activated != current_activated:
                previous_activated = current_activated
                model.advance_model()
                current_activated = model.get_num_activated_nodes()
                delta_nodes_over_time.append(current_activated - previous_activated)

            max_length = max(max_length, len(delta_nodes_over_time))
            all_trials_delta_nodes.append(delta_nodes_over_time)

        padded_trials = [
            trial + [0] * (max_length - len(trial)) for trial in all_trials_delta_nodes
        ]

        median_delta = np.median(padded_trials, axis=0)

        if plot_iqr:
            iqr_values = [
                np.percentile(
                    [trial[i] if i < len(trial) else 0 for trial in padded_trials],
                    [25, 75],
                )
                for i in range(max_length)
            ]
            lower_quartile, upper_quartile = zip(*iqr_values)

            for y in median_delta:
                plt.axhline(y=y, color="gray", linestyle="--", alpha=0.2)

            plt.fill_between(
                range(len(lower_quartile)),
                lower_quartile,
                upper_quartile,
                color="#7daec7",
                alpha=0.3,
            )

        plt.plot(median_delta, label="Median Delta Nodes", color="#3f83a6")

    plt.xlabel("Iteration")
    plt.ylabel("Delta Nodes Infected")
    plt.title("Diffusion Process Over Time")
    plt.legend()
    plt.show()
"""
