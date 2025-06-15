#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import gzip
import heapq
import random
import time
import typing as t
import warnings
from collections import defaultdict
from functools import wraps

import matplotlib.pyplot as plt
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as Mc
import networkx as nx
import numpy as np
import pooch
from coloraide import Color
from tqdm.notebook import tqdm, trange

from cynetdiff.models import DiffusionModel
from cynetdiff.utils import networkx_to_ic_model

DiffusionGraphT = t.Union[nx.Graph, nx.DiGraph]
SeedSetT = set[int]
MethodsT = t.Literal["cynetdiff", "ndlib", "python"]

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


def diffuse_python(
    graph: DiffusionGraphT,
    seeds: SeedSetT,
    num_samples: int,
    *,
    progress_bar: bool = True,
) -> float:
    if not graph.is_directed():
        graph = graph.to_directed()

    range_obj = trange if progress_bar else range

    res = 0.0
    seeds = seeds
    for _ in range_obj(num_samples):  # type: ignore
        res += float(sum(len(level) for level in independent_cascade(graph, seeds)))

    return res / num_samples


# NDlib diffusion wrapper


@timing
def diffuse_ndlib(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    model = ep.IndependentCascadesModel(graph)

    config = Mc.Configuration()

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
def diffuse_cynetdiff(graph: DiffusionGraphT, seeds: SeedSetT, num_samples: int) -> float:
    model, _ = networkx_to_ic_model(graph)
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
    backends_to_run: t.Optional[t.List[MethodsT]] = None,
) -> None:
    """
    A simple benchmark function for three different diffusion
    implementations.
    """

    if backends_to_run is None:
        backends_to_run = [
            "cynetdiff",
            "ndlib",
            "python",
        ]

    # Number of seed nodes
    print(f"Number of randomly chosen seed nodes: {num_seeds:,d}")
    print(f"Graph nodes: {graph.number_of_nodes():,d}")
    print(f"Graph edges: {graph.number_of_edges():,d}")
    print(f"Number of trials: {num_trials:,d}")
    seeds = random.sample(list(graph.nodes()), num_seeds)

    if "ndlib" in backends_to_run:
        print("Starting diffusion with NDlib.")
        avg_ndlib = diffuse_ndlib(graph, seeds, num_trials)

    if "python" in backends_to_run:
        print("Starting diffusion with pure Python.")
        avg_python = timing(diffuse_python)(graph, seeds, num_trials)

    if "cynetdiff" in backends_to_run:
        print("Starting diffusion with CyNetDiff.")
        avg_cynetdiff = diffuse_cynetdiff(graph, seeds, num_trials)

    if "ndlib" in backends_to_run:
        print("NDlib computed influence:", avg_ndlib)

    if "python" in backends_to_run:
        print("Pure Python computed influence:", avg_python)

    if "cynetdiff" in backends_to_run:
        print("CyNetDiff computed influence:", avg_cynetdiff)


@timing
def diffusion_get_frequencies(graph: DiffusionGraphT, seeds: t.Set[int], num_trials: int = 1_000) -> t.Dict[int, int]:
    """
    Get the frequency dict from the network diffusion.
    """

    model, _ = networkx_to_ic_model(graph)
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
        color = Color(color_str)
        # I think this sets the transparency.
        color[3] = 0.5
        light_colors.append(color)

    max_length = 0
    all_graphs_data: t.List[t.Tuple[str, t.List[t.List[int]]]] = []

    # Get all the data for graphs
    for model_name, graph, seeds in model_data_tuples:
        model, _ = networkx_to_ic_model(graph)
        model.set_seeds(seeds)

        all_trials_infected_nodes: t.List[t.List[int]] = []

        for _ in range(num_trials):
            model.reset_model()

            infected_nodes_over_time: t.List[int] = [model.get_num_activated_nodes()]
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
        padded_trials = [trial + [trial[-1]] * (max_length - len(trial)) for trial in graph_data]
        padded_all_graphs_data.append((model_name, padded_trials))

    # Graph Data
    for idx, (model_name, graph_data) in enumerate(padded_all_graphs_data):
        mean_infected = np.mean(graph_data, axis=0)

        # print(model_name, idx, mean_infected)

        if plot_iqr:
            iqr_values = [
                np.percentile(
                    [trial[i] if i < len(trial) else trial[-1] for trial in graph_data],
                    [25, 75],
                )
                for i in range(max_length)
            ]
            lower_quartile, upper_quartile = zip(*iqr_values, strict=True)

            for y in mean_infected:
                plt.axhline(y=y, color=colors[idx], linestyle="--", alpha=0.2)

            plt.fill_between(
                range(max_length),
                lower_quartile,
                upper_quartile,
                color=light_colors[idx],
                alpha=0.3,
            )

        plt.plot(mean_infected, label=f"{model_name} Influence", color=colors[idx])

    plt.xlabel("Model Iteration")
    plt.ylabel("Activated Nodes")
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
        model, _ = networkx_to_ic_model(graph)
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

_T = t.TypeVar("_T")


class Singleton(type, t.Generic[_T]):
    """
    Singleton metaclass, adapted from here:
    https://stackoverflow.com/a/75308084/2923069
    """

    _instances: t.Dict[Singleton[_T], _T] = {}

    def __call__(cls, *args: t.Any, **kwargs: t.Any) -> _T:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class GraphDownloader(metaclass=Singleton):
    """
    Registry for all the example data stored on Sariel's site, with corresponding hashes.
    See: https://sarielhp.org/p/24/frechet_ve/examples/
    """

    __slots__ = ("__graphs", "__file_fetcher", "__registry")

    __graphs: t.Dict[str, DiffusionGraphT]
    __file_fetcher: pooch.Pooch
    __registry: t.Dict[str, str]

    def __init__(self) -> None:
        self.__registry = {
            "facebook_combined.txt.gz": "125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7",
            "twitter_combined.txt.gz": "d9f99b0e6a53b9204b8c215f41b3c10fb99a1e1e783858c012b06d0d3d4bd129",
            "wiki-Vote.txt.gz": "7d3e53626e14b8b09fb3b396bece9d481ad606bd64ceab066349ff57d4ada7fc",
            "soc-Epinions1.txt.gz": "69a2dab71fa5e3a0715487599fc16ca17ddc847379325a6c765bbad6e3e36938",
        }

        self.__file_fetcher = pooch.create(
            # Use the default cache folder for the operating system
            path=pooch.os_cache("snap_graphs"),
            base_url="http://snap.stanford.edu/data/",
            # The registry specifies the files that can be fetched
            registry=self.__registry,
        )

        self.__graphs = {}

    def list_graphs(self) -> t.List[str]:
        """
        Get list of graphs that can be returned by this downloader.
        """

        return list(self.__registry.keys())

    def get_graph(self, name: str) -> DiffusionGraphT:
        """
        Get the graph named "name" from this registry.
        """

        if name not in self.__registry:
            raise ValueError(f'File with name "{name}" not found in registry.')

        if name not in self.__graphs:
            graph_data_file = self.__file_fetcher.fetch(name, progressbar=True)

            with gzip.open(graph_data_file, "r") as file:
                new_graph = nx.read_edgelist(
                    file,
                    create_using=nx.Graph(),
                    nodetype=int,
                )

            self.__graphs[name] = new_graph

        # Return a copy so that in-place operations don't mess up the cached graphs.
        return self.__graphs[name].copy()


# Celf algorithm


def compute_marginal_gain(
    cynetdiff_model: DiffusionModel,
    ndlib_model: t.Any,
    graph: DiffusionGraphT,
    new_node: int,
    seeds: set[int],
    num_trials: int,
    method: MethodsT,
) -> float:
    """
    Compute the marginal gain in the spread of influence by adding a new node to the set of seed nodes,
    by summing the differences of spreads for each trial and then taking the average.

    Parameters:
    - model: The model used for simulating the spread of influence.
    - new_node: The new node to consider adding to the set of seed nodes.
    - seeds: The current set of seed nodes.
    - num_trials: The number of trials to average the spread of influence over.

    Returns:
    - The average marginal gain in the spread of influence by adding the new node.
    """

    if method == "cynetdiff":
        original_spread = 0
        new_spread = 0
        # If no seeds at the beginning, original spread is always just zero.
        if len(seeds) > 0:
            cynetdiff_model.set_seeds(seeds)

            for _ in range(num_trials):
                cynetdiff_model.reset_model()
                cynetdiff_model.advance_until_completion()
                original_spread += cynetdiff_model.get_num_activated_nodes()

        new_seeds = seeds.union({new_node})
        cynetdiff_model.set_seeds(new_seeds)

        for _ in range(num_trials):
            cynetdiff_model.reset_model()
            cynetdiff_model.advance_until_completion()
            new_spread += cynetdiff_model.get_num_activated_nodes()

        return (new_spread - original_spread) / num_trials

    elif method == "ndlib":
        total_activated_old = 0.0
        total_activated_new = 0.0

        if len(seeds) > 0:
            for _ in range(num_trials):
                ndlib_model.reset(seeds)
                prev_iter_count = ndlib_model.iteration()["node_count"]
                curr_iter_count = ndlib_model.iteration()["node_count"]

                while prev_iter_count != curr_iter_count:
                    prev_iter_count = curr_iter_count
                    curr_iter_count = ndlib_model.iteration()["node_count"]

                total_activated_old += curr_iter_count[2]

        new_seeds = seeds.union({new_node})

        for _ in range(num_trials):
            ndlib_model.reset(new_seeds)
            prev_iter_count = ndlib_model.iteration()["node_count"]
            curr_iter_count = ndlib_model.iteration()["node_count"]

            while prev_iter_count != curr_iter_count:
                prev_iter_count = curr_iter_count
                curr_iter_count = ndlib_model.iteration()["node_count"]

            total_activated_new += curr_iter_count[2]

        return (total_activated_new - total_activated_old) / num_trials

    elif method == "python":
        old_val = 0.0

        if len(seeds) > 0:
            old_val = diffuse_python(graph, seeds, num_trials, progress_bar=False)

        new_val = diffuse_python(graph, seeds.union({new_node}), num_trials, progress_bar=False)

        return new_val - old_val

    else:
        raise ValueError(f'Invalid method "{method}"')


@timing
def celf(
    graph: DiffusionGraphT, k: int, method: MethodsT, num_trials: int = 1_000
) -> t.Tuple[t.Set[int], t.List[float]]:
    """
    Input: graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    Code adapted from this blog post:
    https://hautahi.com/im_greedycelf
    """

    print(f"Starting CELF algorithm with {method} backend.")
    # Make cynetdiff model
    cynetdiff_model, _ = networkx_to_ic_model(graph)

    # NDLib Model
    ndlib_model = ep.IndependentCascadesModel(graph)

    config = Mc.Configuration()

    # Assume that thresholds were already set.
    for u, v, data in graph.edges(data=True):
        config.add_edge_configuration("threshold", (u, v), data["activation_prob"])

    # Don't randomly infect anyone to start, just use given seeds.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config.add_model_parameter("fraction_infected", 0.0)
        ndlib_model.set_initial_status(config)

    # Prepare graph
    dir_graph = graph
    if not dir_graph.is_directed():
        dir_graph = dir_graph.to_directed()

    # Run the CELF algorithm
    marg_gain = []

    # First, compute all marginal gains
    print("Computing initial marginal gains.")
    for node in tqdm(list(dir_graph.nodes())):
        marg_gain.append(
            (
                -compute_marginal_gain(
                    cynetdiff_model,
                    ndlib_model,
                    dir_graph,
                    node,
                    set(),
                    num_trials,
                    method,
                ),
                node,
            )
        )

    heapq.heapify(marg_gain)

    max_mg, selected_node = heapq.heappop(marg_gain)
    seeds = {selected_node}
    spread = -max_mg
    spreads = [spread]

    print("Performing greedy selection.")
    for _ in trange(k - 1):
        while True:
            current_mg, current_node = heapq.heappop(marg_gain)
            new_mg_neg = -compute_marginal_gain(
                cynetdiff_model,
                ndlib_model,
                dir_graph,
                current_node,
                seeds,
                num_trials,
                method,
            )

            if new_mg_neg <= current_mg:
                break
            else:
                heapq.heappush(marg_gain, (current_mg, current_node))

        spread += -new_mg_neg
        seeds.add(current_node)
        spreads.append(spread)

    return seeds, spreads


def draw_graph(
    graph: DiffusionGraphT,
    frequencies: t.Optional[t.Dict[int, int]] = None,
    *,
    layout_prog: str = "fdp",
) -> None:
    agraph = nx.nx_agraph.to_agraph(graph)  # convert to a graphviz graph
    agraph.node_attr["width"] = 0.5
    agraph.node_attr["shape"] = "circle"

    if frequencies is not None:
        heat_iterpolator = Color.interpolate(["blue", "red"], space="srgb")

        max_freq = max(frequencies.values())
        for node in graph.nodes():
            freq = frequencies.get(node, 0)
            viz_node = agraph.get_node(node)
            viz_node.attr["fillcolor"] = heat_iterpolator(freq / max_freq).to_string(hex=True)
            viz_node.attr["style"] = "filled"
            viz_node.attr["fontcolor"] = "white"

    agraph.layout(prog=layout_prog)
    return agraph
