import random

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
import pandas as pd
from cynetdiff.utils import networkx_to_ic_model


def independent_cascade(G: nx.Graph | nx.DiGraph, seeds: list[int]) -> list[list[int]]:
    if not G.is_directed():
        G = G.to_directed()

    visited = set(seeds)

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    res = []
    current_layer = seeds.copy()

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


### TODO ignore the crap above
def diffuse_python(
    graph: nx.Graph | nx.DiGraph, seeds: set[int], num_samples: int
) -> float:
    res = 0.0
    seeds_list = list(seeds)
    for _ in range(num_samples):
        res += sum(len(level) for level in independent_cascade(graph, seeds_list))

    return res / num_samples


def diffuse_CyNetDiff(
    graph: nx.Graph | nx.DiGraph, seeds: set[int], num_samples: int
) -> float:
    model = networkx_to_ic_model(graph)
    model.set_seeds(seeds)

    total_activated = 0.0

    for _ in range(num_samples):
        model.reset_model()
        model.advance_until_completion()
        total_activated += model.get_num_activated_nodes()

    return total_activated / num_samples


def diffuse_ndlib(
    graph: nx.Graph | nx.DiGraph, seeds: set[int], num_samples: int
) -> float:
    model = ep.IndependentCascadesModel(graph)

    config = mc.Configuration()

    # Assume that thresholds were already set.
    for u, v, data in graph.edges(data=True):
        config.add_edge_configuration("threshold", (u, v), data["threshold"])
    # print(seeds)
    # config.add_model_parameter("infected", seeds)

    model.set_initial_status(config)
    total_infected = 0.0

    for _ in range(num_samples):
        model.reset(seeds)
        prev_iter_count = model.iteration()["node_count"]
        curr_iter_count = model.iteration()["node_count"]

        while prev_iter_count != curr_iter_count:
            prev_iter_count = curr_iter_count
            curr_iter_count = model.iteration()["node_count"]
        # print(curr_iter_count[2])
        total_infected += curr_iter_count[2]

    return total_infected / num_samples


import time


def time_diffusion(func, graph, seeds, num_samples):
    start = time.perf_counter()
    diffused = func(graph, seeds, num_samples)
    end = time.perf_counter()
    return func.__name__, diffused, end - start


def main() -> None:
    # TODO parameterize these benchmarks with
    # https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.random_graphs

    n_values = [10_000]
    k_values = [1, 2, 5, 10, 20]
    frac_values = [0.007]
    num_samples_values = [1000]

    results = []

    for n, k, frac, num_samples in zip(
        n_values, k_values, frac_values, num_samples_values
    ):
        g = nx.fast_gnp_random_graph(n, frac)
        nx.set_edge_attributes(g, 0.1, "threshold")
        seeds = set(random.sample(list(g.nodes()), k))

        for func in [diffuse_CyNetDiff, diffuse_ndlib, diffuse_python]:
            # Print the parameters before running
            print(
                f"Running {func.__name__} with n={n}, k={k}, frac={frac}, num_samples={num_samples}"
            )

            model_name, diffused, time_taken = time_diffusion(
                func, g, seeds, num_samples
            )

            # Print the results after running
            print(
                f"Completed {model_name}: Diffused = {diffused}, Time = {time_taken:.4f} seconds"
            )

            results.append(
                {
                    "n": n,
                    "k": k,
                    "frac": frac,
                    "num_samples": num_samples,
                    "model": model_name,
                    "diffused": diffused,
                    "time": time_taken,
                }
            )

    df = pd.DataFrame(results)
    print(df)


if __name__ == "__main__":
    main()
