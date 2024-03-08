import copy
import random
import typing as t

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx

from src.cynetdiff.utils import networkx_to_ic_model

def independent_cascade(
    G: nx.Graph | nx.DiGraph,
    seeds: t.Iterable[int],
    *,
    steps: int = 0,
    random_seed: t.Any = None,
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
  >>>   (4,6), (5,3), (5,4), (5,6), (6,4), (6,5)], act_prob=0.2)
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

    # init activation probabilities
    for u, v, data in DG.edges(data=True):
        act_prob = data.setdefault("act_prob", 0.1)
        # if "act_prob" not in data:
        #    data["act_prob"] = 0.1
        if act_prob > 1.0:
            raise Exception(
                f"edge activation probability: {act_prob} cannot be larger than 1."
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


def diffuse_ndleafy(
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
    print(func.__name__, " diffused:", diffused, "time: ", end - start)


def main(n, k, frac, num_samples, model_type) -> None:
    # TODO parameterize these benchmarks with
    # https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.random_graphs

    g = nx.fast_gnp_random_graph(n, frac)
    nx.set_edge_attributes(g, 0.1, "threshold")
    seeds = set(random.sample(list(g.nodes()), k))

    if model_type == 'diffuse_ndleafy':
        time_diffusion(diffuse_ndleafy, g, seeds, num_samples)
    elif model_type == 'diffuse_ndlib':
        time_diffusion(diffuse_ndlib, g, seeds, num_samples)
    elif model_type == 'diffuse_python':
        time_diffusion(diffuse_python, g, seeds, num_samples)


def get_input(prompt):
    """Utility function to get and parse input from the user."""
    try:
        return [item.strip() for item in input(prompt).split(',')]
    except ValueError:
        raise ValueError("Invalid input format.")


def parse_input_list(input_list, data_type):
    """Converts a list of string inputs to a list of specified data type."""
    try:
        return [data_type(item) for item in input_list]
    except ValueError:
        raise ValueError(f"Invalid input. Expected list of {data_type.__name__}.")


def validate_inputs(
        n_list, k_list, frac_list, num_samples_list, model_types_list
        ):
    """Validate the length of input lists and model types."""
    if not (len(n_list) == len(k_list) == len(frac_list) ==
            len(num_samples_list) == len(model_types_list)):
        raise ValueError("All input lists must be of equal length.")

    valid_model_types = {'diffuse_ndleafy', 'diffuse_ndlib', 'diffuse_python'}
    if not all(model_type in valid_model_types for model_type in model_types_list):
        raise ValueError(f"Invalid model type. Valid model types are: {valid_model_types}")


if __name__ == "__main__":
    try:
        n_list = parse_input_list(get_input("Enter values for n (comma-separated): "), int)

        k_list = parse_input_list(get_input("Enter values for k (comma-separated): "), int)

        frac_list = parse_input_list(get_input("Enter values for frac (comma-separated): "), float)

        num_samples_list = parse_input_list(get_input("Enter values for num_samples (comma-separated): "), int)

        model_types_list = get_input("Enter model types (comma-separated, choose from diffuse_ndleafy,diffuse_ndlib, diffuse_python): ")

        validate_inputs(n_list, k_list, frac_list, num_samples_list, model_types_list)

        for n, k, frac, num_samples, model_type in zip(n_list, k_list, frac_list, num_samples_list, model_types_list):
            print(f"Running {model_type} with n={n}, k={k}, frac={frac},num_samples={num_samples}")
            main(n, k, frac, num_samples, model_type)

    except Exception as e:
        print(f"Error: {e}")
