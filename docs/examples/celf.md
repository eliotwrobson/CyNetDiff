# Influence Maximization

## Motivation and Problem Definition

Suppose we are interested in computing the seed set which activates the most number of nodes in expectation.
This problem is known as [influence maximization](https://snap-stanford.github.io/cs224w-notes/network-methods/influence-maximization).
Given a graph $G = (V,E)$ and a fixed budget $k$, we wish to choose a seed set $S \subseteq V$ such that $|S| = k$ and the influence
$\sigma(S)$ on the graph is maximized.

## Overview of CELF

 A well-known algorithm for this problem is the [CELF](https://hautahi.com/im_greedycelf) algorithm, which is
an optimized version of the greedy algorithm. At a high level, this algorithm works by greedily adding nodes to
the seed set which has the highest marginal gain. This is repeated until the budget is exhausted.

## Computing Marginal Gains

To implement this, we first need a specialized function to compute the marginal gain of adding a node to a
given seed set. In this example, we focus on the independent cascade model.
`CyNetDiff` allows us to accomplish this task very quickly:

```python
from cynetdiff.models import DiffusionModel

def compute_marginal_gain(
    model: DiffusionModel,
    new_node: int,
    seeds: set[int],
    num_trials: int = 1_000,
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

    original_spread = 0
    new_spread = 0
    # If no seeds at the beginning, original spread is always just zero.
    # Prevents wasted work in cases where the seed set is empty.
    if len(seeds) > 0:
        model.set_seeds(seeds)

        for _ in range(num_trials):
            model.reset_model()
            model.advance_until_completion()
            original_spread += model.get_num_activated_nodes()

    new_seeds = seeds.union({new_node})
    model.set_seeds(new_seeds)

    for _ in range(num_trials):
        model.reset_model()
        model.advance_until_completion()
        new_spread += model.get_num_activated_nodes()

    # Avoid floating point division until the very end.
    return (new_spread - original_spread) / num_trials
```

## Main Algorithm

With this function, implementing the rest of the algorithm is straightforward.

```python
from tqdm import tqdm, trange
from cynetdiff.utils import networkx_to_ic_model


def celf(
    model: DiffusionModel, k: int, num_trials: int = 1_000
) -> t.Tuple[t.Set[int], t.List[float]]:
    """
    Input: graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    Code adapted from this blog post:
    https://hautahi.com/im_greedycelf
    """

    # Run the CELF algorithm
    marg_gain = []

    print("Computing marginal gains.")
    # First, compute all marginal gains
    for node in tqdm(list(dir_graph.nodes())):
        marg_gain.append(
            (
                -compute_marginal_gain(
                    model,
                    node,
                    set(),
                    num_trials,
                ),
                node,
            )
        )

    # Convert to heap
    heapq.heapify(marg_gain)

    max_mg, selected_node = heapq.heappop(marg_gain)
    S = [selected_node]
    spread = -max_mg
    spreads = [spread]

    print("Greedily selecting nodes.")
    # Greedily select remaining nodes
    for _ in trange(k - 1):
        while True:
            current_mg, current_node = heapq.heappop(marg_gain)
            new_mg_neg = -compute_marginal_gain(
                model,
                current_node,
                S,
                num_trials,
            )

            if new_mg_neg > current_mg:
                break
            else:
                heapq.heappush(marg_gain, (current_mg, current_node))

        spread += -new_mg_neg
        S.append(current_node)
        spreads.append(spread)

    # Return the maximizing set S and the increasing spread values.
    return S, spreads
```

## Running the Algorithm
To see this in action, we can create a model as before and run our algorithm.
We'll be using the [trivalency weighting scheme](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2010-2_v2.pdf)
to set our activation probabilities:

```python
import networkx as nx
from cynetdiff.utils import networkx_to_ic_model, set_activation_random_sample

# Randomly generate the graph
celf_graph = nx.random_regular_graph(7, 5_000).to_directed()
# Set activation probabilites
set_activation_random_sample(celf_graph, {0.1, 0.01, 0.001})
# Create corresponding model
celf_ic_model = networkx_to_ic_model(celf_graph)

num_seeds = 20
# Get best seed set returned by the algorithm
celf_ic_seeds, ic_marg_gains = celf(celf_ic_model, num_seeds)
```

## Trying Different Models
The `celf` function also works with the Linear Threshold diffusion model:

```python
# First, remove the old edge data
for n1, n2, d in graph.edges(data=True):
    d.clear()

# Next, create the model using the default weighting scheme.
celf_lt_model = networkx_to_lt_model(celf_graph)

num_seeds = 20
# Get best seed set returned by the algorithm
celf_lt_seeds, lt_marg_gains = celf(celf_lt_model, num_seeds)
```
