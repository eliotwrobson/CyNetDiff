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

```python
import typing as t
from cynetdiff.models import DiffusionModel
from tqdm import tqdm, trange
import heapq


def celf(
    model: DiffusionModel, n: int, k: int, num_trials: int = 1_000
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
    for node in trange(n):
        marg_gain.append(
            (
                -model.compute_marginal_gains([node], [], num_trials),
                node,
            )
        )

    # Convert to heap
    heapq.heapify(marg_gain)

    max_mg, selected_node = heapq.heappop(marg_gain)
    seeds = [selected_node]
    spread = -max_mg
    spreads = [spread]

    print("Greedily selecting nodes.")
    # Greedily select remaining nodes
    for _ in trange(k - 1):
        while True:
            current_mg, current_node = heapq.heappop(marg_gain)
            new_mg_neg = -model.compute_marginal_gains(seeds, [current_node], num_trials)[1]

            if new_mg_neg <= current_mg:
                break
            else:
                heapq.heappush(marg_gain, (new_mg_neg, current_node))

        spread += -new_mg_neg
        seeds.append(current_node)
        spreads.append(spread)

    # Return the maximizing set S and the increasing spread values.
    return seeds, spreads
```

## Running the Algorithm
To see this in action, we can create a model as before and run our algorithm.
We'll be using the [trivalency weighting scheme](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2010-2_v2.pdf)
to set our activation probabilities:

```python
import networkx as nx
from cynetdiff.utils import networkx_to_ic_model, set_activation_random_sample

n = 5_000
# Randomly generate the graph
ic_graph = nx.random_regular_graph(7, n).to_directed()
# Set activation probabilites
set_activation_random_sample(ic_graph, {0.1, 0.01, 0.001})
# Create corresponding model
celf_ic_model, _ = networkx_to_ic_model(ic_graph)

num_seeds = 20
# Get best seed set returned by the algorithm
celf_ic_seeds, ic_marg_gains = celf(celf_ic_model, n, num_seeds)
```

## Trying Different Models
The `celf` function also works with the Linear Threshold diffusion model:

```python
from cynetdiff.utils import networkx_to_lt_model

n = 5_00
# Randomly generate the graph
lt_graph = nx.random_regular_graph(7, n).to_directed()

# Set weighting scheme
celf_lt_model, _ = networkx_to_lt_model(lt_graph)

num_seeds = 20
# Get best seed set returned by the algorithm
celf_lt_seeds, lt_marg_gains = celf(celf_lt_model, n, num_seeds)
```
