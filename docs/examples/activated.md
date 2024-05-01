# Observing Activated Nodes

## Motivation and Starter Code

Instead of just the number of activated nodes, suppose that we want
to get the nodes themselves. We start with similar code as the basic
example from the README:

```python
import random
import networkx as nx
from cynetdiff.utils import networkx_to_ic_model

# Randomly generate the graph
n = 1_000
p = 0.05
k = 10
graph = nx.watts_strogatz_graph(n, k, p)

# Randomly choose seed nodes
k = 10
nodes = list(graph.nodes)
seeds = random.sample(nodes, k)

# Set the activation probability uniformly and set seeds
model, _ = networkx_to_ic_model(graph, activation_prob=0.2)
model.set_seeds(seeds)
```

## Obtaining the activated nodes

Next, we run the model until completion and can view the
nodes that were activated:

```python
model.advance_until_completion()
activated_nodes = set(model.get_activated_nodes())
```

## Counting the number of activations

This functionality is useful in cases where we are
interested in how often particular nodes are activated.

Over `n_sim = 1_000`, we can record the number of simulations
where a node is activated:

```python
n_sim = 1_000
times_seen = dict()

for _ in range(n_sim):
    model.reset_model()
    model.advance_until_completion()

    for activated_node in model.get_activated_nodes():
        if activated_node not in times_seen:
            times_seen[activated_node] = 0

        # Increment seen counter for each node
        times_seen[activated_node] += 1
```

## Getting the most frequently activated nodes

The `times_seen` dictionary now contains the number of simulations
each node has been activated in. To get the top `l = 50` activated nodes,
we can run the following:

```python
nodes_sorted_by_activation_frequency = sorted(
    times_seen.items(), key=lambda pair: -pair[1]
)
l = 50
print(nodes_sorted_by_activation_frequency[:l])
```
