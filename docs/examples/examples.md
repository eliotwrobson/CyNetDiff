# Basic Usage

## Getting layers of activated nodes

```python
import random
import networkx as nx
from cynetdiff.utils import networkx_to_ic_model

# Randomly generate the graph
n = 1_000
p = 0.05
graph = nx.fast_gnp_random_graph(n, p)

# Randomly choose seed nodes
k = 10
nodes = list(graph.nodes)
seeds = random.sample(nodes, k)

# Set the activation probability uniformly and set seeds
model = networkx_to_ic_model(graph, activation_prob=0.2)
model.set_seeds(seeds)

# Create a set of layers
layers = []
last_activated = list(model.get_newly_activated_nodes())

# While there are still nodes activated, advance the model and
# add to the layers list.
while last_activated:
    layers.append(last_activated)
    model.advance_model()
    last_activated = sorted(model.get_newly_activated_nodes())

# Verify the results
assert sum(map(len, layers)) == model.get_num_activated_nodes()
```
