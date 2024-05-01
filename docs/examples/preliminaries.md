# Network Diffusion Preliminaries

Network diffusion models aim to capture the spread of trends through networks.
At discrete time steps, new nodes are activated according to the parameters
of the model. Here, we will introduce the Independent Cascade (IC) and Linear Threshold
(LT) models, along with the corresponding primitives provided by `CyNetDiff`.

## Model Context

We first discuss traits common to all network diffusion models. Given a (possibly directed) graph $G=(V,E)$,
the process starts with a set $S \subseteq V$, called the _seed set_, being the initially active nodes.
The model proceeds in discrete time steps, where nodes are activated according to rules set by the different
models. The process runs until no further activation is possible.

In `CyNetDiff`, diffusion models are represented by a model class. In the following, suppose we have an instance
of an existing diffusion `model`. To set the initially active nodes from a set `seed_set`, simply run:

```python
model.set_seeds(seed_set)
```

To advance the diffusion process one time step, run:

```python
model.advance_model()
```

This function has no effect if no activations are possible.

To run the diffusion model until completion (i.e., until no more activations are possible), run:

```python
model.advance_until_completion()
```

A generator for the newly activated nodes can be obtained by:
```python
model.get_newly_activated_nodes()
```
Note that at time step zero, this yields the same elements as `seed_set`.

A generator for all activated nodes (not just ones in the latest time step) can be obtained by:
```python
model.get_newly_activated_nodes()
```

Finally, to reset the model to it's initial state (only seed nodes are activated), run:
```python
model.reset_model()
```

All network diffusion models are stochastic, in that the nodes activated depend on random variables as
well as the model parameters. Thus, resetting and re-running the diffusion process is important for
obtaining useful information from these models.


## Independent Cascade Model

In the independent cascade model, when a node $v \in V$ first becomes active, it will be given a single chance
to activate each currently inactive neighbor $u$. The activation succeeds with probability $p_{v,u}$
(independent of the history thus far). If $u$ has multiple newly activated neighbors, their attempts occur in
an arbitrary order. If $v$ succeeds, then $u$ will become active; but whether or not $v$ succeeds, it cannot make
any further attempts to activate $u$ in subsequent rounds.

`CyNetDiff` supports simulating the above process by the use of a model class. This class stores the state
of the diffusion model, and can be manually advanced by repeated function calls. To start, if we have a
`NetworkX` graph `graph`, we can create a corresponding model with the following:

```python
from cynetdiff.utils import networkx_to_ic_model
model, _ = networkx_to_ic_model(graph)
```

By default, the activation probabilities are all set to a default of `0.1`. There are a number of different
commonly-used weighting schemes which can be set by `CyNetDiff`.

### Weighting Schemes

In the IC model, the weighting scheme is how the probabilities $p_{v,w}$ are chosen. There are three common weighting schemes handled by utility functions from `CyNetDiff`:

- We can choose the weights $p_{v,u} \in [0,1]$, uniformly at random. This can be set by calling the function `set_activation_uniformly_random(graph)`.
- We can choose the weights $p_{v,u}$, uniformly at random from a set `weight_set`. This can be set by calling the function `set_activation_random_sample(graph, weight_set)`.
- We can choose the weights $p_{v,u} = \frac{1}{|N^{in}(u)|}$ as the inverse of the in-degree of $u$. This can be set by calling the function `set_activation_weighted_cascade(graph)`. The input graph must be _directed_ to use this weighting scheme.

To apply a given weighting scheme, we simply call the corresponding helper function from the utility module
of `CyNetDiff`. For example, to use the well-known trivalency weighting scheme, simply call:

```python
from cynetdiff.utils import set_activation_random_sample, networkx_to_ic_model
# Setting weights is done in-place.
set_activation_random_sample(small_world_graph, {0.1, 0.01, 0.001})
# Make sure to set the weights before creating the model.
model, _ = networkx_to_ic_model(small_world_graph)
```

Custom weights can be set under the `"activation_prob"` data key on each edge in the starting `NetworkX` graph.

## Linear Threshold Model

In the linear threshold model, each directed edge $(v,u)$ is given an influence $w_{v,u} \in [0,1]$ such that
$\sum_{v \in N^{in}(u)} w_{v,u} \leq 1$, so that the total incoming influence at every node is at
most one. Each node $u \in V$ is assigned an activation threshold $\theta_u \in (0,1]$ uniformly
at random. Let $A_t$ denote the nodes active at time step $t$. A node $u$ becomes active at time step $t + 1$ if
$$
    \sum_{v \in N^{in}(u) \cap A_t} w_{v,u} \geq \theta_u.
$$
In words, this occurs if the influence of $u$'s active in-neighbors is at least $\theta_u$.

To create a linear threshold model from an `NetworkX` graph `graph`, simply run:

```python
from cynetdiff.utils import networkx_to_lt_model
model, _ = networkx_to_lt_model(graph)
```

By default, the graph is made directed if it is not, and influences are set to the inverse of the in-degree,
i.e. $w_{v,u} = \frac{1}{|N^{in}(u)|}$. Custom influence values can be set under the `"influence"` data key on each edge
in the starting `NetworkX` graph.
