# CyNetDiff
[![PyPI version](https://badge.fury.io/py/cynetdiff.svg)](https://badge.fury.io/py/cynetdiff)
[![tests](https://github.com/eliotwrobson/CyNetDiff/actions/workflows/tests.yml/badge.svg)](https://github.com/eliotwrobson/CyNetDiff/actions/workflows/tests.yml)
[![docs](https://github.com/eliotwrobson/CyNetDiff/actions/workflows/docs.yml/badge.svg)](https://github.com/eliotwrobson/CyNetDiff/actions/workflows/docs.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cynetdiff)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10801504.svg)](https://doi.org/10.5281/zenodo.10801504)

- **Documentation**: https://eliotwrobson.github.io/CyNetDiff/

Network diffusion processes aim to model the spread of trends through social networks, represented using graphs. Experimental work with these models usually involves simulating these processes many times over large graphs, which can be computationally very expensive. To address this, CyNetDiff is a Cython module implementing the independent cascade and linear threshold models. Development has been focused on performance, while still giving an intuitive, high-level interface to assist in research tasks. To learn more about these specific models, read
[this book chapter](https://www.researchgate.net/publication/300470631_The_Independent_Cascade_and_Linear_Threshold_Models)
or the [preliminaries page in the documentation](https://eliotwrobson.github.io/CyNetDiff/examples/preliminaries/).

## Quick Start

### Installation
```sh
pip install cynetdiff
```
*Note:* The installation includes a build step that requires having a C++ complier installed.

### Basic Usage
We can run models over graphs we define, using pre-defined weighting schemes. Here is a simple
example:
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
model, _ = networkx_to_ic_model(graph, activation_prob=0.2)
model.set_seeds(seeds)

# Run a single diffusion process until completion
model.advance_until_completion()

# Get the number of nodes activated
model.get_num_activated_nodes()
```

The output from the last line is the number of nodes activated in a single
simulation of the model. To get the average number of activated nodes
across `n_sim = 1_000` simulations, we can replace the last line in the
above with the following:

```python
n_sim = 1_000
total = 0.0

for _ in range(n_sim):
    # Resetting the model doesn't change the initial seed set used.
    model.reset_model()
    model.advance_until_completion()
    total += model.get_num_activated_nodes()

avg = total / n_sim
```

### Advanced Usage
See the [documentation](https://eliotwrobson.github.io/CyNetDiff/examples/activated/).

## Citation

If you use this code in your research, please use the following citation:
```bibtex
@article{DBLP:journals/pvldb/RobsonRU24,
  author       = {Eilot W. Robson and
                  Dhemath Reddy and
                  Abhishek Kumar Umrawal},
  title        = {CyNetDiff: {A} Python Library for Accelerated Implementation of Network
                  Diffusion Models},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {17},
  number       = {12},
  pages        = {4409--4412},
  year         = {2024},
  url          = {https://www.vldb.org/pvldb/vol17/p4409-umrawal.pdf},
  timestamp    = {Thu, 19 Sep 2024 13:09:38 +0200},
  biburl       = {https://dblp.org/rec/journals/pvldb/RobsonRU24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Project Status

This project is still considered in a beta stage of development. As such,
the API could still change to facilitate easier community adoption.

All feedback is greatly appreciated!

## Contributing

Contributions are always welcome! Take a look at the [contributing guide](CONTRIBUTING.md).
