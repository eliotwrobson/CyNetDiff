# CyNetDiff
[![tests](https://github.com/eliotwrobson/CyNetDiff/actions/workflows/tests.yml/badge.svg)](https://github.com/eliotwrobson/CyNetDiff/actions/workflows/tests.yml)
![PyPI - Version](https://img.shields.io/pypi/v/cynetdiff)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cynetdiff)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10801504.svg)](https://doi.org/10.5281/zenodo.10801504)


This is a performance-focused library implementing algorithms for simulating network
diffusion processes, written in Cython.

## Project Status

This project is still considered in an alpha stage of development. As such,
the API is still relatively undocumented, not yet fully featured, and
could still change.

Use at your own risk, however all feedback is greatly appreciated!

## Development

Package development is being done with Poetry. After cloning the repo,
you will first need to add `"cython"` to the build dependencies.
Make sure not to commit this change, as it just enables regeneration
of the C++ files from Cython code.

Then, the following command to build the project:
```
poetry install
```

To run tests on the newly compiled code:
```
poetry run pytest
```
