# CyNetDiff
[![Run all tests](https://github.com/eliotwrobson/CyNetDiff/actions/workflows/tests.yml/badge.svg)](https://github.com/eliotwrobson/CyNetDiff/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a performance-focused library implementing algorithms for simulating network
diffusion processes, written in Cython. 

## Development

Package development is being done with Poetry. After cloning the repo,
run the following command to build the project:
```
poetry install
```

Then to run tests on the newly compiled code, run:
```
poetry run pytest
```
