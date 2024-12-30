# Contributing to CyNetDiff

## Code of Conduct

When interacting with other users and maintainers, please be sure to abide by
the [Code of Conduct](CODE_OF_CONDUCT.md).

## Submitting an issue

### Bug reports

If you are submitting a bug report, please answer the following questions:

1. What version of CyNetDiff were you using?
2. What were you doing?
3. What did you expect to happen?
4. What happened instead?

Please provide any code to reproduce the issue, if possible.

## New features

If you are requesting a new feature or change or behavior, please describe what
you are looking for, and what implementation would make this feature beneficial
for your use case.

## Modifying the codebase

CyNetDiff is an open-source project under the MIT License, so you are welcome and
encouraged to modify the codebase with new fixes and enhancements. Please
observe the following guidelines when submitting pull requests for new fixes or
features:

1. All new code must be formatted with [ruff](https://github.com/astral-sh/ruff). The .vscode directory in this repository is configured to autoformat with ruff on save if you are using VSCode.

2. Whether you are introducing a bug fix or a new feature, you *must* add tests to verify that your code additions function correctly and break nothing else.

3. Make sure that all new code includes updated docstrings in the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).

4. If you are adding a new feature or changing behavior, please update the documentation appropriately with the relevant information. To run the documentation site locally, install the documentation dependencies with:

```sh
poetry install --with docs
```

Then, start the local server with the following command:

```sh
poetry run mkdocs serve
```
### Using Poetry

This project is developed using [poetry](https://python-poetry.org/). It is strongly recommended for local development.

### Installing project dependencies

To install the local version of the library with development dependencies, run the following:

```sh
poetry install --with dev
```

### Cython Generation

The Cython code for this project is generated and then committed to the repository. This code generation is not enabled by default. To enable this, add `"cython"` to the build dependencies in `pyproject.toml`.
Make sure not to commit this change, as it just enables regeneration
of the C++ files from Cython code.

### Running unit tests

The unit tests use [pytest](https://docs.pytest.org/en/8.0.x/).

```sh
poetry run pytest
```
