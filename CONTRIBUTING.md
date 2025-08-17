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

1. All new Python code must be formatted with [ruff](https://github.com/astral-sh/ruff). The .vscode directory in this repository is configured to autoformat with ruff on save if you are using VSCode. All Cython code must pass a check from [cython-lint](https://github.com/MarcoGorelli/cython-lint).

2. Whether you are introducing a bug fix or a new feature, you *must* add tests to verify that your code additions function correctly and break nothing else.

3. Make sure that all new code includes updated docstrings in the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).

4. If you are adding a new feature or changing behavior, please update the documentation appropriately with the relevant information. To run the documentation site locally, install the documentation dependencies with:

```sh
pdm install -G docs --no-self
```

Then, start the local server with the following command:

```sh
pdm run mkdocs gh-deploy --force
```
### Using PDM

This project is developed using [PDM](https://pdm-project.org). It is strongly recommended for local development.

### Installing project dependencies

To install the local version of the library with all development dependencies, run the following:

```sh
pdm install -G :all --without docs
```

This will build the project locally and install dependencies. To rebuild the project after making changes,
run `pdm install` again.

### Running unit tests

The unit tests use [pytest](https://docs.pytest.org/en/8.0.x/).

```sh
pdm run pytest
```
