import array
import typing as t
from collections.abc import Sequence

import numpy as np

SeedLike = t.Union[int, np.integer, Sequence[int], np.random.SeedSequence]
RNGLike = t.Union[np.random.Generator, np.random.BitGenerator]
RNGType = t.Union[SeedLike, RNGLike, None]

class DiffusionModel:
    """
    Base class for Diffusion Models. This class provides an interface for advancing,
    resetting, and retrieving newly activated nodes. Nodes for the graph of the diffusion
    process have labels in [0, n-1] and the graph is represented in compressed
    sparse row format.
    """

    def set_rng(self, rng: RNGType = None):
        """
        Sets the random number generator for the model. If not set, creates a new generator
        by default.

        Parameters
        ----------
        rng : SeedLike | RNGLike | None
            Random number generator to use for the model.

        Examples
        ----------
        >>> rng_seed = 42
        >>> model.set_rng(rng_seed)
        >>> model.advance_until_completion()
        >>> old_num = model.get_num_activated_nodes()
        >>> model.set_rng(rng_seed)
        >>> model.advance_until_completion()
        >>> old_num == model.get_num_activated_nodes()
        True
        """

    def advance_model(self) -> None:
        """
        Advances the diffusion model by one step. Since these diffusion models
        are progressive, the number of activated nodes cannot decrease.

        Examples
        ----------
        >>> old_num_active = model.get_num_activated_nodes()
        >>> model.advance_model()
        >>> model.get_num_activated_nodes() >= old_num_active
        True
        """

    def reset_model(self) -> None:
        """
        Resets the model to the original set of seed nodes. This is useful if
        running many simulations over the same original seed set. If randomized
        activation is enabled, resetting the model will perform the randomized
        activation step.

        Examples
        ----------
        >>> model.set_seeds([0, 1, 2])
        >>> model.advance_until_completion()
        >>> model.get_num_activated_nodes()
        10
        >>> model.reset_model()
        >>> model.get_num_activated_nodes()
        3
        """

    def advance_until_completion(self) -> None:
        """
        Continuously advances the model until the diffusion process is complete.

        Examples
        ----------
        >>> model.advance_until_completion()
        >>> activated_nodes = model.get_num_activated_nodes()
        >>> model.advance_model()
        >>> activated_nodes == model.get_num_activated_nodes()
        True
        """

    def get_newly_activated_nodes(self) -> t.Generator[int, None, None]:
        """
        A generator yielding the nodes that were newly activated in the last
        iteration of the model. If the model has not yet been run, this is
        just the current seed nodes.

        Yields
        ----------
        int
            The label of a node that was newly activated.

        Examples
        ----------
        >>> model.set_seeds([0, 1, 2])
        >>> set(model.get_newly_activated_nodes())
        {0, 1, 2}
        >>> model.advance_until_completion()
        >>> len(set(model.get_newly_activated_nodes()))
        0
        """

    def set_seeds(self, seeds: t.Iterable[int], seed_probs: t.Optional[t.Iterable[float]] = None) -> None:
        """
        Sets the initial active nodes (seeds) for the diffusion process. Must
        be valid nodes in the graph. If activation probabilities are set, they
        represent the probability of activation for each seed node. Must be
        in the range [0.0, 1.0].

        Parameters
        ----------
        seeds : Iterable[int]
            Seeds to set as initially active.

        seed_probs : Optional[Iterable[float]]
            Activation probabilities for each seed node.
            Entries must be in the range [0.0, 1.0]. Length must be equal to seeds.
            If not set, seeds are always active.

        Raises
        ------
        ValueError
            If a node in the seed set is invalid (not in the graph).

        Examples
        ----------
        >>> model.set_seeds([0, 1, 2])
        >>> set(model.get_activated_nodes())
        {0, 1, 2}
        """

    def get_num_activated_nodes(self) -> int:
        """
        Returns the total number of activated nodes in the model.

        Returns
        ----------
        int
            Total number of activated nodes.

        Examples
        ----------
        >>> model.set_seeds([0, 1, 2])
        >>> model.get_num_activated_nodes()
        3
        """

    def get_activated_nodes(self) -> t.Generator[int, None, None]:
        """
        Yields all activated nodes.

        Yields
        ----------
        int
            All of the currently activated nodes.

        Examples
        ----------
        >>> model.set_seeds([0, 1, 2])
        >>> for node in model.get_activated_nodes()
        ...    print(node)
        0
        1
        2
        """

    def compute_payoffs(self) -> float:
        """
        Computes the payoffs of each active node.
        Payoffs are defaulted to 1.0 if not set.

        Returns
        ----------
        float
            Sum of payoffs for all activated nodes.

        Examples
        ----------
        >>> model.set_seeds([0, 1, 2])
        >>> model.compute_payoffs()
        3.0
        """

class IndependentCascadeModel(DiffusionModel):
    """
    A Diffusion Model representing the Independent Cascade process. This class is a
    subclass of the DiffusionModel and provides specific implementations for the
    Independent Cascade diffusion process.

    Parameters
    ----------
    starts : array.array
        An array of start indices for each node's edges in the edge array. Type
        of array elements must be `unsigned int`.
    edges : array.array
        An array of edges represented as integer indices of nodes. Type
        of array elements must be `unsigned int`.
    payoffs : array.array
        An array of payoffs for each node if activated. Type of array elements must be `float`.
    activation_prob : float, optional
        Uniform activation probability for the Independent Cascade model.
        Defaults to `0.1`. Should not be set if `activation_probs` is set.
        Must be in [`0.0`,`1.0`].
    activation_probs : array.array, optional
        Set individual activation probabilities for the Independent Cascade model.
        Overrides `activation_prob`. Array elements must be `float`s in [`0.0`,`1.0`].
    rng : np.random.Generator | np.random.BitGenerator | None, optional
        Random number generator to use for the model.
        If not set, creates a new generator by default.
    _edge_probabilities : array.array, optional
        An array of success probabilities for each edge, default is None.
    """

    def __init__(
        self,
        starts: array.array,
        edges: array.array,
        *,
        activation_prob: float = 0.1,
        activation_probs: t.Optional[array.array] = None,
        payoffs: t.Optional[array.array] = None,
        rng: RNGType = None,
        _edge_probabilities: t.Optional[array.array] = None,
    ) -> None: ...
    def compute_marginal_gains(
        self, seed_set: t.Iterable[int], new_seeds: t.List[int], num_trials: int
    ) -> t.List[float]:
        """
        Computes the marginal gain of adding each seed in new_seeds on top of the original seed_set.
        Averages over num_trials number of randomized activations. Scores are computed using payoffs
        if set, otherwise the number of activated nodes is used.

        Parameters
        ----------
        seed_set : Iterable[int]
            An iterable representing the current seed set. Can be empty.
        new_seeds : List[int]
            New seeds to compute marginal gains on. Can be empty.
        num_trials : int
            Number of randomized trials to run.

        Returns
        ----------
        List[float]
            List containing computed marginal gains. First entry is average influence of the
            starting seed set. Following entries are marginal gains with the addition of vertices
            from new_seeds in order. Has length len(new_seeds)+1.

        Examples
        ----------
        >>> model.compute_marginal_gains([0, 1, 2], [3, 4], 100)
        [3.0, 1.0, 1.0]
        """

class LinearThresholdModel(DiffusionModel):
    """
    A Diffusion Model representing the Linear Threshold process. This class is a
    subclass of the DiffusionModel and provides specific implementations for the
    Linear Threshold diffusion process.

    Parameters
    ----------
    starts : array.array
        An array of start indices for each node's edges in the edge array. Type
        of array elements must be `unsigned int`.
    edges : array.array
        An array of edges represented as integer indices of nodes. Type
        of array elements must be `unsigned int`.
    payoffs : array.array
        An array of payoffs for each node if activated. Type of array elements must be `float`.
    influence : array.array, optional
        An array of influence values for each edge. Array elements must be
        `float`s in [`0.0`,`1.0`]. If not set, the inverse of the in-degree of a node
        is used for the influence.
    rng : np.random.Generator | np.random.BitGenerator | None, optional
        Random number generator to use for the model. If not set, creates a new generator by default.
    """

    def __init__(
        self,
        starts: array.array,
        edges: array.array,
        *,
        payoffs: t.Optional[array.array] = None,
        influence: t.Optional[array.array] = None,
        rng: RNGType = None,
    ) -> None: ...
    def _assign_thresholds(self, node_thresholds: array.array) -> None:
        """
        Assigns activation thresholds from the given array. Should mainly be used for
        testing.
        """
        ...

    def compute_marginal_gains(
        self,
        seed_set: t.Iterable[int],
        new_seeds: t.List[int],
        num_trials: int,
        *,
        _node_thresholds: t.Optional[array.array] = None,
    ) -> t.List[float]:
        """
        Computes the marginal gain of adding each seed in new_seeds on top of the original seed_set.
        Averages over num_trials number of randomized activations. Scores are computed using payoffs
        if set, otherwise the number of activated nodes is used.

        Parameters
        ----------
        seed_set : Iterable[int]
            An iterable representing the current seed set. Can be empty.
        new_seeds : List[int]
            New seeds to compute marginal gains on. Can be empty.
        num_trials : int
            Number of randomized trials to run.

        Returns
        ----------
        List[float]
            List containing computed marginal gains. First entry is average influence of the
            starting seed set. Following entries are marginal gains with the addition of vertices
            from new_seeds in order. Has length len(new_seeds)+1.

        Examples
        ----------
        >>> model.compute_marginal_gains([0, 1, 2], [3, 4], 100)
        [3.0, 1.0, 1.0]
        """
