import array
import typing as t

class DiffusionModel:
    """
    Base class for Diffusion Models. This class provides an interface for advancing,
    resetting, and retrieving newly activated nodes. Nodes for the graph of the diffusion
    process have labels in [0, n-1] and the graph is represented in compressed
    sparse row format.
    """

    def advance_model(self) -> None:
        """
        Advances the diffusion model by one step.
        """

    def reset_model(self) -> None:
        """
        Resets the model to the original set of seed nodes. This is useful if
        running many simulations over the same original seed set.
        """

    def advance_until_completion(self) -> None:
        """
        Continuously advances the model until the diffusion process is complete.
        """

    def get_newly_activated_nodes(self) -> t.Generator[int, None, None]:
        """
        A generator yielding the nodes that were newly activated in the last
        iteration of the model.

        Yields
        ----------
        int
            The label of a node that was newly activated.
        """

    def set_seeds(self, seeds: t.Iterable[int]) -> None:
        """
        Sets the initial active nodes (seeds) for the diffusion process. Must
        be valid nodes in the graph.

        Parameters
        ----------
        seeds : Iterable[int]
            Seeds to set as initially active.

        Raises
        ------
        ValueError
            If a node in the seed set is invalid (not in the graph).
        """

    def get_num_activated_nodes(self) -> int:
        """
        Returns the total number of activated nodes in the model.

        Returns
        ----------
        int
            Total number of activated nodes.
        """

    def get_activated_nodes(self) -> t.Generator[int, None, None]:
        """
        Yields all activated nodes.

        Yields
        ----------
        int
            All of the currently activated nodes.
        """

    # TODO add function that computes payoffs.

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
        _edge_probabilities: t.Optional[array.array] = None,
    ) -> None: ...
    def compute_marginal_gains(
        self, seed_set: t.Iterable[int], new_seeds: t.List[int], num_trials: int
    ) -> t.List[float]:
        """
        Computes the marginal gain of adding each seed in new_seeds on top of the original seed_set.
        Averages over num_trials number of randomized activations.

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
    """

    def __init__(
        self,
        starts: array.array,
        edges: array.array,
        *,
        payoffs: t.Optional[array.array] = None,
        influence: t.Optional[array.array] = None,
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
        Averages over num_trials number of randomized activations.

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
        """
