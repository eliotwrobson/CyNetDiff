import array
import typing as t

class DiffusionModel:
    """
    Base class for Diffusion Models. This class provides an interface for advancing,
    resetting, and retrieving newly activated nodes.

    Methods
    -------
    advance_model():
        Advances the diffusion model by one step.

    reset_model():
        Resets the diffusion model to its initial state.

    advance_until_completion():
        Continuously advances the model until the diffusion process is complete.

    get_newly_activated_nodes() -> Generator[int, None, None]:
        Yields the nodes that were newly activated in the last advancement.
    """

    def advance_model(self) -> None: ...
    def reset_model(self) -> None: ...
    def advance_until_completion(self) -> None: ...
    def get_newly_activated_nodes(self) -> t.Generator[int, None, None]: ...

class IndependentCascadeModel(DiffusionModel):
    """
    A Diffusion Model representing the Independent Cascade process. This class is a
    subtype of the DiffusionModel and provides specific implementations for the
    Independent Cascade diffusion process.

    Parameters
    ----------
    starts : array.array
        An array of start indices for each node's edges in the edge array.
    edges : array.array
        An array of edges represented as integer indices of nodes.
    threshold : float, optional
        Threshold for the Independent Cascade model, default is 0.1.
    edge_probabilities : Optional[array.array], optional
        An array of success probabilities for each edge, default is None.

    Methods
    -------
    set_seeds(seeds: Iterable[int]):
        Sets the initial active nodes (seeds) for the diffusion process.

    get_num_activated_nodes() -> int:
        Returns the number of activated nodes in the model.
    """

    def __init__(
        self,
        starts: array.array,
        edges: array.array,
        *,
        threshold: float = 0.1,
        edge_probabilities: t.Optional[array.array] = None,
    ) -> None: ...
    def set_seeds(self, seeds: t.Iterable[int]) -> None: ...
    def get_num_activated_nodes(self) -> int: ...

class LinearThresholdModel(DiffusionModel):
    """
    A Diffusion Model representing the Linear Threshold process. This class is a
    subtype of the DiffusionModel and provides specific implementations for the
    Linear Threshold diffusion process.

    Parameters
    ----------
    successors : array.array
        An array of successor nodes for each node.
    successor_starts : array.array
        An array of start indices for each node's successors in the successor array.
    predecessors : array.array
        An array of predecessor nodes for each node.
    predecessor_starts : array.array
        An array of start indices for each node's predecessors in the predecessor array.
    threshold : array.array
        An array of thresholds for each node.
    influence : Optional[array.array], optional
        An array of influence values for each edge, default is None.

    Methods
    -------
    set_seeds(seeds: Iterable[int]):
        Sets the initial active nodes (seeds) for the diffusion process.

    get_num_activated_nodes() -> int:
        Returns the number of activated nodes in the model.
    """

    def __init__(
        self,
        successors: array.array,
        successor_starts: array.array,
        predecessors: array.array,
        predecessor_starts: array.array,
        threshold: array.array,
        *,
        influence: t.Optional[array.array] = None,
    ) -> None: ...
    def set_seeds(self, seeds: t.Iterable[int]) -> None: ...
    def get_num_activated_nodes(self) -> int: ...
