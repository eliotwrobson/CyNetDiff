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
        Sets the initial active nodes (seeds) for the diffusion process.

        Parameters
        ----------
        seeds : Iterable[int]
            Seeds to set as initially active.
        """

    def get_num_activated_nodes(self) -> int:
        """
        Returns the total number of activated nodes in the model.

        Returns
        ----------
        int
            Total number of activated nodes.
        """

# TODO rename some of the parameters
class IndependentCascadeModel(DiffusionModel):
    """
    A Diffusion Model representing the Independent Cascade process. This class is a
    subclass of the DiffusionModel and provides specific implementations for the
    Independent Cascade diffusion process.

    Parameters
    ----------
    starts : array.array
        An array of start indices for each node's edges in the edge array. Type
        of array elements must be unsigned int.
    edges : array.array
        An array of edges represented as integer indices of nodes. Type
        of array elements must be unsigned int.
    threshold : float, optional
        Uniform threshold for the Independent Cascade model. Defaults to 0.1.
        Should not be set if edge_thresholds is set.
    edge_thresholds : array.array, optional
        Set individual thresholds for the Independent Cascade model.
        Overrides threshold.
    edge_probabilities : array.array, optional
        An array of success probabilities for each edge, default is None.
    """

    def __init__(
        self,
        starts: array.array,
        edges: array.array,
        *,
        threshold: float = 0.1,
        edge_thresholds: t.Optional[array.array] = None,
        edge_probabilities: t.Optional[array.array] = None,
    ) -> None: ...

class LinearThresholdModel(DiffusionModel):
    """
    A Diffusion Model representing the Linear Threshold process. This class is a
    subclass of the DiffusionModel and provides specific implementations for the
    Linear Threshold diffusion process.

    Parameters
    ----------
    successors : array.array
        An array of successor nodes for each node. Type
        of array elements must be unsigned int.
    successor_starts : array.array
        An array of start indices for each node's successors in the successor array.
        Type of array elements must be unsigned int.
    predecessors : array.array
        An array of predecessor nodes for each node. Type
        of array elements must be unsigned int.
    predecessor_starts : array.array
        An array of start indices for each node's predecessors in the predecessor array.
        Type of array elements must be unsigned int.
    thresholds : array.array, optional
        An array of thresholds for each node. Type
        of array elements must be float.
    influence : array.array, optional
        An array of influence values for each edge. Array elements must be
        floats in [0,1]. If not set, the inverse of the in-degree of a node
        is used for the influence.
    """

    def __init__(
        self,
        successors: array.array,
        successor_starts: array.array,
        predecessors: array.array,
        predecessor_starts: array.array,
        *,
        influence: t.Optional[array.array] = None,
        thresholds: t.Optional[array.array] = None,
    ) -> None: ...
    def reassign_threshold(self) -> None: ...
