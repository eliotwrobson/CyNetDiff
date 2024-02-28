import array
import typing as t

import typing_extensions as te

class DiffusionModel:
    def advance_model(self) -> None: ...
    def reset_model(self) -> None: ...
    def advance_until_completion(self) -> None: ...
    def get_newly_activated_nodes(self) -> t.Generator[int, None, None]: ...

class IndependentCascadeModel(DiffusionModel):
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
