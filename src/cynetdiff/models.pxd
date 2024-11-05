from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.unordered_map cimport unordered_map as cmap
from libcpp.vector cimport vector as cvector

cdef class DiffusionModel:
    cdef readonly float[:] payoffs

    cpdef void advance_model(self)
    cpdef void reset_model(self)
    cpdef void advance_until_completion(self)
    cdef float _compute_payoff(
        self,
        cdeque[unsigned int]& activated_nodes,
        float[:] payoffs,
    )

cdef class IndependentCascadeModel(DiffusionModel):
    cdef readonly unsigned int[:] starts
    cdef readonly unsigned int[:] edges

    # Setting the activation threshold uniformly or non-uniformly
    cdef float activation_prob
    cdef readonly float[:] activation_probs

    # Mostly for testing
    cdef readonly float[:] _edge_probabilities

    # Model simulation data structures
    cdef cdeque[unsigned int] work_deque
    cdef cset[unsigned int] seen_set
    cdef cset[unsigned int] original_seeds

    cdef int _activation_succeeds(self, unsigned int edge_idx) except -1 nogil

    cdef int _advance_model(
        self,
        cdeque[unsigned int]& work_deque,
        cset[unsigned int]& seen_set
    ) except -1 nogil

    cdef cvector[float] _compute_marginal_gains(
        self,
        cvector[unsigned int]& original_seeds,
        cvector[unsigned int]& new_seeds,
        unsigned int num_trials
    )

cdef class LinearThresholdModel(DiffusionModel):
    # Core model parameters
    cdef readonly unsigned int[:] starts
    cdef readonly unsigned int[:] edges
    cdef readonly float[:] influence

    # Model simulation data structures
    cdef cdeque[unsigned int] work_deque
    cdef cset[unsigned int] seen_set
    cdef cset[unsigned int] original_seeds
    cdef cmap[unsigned int, float] thresholds
    cdef cmap[unsigned int, float] buckets

    # Mostly for testing
    cpdef void _assign_thresholds(self, float[:] _node_thresholds)

    cdef int _advance_model(
        self,
        cdeque[unsigned int]& work_deque,
        cset[unsigned int]& seen_set,
        cmap[unsigned int, float]& thresholds,
        cmap[unsigned int, float]& buckets,
    ) except -1 nogil

    cdef cvector[float] _compute_marginal_gains(
        self,
        cvector[unsigned int]& original_seeds,
        cvector[unsigned int]& new_seeds,
        unsigned int num_trials,
        float[:] _node_thresholds
    )
