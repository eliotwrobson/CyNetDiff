from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
from libcpp cimport bool

cdef class DiffusionModel:
    cpdef void advance_model(self)
    cpdef void reset_model(self)
    cpdef void advance_until_completion(self)

cdef class IndependentCascadeModel(DiffusionModel):
    cdef readonly unsigned int[:] starts
    cdef readonly unsigned int[:] edges

    # Setting the activation threshold uniformly or non-uniformly
    cdef float threshold
    cdef readonly float[:] edge_thresholds

    # Mostly for testing
    cdef readonly float[:] edge_probabilities

    cdef cdeque[unsigned int] work_deque
    cdef cset[unsigned int] seen_set
    cdef cset[unsigned int] original_seeds

    cdef int __activation_succeeds(self, unsigned int edge_idx) except -1 nogil
    cdef int __advance_model(self, cdeque[unsigned int]& work_deque, cset[unsigned int]& seen_set) except -1 nogil

cdef class LinearThresholdModel(DiffusionModel):
    # Core model parameters
    cdef readonly unsigned int[:] successors
    cdef readonly unsigned int[:] successor_starts
    cdef readonly unsigned int[:] predecessors
    cdef readonly unsigned int[:] predecessor_starts
    cdef readonly float[:] threshold
    cdef readonly float[:] influence

    cdef cdeque[unsigned int] work_deque
    cdef cset[unsigned int] seen_set
    cdef cset[unsigned int] original_seeds

    cdef int __activation_succeeds(self, unsigned int vtx_idx, const cset[unsigned int]& seen_set) except -1 nogil
    cdef int __advance_model(self, cdeque[unsigned int]& work_deque, cset[unsigned int]& seen_set) except -1 nogil
