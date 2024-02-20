from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
from libcpp cimport bool

cdef class DiffusionModel:
    cpdef void advance_model(self)
    cpdef void reset_model(self)
    cpdef void advance_until_completion(self)

cdef class IndependentCascadeModel(DiffusionModel):
    cdef array.array starts
    cdef array.array edges
    cdef array.array edge_probabilities
    cdef double threshhold
    cdef cdeque[unsigned int] work_deque
    cdef cset[unsigned int] seen_set
    cdef bool activation_succeeds(self, unsigned int edge_idx)
    cdef void __advance_model(self, cdeque[unsigned int]& work_deque, cset[unsigned int]& seen_set)
