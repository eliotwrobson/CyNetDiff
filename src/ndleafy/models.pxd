from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset

cdef class DiffusionModel:
    cpdef void advance_model(self)
    cpdef void reset_model(self)


cdef class IndependentCascadeModel(DiffusionModel):
    cdef array.array starts
    cdef array.array edges
    cdef cdeque[unsigned int] work_deque
    cdef cset[unsigned int] seen_set
