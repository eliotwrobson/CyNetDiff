from cpython cimport array
import array

cdef class DiffusionModel:
    cpdef readonly array.array get_newly_activated_nodes(self):
        pass

    cpdef void advance_model(self):
        pass

    cpdef void reset_model(self):
        pass

cdef class IndependentCascadeModel(DiffusionModel):

    #cdef readonly int[:] starts
    #cdef readonly int[:] edges

    def __cinit__(self, array.array starts, array.array edges):
        self.starts = starts
        self.edges = edges
