from cpython cimport array

cdef class DiffusionModel:
    cpdef readonly array.array get_newly_activated_nodes(self)
    cpdef void advance_model(self)
    cpdef void reset_model(self)


cdef class IndependentCascadeModel(DiffusionModel):
    cdef readonly int[:] starts
    cdef readonly int[:] edges
