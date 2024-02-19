from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
import array

cdef class DiffusionModel:
    def get_newly_activated_nodes(self):
        raise NotImplementedError

    cpdef void advance_model(self):
        # Function used to advance the model one time step.
        raise NotImplementedError

    cpdef void reset_model(self):
        raise NotImplementedError

cdef class IndependentCascadeModel(DiffusionModel):
    # Functions that interface with the Python side of things
    def __cinit__(self, array.array starts, array.array edges):
        self.starts = starts
        self.edges = edges

    def initialize_model(self, seeds):
        for seed in seeds:
            self.work_deque.push_back(seed)
            self.seen_set.insert(seed)

    def get_newly_activated_nodes(self):
        for new_node in self.work_deque:
            yield new_node

    # Functions that actually advance the model
    cpdef void advance_model(self):
        cdef unsigned int q = self.work_deque.size()
        cdef unsigned int num_starts = len(self.starts)
        cdef unsigned int num_edges = len(self.edges)

        # Working variables
        cdef unsigned int node
        cdef unsigned int range_end
        cdef unsigned int child

        for _ in range(q):
            node = self.work_deque.front()
            self.work_deque.pop_front()

            range_end = num_edges
            if node + 1 < num_starts:
                range_end = self.starts.data.as_ints[node + 1]

            for i in range(self.starts.data.as_ints[node], range_end):
                child = self.edges.data.as_ints[i]

                # Child is _not_ in the seen set
                if self.seen_set.find(child) == self.seen_set.end():
                    self.work_deque.push_back(child)
                    self.seen_set.insert(child)
