# Start with imports
from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
from libcpp cimport bool
from cython.parallel import parallel, prange
import array
import random

# Next, utility functions
# TODO move these to a separate file later
# From https://narkive.com/Fjs6xpVv:2.890.139
from libc.stdlib cimport rand, RAND_MAX
cdef double RAND_SCALE = 1.0 / RAND_MAX

cdef inline double next_rand():
    return rand() * RAND_SCALE

# Now, the actual classes we care about

cdef class DiffusionModel:
    def get_newly_activated_nodes(self):
        raise NotImplementedError

    cpdef void advance_model(self):
        # Function used to advance the model one time step.
        raise NotImplementedError

    cpdef void reset_model(self):
        raise NotImplementedError

    cpdef void advance_until_completion(self):
        raise NotImplementedError


cdef class IndependentCascadeModel(DiffusionModel):
    # Functions that interface with the Python side of things
    def __cinit__(
            self,
            array.array starts,
            array.array edges,
            double threshhold = 0.1,
            array.array edge_probabilities = None
        ):

        self.starts = starts
        self.edges = edges
        self.threshhold = threshhold
        self.edge_probabilities = edge_probabilities

    def initialize_model(self, seeds):
        self.work_deque.clear()
        self.seen_set.clear()
        for seed in seeds:
            self.work_deque.push_back(seed)
            self.seen_set.insert(seed)

    def get_newly_activated_nodes(self):
        for new_node in self.work_deque:
            yield new_node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    cdef inline bool activation_succeeds(self, unsigned int edge_idx):
        if self.edge_probabilities is None:
            return next_rand() <= self.threshhold

        return self.edge_probabilities[edge_idx] <= self.threshhold

    # Functions that actually advance the model
    cpdef void advance_until_completion(self):
        while self.work_deque.size() > 0:
            self.__advance_model(self.work_deque, self.seen_set)

    cpdef void advance_model(self):
        self.__advance_model(self.work_deque, self.seen_set)

    # Internal-only function to advance,
    # TODO change this to not use internal attributes directly.
    # This should make coding up a parellel runner easier
    cdef void __advance_model(self, cdeque[unsigned int]& work_deque, cset[unsigned int]& seen_set):
        cdef unsigned int q = work_deque.size()
        cdef unsigned int num_starts = len(self.starts)
        cdef unsigned int num_edges = len(self.edges)

        # Working variables
        cdef unsigned int node
        cdef unsigned int range_end
        cdef unsigned int child

        for _ in range(q):
            node = work_deque.front()
            work_deque.pop_front()

            range_end = num_edges
            if node + 1 < num_starts:
                range_end = self.starts.data.as_ints[node + 1]

            for i in range(self.starts.data.as_ints[node], range_end):
                if not self.activation_succeeds(i):
                    continue

                child = self.edges.data.as_ints[i]

                # Child is _not_ in the seen set
                if seen_set.find(child) == seen_set.end():
                    work_deque.push_back(child)
                    seen_set.insert(child)
