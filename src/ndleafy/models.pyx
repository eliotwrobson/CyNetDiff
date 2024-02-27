# Start with imports
from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
from libcpp cimport bool
from cython.parallel import parallel, prange
from cython.operator import dereference

import array
import random

# Next, utility functions
# TODO move these to a separate file later
# From https://narkive.com/Fjs6xpVv:2.890.139
from libc.stdlib cimport rand, RAND_MAX
cdef double RAND_SCALE = 1.0 / RAND_MAX

cdef inline double next_rand() nogil:
    return rand() * RAND_SCALE

# Now, the actual classes we care about

# First, the DiffusionModel base class
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

    cpdef float run_in_parallel(self, unsigned int k):
        raise NotImplementedError

# IC Model
cdef class IndependentCascadeModel(DiffusionModel):
    # Functions that interface with the Python side of things
    def __cinit__(
            self,
            array.array starts,
            array.array edges,
            *,
            double threshold = 0.1,
            array.array edge_probabilities = None
        ):

        self.starts = starts
        self.edges = edges
        self.threshold = threshold
        self.edge_probabilities = edge_probabilities

        if self.edge_probabilities is not None:
            assert len(self.edges) == len(self.edge_probabilities)

    def set_seeds(self, seeds):
        self.original_seeds.clear()
        for seed in seeds:
            self.original_seeds.insert(seed)

        self.reset_model()

    cpdef void reset_model(self):
        self.work_deque.clear()
        self.seen_set.clear()

        for seed in self.original_seeds:
            self.work_deque.push_back(seed)
            self.seen_set.insert(seed)

    def get_newly_activated_nodes(self):
        for new_node in self.work_deque:
            yield new_node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    cdef inline int __activation_succeeds(self, unsigned int edge_idx) except -1 nogil:
        if self.edge_probabilities is None:
            if next_rand() <= self.threshold:
                return 1
            return 0

        if self.edge_probabilities[edge_idx] <= self.threshold:
            return 1
        return 0

    # Functions that actually advance the model
    cpdef void advance_until_completion(self):
        while self.work_deque.size() > 0:
            self.__advance_model(self.work_deque, self.seen_set)

    cpdef void advance_model(self):
        self.__advance_model(self.work_deque, self.seen_set)

    # Internal-only function to advance,
    # returns an int to allow for exceptions
    cdef int __advance_model(self, cdeque[unsigned int]& work_deque, cset[unsigned int]& seen_set) except -1 nogil:
        cdef unsigned int q = work_deque.size()

        # Working variables
        cdef unsigned int node
        cdef unsigned int range_end
        cdef unsigned int child

        for _ in range(q):
            node = work_deque.front()
            work_deque.pop_front()

            range_end = len(self.edges)
            if node + 1 < len(self.starts):
                range_end = self.starts[node + 1]

            for i in range(self.starts[node], range_end):
                if self.__activation_succeeds(i) == 0:
                    continue

                child = self.edges[i]

                # Child is _not_ in the seen set
                if seen_set.find(child) == seen_set.end():
                    work_deque.push_back(child)
                    seen_set.insert(child)


    cpdef float run_in_parallel(self, unsigned int k):
        cdef float res = 0.0
        cdef cdeque[unsigned int]* local_work_deque
        cdef cset[unsigned int]* local_seen_set
        cdef unsigned int j
        cdef unsigned int seed

        with nogil, parallel():
            local_work_deque = new cdeque[unsigned int]()
            local_seen_set = new cset[unsigned int]()

            for j in prange(k, schedule="guided"):
                local_work_deque.clear()
                local_seen_set.clear()

                # TODO replace this with a more efficient copying data structure
                for seed in self.original_seeds:
                    local_work_deque.push_back(seed)
                    local_seen_set.insert(seed)

                while local_work_deque.size() > 0:
                    self.__advance_model(dereference(local_work_deque), dereference(local_seen_set))

                res += local_seen_set.size()

            del local_work_deque
            del local_seen_set

        return res / k

# LT Model
cdef class LinearThresholdModel(DiffusionModel):
    # Functions that interface with the Python side of things
    def __cinit__(
            self,
            array.array successors,
            array.array successor_starts,
            array.array predecessors,
            array.array predecessor_starts,
            array.array threshold
            *,
            array.array influence = None,
        ):

        self.successors = successors
        self.successor_starts = successor_starts
        self.predecessors = predecessors
        self.predecessor_starts = predecessor_starts
        self.threshold = threshold
        self.influence = influence

        assert len(self.successor_starts) == len(self.predecessor_starts)
        assert len(self.predecessors) == len(self.successors)
        assert len(self.successor_starts) == len(self.threshold)

        if influence is not None:
            assert len(self.predecessors) == len(self.influence)

    def set_seeds(self, seeds):
        self.original_seeds.clear()
        for seed in seeds:
            self.original_seeds.insert(seed)

        self.reset_model()

    cpdef void reset_model(self):
        self.work_deque.clear()
        self.seen_set.clear()

        for seed in self.original_seeds:
            self.work_deque.push_back(seed)
            self.seen_set.insert(seed)

    def get_newly_activated_nodes(self):
        for new_node in self.work_deque:
            yield new_node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    # Functions that actually advance the model
    cpdef void advance_until_completion(self):
        while self.work_deque.size() > 0:
            self.__advance_model(self.work_deque, self.seen_set)

    cpdef void advance_model(self):
        self.__advance_model(self.work_deque, self.seen_set)

    cdef inline int __activation_succeeds(self, unsigned int vtx_idx, const cset[unsigned int]& seen_set) except -1 nogil:
        cdef unsigned int i
        cdef unsigned int range_end
        cdef unsigned int parent

        range_end = len(self.predecessors)
        if vtx_idx + 1 < len(self.predecessor_starts):
            range_end = self.predecessor_starts[vtx_idx + 1]

        cdef float influence_sum = 0.0

        for i in range(self.predecessor_starts[vtx_idx], range_end):
            parent = self.predecessors[i]
            # Parent is in the seen set
            if seen_set.find(parent) != seen_set.end():
                if self.influence is None:
                    #NOTE shouldn't need a nonzero check here, because if the in-degree is
                    #0 this should never be hit.
                    influence_sum += 1.0 / (range_end - self.predecessor_starts[vtx_idx])
                else:
                    influence_sum += self.influence[i]

        if influence_sum >= self.threshold[vtx_idx]:
            return 1
        return 0

    # Internal-only function to advance,
    # returns an int to allow for exceptions
    cdef int __advance_model(self, cdeque[unsigned int]& work_deque, cset[unsigned int]& seen_set) except -1 nogil:
        cdef unsigned int q = work_deque.size()

        # Working variables
        cdef unsigned int node
        cdef unsigned int range_end
        cdef unsigned int child
        cdef unsigned int i

        # Use temporary seen set because we don't want this feeding into
        cdef cset[unsigned int] seen_this_iter

        for _ in range(q):
            node = work_deque.front()
            work_deque.pop_front()

            range_end = len(self.successors)
            if node + 1 < len(self.successor_starts):
                range_end = self.successor_starts[node + 1]

            for i in range(self.successor_starts[node], range_end):
                child = self.successors[i]

                if self.__activation_succeeds(child, seen_set) == 0:
                    continue

                # Child has _not_ been seen yet
                if (
                    seen_this_iter.find(child) == seen_this_iter.end()
                    and seen_set.find(child) == seen_set.end()
                ):
                    work_deque.push_back(child)
                    seen_this_iter.insert(child)

        for num in seen_this_iter:
            # Assert the numbers are not in the seen set
            assert seen_set.find(num) == seen_set.end()
            seen_set.insert(num)

    #TODO before doing parallel running, make sure to set random thresholds
