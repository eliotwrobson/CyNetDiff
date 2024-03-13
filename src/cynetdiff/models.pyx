# Start with imports
from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset

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


# IC Model
cdef class IndependentCascadeModel(DiffusionModel):
    # Functions that interface with the Python side of things
    def __cinit__(
            self,
            array.array starts,
            array.array edges,
            *,
            double activation_prob = 0.1,
            array.array activation_probs = None,
            array.array _edge_probabilities = None
        ):

        self.starts = starts
        self.edges = edges
        self._edge_probabilities = _edge_probabilities
        self.activation_prob = activation_prob
        self.activation_probs = activation_probs

        if self._edge_probabilities is not None:
            assert len(self.edges) == len(self._edge_probabilities)

        if self.activation_probs is not None:
            assert len(self.edges) == len(self.activation_probs)

    def set_seeds(self, seeds):
        self.original_seeds.clear()
        for seed in seeds:
            self.original_seeds.insert(seed)

        self.reset_model()

    cpdef void reset_model(self):
        self.work_deque.assign(self.original_seeds.begin(), self.original_seeds.end())
        self.seen_set.clear()
        self.seen_set.insert(self.original_seeds.begin(), self.original_seeds.end())

    def get_newly_activated_nodes(self):
        for node in self.work_deque:
            yield node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    cdef inline int __activation_succeeds(self, unsigned int edge_idx) except -1 nogil:
        cdef float activation_prob

        if self.activation_probs is not None:
            activation_prob = self.activation_probs[edge_idx]
        else:
            activation_prob = self.activation_prob

        # TODO get rid of edge probabilities
        if self._edge_probabilities is None:
            if next_rand() <= activation_prob:
                return 1
            return 0

        if self._edge_probabilities[edge_idx] <= activation_prob:
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


# LT Model
cdef class LinearThresholdModel(DiffusionModel):
    # Functions that interface with the Python side of things
    def __cinit__(
            self,
            array.array successors,
            array.array successor_starts,
            array.array predecessors,
            array.array predecessor_starts,
            *,
            array.array influence = None,
            array.array thresholds = None
        ):

        self.successors = successors
        self.successor_starts = successor_starts
        self.predecessors = predecessors
        self.predecessor_starts = predecessor_starts
        self.influence = influence

        assert len(self.successor_starts) == len(self.predecessor_starts)
        # NOTE Assertion below is true because all out-edges must appear as in-edges
        # somewhere else.
        assert len(self.predecessors) == len(self.successors)

        if influence is not None:
            assert len(self.predecessors) == len(self.influence)


        self.thresholds.resize(len(self.successor_starts))

        if thresholds is not None:
            assert len(self.successor_starts) == len(thresholds)

            for i in range(len(thresholds)):
                self.thresholds[i] = thresholds[i]
        else:
            # Make sure to reassign the size before running this.
            self.reassign_thresholds()

    def set_seeds(self, seeds):
        self.original_seeds.clear()
        for seed in seeds:
            self.original_seeds.insert(seed)

        self.reset_model()


    cpdef void reassign_thresholds(self):
        for i in range(self.thresholds.size()):
            self.thresholds[i] = next_rand()

    cpdef void reset_model(self):
        self.work_deque.assign(self.original_seeds.begin(), self.original_seeds.end())
        self.seen_set.clear()
        self.seen_set.insert(self.original_seeds.begin(), self.original_seeds.end())

    def get_newly_activated_nodes(self):
        for node in self.work_deque:
            yield node

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

        if influence_sum >= self.thresholds[vtx_idx]:
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
