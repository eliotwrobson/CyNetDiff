# Start with imports
from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.algorithm cimport fill
from libcpp.vector cimport vector as cvector
from libcpp.unordered_map cimport unordered_map as cmap

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

        # NOTE don't need to store random number since only one is drawn for each edge.
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
            array.array starts,
            array.array edges,
            *,
            array.array influence = None,
            array.array thresholds = None
        ):

        cdef unsigned int i

        self.starts = starts
        self.edges = edges

        cdef unsigned int n = len(self.starts)
        cdef unsigned int m = len(self.edges)
        cdef cvector[unsigned int] in_degrees

        # Setting the influence sent across each edge
        if influence is not None:
            # If provided, copy from user code
            assert m == len(influence)
            self.influence = influence
        else:
            # Otherwise, default to 1/in_degree
            in_degrees.resize(n)
            fill(in_degrees.begin(), in_degrees.end(), 0)

            for i in range(m):
                in_degrees[self.edges[i]] += 1

            influence_arr = array.array("f")

            for i in range(m):
                influence_arr.append(1.0 / in_degrees[self.edges[i]])

            self.influence = influence_arr


        # Setting activation threshold at each node if provided
        if thresholds is not None:
            # If provided, copy from user code
            assert n == len(thresholds)

            # Make a copy to avoid destroying memory on resets.
            for i in range(n):
                self.thresholds[i] = thresholds[i]

    def set_seeds(self, seeds):
        self.original_seeds.clear()
        for seed in seeds:
            self.original_seeds.insert(seed)

        self.reset_model()

    # TODO figure out if we want to refactor this.
    cpdef void reassign_thresholds(self):
        self.thresholds.clear()

    cpdef void reset_model(self):
        self.work_deque.assign(self.original_seeds.begin(), self.original_seeds.end())
        self.seen_set.clear()
        self.seen_set.insert(self.original_seeds.begin(), self.original_seeds.end())
        self.buckets.clear()

    def get_newly_activated_nodes(self):
        for node in self.work_deque:
            yield node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    # Functions that actually advance the model
    cpdef void advance_until_completion(self):
        while self.work_deque.size() > 0:
            self.__advance_model(self.work_deque, self.seen_set, self.thresholds, self.buckets)

    cpdef void advance_model(self):
        self.__advance_model(self.work_deque, self.seen_set, self.thresholds, self.buckets)

    cdef int __advance_model(
        self,
        cdeque[unsigned int]& work_deque,
        cset[unsigned int]& seen_set,
        cmap[unsigned int, float]& thresholds,
        cmap[unsigned int, float]& buckets,
    ) except -1 nogil:

        # Internal-only function to advance,
        # returns an int to allow for exceptions

        cdef unsigned int q = work_deque.size()

        # Working variables
        cdef unsigned int node
        cdef unsigned int range_end
        cdef unsigned int child
        cdef unsigned int edge_idx
        cdef float threshold

        for _ in range(q):
            node = work_deque.front()
            work_deque.pop_front()

            range_end = len(self.edges)
            if node + 1 < len(self.starts):
                range_end = self.starts[node + 1]

            for edge_idx in range(self.starts[node], range_end):
                child = self.edges[edge_idx]

                # Child has _not_ been activated yet
                if seen_set.find(child) == seen_set.end():
                    child = self.edges[edge_idx]

                    # Lazy evaluation for buckets and thresholds
                    if buckets.count(child) == 0:
                        buckets[child] = 0.0

                    if thresholds.count(child) == 0:
                        thresholds[child] = next_rand()

                    threshold = thresholds[child]

                    # Function is written so that each edge is traversed _once_
                    assert buckets[child] < threshold

                    buckets[child] += self.influence[edge_idx]

                    # Skip if we don't have enough influence yet.
                    if buckets[child] < threshold:
                        continue

                    work_deque.push_back(child)
                    seen_set.insert(child)
