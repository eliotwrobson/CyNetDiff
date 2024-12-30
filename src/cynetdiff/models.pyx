# Start with imports
from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.algorithm cimport fill
from libcpp.vector cimport vector as cvector
from libcpp.unordered_map cimport unordered_map as cmap

cimport cython
import numpy.random as npr
cimport numpy.random as npr
from numpy.random.c_distributions cimport random_standard_uniform
from cpython.pycapsule cimport PyCapsule_GetPointer

cdef const char *capsule_name = "BitGenerator"

# First, the DiffusionModel base class
cdef class DiffusionModel:
    def set_rng(self, rng = None):
        self._rng = npr.default_rng(rng)
        self.bitgen_state = <npr.bitgen_t*>PyCapsule_GetPointer(
            self._rng.bit_generator.capsule,
            capsule_name
        )

    def get_newly_activated_nodes(self):
        raise NotImplementedError

    def get_activated_nodes(self):
        raise NotImplementedError

    cpdef void advance_model(self):
        # Function used to advance the model one time step.
        raise NotImplementedError

    cpdef void reset_model(self):
        raise NotImplementedError

    cpdef void advance_until_completion(self):
        raise NotImplementedError

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float _compute_payoff_set(
        self,
        cset[unsigned int]& activated_nodes,
        float[:] payoffs,
    ):
        cdef float result = 0.0

        if payoffs is not None:
            for node in activated_nodes:
                result += payoffs[node]
        else:
            result += <float>activated_nodes.size()

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float _compute_payoff(
        self,
        cdeque[unsigned int]& activated_nodes,
        float[:] payoffs,
    ):
        cdef float result = 0.0

        if payoffs is not None:
            for node in activated_nodes:
                result += payoffs[node]
        else:
            result += <float>activated_nodes.size()

        return result

    def set_seeds(self, seeds, seed_probs=None):
        self.original_seeds.clear()
        self.seed_probs.clear()

        n = len(self.starts)

        for seed in seeds:
            if not (isinstance(seed, int) and 0 <= seed < n):
                raise ValueError(
                    f"Invalid seed node: {seed}. Must be in the range [0, {n-1}]"
                )
            self.original_seeds.push_back(seed)

        if seed_probs is not None:
            for prob in seed_probs:
                if not (isinstance(prob, float) and 0.0 <= prob <= 1.0):
                    raise ValueError(
                        f"Invalid activation probability: {prob}. "
                        "Must be in the range [0.0, 1.0]"
                    )
                self.seed_probs.push_back(prob)

        if self.seed_probs.size() > 0 and self.seed_probs.size() != self.original_seeds.size():
            raise ValueError(
                "Activation probabilities must be provided for each seed node."
            )

        self.reset_model()

# IC Model
cdef class IndependentCascadeModel(DiffusionModel):
    # Functions that interface with the Python side of things
    def __cinit__(
        self,
        unsigned int[:] starts not None,
        unsigned int[:] edges not None,
        *,
        double activation_prob = 0.1,
        float[:] activation_probs = None,
        float[:] payoffs = None,
        float[:] _edge_probabilities = None,
        rng = None,
    ):

        self.starts = starts
        self.edges = edges
        self.activation_prob = activation_prob
        self.activation_probs = activation_probs
        self.payoffs = payoffs

        self.set_rng(rng)

        self._edge_probabilities = _edge_probabilities

        if self._edge_probabilities is not None:
            assert len(self.edges) == len(self._edge_probabilities)

        if self.activation_probs is not None:
            assert len(self.edges) == len(self.activation_probs)

        if self.payoffs is not None:
            assert len(self.starts) == len(self.payoffs)

    cpdef void reset_model(self):
        self.seen_set.clear()
        self.work_deque.clear()

        # Reset the work deque
        if len(self.seed_probs) == 0:
            self.work_deque.assign(self.original_seeds.begin(), self.original_seeds.end())
            self.seen_set.insert(self.original_seeds.begin(), self.original_seeds.end())
        else:
            for i in range(self.original_seeds.size()):
                if random_standard_uniform(self.bitgen_state) <= self.seed_probs[i]:
                    self.work_deque.push_back(self.original_seeds[i])
                    self.seen_set.insert(self.original_seeds[i])

    def get_newly_activated_nodes(self):
        for node in self.work_deque:
            yield node

    def get_activated_nodes(self):
        for node in self.seen_set:
            yield node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    def compute_payoffs(self):
        return self._compute_payoff_set(self.seen_set, self.payoffs)

    def compute_marginal_gains(self, seed_set, new_seeds, num_trials):
        cdef cvector[unsigned int] original_seeds
        cdef cvector[unsigned int] new_seeds_vec
        n = len(self.starts)

        new_seeds_set = set(new_seeds)

        if len(new_seeds) != len(new_seeds_set):
            raise ValueError(
                "New seeds set must have all unique elements."
            )

        for seed in seed_set:
            if not (isinstance(seed, int) and 0 <= seed < n):
                raise ValueError(
                    f"Invalid seed node: {seed}. Must be in the range [0, {n-1}]"
                )
            elif seed in new_seeds_set:
                raise ValueError(
                    f"New seed {seed} should not be contained within the seed set."
                )
            original_seeds.push_back(seed)

        for new_seed in new_seeds:
            if not (isinstance(new_seed, int) and 0 <= new_seed < n):
                raise ValueError(
                    f"Invalid new seed {new_seed}. "
                    f"Must be integer in the range [0, {n-1}]"
                )
            new_seeds_vec.push_back(new_seed)

        cdef cvector[float] results = self._compute_marginal_gains(
            original_seeds, new_seeds_vec, num_trials
        )

        return [
            num for num in results
        ]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cvector[float] _compute_marginal_gains(
        self,
        cvector[unsigned int]& original_seeds,
        cvector[unsigned int]& new_seeds,
        unsigned int num_trials
    ):
        cdef cdeque[unsigned int] work_deque
        cdef cset[unsigned int] seen_set

        cdef cvector[float] results
        results = cvector[float](new_seeds.size()+1, 0.0)

        cdef unsigned int new_seed
        cdef unsigned int i

        for _ in range(num_trials):
            work_deque.assign(original_seeds.begin(), original_seeds.end())
            seen_set.clear()
            seen_set.insert(original_seeds.begin(), original_seeds.end())

            while work_deque.size() > 0:
                results[0] += self._compute_payoff(work_deque, self.payoffs)
                self._advance_model(work_deque, seen_set)

            for i in range(1, new_seeds.size()+1):
                new_seed = new_seeds[i-1]

                # No marginal gain unless we're activating a new node
                if seen_set.find(new_seed) == seen_set.end():
                    work_deque.push_back(new_seed)
                    seen_set.insert(new_seed)

                    while work_deque.size() > 0:
                        results[i] += self._compute_payoff(work_deque, self.payoffs)
                        self._advance_model(work_deque, seen_set)

        for i in range(results.size()):
            results[i] /= <float>num_trials

        return results

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline int _activation_succeeds(self, unsigned int edge_idx) except -1 nogil:
        cdef float activation_prob

        if self.activation_probs is not None:
            activation_prob = self.activation_probs[edge_idx]
        else:
            activation_prob = self.activation_prob

        # NOTE don't need to store random number since only one is drawn for each edge.
        if self._edge_probabilities is None:
            if random_standard_uniform(self.bitgen_state) <= activation_prob:
                return 1
            return 0

        if self._edge_probabilities[edge_idx] <= activation_prob:
            return 1
        return 0

    # Functions that actually advance the model
    cpdef void advance_until_completion(self):
        while self.work_deque.size() > 0:
            self._advance_model(self.work_deque, self.seen_set)

    cpdef void advance_model(self):
        self._advance_model(self.work_deque, self.seen_set)

    # Internal-only function to advance,
    # returns an int to allow for exceptions
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _advance_model(
        self,
        cdeque[unsigned int]& work_deque,
        cset[unsigned int]& seen_set
    ) except -1 nogil:
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
                if self._activation_succeeds(i) == 0:
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
        unsigned int[:] starts not None,
        unsigned int[:] edges not None
        *,
        float[:] influence = None,
        float[:] payoffs = None,
        rng = None,
    ):
        self.starts = starts
        self.edges = edges
        self.payoffs = payoffs

        self.set_rng(rng)

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

            for out_node in self.edges:
                in_degrees[out_node] += 1

            influence_arr = array.array("f")
            influence_arr.extend(
                1.0 / in_degrees[out_node]
                for out_node in self.edges
            )

            self.influence = influence_arr

        # Verify payoffs
        if self.payoffs is not None:
            assert len(self.starts) == len(self.payoffs)

    cpdef void _assign_thresholds(self, float[:] _node_thresholds):
        # If provided, copy from user code
        cdef unsigned int n = len(self.starts)
        assert n == len(_node_thresholds)

        # Make a copy to avoid destroying memory on resets.
        for i in range(n):
            self.thresholds[i] = _node_thresholds[i]

    cpdef void reset_model(self):
        # Clear old data structures
        self.seen_set.clear()
        self.work_deque.clear()
        self.buckets.clear()
        self.thresholds.clear()

        # Reset the work deque
        if len(self.seed_probs) == 0:
            self.work_deque.assign(self.original_seeds.begin(), self.original_seeds.end())
            self.seen_set.insert(self.original_seeds.begin(), self.original_seeds.end())
        else:
            for i in range(self.original_seeds.size()):
                if random_standard_uniform(self.bitgen_state) <= self.seed_probs[i]:
                    self.work_deque.push_back(self.original_seeds[i])
                    self.seen_set.insert(self.original_seeds[i])

    def get_newly_activated_nodes(self):
        for node in self.work_deque:
            yield node

    def get_activated_nodes(self):
        for node in self.seen_set:
            yield node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    def compute_payoffs(self):
        return self._compute_payoff_set(self.seen_set, self.payoffs)

    def compute_marginal_gains(
        self,
        seed_set,
        new_seeds,
        num_trials,
        *,
        _node_thresholds=None
    ):
        cdef cvector[unsigned int] original_seeds
        cdef cvector[unsigned int] new_seeds_vec
        n = len(self.starts)

        new_seeds_set = set(new_seeds)

        if len(new_seeds) != len(new_seeds_set):
            raise ValueError(
                "New seeds set must have all unique elements."
            )

        for seed in seed_set:
            if not (isinstance(seed, int) and 0 <= seed < n):
                raise ValueError(
                    f"Invalid seed node: {seed}. Must be in the range [0, {n-1}]"
                )
            elif seed in new_seeds_set:
                raise ValueError(
                    f"New seed {seed} should not be contained within the seed set."
                )
            original_seeds.push_back(seed)

        for new_seed in new_seeds:
            if not (isinstance(new_seed, int) and 0 <= new_seed < n):
                raise ValueError(
                    f"Invalid new seed {new_seed}. "
                    f"Must be integer in the range [0, {n-1}]"
                )
            new_seeds_vec.push_back(new_seed)

        cdef cvector[float] results = self._compute_marginal_gains(
            original_seeds, new_seeds_vec, num_trials, _node_thresholds
        )

        return [
            num for num in results
        ]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cvector[float] _compute_marginal_gains(
        self,
        cvector[unsigned int]& original_seeds,
        cvector[unsigned int]& new_seeds,
        unsigned int num_trials,
        float[:] _node_thresholds
    ):
        cdef cdeque[unsigned int] work_deque
        cdef cset[unsigned int] seen_set
        cdef cmap[unsigned int, float] thresholds
        cdef cmap[unsigned int, float] buckets

        cdef cvector[float] results
        results = cvector[float](new_seeds.size()+1, 0.0)

        cdef unsigned int n = len(self.starts)
        cdef unsigned int new_seed
        cdef unsigned int i

        # Copy initial thresholds if provided
        if _node_thresholds is not None:
            assert n == len(_node_thresholds)

            # Make a copy to avoid destroying memory on resets.
            for i in range(n):
                thresholds[i] = _node_thresholds[i]

        for _ in range(num_trials):
            work_deque.assign(original_seeds.begin(), original_seeds.end())
            seen_set.clear()
            seen_set.insert(original_seeds.begin(), original_seeds.end())
            buckets.clear()

            while work_deque.size() > 0:
                results[0] += self._compute_payoff(work_deque, self.payoffs)
                self._advance_model(work_deque, seen_set, thresholds, buckets)

            for i in range(1, new_seeds.size()+1):
                new_seed = new_seeds[i-1]

                # No marginal gain unless we're activating a new node
                if seen_set.find(new_seed) == seen_set.end():
                    work_deque.push_back(new_seed)
                    seen_set.insert(new_seed)

                    while work_deque.size() > 0:
                        results[i] += self._compute_payoff(work_deque, self.payoffs)
                        self._advance_model(work_deque, seen_set, thresholds, buckets)

            # Clear thresholds at the end to allow seeding.
            thresholds.clear()

        for i in range(results.size()):
            results[i] /= <float>num_trials

        return results

    # Functions that actually advance the model
    cpdef void advance_until_completion(self):
        while self.work_deque.size() > 0:
            self._advance_model(
                self.work_deque, self.seen_set, self.thresholds, self.buckets
            )

    cpdef void advance_model(self):
        self._advance_model(
            self.work_deque, self.seen_set, self.thresholds, self.buckets
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _advance_model(
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
                        thresholds[child] = random_standard_uniform(self.bitgen_state)
                        while thresholds[child] == 0.0:
                            thresholds[child] = random_standard_uniform(
                                self.bitgen_state
                            )

                    threshold = thresholds[child]
                    # Function is written so that each edge is traversed _once_
                    assert buckets[child] < threshold

                    buckets[child] += self.influence[edge_idx]

                    # Skip if we don't have enough influence yet.
                    if buckets[child] < threshold:
                        continue

                    work_deque.push_back(child)
                    seen_set.insert(child)
