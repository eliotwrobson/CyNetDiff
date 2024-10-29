# Start with imports
from cpython cimport array
from libcpp.deque cimport deque as cdeque
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.algorithm cimport fill
from libcpp.vector cimport vector as cvector
from libcpp.unordered_map cimport unordered_map as cmap

cimport cython

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

    def get_activated_nodes(self):
        raise NotImplementedError

    cpdef void advance_model(self):
        # Function used to advance the model one time step.
        raise NotImplementedError

    cpdef void reset_model(self):
        raise NotImplementedError

    cpdef void advance_until_completion(self):
        raise NotImplementedError

    cdef float _compute_payoff(
        self,
        cset[unsigned int]& new_set,
        cset[unsigned int]& old_set,
        float[:] payoffs,
    ):
        cdef float result = 0.0

        if payoffs is not None:
            for node in new_set:
                if old_set.find(node) == old_set.end():
                    result += payoffs[node]
        else:
            result += new_set.size() - old_set.size()

        return result

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
        float[:] _edge_probabilities = None
    ):

        self.starts = starts
        self.edges = edges
        self.activation_prob = activation_prob
        self.activation_probs = activation_probs
        self.payoffs = payoffs

        self._edge_probabilities = _edge_probabilities

        if self._edge_probabilities is not None:
            assert len(self.edges) == len(self._edge_probabilities)

        if self.activation_probs is not None:
            assert len(self.edges) == len(self.activation_probs)

        if self.payoffs is not None:
            assert len(self.starts) == len(self.payoffs)

    def set_seeds(self, seeds):
        self.original_seeds.clear()
        n = len(self.starts)

        for seed in seeds:
            if not (isinstance(seed, int) and 0 <= seed < n):
                raise ValueError(
                    f"Invalid seed node: {seed}. Must be in the range [0, {n-1}]"
                )
            self.original_seeds.insert(seed)

        self.reset_model()

    cpdef void reset_model(self):
        self.work_deque.assign(self.original_seeds.begin(), self.original_seeds.end())
        self.seen_set.clear()
        self.seen_set.insert(self.original_seeds.begin(), self.original_seeds.end())

    def get_newly_activated_nodes(self):
        for node in self.work_deque:
            yield node

    def get_activated_nodes(self):
        for node in self.seen_set:
            yield node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    def compute_marginal_gain(self, seed_set, new_seed, num_trials):
        cdef cset[unsigned int] original_seeds
        n = len(self.starts)

        for seed in seed_set:
            if not (isinstance(seed, int) and 0 <= seed < n):
                raise ValueError(
                    f"Invalid seed node: {seed}. Must be in the range [0, {n-1}]"
                )
            elif seed == new_seed:
                raise ValueError(
                    f"new_seed {new_seed} should not be contained within the seed set."
                )
            original_seeds.insert(seed)

        if new_seed is None:
            new_seed = n
            # Special value to compute marginal gain differently.
        else:
            if not (isinstance(new_seed, int) and 0 <= new_seed < n):
                raise ValueError(
                    f"Invalid new_seed: {new_seed}. Must be in the range [0, {n-1}]"
                )

        return self._compute_marginal_gain(
            original_seeds, new_seed, num_trials
        )

    cdef float _compute_marginal_gain(
        self,
        cset[unsigned int]& original_seeds,
        unsigned int new_seed,
        unsigned int num_trials
    ):
        cdef cdeque[unsigned int] work_deque
        cdef cset[unsigned int] seen_set
        cdef cset[unsigned int] new_seen_set

        cdef float result = 0.0
        cdef unsigned int n = len(self.starts)

        for _ in range(num_trials):
            work_deque.assign(original_seeds.begin(), original_seeds.end())
            seen_set.clear()
            seen_set.insert(original_seeds.begin(), original_seeds.end())

            while work_deque.size() > 0:
                self._advance_model(work_deque, seen_set)

            if new_seed == n:
                # Use empty new seen set to always return marginal gain.
                result += self._compute_payoff(seen_set, new_seen_set, self.payoffs)

            # No marginal gain unless we're activating a new node
            elif seen_set.find(new_seed) == seen_set.end():
                new_seen_set.clear()
                new_seen_set.insert(seen_set.begin(), seen_set.end())

                work_deque.push_back(new_seed)
                new_seen_set.insert(new_seed)

                while work_deque.size() > 0:
                    self._advance_model(work_deque, new_seen_set)

                result += self._compute_payoff(new_seen_set, seen_set, self.payoffs)

        return result / num_trials

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
            if next_rand() <= activation_prob:
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
    ):
        self.starts = starts
        self.edges = edges
        self.payoffs = payoffs

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

    def set_seeds(self, seeds):
        self.original_seeds.clear()
        n = len(self.starts)

        for seed in seeds:
            if not (isinstance(seed, int) and 0 <= seed < n):
                raise ValueError(
                    f"Invalid seed node: {seed}. Must be in the range [0, {n-1}]"
                )
            self.original_seeds.insert(seed)

        self.reset_model()

    cpdef void _assign_thresholds(self, float[:] _node_thresholds):
        # If provided, copy from user code
        cdef unsigned int n = len(self.starts)
        assert n == len(_node_thresholds)

        # Make a copy to avoid destroying memory on resets.
        for i in range(n):
            self.thresholds[i] = _node_thresholds[i]

    cpdef void reset_model(self):
        self.work_deque.assign(self.original_seeds.begin(), self.original_seeds.end())
        self.seen_set.clear()
        self.seen_set.insert(self.original_seeds.begin(), self.original_seeds.end())
        self.buckets.clear()
        self.thresholds.clear()

    def get_newly_activated_nodes(self):
        for node in self.work_deque:
            yield node

    def get_activated_nodes(self):
        for node in self.seen_set:
            yield node

    def get_num_activated_nodes(self):
        return self.seen_set.size()

    def compute_marginal_gain(
        self,
        seed_set,
        new_seed,
        num_trials,
        *,
        _node_thresholds=None
    ):
        cdef cset[unsigned int] original_seeds
        n = len(self.starts)

        for seed in seed_set:
            if not (isinstance(seed, int) and 0 <= seed < n):
                raise ValueError(
                    f"Invalid seed node: {seed}. Must be in the range [0, {n-1}]"
                )
            elif seed == new_seed:
                raise ValueError(
                    f"new_seed {new_seed} should not be contained within the seed set."
                )
            original_seeds.insert(seed)

        if new_seed is None:
            new_seed = n
            # Special value to compute marginal gain differently.
        else:
            if not (isinstance(new_seed, int) and 0 <= new_seed < n):
                raise ValueError(
                    f"Invalid new_seed: {new_seed}. Must be in the range [0, {n-1}]"
                )

        return self._compute_marginal_gain(
            original_seeds, new_seed, num_trials, _node_thresholds
        )

    cdef float _compute_marginal_gain(
        self,
        cset[unsigned int]& original_seeds,
        unsigned int new_seed,
        unsigned int num_trials,
        float[:] _node_thresholds
    ):
        cdef cdeque[unsigned int] work_deque
        cdef cset[unsigned int] seen_set
        cdef cset[unsigned int] new_seen_set
        cdef cmap[unsigned int, float] thresholds
        cdef cmap[unsigned int, float] buckets

        cdef float result = 0.0
        cdef unsigned int n = len(self.starts)

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
                self._advance_model(work_deque, seen_set, thresholds, buckets)

            if new_seed == n:
                # Use empty new seen set to always return marginal gain.
                result += self._compute_payoff(seen_set, new_seen_set, self.payoffs)

            # No marginal gain unless we're activating a new node
            elif seen_set.find(new_seed) == seen_set.end():
                new_seen_set.clear()
                new_seen_set.insert(seen_set.begin(), seen_set.end())

                work_deque.push_back(new_seed)
                new_seen_set.insert(new_seed)

                while work_deque.size() > 0:
                    self._advance_model(work_deque, new_seen_set, thresholds, buckets)

                result += self._compute_payoff(new_seen_set, seen_set, self.payoffs)

            # Clear thresholds at the end to allow seeding.
            thresholds.clear()

        return result / num_trials

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
                        thresholds[child] = next_rand()
                        while thresholds[child] == 0.0:
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
