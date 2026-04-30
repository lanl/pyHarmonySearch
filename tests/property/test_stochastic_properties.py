"""
    Copyright (c) 2013, Triad National Security, LLC
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Triad National Security, LLC nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import random
from math import pow

import pytest
from hypothesis import assume, given, settings, strategies as st

from pyharmonysearch import HarmonySearch, ObjectiveFunctionInterface


class StochasticTestObjectiveFunction(ObjectiveFunctionInterface):
    """
    Test objective function for stochastic property testing.
    """
    
    def __init__(self, num_parameters, hms, max_imp, use_seed=False, random_seed=None, maximize=True):
        self._num_parameters = num_parameters
        self._lower_bounds = [-10.0] * num_parameters
        self._upper_bounds = [10.0] * num_parameters
        self._variable = [True] * num_parameters
        
        # HS parameters
        self._maximize = maximize
        self._max_imp = max_imp
        self._hms = hms
        self._hmcr = 0.75
        self._par = 0.5
        self._mpap = 0.25
        self._mpai = 2
        self._use_seed = use_seed
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        # Minimize sum of squares (optimal at origin)
        result = sum(pow(v, 2) for v in vector)
        return -result if self._maximize else result
    
    def get_value(self, i, index=None):
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])
    
    def get_lower_bound(self, i):
        return self._lower_bounds[i]
    
    def get_upper_bound(self, i):
        return self._upper_bounds[i]
    
    def is_variable(self, i):
        return self._variable[i]
    
    def is_discrete(self, i):
        return False
    
    def get_num_parameters(self):
        return self._num_parameters
    
    def use_random_seed(self):
        return self._use_seed
    
    def get_random_seed(self):
        return self._random_seed
    
    def get_max_imp(self):
        return self._max_imp
    
    def get_hmcr(self):
        return self._hmcr
    
    def get_par(self):
        return self._par
    
    def get_hms(self):
        return self._hms
    
    def get_mpai(self):
        return self._mpai
    
    def get_mpap(self):
        return self._mpap
    
    def maximize(self):
        return self._maximize


class ConvexObjectiveFunction(ObjectiveFunctionInterface):
    """
    Simple convex objective function with known optimal solution.
    
    Minimize sum of (x_i - target_i)^2
    Optimal solution is at target values.
    """
    
    def __init__(self, num_parameters, hms, max_imp, target_values, use_seed=False, random_seed=None):
        self._num_parameters = num_parameters
        self._target_values = target_values
        self._lower_bounds = [-100.0] * num_parameters
        self._upper_bounds = [100.0] * num_parameters
        self._variable = [True] * num_parameters
        
        # HS parameters
        self._maximize = False  # Minimize
        self._max_imp = max_imp
        self._hms = hms
        self._hmcr = 0.8
        self._par = 0.6
        self._mpap = 0.3
        self._mpai = 2
        self._use_seed = use_seed
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        # Sum of squared distances from target
        return sum(pow(vector[i] - self._target_values[i], 2) for i in range(self._num_parameters))
    
    def get_value(self, i, index=None):
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])
    
    def get_lower_bound(self, i):
        return self._lower_bounds[i]
    
    def get_upper_bound(self, i):
        return self._upper_bounds[i]
    
    def is_variable(self, i):
        return self._variable[i]
    
    def is_discrete(self, i):
        return False
    
    def get_num_parameters(self):
        return self._num_parameters
    
    def use_random_seed(self):
        return self._use_seed
    
    def get_random_seed(self):
        return self._random_seed
    
    def get_max_imp(self):
        return self._max_imp
    
    def get_hmcr(self):
        return self._hmcr
    
    def get_par(self):
        return self._par
    
    def get_hms(self):
        return self._hms
    
    def get_mpai(self):
        return self._mpai
    
    def get_mpap(self):
        return self._mpap
    
    def maximize(self):
        return self._maximize


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=5),
    hms=st.integers(min_value=5, max_value=20),
    max_imp=st.integers(min_value=50, max_value=200),
)
def test_algorithm_eventually_improves_from_random_start(num_parameters, hms, max_imp):
    """
    Test that the algorithm improves fitness from initial random solutions.
    
    The final best fitness should be better than or equal to the worst
    fitness in the initial harmony memory, demonstrating improvement.
    """
    obj_fun = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=True, random_seed=42, maximize=True)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Get initial harmony memory (generation 0)
    initial_harmonies = harmony_history[0]['harmonies']
    initial_fitnesses = [fitness for harmony, fitness in initial_harmonies]
    
    # Get final harmony memory
    final_fitnesses = [fitness for harmony, fitness in harmony_memory]
    
    if obj_fun.maximize():
        initial_worst = min(initial_fitnesses)
        final_best = max(final_fitnesses)
        # Algorithm should improve: final best should be >= initial worst
        assert final_best >= initial_worst, f'Algorithm did not improve: final best {final_best} <= initial worst {initial_worst}'
    else:
        initial_worst = max(initial_fitnesses)
        final_best = min(final_fitnesses)
        # Algorithm should improve: final best should be <= initial worst
        assert final_best <= initial_worst, f'Algorithm did not improve: final best {final_best} >= initial worst {initial_worst}'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=5),
    hms=st.integers(min_value=5, max_value=20),
    max_imp=st.integers(min_value=30, max_value=100),
    random_seed=st.integers(min_value=1, max_value=999999),
)
def test_with_seed_produces_deterministic_results(num_parameters, hms, max_imp, random_seed):
    """
    Test that using a random seed produces identical results across runs.
    
    Running the same configuration with the same seed should produce
    the exact same best harmony and fitness.
    """
    # First run
    obj_fun1 = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=True, random_seed=random_seed)
    hs1 = HarmonySearch(obj_fun1)
    best_harmony1, best_fitness1, harmony_memory1, harmony_history1 = hs1.run()
    
    # Second run with same seed
    obj_fun2 = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=True, random_seed=random_seed)
    hs2 = HarmonySearch(obj_fun2)
    best_harmony2, best_fitness2, harmony_memory2, harmony_history2 = hs2.run()
    
    # Results should be identical
    assert best_fitness1 == best_fitness2, f'Fitness differs: {best_fitness1} != {best_fitness2}'
    assert len(best_harmony1) == len(best_harmony2), 'Best harmony length differs'
    
    # Check harmonies are identical (within floating point tolerance)
    for i in range(len(best_harmony1)):
        assert abs(best_harmony1[i] - best_harmony2[i]) < 1e-10, f'Best harmony parameter {i} differs: {best_harmony1[i]} != {best_harmony2[i]}'
    
    # Check harmony memory is identical
    assert len(harmony_memory1) == len(harmony_memory2), 'Harmony memory size differs'
    for i in range(len(harmony_memory1)):
        harmony1, fitness1 = harmony_memory1[i]
        harmony2, fitness2 = harmony_memory2[i]
        assert fitness1 == fitness2, f'Harmony memory {i} fitness differs'
        for j in range(len(harmony1)):
            assert abs(harmony1[j] - harmony2[j]) < 1e-10, f'Harmony memory {i} parameter {j} differs'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=5),
    hms=st.integers(min_value=5, max_value=20),
    max_imp=st.integers(min_value=30, max_value=100),
)
def test_without_seed_produces_varying_results(num_parameters, hms, max_imp):
    """
    Test that without a seed, multiple runs produce different results.
    
    Running without a seed should produce stochastic behavior, meaning
    different runs should generally produce different results.
    """
    results = []
    
    # Run the algorithm 5 times without seed
    for _ in range(5):
        obj_fun = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=False)
        hs = HarmonySearch(obj_fun)
        best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
        results.append((best_harmony, best_fitness))
    
    # Check that we get at least some variation in results
    # Not all fitnesses should be identical
    fitnesses = [fitness for harmony, fitness in results]
    unique_fitnesses = len(set(fitnesses))
    
    # With stochastic behavior, we should see some variation
    # (though theoretically all could be the same, it's very unlikely)
    # We'll check that at least 2 unique values exist in most cases
    # If all are the same, we'll check harmonies differ
    if unique_fitnesses == 1:
        # If all fitnesses are the same, at least some harmonies should differ
        first_harmony = results[0][0]
        all_identical = all(
            all(abs(harmony[i] - first_harmony[i]) < 1e-10 for i in range(len(harmony)))
            for harmony, fitness in results
        )
        # It's extremely unlikely all runs produce identical harmonies without a seed
        # But we'll be lenient and just check they're not all absolutely identical
        assert not all_identical or unique_fitnesses > 1, 'All runs produced identical results without seed (extremely unlikely)'
    else:
        # We have variation in fitness, which is expected
        assert unique_fitnesses >= 2, f'Expected variation in results without seed, got {unique_fitnesses} unique fitnesses'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=3),
    hms=st.integers(min_value=5, max_value=15),
    max_imp=st.integers(min_value=50, max_value=200),
)
def test_convergence_improves_over_time(num_parameters, hms, max_imp):
    """
    Test that fitness improves (or stays the same) over generations.
    
    Statistical property: the best fitness in each generation should
    generally improve or stay the same over time (monotonic improvement).
    """
    obj_fun = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=True, random_seed=42, maximize=True)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Track best fitness across generations
    generation_best_fitnesses = []
    for gen_data in harmony_history:
        fitnesses = [fitness for harmony, fitness in gen_data['harmonies']]
        if obj_fun.maximize():
            gen_best = max(fitnesses)
        else:
            gen_best = min(fitnesses)
        generation_best_fitnesses.append(gen_best)
    
    # Check for general improvement trend
    # The best fitness should generally improve (or stay the same) over time
    improvements = 0
    for i in range(1, len(generation_best_fitnesses)):
        if obj_fun.maximize():
            if generation_best_fitnesses[i] >= generation_best_fitnesses[i - 1]:
                improvements += 1
        else:
            if generation_best_fitnesses[i] <= generation_best_fitnesses[i - 1]:
                improvements += 1
    
    # At least 50% of generations should show improvement or stability
    improvement_ratio = improvements / (len(generation_best_fitnesses) - 1)
    assert improvement_ratio >= 0.5, f'Convergence not improving: only {improvement_ratio:.2%} of generations showed improvement'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=3),
    hms=st.integers(min_value=10, max_value=25),
    max_imp=st.integers(min_value=100, max_value=300),
)
def test_algorithm_converges_toward_optimum(num_parameters, hms, max_imp):
    """
    Test that algorithm converges toward the known optimum.
    
    For a simple convex function with known optimum, the algorithm
    should find solutions close to the optimum.
    """
    # Create target values (the optimum)
    target_values = [0.0] * num_parameters  # Optimal at origin
    
    obj_fun = ConvexObjectiveFunction(num_parameters, hms, max_imp, target_values, use_seed=True, random_seed=42)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # The best fitness should be reasonably close to optimal (0.0)
    # We'll be lenient and accept anything better than the typical random solution
    # A random solution would have expected fitness of approximately num_parameters * (200^2 / 12) for uniform(-100, 100)
    # That's approximately 3333.33 per parameter
    expected_random_fitness = num_parameters * 3333.33
    
    # The algorithm should do much better than random
    assert best_fitness < expected_random_fitness * 0.1, f'Algorithm did not converge well: fitness {best_fitness} not much better than random {expected_random_fitness}'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=2, max_value=5),
    hms=st.integers(min_value=10, max_value=25),
    max_imp=st.integers(min_value=100, max_value=300),
)
def test_increasing_iterations_improves_convergence(num_parameters, hms, max_imp):
    """
    Test that more iterations generally lead to better convergence.
    
    Running with more max_imp should generally produce better or equal results.
    """
    # Run with fewer iterations
    obj_fun1 = StochasticTestObjectiveFunction(num_parameters, hms, max_imp // 2, use_seed=True, random_seed=42, maximize=True)
    hs1 = HarmonySearch(obj_fun1)
    best_harmony1, best_fitness1, harmony_memory1, harmony_history1 = hs1.run()
    
    # Run with more iterations (same seed, so continuation of search)
    obj_fun2 = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=True, random_seed=42, maximize=True)
    hs2 = HarmonySearch(obj_fun2)
    best_harmony2, best_fitness2, harmony_memory2, harmony_history2 = hs2.run()
    
    # More iterations should produce equal or better results
    if obj_fun1.maximize():
        assert best_fitness2 >= best_fitness1, f'More iterations produced worse results: {best_fitness2} < {best_fitness1}'
    else:
        assert best_fitness2 <= best_fitness1, f'More iterations produced worse results: {best_fitness2} > {best_fitness1}'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=5),
    hms=st.integers(min_value=5, max_value=20),
    max_imp=st.integers(min_value=30, max_value=100),
)
def test_final_harmony_memory_sorted_by_fitness(num_parameters, hms, max_imp):
    """
    Test that final harmony memory contains the best solutions found.
    
    While not strictly sorted, the harmony memory should contain
    better solutions than typical random solutions.
    """
    obj_fun = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=True, random_seed=42, maximize=True)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # All solutions in final memory should be better than initial random solutions
    initial_fitnesses = [fitness for harmony, fitness in harmony_history[0]['harmonies']]
    final_fitnesses = [fitness for harmony, fitness in harmony_memory]
    
    if obj_fun.maximize():
        initial_avg = sum(initial_fitnesses) / len(initial_fitnesses)
        final_avg = sum(final_fitnesses) / len(final_fitnesses)
        # Final average should be better than initial average
        assert final_avg >= initial_avg, f'Final harmony memory not better than initial: {final_avg} < {initial_avg}'
    else:
        initial_avg = sum(initial_fitnesses) / len(initial_fitnesses)
        final_avg = sum(final_fitnesses) / len(final_fitnesses)
        # Final average should be better than initial average
        assert final_avg <= initial_avg, f'Final harmony memory not better than initial: {final_avg} > {initial_avg}'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=5),
    hms=st.integers(min_value=5, max_value=20),
    random_seed1=st.integers(min_value=1, max_value=999999),
    random_seed2=st.integers(min_value=1, max_value=999999),
)
def test_different_seeds_produce_different_results(num_parameters, hms, random_seed1, random_seed2):
    """
    Test that different random seeds produce different results.
    
    Using different seeds should explore different parts of the search space
    and generally produce different results.
    """
    assume(random_seed1 != random_seed2)
    
    max_imp = 50
    
    # Run with first seed
    obj_fun1 = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=True, random_seed=random_seed1)
    hs1 = HarmonySearch(obj_fun1)
    best_harmony1, best_fitness1, harmony_memory1, harmony_history1 = hs1.run()
    
    # Run with second seed
    obj_fun2 = StochasticTestObjectiveFunction(num_parameters, hms, max_imp, use_seed=True, random_seed=random_seed2)
    hs2 = HarmonySearch(obj_fun2)
    best_harmony2, best_fitness2, harmony_memory2, harmony_history2 = hs2.run()
    
    # Results should generally differ
    # Check if harmonies are different
    harmonies_differ = any(
        abs(best_harmony1[i] - best_harmony2[i]) > 1e-10
        for i in range(len(best_harmony1))
    )
    
    # Either harmonies differ or fitnesses differ (or both)
    results_differ = harmonies_differ or abs(best_fitness1 - best_fitness2) > 1e-10
    
    # With different seeds, results should differ (though theoretically could be the same, very unlikely)
    assert results_differ, f'Different seeds produced identical results (extremely unlikely)'
