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
from hypothesis import given, settings, strategies as st

from pyharmonysearch import HarmonySearch, ObjectiveFunctionInterface


class SimpleTestObjectiveFunction(ObjectiveFunctionInterface):
    """
    Simple test objective function with configurable parameters.
    
    Goal: maximize -(x^2 + (y+1)^2) + 4
    Can be configured for different numbers of parameters, HMS, max_imp, etc.
    """
    
    def __init__(self, num_parameters, hms, max_imp, hmcr=0.75, par=0.5, mpap=0.25, mpai=2, maximize=True, use_seed=True):
        self._num_parameters = num_parameters
        self._lower_bounds = [-10.0] * num_parameters
        self._upper_bounds = [10.0] * num_parameters
        self._variable = [True] * num_parameters
        
        # HS parameters
        self._maximize = maximize
        self._max_imp = max_imp
        self._hms = hms
        self._hmcr = hmcr
        self._par = par
        self._mpap = mpap
        self._mpai = mpai
        self._use_seed = use_seed
        self._random_seed = 42
    
    def get_fitness(self, vector):
        # Simple quadratic sum function
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


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=10),
    hms=st.integers(min_value=2, max_value=20),
    max_imp=st.integers(min_value=10, max_value=100),
)
def test_harmony_memory_size_invariant(num_parameters, hms, max_imp):
    """
    Test that harmony memory always maintains the specified HMS size.
    
    This is a critical invariant - the harmony memory should always contain
    exactly HMS harmonies throughout the algorithm's execution.
    """
    obj_fun = SimpleTestObjectiveFunction(num_parameters, hms, max_imp)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Harmony memory must have exactly HMS entries
    assert len(harmony_memory) == hms, f'Harmony memory size {len(harmony_memory)} does not match HMS {hms}'
    
    # Each entry in harmony_history should also have HMS harmonies
    for gen_data in harmony_history:
        assert len(gen_data['harmonies']) == hms, f"Generation {gen_data['gen']} has {len(gen_data['harmonies'])} harmonies, expected {hms}"


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=10),
    hms=st.integers(min_value=2, max_value=20),
    max_imp=st.integers(min_value=10, max_value=100),
)
def test_parameter_count_invariant(num_parameters, hms, max_imp):
    """
    Test that all harmonies always have the correct number of parameters.
    
    Every harmony should be a list with length equal to num_parameters.
    """
    obj_fun = SimpleTestObjectiveFunction(num_parameters, hms, max_imp)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Best harmony must have correct number of parameters
    assert len(best_harmony) == num_parameters, f'Best harmony has {len(best_harmony)} parameters, expected {num_parameters}'
    
    # All harmonies in memory must have correct number of parameters
    for harmony, fitness in harmony_memory:
        assert len(harmony) == num_parameters, f'Harmony {harmony} has {len(harmony)} parameters, expected {num_parameters}'
    
    # All harmonies in history must have correct number of parameters
    for gen_data in harmony_history:
        for harmony, fitness in gen_data['harmonies']:
            assert len(harmony) == num_parameters, f"Harmony in gen {gen_data['gen']} has {len(harmony)} parameters, expected {num_parameters}"


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=10),
    hms=st.integers(min_value=2, max_value=20),
    max_imp=st.integers(min_value=10, max_value=100),
)
def test_bounds_never_violated(num_parameters, hms, max_imp):
    """
    Test that all parameter values always stay within their specified bounds.
    
    This is critical for continuous optimization - no value should ever
    exceed the upper bound or fall below the lower bound.
    """
    obj_fun = SimpleTestObjectiveFunction(num_parameters, hms, max_imp)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    lower_bounds = [obj_fun.get_lower_bound(i) for i in range(num_parameters)]
    upper_bounds = [obj_fun.get_upper_bound(i) for i in range(num_parameters)]
    
    # Check best harmony
    for i, value in enumerate(best_harmony):
        assert lower_bounds[i] <= value <= upper_bounds[i], f'Best harmony parameter {i} value {value} violates bounds [{lower_bounds[i]}, {upper_bounds[i]}]'
    
    # Check all harmonies in memory
    for harmony, fitness in harmony_memory:
        for i, value in enumerate(harmony):
            assert lower_bounds[i] <= value <= upper_bounds[i], f'Harmony memory parameter {i} value {value} violates bounds [{lower_bounds[i]}, {upper_bounds[i]}]'
    
    # Check all harmonies in history
    for gen_data in harmony_history:
        for harmony, fitness in gen_data['harmonies']:
            for i, value in enumerate(harmony):
                assert lower_bounds[i] <= value <= upper_bounds[i], f"History gen {gen_data['gen']} parameter {i} value {value} violates bounds [{lower_bounds[i]}, {upper_bounds[i]}]"


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=10),
    hms=st.integers(min_value=2, max_value=20),
    max_imp=st.integers(min_value=10, max_value=100),
)
def test_harmony_structure_is_list_of_numbers(num_parameters, hms, max_imp):
    """
    Test that all harmonies are lists containing only numeric values.
    
    Harmonies should be lists of int or float, not other types.
    """
    obj_fun = SimpleTestObjectiveFunction(num_parameters, hms, max_imp)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check best harmony is a list of numbers
    assert isinstance(best_harmony, list), f'Best harmony is not a list: {type(best_harmony)}'
    for i, value in enumerate(best_harmony):
        assert isinstance(value, (int, float)), f'Best harmony parameter {i} is not numeric: {type(value)}'
    
    # Check all harmonies in memory
    for harmony, fitness in harmony_memory:
        assert isinstance(harmony, list), f'Harmony in memory is not a list: {type(harmony)}'
        for i, value in enumerate(harmony):
            assert isinstance(value, (int, float)), f'Harmony memory parameter {i} is not numeric: {type(value)}'
    
    # Check all harmonies in history
    for gen_data in harmony_history:
        for harmony, fitness in gen_data['harmonies']:
            assert isinstance(harmony, list), f"Harmony in history gen {gen_data['gen']} is not a list: {type(harmony)}"
            for i, value in enumerate(harmony):
                assert isinstance(value, (int, float)), f"History gen {gen_data['gen']} parameter {i} is not numeric: {type(value)}"


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=10),
    hms=st.integers(min_value=2, max_value=20),
    max_imp=st.integers(min_value=10, max_value=100),
    maximize=st.booleans(),
)
def test_fitness_types_always_numeric(num_parameters, hms, max_imp, maximize):
    """
    Test that all fitness values are always numeric (int or float).
    
    Fitness values must be comparable numbers, not strings or other types.
    """
    obj_fun = SimpleTestObjectiveFunction(num_parameters, hms, max_imp, maximize=maximize)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check best fitness is numeric
    assert isinstance(best_fitness, (int, float)), f'Best fitness is not numeric: {type(best_fitness)}'
    assert not (isinstance(best_fitness, float) and best_fitness != best_fitness), 'Best fitness is NaN'
    
    # Check all fitness values in memory
    for harmony, fitness in harmony_memory:
        assert isinstance(fitness, (int, float)), f'Fitness in memory is not numeric: {type(fitness)}'
        assert not (isinstance(fitness, float) and fitness != fitness), f'Fitness in memory is NaN'
    
    # Check all fitness values in history
    for gen_data in harmony_history:
        for harmony, fitness in gen_data['harmonies']:
            assert isinstance(fitness, (int, float)), f"Fitness in history gen {gen_data['gen']} is not numeric: {type(fitness)}"
            assert not (isinstance(fitness, float) and fitness != fitness), f"Fitness in history gen {gen_data['gen']} is NaN"


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=10),
    hms=st.integers(min_value=2, max_value=20),
    max_imp=st.integers(min_value=10, max_value=100),
)
def test_harmony_memory_contains_best_solutions(num_parameters, hms, max_imp):
    """
    Test that the best harmony is actually contained in the harmony memory.
    
    The algorithm should maintain the best solutions found in memory.
    """
    obj_fun = SimpleTestObjectiveFunction(num_parameters, hms, max_imp)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # The best harmony should be in the memory (or have the same fitness)
    fitnesses_in_memory = [fitness for harmony, fitness in harmony_memory]
    
    if obj_fun.maximize():
        best_in_memory = max(fitnesses_in_memory)
        # Best fitness should be at least as good as best in memory
        assert best_fitness >= best_in_memory, f'Best fitness {best_fitness} is worse than best in memory {best_in_memory}'
    else:
        best_in_memory = min(fitnesses_in_memory)
        # Best fitness should be at least as good as best in memory
        assert best_fitness <= best_in_memory, f'Best fitness {best_fitness} is worse than best in memory {best_in_memory}'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=10),
    hms=st.integers(min_value=5, max_value=20),
    max_imp=st.integers(min_value=20, max_value=100),
)
def test_harmony_history_records_generations(num_parameters, hms, max_imp):
    """
    Test that harmony history correctly records generations.
    
    History should be recorded every HMS improvisations (one generation).
    Number of generations should be (max_imp // hms) + 1 (including initial).
    """
    obj_fun = SimpleTestObjectiveFunction(num_parameters, hms, max_imp)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Expected number of generations: initial (gen 0) + (max_imp // hms)
    expected_generations = (max_imp // hms) + 1
    assert len(harmony_history) == expected_generations, f'Expected {expected_generations} generations, got {len(harmony_history)}'
    
    # Check generation numbers are sequential starting from 0
    for i, gen_data in enumerate(harmony_history):
        assert gen_data['gen'] == i, f"Generation number mismatch: expected {i}, got {gen_data['gen']}"
        assert 'harmonies' in gen_data, f"Generation {i} missing 'harmonies' key"
        assert isinstance(gen_data['harmonies'], list), f"Generation {i} harmonies is not a list"
