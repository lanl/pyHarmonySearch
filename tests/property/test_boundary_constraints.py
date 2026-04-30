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
from bisect import bisect_left
from math import pow

import pytest
from hypothesis import assume, given, settings, strategies as st

from pyharmonysearch import HarmonySearch, ObjectiveFunctionInterface


class ContinuousObjectiveFunction(ObjectiveFunctionInterface):
    """
    Test objective function with continuous parameters and configurable bounds.
    """
    
    def __init__(self, num_parameters, hms, max_imp, lower_bounds, upper_bounds, hmcr=0.75, par=0.5, mpap=0.25):
        self._num_parameters = num_parameters
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self._variable = [True] * num_parameters
        
        # HS parameters
        self._maximize = True
        self._max_imp = max_imp
        self._hms = hms
        self._hmcr = hmcr
        self._par = par
        self._mpap = mpap
        self._mpai = 2
        self._use_seed = True
        self._random_seed = 42
    
    def get_fitness(self, vector):
        # Simple sum of squares
        return -sum(pow(v, 2) for v in vector)
    
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


class DiscreteObjectiveFunction(ObjectiveFunctionInterface):
    """
    Test objective function with discrete parameters.
    """
    
    def __init__(self, num_parameters, hms, max_imp, discrete_values, hmcr=0.75, par=0.5, mpai=3):
        self._num_parameters = num_parameters
        self._discrete_values = discrete_values
        self._variable = [True] * num_parameters
        
        # HS parameters
        self._maximize = True
        self._max_imp = max_imp
        self._hms = hms
        self._hmcr = hmcr
        self._par = par
        self._mpap = 0.25
        self._mpai = mpai
        self._use_seed = True
        self._random_seed = 42
    
    def get_fitness(self, vector):
        # Simple sum of squares
        return -sum(pow(v, 2) for v in vector)
    
    def get_value(self, i, j=None):
        if j is not None:
            return self._discrete_values[i][j]
        return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
    
    def get_index(self, i, v):
        discrete_list = self._discrete_values[i]
        idx = bisect_left(discrete_list, v)
        if idx != len(discrete_list) and discrete_list[idx] == v:
            return idx
        raise ValueError(f'Value {v} not found in discrete values for parameter {i}')
    
    def get_num_discrete_values(self, i):
        return len(self._discrete_values[i])
    
    def get_lower_bound(self, i):
        raise NotImplementedError('Lower bound not defined for discrete variables')
    
    def get_upper_bound(self, i):
        raise NotImplementedError('Upper bound not defined for discrete variables')
    
    def is_variable(self, i):
        return self._variable[i]
    
    def is_discrete(self, i):
        return True
    
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


class MixedObjectiveFunction(ObjectiveFunctionInterface):
    """
    Test objective function with mixed continuous and discrete parameters.
    """
    
    def __init__(self, continuous_params, discrete_params, hms, max_imp, hmcr=0.75, par=0.5):
        self._continuous_params = continuous_params  # List of (lower, upper) tuples
        self._discrete_params = discrete_params      # List of value lists
        self._num_parameters = len(continuous_params) + len(discrete_params)
        self._variable = [True] * self._num_parameters
        
        # HS parameters
        self._maximize = True
        self._max_imp = max_imp
        self._hms = hms
        self._hmcr = hmcr
        self._par = par
        self._mpap = 0.25
        self._mpai = 3
        self._use_seed = True
        self._random_seed = 42
    
    def get_fitness(self, vector):
        return -sum(pow(v, 2) for v in vector)
    
    def get_value(self, i, j=None):
        if self.is_discrete(i):
            discrete_idx = i - len(self._continuous_params)
            if j is not None:
                return self._discrete_params[discrete_idx][j]
            return self._discrete_params[discrete_idx][random.randint(0, len(self._discrete_params[discrete_idx]) - 1)]
        else:
            lower, upper = self._continuous_params[i]
            return random.uniform(lower, upper)
    
    def get_index(self, i, v):
        if not self.is_discrete(i):
            raise ValueError(f'Parameter {i} is not discrete')
        discrete_idx = i - len(self._continuous_params)
        discrete_list = self._discrete_params[discrete_idx]
        idx = bisect_left(discrete_list, v)
        if idx != len(discrete_list) and discrete_list[idx] == v:
            return idx
        raise ValueError(f'Value {v} not found in discrete values for parameter {i}')
    
    def get_num_discrete_values(self, i):
        if self.is_discrete(i):
            discrete_idx = i - len(self._continuous_params)
            return len(self._discrete_params[discrete_idx])
        return float('+inf')
    
    def get_lower_bound(self, i):
        if self.is_discrete(i):
            raise ValueError(f'Parameter {i} is discrete')
        return self._continuous_params[i][0]
    
    def get_upper_bound(self, i):
        if self.is_discrete(i):
            raise ValueError(f'Parameter {i} is discrete')
        return self._continuous_params[i][1]
    
    def is_variable(self, i):
        return self._variable[i]
    
    def is_discrete(self, i):
        return i >= len(self._continuous_params)
    
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
    hms=st.integers(min_value=3, max_value=15),
    max_imp=st.integers(min_value=20, max_value=80),
    lower_bound=st.floats(min_value=-1000.0, max_value=-0.1),
    upper_bound=st.floats(min_value=0.1, max_value=1000.0),
)
def test_continuous_parameters_stay_within_bounds(num_parameters, hms, max_imp, lower_bound, upper_bound):
    """
    Test that continuous parameters always stay within [lower, upper] bounds.
    
    Tests with wide range of bounds to ensure the algorithm respects them
    in all cases, including pitch adjustment.
    """
    lower_bounds = [lower_bound] * num_parameters
    upper_bounds = [upper_bound] * num_parameters
    
    obj_fun = ContinuousObjectiveFunction(num_parameters, hms, max_imp, lower_bounds, upper_bounds)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check all harmonies in memory respect bounds
    for harmony, fitness in harmony_memory:
        for i, value in enumerate(harmony):
            assert lower_bound <= value <= upper_bound, f'Parameter {i} value {value} violates bounds [{lower_bound}, {upper_bound}]'
    
    # Check all harmonies in history respect bounds
    for gen_data in harmony_history:
        for harmony, fitness in gen_data['harmonies']:
            for i, value in enumerate(harmony):
                assert lower_bound <= value <= upper_bound, f"Gen {gen_data['gen']} parameter {i} value {value} violates bounds [{lower_bound}, {upper_bound}]"


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=1, max_value=5),
    hms=st.integers(min_value=3, max_value=15),
    max_imp=st.integers(min_value=20, max_value=80),
    num_discrete_values=st.integers(min_value=3, max_value=20),
)
def test_discrete_parameters_in_valid_value_sets(num_parameters, hms, max_imp, num_discrete_values):
    """
    Test that discrete parameters always take on values from their valid sets.
    
    Discrete parameters should never produce values outside their defined set.
    """
    # Create discrete value sets for each parameter
    discrete_values = []
    for i in range(num_parameters):
        # Generate sorted unique values
        values = sorted(set([random.uniform(-50, 50) for _ in range(num_discrete_values)]))
        discrete_values.append(values)
    
    obj_fun = DiscreteObjectiveFunction(num_parameters, hms, max_imp, discrete_values)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check all harmonies in memory have valid discrete values
    for harmony, fitness in harmony_memory:
        for i, value in enumerate(harmony):
            assert value in discrete_values[i], f'Parameter {i} value {value} not in valid set {discrete_values[i]}'
    
    # Check all harmonies in history have valid discrete values
    for gen_data in harmony_history:
        for harmony, fitness in gen_data['harmonies']:
            for i, value in enumerate(harmony):
                assert value in discrete_values[i], f"Gen {gen_data['gen']} parameter {i} value {value} not in valid set"


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    hms=st.integers(min_value=3, max_value=15),
    max_imp=st.integers(min_value=30, max_value=80),
    mpap=st.floats(min_value=0.01, max_value=0.99),
    par=st.floats(min_value=0.1, max_value=0.9),
)
def test_pitch_adjustment_maintains_bounds_continuous(hms, max_imp, mpap, par):
    """
    Test that pitch adjustment for continuous parameters maintains bounds.
    
    Pitch adjustment should never cause a parameter to exceed its bounds,
    regardless of the mpap (maximum pitch adjustment proportion) value.
    """
    num_parameters = 3
    lower_bounds = [-100.0, -50.0, -10.0]
    upper_bounds = [100.0, 50.0, 10.0]
    
    obj_fun = ContinuousObjectiveFunction(num_parameters, hms, max_imp, lower_bounds, upper_bounds, par=par, mpap=mpap)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check all harmonies respect bounds after pitch adjustment
    for harmony, fitness in harmony_memory:
        for i, value in enumerate(harmony):
            assert lower_bounds[i] <= value <= upper_bounds[i], f'Parameter {i} value {value} violates bounds [{lower_bounds[i]}, {upper_bounds[i]}] after pitch adjustment'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    hms=st.integers(min_value=3, max_value=15),
    max_imp=st.integers(min_value=30, max_value=80),
    mpai=st.integers(min_value=1, max_value=10),
    par=st.floats(min_value=0.1, max_value=0.9),
)
def test_pitch_adjustment_maintains_valid_discrete_values(hms, max_imp, mpai, par):
    """
    Test that pitch adjustment for discrete parameters maintains valid values.
    
    Pitch adjustment should only produce values that are in the valid discrete set,
    regardless of the mpai (maximum pitch adjustment index) value.
    """
    num_parameters = 2
    discrete_values = [
        list(range(-10, 11)),  # Integers from -10 to 10
        [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
    ]
    
    obj_fun = DiscreteObjectiveFunction(num_parameters, hms, max_imp, discrete_values, par=par, mpai=mpai)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check all harmonies have valid discrete values after pitch adjustment
    for harmony, fitness in harmony_memory:
        for i, value in enumerate(harmony):
            assert value in discrete_values[i], f'Parameter {i} value {value} not in valid discrete set after pitch adjustment'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    hms=st.integers(min_value=3, max_value=15),
    max_imp=st.integers(min_value=30, max_value=80),
    hmcr=st.floats(min_value=0.1, max_value=0.99),
)
def test_random_selection_produces_valid_continuous_values(hms, max_imp, hmcr):
    """
    Test that random selection always produces values within bounds.
    
    When random selection is used (not memory consideration), values should
    still be within the specified bounds.
    """
    num_parameters = 3
    lower_bounds = [-75.0, -25.0, -5.0]
    upper_bounds = [75.0, 25.0, 5.0]
    
    # Use low HMCR to increase random selection probability
    obj_fun = ContinuousObjectiveFunction(num_parameters, hms, max_imp, lower_bounds, upper_bounds, hmcr=hmcr)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check all values are within bounds
    for harmony, fitness in harmony_memory:
        for i, value in enumerate(harmony):
            assert lower_bounds[i] <= value <= upper_bounds[i], f'Parameter {i} value {value} from random selection violates bounds [{lower_bounds[i]}, {upper_bounds[i]}]'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    hms=st.integers(min_value=3, max_value=15),
    max_imp=st.integers(min_value=30, max_value=80),
    hmcr=st.floats(min_value=0.1, max_value=0.99),
)
def test_random_selection_produces_valid_discrete_values(hms, max_imp, hmcr):
    """
    Test that random selection produces values from valid discrete sets.
    
    Random selection for discrete parameters should only return values
    from the defined discrete value set.
    """
    num_parameters = 2
    discrete_values = [
        list(range(-20, 21, 2)),  # Even integers from -20 to 20
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    ]
    
    # Use low HMCR to increase random selection probability
    obj_fun = DiscreteObjectiveFunction(num_parameters, hms, max_imp, discrete_values, hmcr=hmcr)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check all values are in valid discrete sets
    for harmony, fitness in harmony_memory:
        for i, value in enumerate(harmony):
            assert value in discrete_values[i], f'Parameter {i} value {value} from random selection not in valid discrete set'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_continuous=st.integers(min_value=1, max_value=3),
    num_discrete=st.integers(min_value=1, max_value=3),
    hms=st.integers(min_value=3, max_value=15),
    max_imp=st.integers(min_value=30, max_value=80),
)
def test_mixed_parameters_all_respect_constraints(num_continuous, num_discrete, hms, max_imp):
    """
    Test that mixed continuous and discrete parameters all respect their constraints.
    
    When both types of parameters are present, each should maintain its
    respective constraints (bounds for continuous, valid sets for discrete).
    """
    # Create continuous parameters with bounds
    continuous_params = [(-10.0 * (i + 1), 10.0 * (i + 1)) for i in range(num_continuous)]
    
    # Create discrete parameters with value sets
    discrete_params = []
    for i in range(num_discrete):
        num_values = random.randint(5, 15)
        values = sorted(set([random.uniform(-30, 30) for _ in range(num_values)]))
        discrete_params.append(values)
    
    obj_fun = MixedObjectiveFunction(continuous_params, discrete_params, hms, max_imp)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check all harmonies respect constraints
    for harmony, fitness in harmony_memory:
        # Check continuous parameters
        for i in range(num_continuous):
            lower, upper = continuous_params[i]
            value = harmony[i]
            assert lower <= value <= upper, f'Continuous parameter {i} value {value} violates bounds [{lower}, {upper}]'
        
        # Check discrete parameters
        for i in range(num_discrete):
            param_idx = num_continuous + i
            value = harmony[param_idx]
            assert value in discrete_params[i], f'Discrete parameter {param_idx} value {value} not in valid set'


@pytest.mark.property
@settings(max_examples=50, deadline=5000)
@given(
    num_parameters=st.integers(min_value=2, max_value=5),
    hms=st.integers(min_value=3, max_value=15),
    max_imp=st.integers(min_value=30, max_value=80),
)
def test_asymmetric_bounds_respected(num_parameters, hms, max_imp):
    """
    Test that asymmetric bounds (different ranges) are all respected.
    
    Each parameter can have different bounds, and all should be respected
    independently.
    """
    # Create asymmetric bounds for each parameter
    lower_bounds = [random.uniform(-100, -1) for _ in range(num_parameters)]
    upper_bounds = [random.uniform(1, 100) for _ in range(num_parameters)]
    
    obj_fun = ContinuousObjectiveFunction(num_parameters, hms, max_imp, lower_bounds, upper_bounds)
    hs = HarmonySearch(obj_fun)
    
    best_harmony, best_fitness, harmony_memory, harmony_history = hs.run()
    
    # Check all harmonies respect their individual bounds
    for harmony, fitness in harmony_memory:
        for i, value in enumerate(harmony):
            assert lower_bounds[i] <= value <= upper_bounds[i], f'Parameter {i} value {value} violates asymmetric bounds [{lower_bounds[i]}, {upper_bounds[i]}]'
