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

import pytest
import random
from math import pow
from bisect import bisect_left

from pyharmonysearch import HarmonySearch, ObjectiveFunctionInterface


@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Autouse fixture to reset the terminating Event before each test.
    This prevents test pollution from the global state used in multiprocessing.
    """
    import sys
    # Get the actual module from sys.modules (not the function exported in __init__.py)
    hs_module = sys.modules['pyharmonysearch.harmony_search']
    # Reset the terminating event
    hs_module.terminating.clear()
    yield
    # Clean up after test
    hs_module.terminating.clear()


@pytest.fixture
def seed_random():
    """
    Fixture to seed random for deterministic tests.
    Returns a function that can be called with a seed value.
    """
    def _seed(seed_value=42):
        random.seed(seed_value)
        return seed_value
    return _seed


class QuadraticObjectiveFunction(ObjectiveFunctionInterface):
    """
    Simple quadratic objective function for testing.
    
    Goal: maximize -(x^2 + (y+1)^2) + 4
    The maximum is 4 at (0, -1).
    
    Both parameters are continuous and variable.
    """
    
    def __init__(self, use_seed=False, random_seed=12345):
        self._lower_bounds = [-10, -10]
        self._upper_bounds = [10, 10]
        self._variable = [True, True]
        
        # HS parameters
        self._maximize = True
        self._max_imp = 100
        self._hms = 10
        self._hmcr = 0.75
        self._par = 0.5
        self._mpap = 0.25
        self._mpai = 2
        self._use_seed = use_seed
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        return -(pow(vector[0], 2) + pow(vector[1] + 1, 2)) + 4
    
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
        return len(self._lower_bounds)
    
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
    Objective function with discrete parameters for testing.
    
    Goal: maximize -(x^2 + (y+1)^2) + 4
    
    x is discrete (integers from -10 to 10)
    y is discrete (values from a fixed list)
    """
    
    def __init__(self, use_seed=False, random_seed=12345):
        self._discrete_values = [
            [x for x in range(-10, 11)],  # x: -10 to 10 inclusive
            [-5, -3, -1, 0, 1, 3, 5]      # y: predefined values
        ]
        self._variable = [True, True]
        
        # HS parameters
        self._maximize = True
        self._max_imp = 100
        self._hms = 10
        self._hmcr = 0.75
        self._par = 0.5
        self._mpap = 0.25
        self._mpai = 3
        self._use_seed = use_seed
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        return -(pow(vector[0], 2) + pow(vector[1] + 1, 2)) + 4
    
    def get_value(self, i, j=None):
        if j is not None:
            return self._discrete_values[i][j]
        return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
    
    def get_index(self, i, v):
        """Binary search for the value in the discrete values list."""
        discrete_list = self._discrete_values[i]
        idx = bisect_left(discrete_list, v)
        if idx != len(discrete_list) and discrete_list[idx] == v:
            return idx
        raise ValueError(f"Value {v} not found in discrete values for parameter {i}")
    
    def get_num_discrete_values(self, i):
        return len(self._discrete_values[i])
    
    def get_lower_bound(self, i):
        # Not called for discrete variables
        raise NotImplementedError("Lower bound not defined for discrete variables")
    
    def get_upper_bound(self, i):
        # Not called for discrete variables
        raise NotImplementedError("Upper bound not defined for discrete variables")
    
    def is_variable(self, i):
        return self._variable[i]
    
    def is_discrete(self, i):
        return True
    
    def get_num_parameters(self):
        return len(self._discrete_values)
    
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
    Objective function with mixed continuous and discrete parameters.
    
    Goal: maximize -(x^2 + (y+1)^2) + 4
    
    x is discrete (integers from -10 to 10)
    y is continuous (range -10 to 10)
    """
    
    def __init__(self, use_seed=False, random_seed=12345):
        self._discrete_values = [[x for x in range(-10, 11)], None]
        self._lower_bounds = [None, -10]
        self._upper_bounds = [None, 10]
        self._variable = [True, True]
        
        # HS parameters
        self._maximize = True
        self._max_imp = 100
        self._hms = 10
        self._hmcr = 0.75
        self._par = 0.5
        self._mpap = 0.25
        self._mpai = 3
        self._use_seed = use_seed
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        return -(pow(vector[0], 2) + pow(vector[1] + 1, 2)) + 4
    
    def get_value(self, i, j=None):
        if self.is_discrete(i):
            if j is not None:
                return self._discrete_values[i][j]
            return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])
    
    def get_index(self, i, v):
        """Binary search for the value in the discrete values list."""
        if not self.is_discrete(i):
            raise ValueError(f"Parameter {i} is not discrete")
        discrete_list = self._discrete_values[i]
        idx = bisect_left(discrete_list, v)
        if idx != len(discrete_list) and discrete_list[idx] == v:
            return idx
        raise ValueError(f"Value {v} not found in discrete values for parameter {i}")
    
    def get_num_discrete_values(self, i):
        if self.is_discrete(i):
            return len(self._discrete_values[i])
        return float('+inf')
    
    def get_lower_bound(self, i):
        return self._lower_bounds[i]
    
    def get_upper_bound(self, i):
        return self._upper_bounds[i]
    
    def is_variable(self, i):
        return self._variable[i]
    
    def is_discrete(self, i):
        return self._discrete_values[i] is not None
    
    def get_num_parameters(self):
        return len(self._variable)
    
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


@pytest.fixture
def simple_objective_function():
    """
    Returns a simple continuous quadratic objective function.
    Uses seeded random for reproducibility.
    """
    return QuadraticObjectiveFunction(use_seed=True, random_seed=42)


@pytest.fixture
def discrete_objective_function():
    """
    Returns an objective function with all discrete parameters.
    Uses seeded random for reproducibility.
    """
    return DiscreteObjectiveFunction(use_seed=True, random_seed=42)


@pytest.fixture
def mixed_objective_function():
    """
    Returns an objective function with mixed continuous and discrete parameters.
    Uses seeded random for reproducibility.
    """
    return MixedObjectiveFunction(use_seed=True, random_seed=42)


@pytest.fixture
def harmony_search_instance(simple_objective_function):
    """
    Returns a HarmonySearch instance initialized with a simple objective function.
    """
    return HarmonySearch(simple_objective_function)


@pytest.fixture
def sample_harmony():
    """
    Returns a sample harmony vector (solution vector).
    """
    return [1.5, -0.8]


@pytest.fixture
def sample_harmony_memory(simple_objective_function):
    """
    Returns a sample harmony memory structure.
    
    Harmony memory is a list of tuples: [(harmony, fitness), ...]
    """
    harmonies = [
        ([0.0, -1.0], 4.0),      # Optimal solution
        ([1.0, -1.0], 3.0),
        ([0.5, -1.5], 3.75),
        ([-1.0, 0.0], 2.0),
        ([2.0, -2.0], -1.0),
        ([0.2, -0.9], 3.95),
        ([-0.5, -1.5], 3.50),
        ([1.5, -0.5], 1.50),
        ([-1.5, 0.0], 1.75),
        ([0.1, -1.1], 3.98),
    ]
    return harmonies


@pytest.fixture
def sample_harmony_history():
    """
    Returns a sample harmony history structure.
    
    Harmony history is a list of dicts: [{'gen': int, 'harmonies': harmony_memory}, ...]
    """
    return [
        {
            'gen': 0,
            'harmonies': [
                ([5.0, 5.0], -60.0),
                ([3.0, 3.0], -25.0),
            ]
        },
        {
            'gen': 1,
            'harmonies': [
                ([2.0, 0.0], -3.0),
                ([1.0, -1.0], 3.0),
            ]
        },
        {
            'gen': 2,
            'harmonies': [
                ([0.0, -1.0], 4.0),
                ([0.5, -0.8], 3.96),
            ]
        },
    ]
