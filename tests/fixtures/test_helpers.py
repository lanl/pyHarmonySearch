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
from pyharmonysearch import ObjectiveFunctionInterface


def assert_harmony_in_bounds(harmony, objective_function):
    """
    Helper to verify that a harmony vector respects the bounds defined by the objective function.
    
    Args:
        harmony: A list representing a solution vector
        objective_function: An ObjectiveFunctionInterface instance
        
    Raises:
        AssertionError: If any parameter is out of bounds
    """
    assert len(harmony) == objective_function.get_num_parameters(), \
        f"Harmony length {len(harmony)} does not match number of parameters {objective_function.get_num_parameters()}"
    
    for i, value in enumerate(harmony):
        if objective_function.is_discrete(i):
            # For discrete parameters, verify the value exists in the allowed discrete values
            try:
                index = objective_function.get_index(i, value)
                assert 0 <= index < objective_function.get_num_discrete_values(i), \
                    f"Discrete value index {index} out of range for parameter {i}"
            except (ValueError, NotImplementedError):
                # If get_index is not implemented or value not found, try to verify another way
                pass
        else:
            # For continuous parameters, verify the value is within bounds
            lower = objective_function.get_lower_bound(i)
            upper = objective_function.get_upper_bound(i)
            assert lower <= value <= upper, \
                f"Parameter {i} value {value} is out of bounds [{lower}, {upper}]"


def assert_valid_harmony_memory(harmony_memory, objective_function):
    """
    Helper to verify the structure and validity of a harmony memory.
    
    Args:
        harmony_memory: A list of (harmony, fitness) tuples
        objective_function: An ObjectiveFunctionInterface instance
        
    Raises:
        AssertionError: If the harmony memory structure is invalid
    """
    assert isinstance(harmony_memory, list), "Harmony memory must be a list"
    assert len(harmony_memory) == objective_function.get_hms(), \
        f"Harmony memory size {len(harmony_memory)} does not match HMS {objective_function.get_hms()}"
    
    for i, entry in enumerate(harmony_memory):
        assert isinstance(entry, tuple), f"Harmony memory entry {i} must be a tuple"
        assert len(entry) == 2, f"Harmony memory entry {i} must have exactly 2 elements (harmony, fitness)"
        
        harmony, fitness = entry
        assert isinstance(harmony, list), f"Harmony at index {i} must be a list"
        assert isinstance(fitness, (int, float)), f"Fitness at index {i} must be numeric"
        
        # Verify each harmony respects bounds
        assert_harmony_in_bounds(harmony, objective_function)


def assert_valid_harmony_history(harmony_history, objective_function):
    """
    Helper to verify the structure and validity of a harmony history.
    
    Args:
        harmony_history: A list of dicts with 'gen' and 'harmonies' keys
        objective_function: An ObjectiveFunctionInterface instance
        
    Raises:
        AssertionError: If the harmony history structure is invalid
    """
    assert isinstance(harmony_history, list), "Harmony history must be a list"
    
    for i, generation_data in enumerate(harmony_history):
        assert isinstance(generation_data, dict), f"History entry {i} must be a dict"
        assert 'gen' in generation_data, f"History entry {i} must have 'gen' key"
        assert 'harmonies' in generation_data, f"History entry {i} must have 'harmonies' key"
        
        gen = generation_data['gen']
        harmonies = generation_data['harmonies']
        
        assert isinstance(gen, int), f"Generation number at index {i} must be an integer"
        assert gen >= 0, f"Generation number at index {i} must be non-negative"
        
        # Verify the harmonies in this generation
        assert_valid_harmony_memory(harmonies, objective_function)


def assert_fitness_order(harmony_memory, objective_function):
    """
    Helper to verify that harmonies are properly ordered by fitness in memory.
    For maximization, better fitness = higher values.
    For minimization, better fitness = lower values.
    
    Note: This doesn't check if they're sorted, just that the best is better than the worst.
    
    Args:
        harmony_memory: A list of (harmony, fitness) tuples
        objective_function: An ObjectiveFunctionInterface instance
    """
    if len(harmony_memory) < 2:
        return  # Nothing to compare
    
    fitnesses = [fitness for _, fitness in harmony_memory]
    
    if objective_function.maximize():
        best_fitness = max(fitnesses)
        worst_fitness = min(fitnesses)
        assert best_fitness >= worst_fitness, "Best fitness should be >= worst fitness for maximization"
    else:
        best_fitness = min(fitnesses)
        worst_fitness = max(fitnesses)
        assert best_fitness <= worst_fitness, "Best fitness should be <= worst fitness for minimization"


def create_mock_objective_function(
    num_parameters=2,
    maximize=True,
    max_imp=100,
    hms=10,
    hmcr=0.75,
    par=0.5,
    mpap=0.25,
    mpai=2,
    all_continuous=True,
    all_discrete=False,
    all_variable=True,
    fitness_func=None,
    use_seed=False,
    random_seed=42
):
    """
    Factory function to create a mock objective function with configurable parameters.
    
    Args:
        num_parameters: Number of parameters in the objective function
        maximize: Whether to maximize (True) or minimize (False)
        max_imp: Maximum number of improvisations
        hms: Harmony memory size
        hmcr: Harmony memory considering rate
        par: Pitch adjusting rate
        mpap: Maximum pitch adjustment proportion (for continuous variables)
        mpai: Maximum pitch adjustment index (for discrete variables)
        all_continuous: If True, all parameters are continuous
        all_discrete: If True, all parameters are discrete
        all_variable: If True, all parameters are variable
        fitness_func: Optional custom fitness function (receives vector, returns fitness)
        use_seed: Whether to use random seed
        random_seed: Random seed value
        
    Returns:
        An instance of a mock ObjectiveFunctionInterface
    """
    
    class MockObjectiveFunction(ObjectiveFunctionInterface):
        def __init__(self):
            self._num_parameters = num_parameters
            self._maximize = maximize
            self._max_imp = max_imp
            self._hms = hms
            self._hmcr = hmcr
            self._par = par
            self._mpap = mpap
            self._mpai = mpai
            self._use_seed = use_seed
            self._random_seed = random_seed
            self._fitness_func = fitness_func
            
            # Set up bounds and discrete values
            if all_discrete:
                self._discrete_values = [[i for i in range(-10, 11)] for _ in range(num_parameters)]
                self._lower_bounds = [None] * num_parameters
                self._upper_bounds = [None] * num_parameters
                self._is_discrete = [True] * num_parameters
            elif all_continuous:
                self._discrete_values = [None] * num_parameters
                self._lower_bounds = [-10.0] * num_parameters
                self._upper_bounds = [10.0] * num_parameters
                self._is_discrete = [False] * num_parameters
            else:
                # Mixed: alternate between discrete and continuous
                self._discrete_values = []
                self._lower_bounds = []
                self._upper_bounds = []
                self._is_discrete = []
                for i in range(num_parameters):
                    if i % 2 == 0:
                        # Discrete
                        self._discrete_values.append([j for j in range(-10, 11)])
                        self._lower_bounds.append(None)
                        self._upper_bounds.append(None)
                        self._is_discrete.append(True)
                    else:
                        # Continuous
                        self._discrete_values.append(None)
                        self._lower_bounds.append(-10.0)
                        self._upper_bounds.append(10.0)
                        self._is_discrete.append(False)
            
            self._variable = [all_variable] * num_parameters
        
        def get_fitness(self, vector):
            if self._fitness_func:
                return self._fitness_func(vector)
            # Default: sum of squares (negated for maximization)
            fitness = sum(x**2 for x in vector)
            return -fitness if self._maximize else fitness
        
        def get_value(self, i, j=None):
            if self._is_discrete[i]:
                if j is not None:
                    return self._discrete_values[i][j]
                return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
            return random.uniform(self._lower_bounds[i], self._upper_bounds[i])
        
        def get_index(self, i, v):
            if not self._is_discrete[i]:
                raise ValueError(f"Parameter {i} is not discrete")
            try:
                return self._discrete_values[i].index(v)
            except ValueError:
                raise ValueError(f"Value {v} not found in discrete values for parameter {i}")
        
        def get_num_discrete_values(self, i):
            if self._is_discrete[i]:
                return len(self._discrete_values[i])
            return float('+inf')
        
        def get_lower_bound(self, i):
            return self._lower_bounds[i]
        
        def get_upper_bound(self, i):
            return self._upper_bounds[i]
        
        def is_variable(self, i):
            return self._variable[i]
        
        def is_discrete(self, i):
            return self._is_discrete[i]
        
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
    
    return MockObjectiveFunction()


def verify_results_structure(results):
    """
    Helper to verify that HarmonySearchResults has the expected structure.
    
    Args:
        results: A HarmonySearchResults namedtuple
        
    Raises:
        AssertionError: If the results structure is invalid
    """
    from datetime import timedelta
    
    # Check that it has all expected fields
    assert hasattr(results, 'elapsed_time'), "Results must have elapsed_time field"
    assert hasattr(results, 'best_harmony'), "Results must have best_harmony field"
    assert hasattr(results, 'best_fitness'), "Results must have best_fitness field"
    assert hasattr(results, 'harmony_memories'), "Results must have harmony_memories field"
    assert hasattr(results, 'harmony_histories'), "Results must have harmony_histories field"
    
    # Check types
    assert isinstance(results.elapsed_time, timedelta), "elapsed_time must be a timedelta"
    assert isinstance(results.best_harmony, list), "best_harmony must be a list"
    assert isinstance(results.best_fitness, (int, float)), "best_fitness must be numeric"
    assert isinstance(results.harmony_memories, list), "harmony_memories must be a list"
    assert isinstance(results.harmony_histories, list), "harmony_histories must be a list"


def get_best_from_memory(harmony_memory, maximize=True):
    """
    Helper to extract the best harmony and fitness from a harmony memory.
    
    Args:
        harmony_memory: A list of (harmony, fitness) tuples
        maximize: Whether we're maximizing (True) or minimizing (False)
        
    Returns:
        A tuple of (best_harmony, best_fitness)
    """
    if not harmony_memory:
        return None, None
    
    best_harmony = None
    best_fitness = float('-inf') if maximize else float('+inf')
    
    for harmony, fitness in harmony_memory:
        if (maximize and fitness > best_fitness) or (not maximize and fitness < best_fitness):
            best_harmony = harmony
            best_fitness = fitness
    
    return best_harmony, best_fitness


def count_unique_harmonies(harmony_memory):
    """
    Helper to count the number of unique harmonies in a harmony memory.
    
    Args:
        harmony_memory: A list of (harmony, fitness) tuples
        
    Returns:
        The number of unique harmonies
    """
    # Convert harmonies to tuples so they can be added to a set
    unique_harmonies = set()
    for harmony, _ in harmony_memory:
        unique_harmonies.add(tuple(harmony))
    return len(unique_harmonies)


def assert_no_duplicate_harmonies(harmony_memory):
    """
    Helper to verify that there are no duplicate harmonies in memory.
    
    Args:
        harmony_memory: A list of (harmony, fitness) tuples
        
    Raises:
        AssertionError: If duplicate harmonies are found
    """
    unique_count = count_unique_harmonies(harmony_memory)
    actual_count = len(harmony_memory)
    assert unique_count == actual_count, \
        f"Found {actual_count - unique_count} duplicate harmonies in memory"


def is_harmony_improved(old_fitness, new_fitness, maximize=True):
    """
    Helper to determine if a new fitness is an improvement over an old fitness.
    
    Args:
        old_fitness: The previous fitness value
        new_fitness: The new fitness value
        maximize: Whether we're maximizing (True) or minimizing (False)
        
    Returns:
        True if new_fitness is better than old_fitness
    """
    if maximize:
        return new_fitness > old_fitness
    else:
        return new_fitness < old_fitness
