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
from pyharmonysearch import HarmonySearch, ObjectiveFunctionInterface
from tests.fixtures.mock_objective_functions import (
    QuadraticObjectiveFunction,
    DiscreteObjectiveFunction,
    MixedObjectiveFunction,
    FixedParameterObjectiveFunction,
    PathologicalObjectiveFunction
)


class TestHarmonySearchInit:
    """Test suite for HarmonySearch.__init__() method."""
    
    def test_init_with_valid_continuous_objective_function(self):
        """Test initialization with a valid continuous objective function."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=10, max_imp=100)
        hs = HarmonySearch(obj_fun)
        
        # Verify the objective function is stored correctly
        assert hs._obj_fun is obj_fun
        assert isinstance(hs._obj_fun, ObjectiveFunctionInterface)
    
    def test_init_with_valid_discrete_objective_function(self):
        """Test initialization with a valid discrete objective function."""
        obj_fun = DiscreteObjectiveFunction(num_parameters=3, hms=15, max_imp=500)
        hs = HarmonySearch(obj_fun)
        
        # Verify the objective function is stored correctly
        assert hs._obj_fun is obj_fun
        assert isinstance(hs._obj_fun, ObjectiveFunctionInterface)
    
    def test_init_with_valid_mixed_objective_function(self):
        """Test initialization with a mixed continuous/discrete objective function."""
        obj_fun = MixedObjectiveFunction(num_parameters=4, hms=20, max_imp=1000)
        hs = HarmonySearch(obj_fun)
        
        # Verify the objective function is stored correctly
        assert hs._obj_fun is obj_fun
        assert isinstance(hs._obj_fun, ObjectiveFunctionInterface)
    
    def test_objective_function_stored_correctly(self):
        """Test that the objective function reference is correctly stored."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=5, hms=30)
        hs = HarmonySearch(obj_fun)
        
        # Verify we can access objective function methods through the stored reference
        assert hs._obj_fun.get_num_parameters() == 5
        assert hs._obj_fun.get_hms() == 30
        assert callable(hs._obj_fun.get_fitness)
        assert callable(hs._obj_fun.get_value)
    
    def test_init_with_maximization_objective(self):
        """Test initialization with a maximization objective function."""
        obj_fun = QuadraticObjectiveFunction(maximize=True)
        hs = HarmonySearch(obj_fun)
        
        assert hs._obj_fun.maximize() is True
    
    def test_init_with_minimization_objective(self):
        """Test initialization with a minimization objective function."""
        obj_fun = QuadraticObjectiveFunction(maximize=False)
        hs = HarmonySearch(obj_fun)
        
        assert hs._obj_fun.maximize() is False
    
    def test_init_with_seeded_objective_function(self):
        """Test initialization with an objective function that uses a random seed."""
        obj_fun = QuadraticObjectiveFunction(random_seed=12345)
        hs = HarmonySearch(obj_fun)
        
        assert hs._obj_fun.use_random_seed() is True
        assert hs._obj_fun.get_random_seed() == 12345
    
    def test_init_with_unseeded_objective_function(self):
        """Test initialization with an objective function without a random seed."""
        obj_fun = QuadraticObjectiveFunction(random_seed=None)
        hs = HarmonySearch(obj_fun)
        
        assert hs._obj_fun.use_random_seed() is False
    
    def test_init_with_fixed_parameter_objective_function(self):
        """Test initialization with an objective function that has fixed parameters."""
        obj_fun = FixedParameterObjectiveFunction(num_parameters=4)
        hs = HarmonySearch(obj_fun)
        
        # Verify the objective function is stored and has fixed parameters
        assert hs._obj_fun is obj_fun
        assert hs._obj_fun.is_variable(0) is True   # Even indices are variable
        assert hs._obj_fun.is_variable(1) is False  # Odd indices are fixed
    
    def test_init_with_pathological_objective_function(self):
        """Test initialization with a pathological objective function (edge case testing)."""
        obj_fun = PathologicalObjectiveFunction(num_parameters=2, max_imp=50)
        hs = HarmonySearch(obj_fun)
        
        # Verify initialization succeeds even with pathological function
        assert hs._obj_fun is obj_fun
        assert hs._obj_fun.get_max_imp() == 50
    
    def test_init_with_various_hms_values(self):
        """Test initialization with various harmony memory sizes."""
        test_values = [5, 10, 20, 50, 100, 200]
        
        for hms in test_values:
            obj_fun = QuadraticObjectiveFunction(hms=hms)
            hs = HarmonySearch(obj_fun)
            assert hs._obj_fun.get_hms() == hms
    
    def test_init_with_various_parameter_counts(self):
        """Test initialization with objective functions having different parameter counts."""
        test_values = [1, 2, 5, 10, 20, 50]
        
        for num_params in test_values:
            obj_fun = QuadraticObjectiveFunction(num_parameters=num_params)
            hs = HarmonySearch(obj_fun)
            assert hs._obj_fun.get_num_parameters() == num_params
    
    def test_init_with_various_max_imp_values(self):
        """Test initialization with various maximum improvisation values."""
        test_values = [10, 100, 500, 1000, 5000, 10000]
        
        for max_imp in test_values:
            obj_fun = QuadraticObjectiveFunction(max_imp=max_imp)
            hs = HarmonySearch(obj_fun)
            assert hs._obj_fun.get_max_imp() == max_imp
    
    def test_init_with_various_hmcr_values(self):
        """Test initialization with various harmony memory considering rates."""
        test_values = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
        
        for hmcr in test_values:
            obj_fun = QuadraticObjectiveFunction(hmcr=hmcr)
            hs = HarmonySearch(obj_fun)
            assert hs._obj_fun.get_hmcr() == hmcr
    
    def test_init_with_various_par_values(self):
        """Test initialization with various pitch adjusting rates."""
        test_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for par in test_values:
            obj_fun = QuadraticObjectiveFunction(par=par)
            hs = HarmonySearch(obj_fun)
            assert hs._obj_fun.get_par() == par
    
    def test_harmony_memory_not_pre_initialized(self):
        """Test that harmony_memory is not initialized in __init__()."""
        obj_fun = QuadraticObjectiveFunction()
        hs = HarmonySearch(obj_fun)
        
        # harmony_memory should not exist as an attribute after __init__
        assert not hasattr(hs, '_harmony_memory')
    
    def test_harmony_history_not_pre_initialized(self):
        """Test that harmony_history is not initialized in __init__()."""
        obj_fun = QuadraticObjectiveFunction()
        hs = HarmonySearch(obj_fun)
        
        # harmony_history should not exist as an attribute after __init__
        assert not hasattr(hs, '_harmony_history')
    
    def test_no_computation_performed_during_init(self):
        """Test that __init__() does not perform any fitness computations."""
        # Create a mock that tracks if get_fitness was called
        obj_fun = QuadraticObjectiveFunction()
        call_count = [0]
        original_get_fitness = obj_fun.get_fitness
        
        def tracked_get_fitness(vector):
            call_count[0] += 1
            return original_get_fitness(vector)
        
        obj_fun.get_fitness = tracked_get_fitness
        
        # Initialize HarmonySearch
        hs = HarmonySearch(obj_fun)
        
        # Verify get_fitness was not called during initialization
        assert call_count[0] == 0
    
    def test_no_random_values_generated_during_init(self):
        """Test that __init__() does not generate any random values."""
        # Create a mock that tracks if get_value was called
        obj_fun = QuadraticObjectiveFunction()
        call_count = [0]
        original_get_value = obj_fun.get_value
        
        def tracked_get_value(i, j=None):
            call_count[0] += 1
            return original_get_value(i, j)
        
        obj_fun.get_value = tracked_get_value
        
        # Initialize HarmonySearch
        hs = HarmonySearch(obj_fun)
        
        # Verify get_value was not called during initialization
        assert call_count[0] == 0
    
    def test_multiple_instances_with_same_objective_function(self):
        """Test creating multiple HarmonySearch instances with the same objective function."""
        obj_fun = QuadraticObjectiveFunction()
        
        hs1 = HarmonySearch(obj_fun)
        hs2 = HarmonySearch(obj_fun)
        hs3 = HarmonySearch(obj_fun)
        
        # All instances should reference the same objective function
        assert hs1._obj_fun is obj_fun
        assert hs2._obj_fun is obj_fun
        assert hs3._obj_fun is obj_fun
        assert hs1._obj_fun is hs2._obj_fun is hs3._obj_fun
    
    def test_multiple_instances_with_different_objective_functions(self):
        """Test creating multiple HarmonySearch instances with different objective functions."""
        obj_fun1 = QuadraticObjectiveFunction(num_parameters=2)
        obj_fun2 = DiscreteObjectiveFunction(num_parameters=3)
        obj_fun3 = MixedObjectiveFunction(num_parameters=4)
        
        hs1 = HarmonySearch(obj_fun1)
        hs2 = HarmonySearch(obj_fun2)
        hs3 = HarmonySearch(obj_fun3)
        
        # Each instance should reference its own objective function
        assert hs1._obj_fun is obj_fun1
        assert hs2._obj_fun is obj_fun2
        assert hs3._obj_fun is obj_fun3
        assert hs1._obj_fun is not hs2._obj_fun
        assert hs2._obj_fun is not hs3._obj_fun
    
    def test_init_preserves_objective_function_state(self):
        """Test that initialization doesn't modify the objective function's state."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=5,
            hms=25,
            max_imp=500,
            hmcr=0.8,
            par=0.3,
            mpap=0.2,
            mpai=3,
            maximize=True,
            random_seed=99999
        )
        
        # Store original values
        original_params = obj_fun.get_num_parameters()
        original_hms = obj_fun.get_hms()
        original_max_imp = obj_fun.get_max_imp()
        original_hmcr = obj_fun.get_hmcr()
        original_par = obj_fun.get_par()
        original_mpap = obj_fun.get_mpap()
        original_mpai = obj_fun.get_mpai()
        original_maximize = obj_fun.maximize()
        original_seed = obj_fun.get_random_seed()
        
        # Initialize HarmonySearch
        hs = HarmonySearch(obj_fun)
        
        # Verify all values remain unchanged
        assert obj_fun.get_num_parameters() == original_params
        assert obj_fun.get_hms() == original_hms
        assert obj_fun.get_max_imp() == original_max_imp
        assert obj_fun.get_hmcr() == original_hmcr
        assert obj_fun.get_par() == original_par
        assert obj_fun.get_mpap() == original_mpap
        assert obj_fun.get_mpai() == original_mpai
        assert obj_fun.maximize() == original_maximize
        assert obj_fun.get_random_seed() == original_seed
    
    def test_init_with_custom_objective_function_subclass(self):
        """Test initialization with a custom ObjectiveFunctionInterface subclass."""
        # Create a minimal custom objective function inline
        import random
        
        class CustomObjectiveFunction(ObjectiveFunctionInterface):
            def get_fitness(self, vector):
                return sum(vector)
            
            def get_value(self, i, j=None):
                return random.uniform(-10, 10)
            
            def get_index(self, i, v):
                raise NotImplementedError()
            
            def get_num_discrete_values(self, i):
                return float('+inf')
            
            def get_lower_bound(self, i):
                return -10.0
            
            def get_upper_bound(self, i):
                return 10.0
            
            def is_variable(self, i):
                return True
            
            def is_discrete(self, i):
                return False
            
            def get_num_parameters(self):
                return 3
            
            def use_random_seed(self):
                return False
            
            def get_random_seed(self):
                return None
            
            def get_max_imp(self):
                return 100
            
            def get_hmcr(self):
                return 0.75
            
            def get_par(self):
                return 0.5
            
            def get_hms(self):
                return 10
            
            def get_mpai(self):
                return 2
            
            def get_mpap(self):
                return 0.25
            
            def maximize(self):
                return False
        
        obj_fun = CustomObjectiveFunction()
        hs = HarmonySearch(obj_fun)
        
        assert hs._obj_fun is obj_fun
        assert isinstance(hs._obj_fun, ObjectiveFunctionInterface)
