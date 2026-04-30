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

"""
Comprehensive unit tests for ObjectiveFunctionInterface.

Tests cover:
1. Abstract interface cannot be instantiated directly
2. All 18 methods raise NotImplementedError when not overridden
3. Method signatures and parameter types
4. Concrete implementations properly override all methods
5. Picklability of implementations (required for multiprocessing)
"""

import inspect
import pickle
from typing import List, Optional, Union

import pytest

from pyharmonysearch import ObjectiveFunctionInterface


# Test data for parametrized tests
ALL_METHODS = [
    'get_fitness',
    'get_value',
    'get_index',
    'get_num_discrete_values',
    'get_lower_bound',
    'get_upper_bound',
    'is_variable',
    'is_discrete',
    'get_num_parameters',
    'use_random_seed',
    'get_random_seed',
    'get_max_imp',
    'get_hmcr',
    'get_par',
    'get_hms',
    'get_mpai',
    'get_mpap',
    'maximize',
]


class TestAbstractInterface:
    """Test that ObjectiveFunctionInterface cannot be instantiated directly."""

    def test_interface_can_be_instantiated(self):
        """
        Test that ObjectiveFunctionInterface can be instantiated.
        It's not a true abstract base class (doesn't use abc.ABC),
        so instantiation is allowed, but all methods raise NotImplementedError.
        """
        # Should not raise - the interface is not using abc.ABC
        obj = ObjectiveFunctionInterface()
        assert isinstance(obj, ObjectiveFunctionInterface)

    def test_interface_is_not_abstract_base_class(self):
        """Test that ObjectiveFunctionInterface doesn't use abc.ABC."""
        import abc

        # Should not be an instance of ABCMeta
        assert not isinstance(type(ObjectiveFunctionInterface), type(abc.ABC))


class TestNotImplementedErrors:
    """Test that all methods raise NotImplementedError when not overridden."""

    @pytest.mark.parametrize(
        'method_name,args,kwargs',
        [
            ('get_fitness', ([1.0, 2.0],), {}),
            ('get_value', (0,), {}),
            ('get_value', (0, 1), {}),  # Test with optional j parameter
            ('get_index', (0, 1.0), {}),
            ('get_num_discrete_values', (0,), {}),
            ('get_lower_bound', (0,), {}),
            ('get_upper_bound', (0,), {}),
            ('is_variable', (0,), {}),
            ('is_discrete', (0,), {}),
            ('get_num_parameters', (), {}),
            ('use_random_seed', (), {}),
            ('get_random_seed', (), {}),
            ('get_max_imp', (), {}),
            ('get_hmcr', (), {}),
            ('get_par', (), {}),
            ('get_hms', (), {}),
            ('get_mpai', (), {}),
            ('get_mpap', (), {}),
            ('maximize', (), {}),
        ],
    )
    def test_method_raises_not_implemented_error(self, method_name, args, kwargs):
        """Test that each method raises NotImplementedError when called on base interface."""
        obj = ObjectiveFunctionInterface()
        method = getattr(obj, method_name)

        with pytest.raises(NotImplementedError) as exc_info:
            method(*args, **kwargs)

        # Verify the error message contains the method name
        assert method_name in str(exc_info.value)


class TestMethodSignatures:
    """Test method signatures and type annotations."""

    def test_get_fitness_signature(self):
        """Test get_fitness signature: (vector: List[Union[int, float]]) -> float"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_fitness)
        params = sig.parameters

        assert 'vector' in params
        assert params['vector'].annotation == List[Union[int, float]]
        assert sig.return_annotation == float

    def test_get_value_signature(self):
        """Test get_value signature: (i: int, j: Optional[int] = None) -> Union[int, float]"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_value)
        params = sig.parameters

        assert 'i' in params
        assert params['i'].annotation == int
        assert 'j' in params
        # Note: Optional[int] is implemented as Union[int, None] in Python's typing module
        assert params['j'].default is None
        assert sig.return_annotation == Union[int, float]

    def test_get_index_signature(self):
        """Test get_index signature: (i: int, v: Union[int, float]) -> int"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_index)
        params = sig.parameters

        assert 'i' in params
        assert params['i'].annotation == int
        assert 'v' in params
        assert params['v'].annotation == Union[int, float]
        assert sig.return_annotation == int

    def test_get_num_discrete_values_signature(self):
        """Test get_num_discrete_values signature: (i: int) -> int"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_num_discrete_values)
        params = sig.parameters

        assert 'i' in params
        assert params['i'].annotation == int
        assert sig.return_annotation == int

    def test_get_lower_bound_signature(self):
        """Test get_lower_bound signature: (i: int) -> float"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_lower_bound)
        params = sig.parameters

        assert 'i' in params
        assert params['i'].annotation == int
        assert sig.return_annotation == float

    def test_get_upper_bound_signature(self):
        """Test get_upper_bound signature: (i: int) -> float"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_upper_bound)
        params = sig.parameters

        assert 'i' in params
        assert params['i'].annotation == int
        assert sig.return_annotation == float

    def test_is_variable_signature(self):
        """Test is_variable signature: (i: int) -> bool"""
        sig = inspect.signature(ObjectiveFunctionInterface.is_variable)
        params = sig.parameters

        assert 'i' in params
        assert params['i'].annotation == int
        assert sig.return_annotation == bool

    def test_is_discrete_signature(self):
        """Test is_discrete signature: (i: int) -> bool"""
        sig = inspect.signature(ObjectiveFunctionInterface.is_discrete)
        params = sig.parameters

        assert 'i' in params
        assert params['i'].annotation == int
        assert sig.return_annotation == bool

    def test_get_num_parameters_signature(self):
        """Test get_num_parameters signature: () -> int"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_num_parameters)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == int

    def test_use_random_seed_signature(self):
        """Test use_random_seed signature: () -> bool"""
        sig = inspect.signature(ObjectiveFunctionInterface.use_random_seed)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == bool

    def test_get_random_seed_signature(self):
        """Test get_random_seed signature: () -> int"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_random_seed)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == int

    def test_get_max_imp_signature(self):
        """Test get_max_imp signature: () -> int"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_max_imp)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == int

    def test_get_hmcr_signature(self):
        """Test get_hmcr signature: () -> float"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_hmcr)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == float

    def test_get_par_signature(self):
        """Test get_par signature: () -> float"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_par)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == float

    def test_get_hms_signature(self):
        """Test get_hms signature: () -> int"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_hms)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == int

    def test_get_mpai_signature(self):
        """Test get_mpai signature: () -> int"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_mpai)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == int

    def test_get_mpap_signature(self):
        """Test get_mpap signature: () -> float"""
        sig = inspect.signature(ObjectiveFunctionInterface.get_mpap)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == float

    def test_maximize_signature(self):
        """Test maximize signature: () -> bool"""
        sig = inspect.signature(ObjectiveFunctionInterface.maximize)
        params = sig.parameters

        assert len(params) == 0
        assert sig.return_annotation == bool

    def test_all_methods_exist(self):
        """Test that all 18 expected methods exist on the interface."""
        for method_name in ALL_METHODS:
            assert hasattr(ObjectiveFunctionInterface, method_name)
            method = getattr(ObjectiveFunctionInterface, method_name)
            assert callable(method)

    def test_method_count(self):
        """Test that the interface has exactly 18 public methods (excluding __init__, etc.)."""
        public_methods = [
            name
            for name in dir(ObjectiveFunctionInterface)
            if not name.startswith('_') and callable(getattr(ObjectiveFunctionInterface, name))
        ]
        assert len(public_methods) == 18


class TestConcreteImplementations:
    """Test that concrete implementations properly override all methods."""

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_implementation_is_instance_of_interface(self, fixture_name, request):
        """Test that concrete implementations are instances of ObjectiveFunctionInterface."""
        obj = request.getfixturevalue(fixture_name)
        assert isinstance(obj, ObjectiveFunctionInterface)

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    @pytest.mark.parametrize('method_name', ALL_METHODS)
    def test_implementation_overrides_method(self, fixture_name, method_name, request):
        """Test that concrete implementations override all interface methods."""
        obj = request.getfixturevalue(fixture_name)

        # Get the method from the concrete class
        concrete_method = getattr(type(obj), method_name)
        # Get the method from the interface
        interface_method = getattr(ObjectiveFunctionInterface, method_name)

        # Verify that the method is overridden (not the same as interface method)
        assert concrete_method is not interface_method

    def test_simple_objective_function_all_methods_callable(self, simple_objective_function):
        """Test that all methods are callable and don't raise NotImplementedError."""
        obj = simple_objective_function

        # Test each method with appropriate arguments
        assert isinstance(obj.get_fitness([1.0, 2.0]), float)
        assert isinstance(obj.get_value(0), (int, float))
        assert isinstance(obj.get_lower_bound(0), float)
        assert isinstance(obj.get_upper_bound(0), float)
        assert isinstance(obj.is_variable(0), bool)
        assert isinstance(obj.is_discrete(0), bool)
        assert isinstance(obj.get_num_parameters(), int)
        assert isinstance(obj.use_random_seed(), bool)
        assert isinstance(obj.get_random_seed(), int)
        assert isinstance(obj.get_max_imp(), int)
        assert isinstance(obj.get_hmcr(), float)
        assert isinstance(obj.get_par(), float)
        assert isinstance(obj.get_hms(), int)
        assert isinstance(obj.get_mpai(), int)
        assert isinstance(obj.get_mpap(), float)
        assert isinstance(obj.maximize(), bool)

    def test_discrete_objective_function_all_methods_callable(self, discrete_objective_function):
        """Test discrete implementation methods."""
        obj = discrete_objective_function

        # Test discrete-specific methods
        assert isinstance(obj.get_fitness([1, 2]), float)
        assert isinstance(obj.get_value(0), (int, float))
        assert isinstance(obj.get_value(0, 5), (int, float))  # With index
        assert isinstance(obj.get_index(0, 5), int)
        assert isinstance(obj.get_num_discrete_values(0), int)
        assert isinstance(obj.is_discrete(0), bool)
        assert obj.is_discrete(0) is True  # Should be discrete

    def test_mixed_objective_function_all_methods_callable(self, mixed_objective_function):
        """Test mixed implementation handles both continuous and discrete parameters."""
        obj = mixed_objective_function

        # Test that it handles both parameter types
        assert obj.is_discrete(0) is True  # First parameter is discrete
        assert obj.is_discrete(1) is False  # Second parameter is continuous

        # Test discrete methods work for discrete parameter
        assert isinstance(obj.get_index(0, 5), int)
        assert isinstance(obj.get_num_discrete_values(0), int)

        # Test continuous methods work for continuous parameter
        assert isinstance(obj.get_lower_bound(1), float)
        assert isinstance(obj.get_upper_bound(1), float)


class TestPicklability:
    """Test picklability of implementations (required for multiprocessing)."""

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_implementation_is_picklable(self, fixture_name, request):
        """Test that concrete implementations can be pickled and unpickled."""
        obj = request.getfixturevalue(fixture_name)

        # Pickle the object
        pickled = pickle.dumps(obj)
        assert pickled is not None

        # Unpickle the object
        unpickled = pickle.loads(pickled)
        assert unpickled is not None
        assert isinstance(unpickled, ObjectiveFunctionInterface)

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_unpickled_object_retains_functionality(self, fixture_name, request):
        """Test that unpickled objects retain full functionality."""
        obj = request.getfixturevalue(fixture_name)

        # Get some values before pickling
        num_params_before = obj.get_num_parameters()
        maximize_before = obj.maximize()
        fitness_before = obj.get_fitness([0.0] * num_params_before)

        # Pickle and unpickle
        pickled = pickle.dumps(obj)
        unpickled = pickle.loads(pickled)

        # Verify values are the same after unpickling
        assert unpickled.get_num_parameters() == num_params_before
        assert unpickled.maximize() == maximize_before
        assert unpickled.get_fitness([0.0] * num_params_before) == fitness_before

    def test_simple_objective_function_pickle_roundtrip(self, simple_objective_function):
        """Test detailed pickle roundtrip for simple objective function."""
        obj = simple_objective_function

        # Test multiple methods before pickling
        vector = [1.5, -0.5]
        fitness = obj.get_fitness(vector)
        hmcr = obj.get_hmcr()
        par = obj.get_par()
        hms = obj.get_hms()

        # Pickle and unpickle
        unpickled = pickle.loads(pickle.dumps(obj))

        # Verify all values match
        assert unpickled.get_fitness(vector) == fitness
        assert unpickled.get_hmcr() == hmcr
        assert unpickled.get_par() == par
        assert unpickled.get_hms() == hms

    def test_discrete_objective_function_pickle_roundtrip(self, discrete_objective_function):
        """Test detailed pickle roundtrip for discrete objective function."""
        obj = discrete_objective_function

        # Test discrete-specific methods
        index_value = obj.get_value(0, 5)
        num_discrete = obj.get_num_discrete_values(0)

        # Pickle and unpickle
        unpickled = pickle.loads(pickle.dumps(obj))

        # Verify discrete methods work correctly
        assert unpickled.get_value(0, 5) == index_value
        assert unpickled.get_num_discrete_values(0) == num_discrete


class TestInterfaceContract:
    """Test the overall interface contract and invariants."""

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_get_num_parameters_returns_positive(self, fixture_name, request):
        """Test that get_num_parameters returns a positive integer."""
        obj = request.getfixturevalue(fixture_name)
        num_params = obj.get_num_parameters()
        assert isinstance(num_params, int)
        assert num_params > 0

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_get_hms_returns_positive(self, fixture_name, request):
        """Test that get_hms returns a positive integer."""
        obj = request.getfixturevalue(fixture_name)
        hms = obj.get_hms()
        assert isinstance(hms, int)
        assert hms > 0

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_get_max_imp_returns_positive(self, fixture_name, request):
        """Test that get_max_imp returns a positive integer."""
        obj = request.getfixturevalue(fixture_name)
        max_imp = obj.get_max_imp()
        assert isinstance(max_imp, int)
        assert max_imp > 0

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_hmcr_in_valid_range(self, fixture_name, request):
        """Test that HMCR is in the range [0, 1]."""
        obj = request.getfixturevalue(fixture_name)
        hmcr = obj.get_hmcr()
        assert isinstance(hmcr, float)
        assert 0.0 <= hmcr <= 1.0

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_par_in_valid_range(self, fixture_name, request):
        """Test that PAR is in the range [0, 1]."""
        obj = request.getfixturevalue(fixture_name)
        par = obj.get_par()
        assert isinstance(par, float)
        assert 0.0 <= par <= 1.0

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_mpap_in_valid_range(self, fixture_name, request):
        """Test that MPAP is in the range [0, 1]."""
        obj = request.getfixturevalue(fixture_name)
        mpap = obj.get_mpap()
        assert isinstance(mpap, float)
        assert 0.0 <= mpap <= 1.0

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_get_value_valid_parameter_index(self, fixture_name, request):
        """Test that get_value works for all valid parameter indices."""
        obj = request.getfixturevalue(fixture_name)
        num_params = obj.get_num_parameters()

        for i in range(num_params):
            value = obj.get_value(i)
            assert isinstance(value, (int, float))

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_is_variable_valid_parameter_index(self, fixture_name, request):
        """Test that is_variable works for all valid parameter indices."""
        obj = request.getfixturevalue(fixture_name)
        num_params = obj.get_num_parameters()

        for i in range(num_params):
            is_var = obj.is_variable(i)
            assert isinstance(is_var, bool)

    @pytest.mark.parametrize('fixture_name', ['simple_objective_function', 'discrete_objective_function', 'mixed_objective_function'])
    def test_is_discrete_valid_parameter_index(self, fixture_name, request):
        """Test that is_discrete works for all valid parameter indices."""
        obj = request.getfixturevalue(fixture_name)
        num_params = obj.get_num_parameters()

        for i in range(num_params):
            is_disc = obj.is_discrete(i)
            assert isinstance(is_disc, bool)

    def test_continuous_parameters_have_bounds(self, simple_objective_function):
        """Test that continuous parameters have valid lower and upper bounds."""
        obj = simple_objective_function
        num_params = obj.get_num_parameters()

        for i in range(num_params):
            if not obj.is_discrete(i):
                lower = obj.get_lower_bound(i)
                upper = obj.get_upper_bound(i)
                assert isinstance(lower, float)
                assert isinstance(upper, float)
                assert lower < upper

    def test_discrete_parameters_have_values(self, discrete_objective_function):
        """Test that discrete parameters have valid discrete values."""
        obj = discrete_objective_function
        num_params = obj.get_num_parameters()

        for i in range(num_params):
            if obj.is_discrete(i):
                num_values = obj.get_num_discrete_values(i)
                assert isinstance(num_values, int)
                assert num_values > 0

                # Test getting values by index
                for j in range(min(num_values, 3)):  # Test first 3 values
                    value = obj.get_value(i, j)
                    assert isinstance(value, (int, float))

                    # Test that we can get the index back
                    index = obj.get_index(i, value)
                    assert index == j

    def test_use_random_seed_consistency(self, simple_objective_function):
        """Test that use_random_seed and get_random_seed are consistent."""
        obj = simple_objective_function

        use_seed = obj.use_random_seed()
        assert isinstance(use_seed, bool)

        if use_seed:
            seed = obj.get_random_seed()
            assert isinstance(seed, int)

    def test_fitness_evaluation_with_valid_vector(self, simple_objective_function):
        """Test that fitness can be evaluated with a valid vector."""
        obj = simple_objective_function
        num_params = obj.get_num_parameters()

        # Create a valid vector
        vector = [obj.get_value(i) for i in range(num_params)]

        # Evaluate fitness
        fitness = obj.get_fitness(vector)
        assert isinstance(fitness, float)

    def test_fitness_evaluation_returns_consistent_values(self, simple_objective_function):
        """Test that fitness evaluation returns consistent values for the same input."""
        obj = simple_objective_function
        vector = [1.0, 2.0]

        fitness1 = obj.get_fitness(vector)
        fitness2 = obj.get_fitness(vector)

        assert fitness1 == fitness2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_get_value_with_none_index_for_continuous(self, simple_objective_function):
        """Test that get_value with j=None works for continuous parameters."""
        obj = simple_objective_function

        # For continuous parameters, j should be ignored
        value1 = obj.get_value(0, None)
        assert isinstance(value1, float)

    def test_get_value_with_explicit_index_for_discrete(self, discrete_objective_function):
        """Test that get_value with explicit index works for discrete parameters."""
        obj = discrete_objective_function

        # Get value at index 0
        value = obj.get_value(0, 0)
        assert isinstance(value, (int, float))

    def test_mixed_parameters_both_types(self, mixed_objective_function):
        """Test that mixed parameters handle both continuous and discrete correctly."""
        obj = mixed_objective_function

        # Parameter 0 is discrete
        assert obj.is_discrete(0) is True
        discrete_value = obj.get_value(0, 0)
        assert isinstance(discrete_value, (int, float))

        # Parameter 1 is continuous
        assert obj.is_discrete(1) is False
        continuous_value = obj.get_value(1)
        assert isinstance(continuous_value, float)
