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
Unit tests for HarmonySearch._random_selection() method.

Tests verify that _random_selection():
1. Calls get_value(i) with the correct parameter index
2. Appends the returned value to the harmony list
3. Works correctly for various parameter indices
4. Modifies the harmony list in-place
5. Handles different value types (int, float)
"""

import pytest
from unittest.mock import Mock, MagicMock, call
from pyharmonysearch import HarmonySearch
from tests.fixtures.test_helpers import create_mock_objective_function


class TestRandomSelectionBasics:
    """Test basic functionality of _random_selection method."""
    
    def test_calls_get_value_with_correct_index_first_parameter(self):
        """Test that _random_selection calls get_value(i) with i=0 for first parameter."""
        mock_obj = create_mock_objective_function(num_parameters=3)
        mock_obj.get_value = Mock(return_value=5.5)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        mock_obj.get_value.assert_called_once_with(0)
    
    def test_calls_get_value_with_correct_index_middle_parameter(self):
        """Test that _random_selection calls get_value(i) with i>0 for middle parameters."""
        mock_obj = create_mock_objective_function(num_parameters=5)
        mock_obj.get_value = Mock(return_value=3.14)
        
        hs = HarmonySearch(mock_obj)
        harmony = [1.0, 2.0]
        
        hs._random_selection(harmony, 2)
        
        mock_obj.get_value.assert_called_once_with(2)
    
    def test_calls_get_value_with_correct_index_last_parameter(self):
        """Test that _random_selection calls get_value(i) with correct index for last parameter."""
        mock_obj = create_mock_objective_function(num_parameters=4)
        mock_obj.get_value = Mock(return_value=7.77)
        
        hs = HarmonySearch(mock_obj)
        harmony = [1.0, 2.0, 3.0]
        
        hs._random_selection(harmony, 3)
        
        mock_obj.get_value.assert_called_once_with(3)
    
    def test_appends_value_to_harmony_list(self):
        """Test that _random_selection appends the returned value to the harmony list."""
        mock_obj = create_mock_objective_function(num_parameters=2)
        expected_value = 42.0
        mock_obj.get_value = Mock(return_value=expected_value)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        assert len(harmony) == 1
        assert harmony[0] == expected_value
    
    def test_appends_value_to_non_empty_harmony_list(self):
        """Test that _random_selection appends to a non-empty harmony list."""
        mock_obj = create_mock_objective_function(num_parameters=3)
        expected_value = 99.9
        mock_obj.get_value = Mock(return_value=expected_value)
        
        hs = HarmonySearch(mock_obj)
        harmony = [1.0, 2.0]
        original_length = len(harmony)
        
        hs._random_selection(harmony, 2)
        
        assert len(harmony) == original_length + 1
        assert harmony[2] == expected_value
        # Verify original values are unchanged
        assert harmony[0] == 1.0
        assert harmony[1] == 2.0


class TestRandomSelectionValueTypes:
    """Test _random_selection with different value types."""
    
    def test_handles_float_values(self):
        """Test that _random_selection correctly handles float values."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_continuous=True)
        float_value = 3.14159
        mock_obj.get_value = Mock(return_value=float_value)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        assert isinstance(harmony[0], float)
        assert harmony[0] == float_value
    
    def test_handles_integer_values(self):
        """Test that _random_selection correctly handles integer values."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True)
        int_value = 42
        mock_obj.get_value = Mock(return_value=int_value)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        assert isinstance(harmony[0], int)
        assert harmony[0] == int_value
    
    def test_handles_negative_values(self):
        """Test that _random_selection correctly handles negative values."""
        mock_obj = create_mock_objective_function(num_parameters=2)
        negative_value = -123.456
        mock_obj.get_value = Mock(return_value=negative_value)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        assert harmony[0] == negative_value
    
    def test_handles_zero_value(self):
        """Test that _random_selection correctly handles zero values."""
        mock_obj = create_mock_objective_function(num_parameters=2)
        mock_obj.get_value = Mock(return_value=0.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        assert harmony[0] == 0.0
    
    def test_handles_very_large_values(self):
        """Test that _random_selection correctly handles very large values."""
        mock_obj = create_mock_objective_function(num_parameters=2)
        large_value = 1e100
        mock_obj.get_value = Mock(return_value=large_value)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        assert harmony[0] == large_value
    
    def test_handles_very_small_values(self):
        """Test that _random_selection correctly handles very small values."""
        mock_obj = create_mock_objective_function(num_parameters=2)
        small_value = 1e-100
        mock_obj.get_value = Mock(return_value=small_value)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        assert harmony[0] == small_value


class TestRandomSelectionInPlaceModification:
    """Test that _random_selection modifies the harmony list in-place."""
    
    def test_modifies_harmony_in_place(self):
        """Test that _random_selection modifies the provided harmony list in-place."""
        mock_obj = create_mock_objective_function(num_parameters=2)
        mock_obj.get_value = Mock(return_value=5.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        original_id = id(harmony)
        
        hs._random_selection(harmony, 0)
        
        # Verify the list object is the same (modified in-place, not replaced)
        assert id(harmony) == original_id
        assert len(harmony) == 1
    
    def test_does_not_return_value(self):
        """Test that _random_selection returns None (modifies in-place only)."""
        mock_obj = create_mock_objective_function(num_parameters=2)
        mock_obj.get_value = Mock(return_value=5.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        result = hs._random_selection(harmony, 0)
        
        assert result is None
    
    def test_preserves_existing_harmony_values(self):
        """Test that _random_selection preserves existing values in the harmony list."""
        mock_obj = create_mock_objective_function(num_parameters=3)
        mock_obj.get_value = Mock(return_value=3.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = [1.5, 2.5]
        
        hs._random_selection(harmony, 2)
        
        # Verify existing values are unchanged
        assert harmony[0] == 1.5
        assert harmony[1] == 2.5
        assert harmony[2] == 3.0


class TestRandomSelectionParameterIndices:
    """Test _random_selection with various parameter indices."""
    
    def test_with_parameter_index_zero(self):
        """Test _random_selection with parameter index i=0."""
        mock_obj = create_mock_objective_function(num_parameters=5)
        mock_obj.get_value = Mock(return_value=10.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        mock_obj.get_value.assert_called_once_with(0)
        assert harmony[0] == 10.0
    
    def test_with_parameter_index_one(self):
        """Test _random_selection with parameter index i=1."""
        mock_obj = create_mock_objective_function(num_parameters=5)
        mock_obj.get_value = Mock(return_value=20.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = [10.0]
        
        hs._random_selection(harmony, 1)
        
        mock_obj.get_value.assert_called_once_with(1)
        assert harmony[1] == 20.0
    
    def test_with_large_parameter_index(self):
        """Test _random_selection with a large parameter index."""
        mock_obj = create_mock_objective_function(num_parameters=100)
        mock_obj.get_value = Mock(return_value=99.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = [float(i) for i in range(99)]
        
        hs._random_selection(harmony, 99)
        
        mock_obj.get_value.assert_called_once_with(99)
        assert harmony[99] == 99.0


class TestRandomSelectionSequentialCalls:
    """Test _random_selection with sequential calls to build a complete harmony."""
    
    def test_sequential_calls_build_harmony(self):
        """Test that sequential calls to _random_selection build a complete harmony."""
        mock_obj = create_mock_objective_function(num_parameters=3)
        values = [1.0, 2.0, 3.0]
        mock_obj.get_value = Mock(side_effect=values)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        for i in range(3):
            hs._random_selection(harmony, i)
        
        assert len(harmony) == 3
        assert harmony == [1.0, 2.0, 3.0]
        assert mock_obj.get_value.call_count == 3
        mock_obj.get_value.assert_has_calls([call(0), call(1), call(2)])
    
    def test_multiple_calls_with_different_indices(self):
        """Test multiple calls with different parameter indices."""
        mock_obj = create_mock_objective_function(num_parameters=5)
        
        # Set up mock to return different values for different indices
        def get_value_side_effect(i):
            return float(i * 10)
        
        mock_obj.get_value = Mock(side_effect=get_value_side_effect)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        # Build harmony in order
        for i in range(5):
            hs._random_selection(harmony, i)
        
        assert len(harmony) == 5
        assert harmony == [0.0, 10.0, 20.0, 30.0, 40.0]


class TestRandomSelectionWithRealObjectiveFunctions:
    """Test _random_selection with real objective function fixtures."""
    
    def test_with_continuous_objective_function(self, simple_objective_function):
        """Test _random_selection with a continuous objective function."""
        hs = HarmonySearch(simple_objective_function)
        harmony = []
        
        # Mock get_value to control output
        expected_value = 5.5
        simple_objective_function.get_value = Mock(return_value=expected_value)
        
        hs._random_selection(harmony, 0)
        
        assert len(harmony) == 1
        assert harmony[0] == expected_value
        simple_objective_function.get_value.assert_called_once_with(0)
    
    def test_with_discrete_objective_function(self, discrete_objective_function):
        """Test _random_selection with a discrete objective function."""
        hs = HarmonySearch(discrete_objective_function)
        harmony = []
        
        # Mock get_value to control output
        expected_value = 7
        discrete_objective_function.get_value = Mock(return_value=expected_value)
        
        hs._random_selection(harmony, 0)
        
        assert len(harmony) == 1
        assert harmony[0] == expected_value
        discrete_objective_function.get_value.assert_called_once_with(0)
    
    def test_with_mixed_objective_function(self, mixed_objective_function):
        """Test _random_selection with a mixed (discrete/continuous) objective function."""
        hs = HarmonySearch(mixed_objective_function)
        harmony = []
        
        # Mock get_value to control output
        expected_values = [5, 3.14]  # First is discrete, second is continuous
        mixed_objective_function.get_value = Mock(side_effect=expected_values)
        
        hs._random_selection(harmony, 0)
        hs._random_selection(harmony, 1)
        
        assert len(harmony) == 2
        assert harmony == expected_values


class TestRandomSelectionEdgeCases:
    """Test _random_selection edge cases and boundary conditions."""
    
    def test_with_single_parameter_function(self):
        """Test _random_selection with a single-parameter objective function."""
        mock_obj = create_mock_objective_function(num_parameters=1)
        mock_obj.get_value = Mock(return_value=42.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        assert len(harmony) == 1
        assert harmony[0] == 42.0
    
    def test_with_many_parameter_function(self):
        """Test _random_selection with a many-parameter objective function."""
        num_params = 50
        mock_obj = create_mock_objective_function(num_parameters=num_params)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        # Build a complete harmony
        for i in range(num_params):
            # Use real get_value instead of mocking for this test
            hs._random_selection(harmony, i)
        
        assert len(harmony) == num_params
        # Verify all values are within bounds
        for i, value in enumerate(harmony):
            if not mock_obj.is_discrete(i):
                lower = mock_obj.get_lower_bound(i)
                upper = mock_obj.get_upper_bound(i)
                assert lower <= value <= upper
    
    def test_get_value_called_exactly_once(self):
        """Test that get_value is called exactly once per _random_selection call."""
        mock_obj = create_mock_objective_function(num_parameters=2)
        mock_obj.get_value = Mock(return_value=1.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = []
        
        hs._random_selection(harmony, 0)
        
        # Verify get_value was called exactly once, not multiple times
        assert mock_obj.get_value.call_count == 1
