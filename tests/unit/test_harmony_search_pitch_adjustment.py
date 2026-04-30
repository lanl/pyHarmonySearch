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
Unit tests for HarmonySearch._pitch_adjustment() method.

This is the most complex method in the HarmonySearch class, handling both discrete 
and continuous variables with different adjustment strategies.

Tests cover:
- Non-variable parameters (early return, no adjustment)
- Discrete variables (index-based adjustments, boundary conditions)
- Continuous variables (proportional adjustments, boundary conditions)
- Edge cases and random behavior verification
"""

import pytest
from unittest.mock import Mock, MagicMock, call, patch
from pyharmonysearch import HarmonySearch
from tests.fixtures.test_helpers import create_mock_objective_function


class TestPitchAdjustmentNonVariable:
    """Test pitch adjustment behavior for non-variable parameters."""
    
    def test_non_variable_parameter_no_adjustment(self):
        """Test that non-variable parameters trigger early return without adjustment."""
        mock_obj = create_mock_objective_function(num_parameters=3, all_variable=False)
        mock_obj._variable = [False, True, True]  # First parameter is not variable
        mock_obj.is_variable = Mock(return_value=False)
        mock_obj.is_discrete = Mock()
        
        hs = HarmonySearch(mock_obj)
        harmony = [5.0]
        
        hs._pitch_adjustment(harmony, 0)
        
        # Should return early without checking if discrete
        mock_obj.is_variable.assert_called_once_with(0)
        mock_obj.is_discrete.assert_not_called()
    
    def test_non_variable_parameter_harmony_unchanged(self):
        """Test that harmony list remains unchanged for non-variable parameters."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_variable=False)
        mock_obj._variable = [False, True]
        mock_obj.is_variable = Mock(return_value=False)
        
        hs = HarmonySearch(mock_obj)
        original_value = 42.5
        harmony = [original_value]
        
        hs._pitch_adjustment(harmony, 0)
        
        # Value should not change
        assert len(harmony) == 1
        assert harmony[0] == original_value


class TestPitchAdjustmentDiscrete:
    """Test pitch adjustment behavior for discrete variables."""
    
    def test_discrete_current_index_retrieved_correctly(self):
        """Test that get_index is called with correct parameters to retrieve current index."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=2)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=5)
        mock_obj.get_num_discrete_values = Mock(return_value=10)
        mock_obj.get_value = Mock(return_value=3)
        mock_obj.get_mpai = Mock(return_value=2)
        
        hs = HarmonySearch(mock_obj)
        current_value = 7
        harmony = [current_value]
        
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1):
                hs._pitch_adjustment(harmony, 0)
        
        # Should call get_index with parameter index and current value
        mock_obj.get_index.assert_called_once_with(0, current_value)
    
    def test_discrete_adjust_down_new_index_calculation(self):
        """Test discrete pitch adjustment down: new_index = current_index - randint(0, min(mpai, current_index))."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=3)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=6)  # current_index = 6
        mock_obj.get_num_discrete_values = Mock(return_value=10)
        mock_obj.get_value = Mock(return_value=-2)
        mock_obj.get_mpai = Mock(return_value=3)
        
        hs = HarmonySearch(mock_obj)
        harmony = [5]
        
        # Mock random to adjust down (< 0.5) with randint returning 2
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=2) as mock_randint:
                hs._pitch_adjustment(harmony, 0)
                
                # Should call randint(0, min(3, 6)) = randint(0, 3)
                mock_randint.assert_called_once_with(0, 3)
                
                # Should call get_value with new_index = 6 - 2 = 4
                mock_obj.get_value.assert_called_once_with(0, 4)
    
    def test_discrete_adjust_up_new_index_calculation(self):
        """Test discrete pitch adjustment up: new_index = current_index + randint(0, min(mpai, num_values - current_index - 1))."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=2)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=4)  # current_index = 4
        mock_obj.get_num_discrete_values = Mock(return_value=10)  # num_values = 10
        mock_obj.get_value = Mock(return_value=8)
        mock_obj.get_mpai = Mock(return_value=2)
        
        hs = HarmonySearch(mock_obj)
        harmony = [3]
        
        # Mock random to adjust up (>= 0.5) with randint returning 1
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.7):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1) as mock_randint:
                hs._pitch_adjustment(harmony, 0)
                
                # Should call randint(0, min(2, 10 - 4 - 1)) = randint(0, 2)
                mock_randint.assert_called_once_with(0, 2)
                
                # Should call get_value with new_index = 4 + 1 = 5
                mock_obj.get_value.assert_called_once_with(0, 5)
    
    def test_discrete_edge_case_current_index_zero(self):
        """Test discrete adjustment when current_index=0 (can only adjust up)."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=3)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=0)  # current_index = 0
        mock_obj.get_num_discrete_values = Mock(return_value=10)
        mock_obj.get_value = Mock(return_value=2)
        mock_obj.get_mpai = Mock(return_value=3)
        
        hs = HarmonySearch(mock_obj)
        harmony = [-10]
        
        # Try to adjust down (< 0.5), but min(mpai, current_index) = min(3, 0) = 0
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=0) as mock_randint:
                hs._pitch_adjustment(harmony, 0)
                
                # Should call randint(0, 0), which can only return 0
                mock_randint.assert_called_once_with(0, 0)
                
                # new_index = 0 - 0 = 0 (stays the same)
                mock_obj.get_value.assert_called_once_with(0, 0)
    
    def test_discrete_edge_case_current_index_max(self):
        """Test discrete adjustment when current_index=max (can only adjust down)."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=3)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=9)  # current_index = 9
        mock_obj.get_num_discrete_values = Mock(return_value=10)  # max_index = 9
        mock_obj.get_value = Mock(return_value=5)
        mock_obj.get_mpai = Mock(return_value=3)
        
        hs = HarmonySearch(mock_obj)
        harmony = [10]
        
        # Try to adjust up (>= 0.5), but min(mpai, num_values - current_index - 1) = min(3, 0) = 0
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.7):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=0) as mock_randint:
                hs._pitch_adjustment(harmony, 0)
                
                # Should call randint(0, 0), which can only return 0
                mock_randint.assert_called_once_with(0, 0)
                
                # new_index = 9 + 0 = 9 (stays the same)
                mock_obj.get_value.assert_called_once_with(0, 9)
    
    def test_discrete_mpai_zero_no_adjustment_possible(self):
        """Test discrete adjustment when mpai=0 (no adjustment possible)."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=0)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=5)
        mock_obj.get_num_discrete_values = Mock(return_value=10)
        mock_obj.get_value = Mock(return_value=0)
        mock_obj.get_mpai = Mock(return_value=0)
        
        hs = HarmonySearch(mock_obj)
        harmony = [3]
        
        # Try to adjust down, but min(0, 5) = 0
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=0) as mock_randint:
                hs._pitch_adjustment(harmony, 0)
                
                # Should call randint(0, 0)
                mock_randint.assert_called_once_with(0, 0)
                
                # new_index = 5 - 0 = 5 (no change)
                mock_obj.get_value.assert_called_once_with(0, 5)
    
    def test_discrete_mpai_greater_than_current_index(self):
        """Test discrete adjustment when mpai > current_index (limited by current_index)."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=10)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=3)  # current_index = 3
        mock_obj.get_num_discrete_values = Mock(return_value=20)
        mock_obj.get_value = Mock(return_value=1)
        mock_obj.get_mpai = Mock(return_value=10)
        
        hs = HarmonySearch(mock_obj)
        harmony = [5]
        
        # Adjust down with large mpai
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=2) as mock_randint:
                hs._pitch_adjustment(harmony, 0)
                
                # Should call randint(0, min(10, 3)) = randint(0, 3)
                mock_randint.assert_called_once_with(0, 3)
    
    def test_discrete_mpai_greater_than_remaining_values(self):
        """Test discrete adjustment when mpai > (num_values - current_index - 1) (limited by boundary)."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=10)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=17)  # current_index = 17
        mock_obj.get_num_discrete_values = Mock(return_value=20)  # remaining = 20 - 17 - 1 = 2
        mock_obj.get_value = Mock(return_value=18)
        mock_obj.get_mpai = Mock(return_value=10)
        
        hs = HarmonySearch(mock_obj)
        harmony = [15]
        
        # Adjust up with large mpai
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.7):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1) as mock_randint:
                hs._pitch_adjustment(harmony, 0)
                
                # Should call randint(0, min(10, 2)) = randint(0, 2)
                mock_randint.assert_called_once_with(0, 2)
    
    def test_discrete_random_controls_up_vs_down_direction(self):
        """Test that random.random() < 0.5 adjusts down, >= 0.5 adjusts up."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=2)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=5)
        mock_obj.get_num_discrete_values = Mock(return_value=10)
        mock_obj.get_value = Mock(return_value=3)
        mock_obj.get_mpai = Mock(return_value=2)
        
        hs = HarmonySearch(mock_obj)
        
        # Test adjust down (random < 0.5)
        harmony_down = [7]
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.49):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1):
                hs._pitch_adjustment(harmony_down, 0)
                # Should call get_value with index 5 - 1 = 4
                assert mock_obj.get_value.call_args_list[-1] == call(0, 4)
        
        # Test adjust up (random >= 0.5)
        harmony_up = [7]
        mock_obj.get_value.reset_mock()
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.50):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1):
                hs._pitch_adjustment(harmony_up, 0)
                # Should call get_value with index 5 + 1 = 6
                mock_obj.get_value.assert_called_once_with(0, 6)
    
    def test_discrete_get_value_called_with_new_index(self):
        """Test that get_value is called with the correct new index after adjustment."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=4)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=8)
        mock_obj.get_num_discrete_values = Mock(return_value=15)
        mock_obj.get_value = Mock(return_value=99)
        mock_obj.get_mpai = Mock(return_value=4)
        
        hs = HarmonySearch(mock_obj)
        harmony = [12]
        
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=3):
                hs._pitch_adjustment(harmony, 0)
                
                # new_index = 8 - 3 = 5
                mock_obj.get_value.assert_called_once_with(0, 5)
                # Harmony should be updated with the new value
                assert harmony[0] == 99
    
    def test_discrete_get_num_discrete_values_called(self):
        """Test that get_num_discrete_values is called to determine bounds for upward adjustment."""
        mock_obj = create_mock_objective_function(num_parameters=2, all_discrete=True, mpai=2)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=True)
        mock_obj.get_index = Mock(return_value=5)
        mock_obj.get_num_discrete_values = Mock(return_value=10)
        mock_obj.get_value = Mock(return_value=8)
        mock_obj.get_mpai = Mock(return_value=2)
        
        hs = HarmonySearch(mock_obj)
        harmony = [6]
        
        # Adjust up requires knowing num_discrete_values
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.7):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1):
                hs._pitch_adjustment(harmony, 0)
                
                mock_obj.get_num_discrete_values.assert_called_once_with(0)
    
    def test_discrete_binary_search_with_get_index(self):
        """Test that get_index performs binary search for discrete value lookup."""
        # Use real mock objective function with actual discrete values
        mock_obj = create_mock_objective_function(num_parameters=1, all_discrete=True, mpai=2)
        # Set up discrete values: [-10, -9, -8, ..., 9, 10]
        mock_obj._discrete_values = [list(range(-10, 11))]
        
        hs = HarmonySearch(mock_obj)
        harmony = [5]  # Value 5 is at index 15 in range(-10, 11)
        
        # The mock's get_index uses list.index() which is essentially a linear search,
        # but we can verify it's called correctly
        with patch.object(mock_obj, 'get_index', wraps=mock_obj.get_index) as spy_get_index:
            with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
                with patch('pyharmonysearch.harmony_search.random.randint', return_value=1):
                    hs._pitch_adjustment(harmony, 0)
                    
                    # Verify get_index was called with the current value
                    spy_get_index.assert_called_once_with(0, 5)
    
    def test_discrete_multiple_sequential_adjustments(self):
        """Test multiple sequential pitch adjustments maintain discrete value integrity."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_discrete=True, mpai=3)
        mock_obj._discrete_values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        
        hs = HarmonySearch(mock_obj)
        harmony = [5]  # Start at index 5, value 5
        
        # First adjustment: move down by 2 (index 5 -> 3, value 3)
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=2):
                hs._pitch_adjustment(harmony, 0)
                assert harmony[0] == 3
        
        # Second adjustment: move up by 1 (index 3 -> 4, value 4)
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.7):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1):
                hs._pitch_adjustment(harmony, 0)
                assert harmony[0] == 4
        
        # Third adjustment: move down by 3 (index 4 -> 1, value 1)
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.2):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=3):
                hs._pitch_adjustment(harmony, 0)
                assert harmony[0] == 1


class TestPitchAdjustmentContinuous:
    """Test pitch adjustment behavior for continuous variables."""
    
    def test_continuous_random_controls_up_vs_down(self):
        """Test that random.random() < 0.5 adjusts down, >= 0.5 adjusts up for continuous variables."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.25)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=0.0)
        mock_obj.get_upper_bound = Mock(return_value=10.0)
        mock_obj.get_mpap = Mock(return_value=0.25)
        
        hs = HarmonySearch(mock_obj)
        
        # Test adjust down (random < 0.5)
        harmony = [5.0]
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.3, 0.5]):
            hs._pitch_adjustment(harmony, 0)
            # value -= (5.0 - 0.0) * 0.5 * 0.25 = 5.0 - 0.625 = 4.375
            assert harmony[0] == pytest.approx(4.375)
        
        # Test adjust up (random >= 0.5)
        harmony = [5.0]
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.7, 0.4]):
            hs._pitch_adjustment(harmony, 0)
            # value += (10.0 - 5.0) * 0.4 * 0.25 = 5.0 + 0.5 = 5.5
            assert harmony[0] == pytest.approx(5.5)
    
    def test_continuous_adjust_down_formula(self):
        """Test continuous adjust down formula: value -= (value - lower) * random * mpap."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.3)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=2.0)
        mock_obj.get_upper_bound = Mock(return_value=20.0)
        mock_obj.get_mpap = Mock(return_value=0.3)
        
        hs = HarmonySearch(mock_obj)
        harmony = [10.0]
        
        # Adjust down: value = 10.0, lower = 2.0, random = 0.6, mpap = 0.3
        # new_value = 10.0 - (10.0 - 2.0) * 0.6 * 0.3 = 10.0 - 1.44 = 8.56
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.3, 0.6]):
            hs._pitch_adjustment(harmony, 0)
            assert harmony[0] == pytest.approx(8.56)
    
    def test_continuous_adjust_up_formula(self):
        """Test continuous adjust up formula: value += (upper - value) * random * mpap."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.4)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=0.0)
        mock_obj.get_upper_bound = Mock(return_value=15.0)
        mock_obj.get_mpap = Mock(return_value=0.4)
        
        hs = HarmonySearch(mock_obj)
        harmony = [7.0]
        
        # Adjust up: value = 7.0, upper = 15.0, random = 0.8, mpap = 0.4
        # new_value = 7.0 + (15.0 - 7.0) * 0.8 * 0.4 = 7.0 + 2.56 = 9.56
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.7, 0.8]):
            hs._pitch_adjustment(harmony, 0)
            assert harmony[0] == pytest.approx(9.56)
    
    def test_continuous_value_at_lower_bound_can_only_go_up(self):
        """Test continuous adjustment at lower bound (difference is 0, so no down adjustment)."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.5)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=1.0)
        mock_obj.get_upper_bound = Mock(return_value=10.0)
        mock_obj.get_mpap = Mock(return_value=0.5)
        
        hs = HarmonySearch(mock_obj)
        harmony = [1.0]  # At lower bound
        
        # Try to adjust down: value = 1.0, lower = 1.0
        # new_value = 1.0 - (1.0 - 1.0) * random * mpap = 1.0 - 0 = 1.0 (no change)
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.3, 0.5]):
            hs._pitch_adjustment(harmony, 0)
            assert harmony[0] == pytest.approx(1.0)
    
    def test_continuous_value_at_upper_bound_can_only_go_down(self):
        """Test continuous adjustment at upper bound (difference is 0, so no up adjustment)."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.5)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=0.0)
        mock_obj.get_upper_bound = Mock(return_value=8.0)
        mock_obj.get_mpap = Mock(return_value=0.5)
        
        hs = HarmonySearch(mock_obj)
        harmony = [8.0]  # At upper bound
        
        # Try to adjust up: value = 8.0, upper = 8.0
        # new_value = 8.0 + (8.0 - 8.0) * random * mpap = 8.0 + 0 = 8.0 (no change)
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.7, 0.5]):
            hs._pitch_adjustment(harmony, 0)
            assert harmony[0] == pytest.approx(8.0)
    
    def test_continuous_mpap_zero_no_adjustment(self):
        """Test continuous adjustment with mpap=0 results in no change."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.0)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=0.0)
        mock_obj.get_upper_bound = Mock(return_value=10.0)
        mock_obj.get_mpap = Mock(return_value=0.0)
        
        hs = HarmonySearch(mock_obj)
        harmony = [5.0]
        
        # Adjust down with mpap=0: value -= (5.0 - 0.0) * random * 0.0 = 5.0 - 0 = 5.0
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.3, 0.8]):
            hs._pitch_adjustment(harmony, 0)
            assert harmony[0] == pytest.approx(5.0)
        
        # Adjust up with mpap=0: value += (10.0 - 5.0) * random * 0.0 = 5.0 + 0 = 5.0
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.7, 0.8]):
            hs._pitch_adjustment(harmony, 0)
            assert harmony[0] == pytest.approx(5.0)
    
    def test_continuous_mpap_one_maximum_adjustment(self):
        """Test continuous adjustment with mpap=1.0 allows maximum adjustment range."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=1.0)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=0.0)
        mock_obj.get_upper_bound = Mock(return_value=10.0)
        mock_obj.get_mpap = Mock(return_value=1.0)
        
        hs = HarmonySearch(mock_obj)
        
        # Adjust down with mpap=1.0 and random=1.0: maximum down adjustment
        harmony = [6.0]
        # new_value = 6.0 - (6.0 - 0.0) * 1.0 * 1.0 = 6.0 - 6.0 = 0.0 (reaches lower bound)
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.3, 1.0]):
            hs._pitch_adjustment(harmony, 0)
            assert harmony[0] == pytest.approx(0.0)
        
        # Adjust up with mpap=1.0 and random=1.0: maximum up adjustment
        harmony = [4.0]
        # new_value = 4.0 + (10.0 - 4.0) * 1.0 * 1.0 = 4.0 + 6.0 = 10.0 (reaches upper bound)
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.7, 1.0]):
            hs._pitch_adjustment(harmony, 0)
            assert harmony[0] == pytest.approx(10.0)
    
    def test_continuous_value_stays_within_bounds_after_adjustment(self):
        """Test that continuous values always stay within bounds after adjustment."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.3)
        mock_obj._lower_bounds = [-5.0]
        mock_obj._upper_bounds = [15.0]
        
        hs = HarmonySearch(mock_obj)
        
        # Test multiple adjustments with various values and random numbers
        test_cases = [
            (0.0, 0.3, 0.9),   # Near lower bound, adjust down
            (14.0, 0.7, 0.9),  # Near upper bound, adjust up
            (7.5, 0.3, 0.5),   # Middle, adjust down
            (7.5, 0.7, 0.5),   # Middle, adjust up
        ]
        
        for initial_value, direction_random, adjustment_random in test_cases:
            harmony = [initial_value]
            with patch('pyharmonysearch.harmony_search.random.random', side_effect=[direction_random, adjustment_random]):
                hs._pitch_adjustment(harmony, 0)
                assert -5.0 <= harmony[0] <= 15.0, f"Value {harmony[0]} out of bounds [-5.0, 15.0]"
    
    def test_continuous_get_lower_bound_called(self):
        """Test that get_lower_bound is called for continuous variable adjustment."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.2)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=1.0)
        mock_obj.get_upper_bound = Mock(return_value=10.0)
        mock_obj.get_mpap = Mock(return_value=0.2)
        
        hs = HarmonySearch(mock_obj)
        harmony = [5.0]
        
        # Adjust down requires lower bound
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.3, 0.5]):
            hs._pitch_adjustment(harmony, 0)
            mock_obj.get_lower_bound.assert_called_once_with(0)
    
    def test_continuous_get_upper_bound_called(self):
        """Test that get_upper_bound is called for continuous variable adjustment."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.2)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=0.0)
        mock_obj.get_upper_bound = Mock(return_value=12.0)
        mock_obj.get_mpap = Mock(return_value=0.2)
        
        hs = HarmonySearch(mock_obj)
        harmony = [5.0]
        
        # Adjust up requires upper bound
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.7, 0.5]):
            hs._pitch_adjustment(harmony, 0)
            mock_obj.get_upper_bound.assert_called_once_with(0)
    
    def test_continuous_get_mpap_called(self):
        """Test that get_mpap is called for continuous variable adjustment."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.35)
        mock_obj.is_variable = Mock(return_value=True)
        mock_obj.is_discrete = Mock(return_value=False)
        mock_obj.get_lower_bound = Mock(return_value=0.0)
        mock_obj.get_upper_bound = Mock(return_value=10.0)
        mock_obj.get_mpap = Mock(return_value=0.35)
        
        hs = HarmonySearch(mock_obj)
        harmony = [5.0]
        
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.3, 0.5]):
            hs._pitch_adjustment(harmony, 0)
            mock_obj.get_mpap.assert_called_once()
    
    def test_continuous_multiple_adjustments_maintain_bounds(self):
        """Test that multiple sequential continuous adjustments maintain bounds."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.4)
        mock_obj._lower_bounds = [0.0]
        mock_obj._upper_bounds = [10.0]
        
        hs = HarmonySearch(mock_obj)
        harmony = [5.0]
        
        # Perform 10 sequential adjustments
        for i in range(10):
            direction = 0.3 if i % 2 == 0 else 0.7  # Alternate between down and up
            adjustment = 0.5 + (i * 0.05)  # Vary adjustment amount
            with patch('pyharmonysearch.harmony_search.random.random', side_effect=[direction, adjustment]):
                hs._pitch_adjustment(harmony, 0)
                # After each adjustment, value should stay in bounds
                assert 0.0 <= harmony[0] <= 10.0, f"Iteration {i}: Value {harmony[0]} out of bounds"


class TestPitchAdjustmentIntegration:
    """Integration tests combining different aspects of pitch adjustment."""
    
    def test_mixed_parameters_discrete_and_continuous(self):
        """Test pitch adjustment works correctly with mixed discrete and continuous parameters."""
        mock_obj = create_mock_objective_function(num_parameters=3, all_continuous=False, all_discrete=False)
        # Parameter 0: discrete, Parameter 1: continuous, Parameter 2: discrete
        mock_obj._is_discrete = [True, False, True]
        mock_obj._discrete_values = [[0, 1, 2, 3, 4, 5], None, [10, 20, 30, 40]]
        mock_obj._lower_bounds = [None, 0.0, None]
        mock_obj._upper_bounds = [None, 100.0, None]
        
        hs = HarmonySearch(mock_obj)
        harmony = [3, 50.0, 30]
        
        # Adjust discrete parameter 0
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.3):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1):
                hs._pitch_adjustment(harmony, 0)
                assert harmony[0] == 2  # Index 3 - 1 = 2, value is 2
        
        # Adjust continuous parameter 1
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.7, 0.5]):
            hs._pitch_adjustment(harmony, 1)
            # value += (100.0 - 50.0) * 0.5 * 0.25 = 50.0 + 6.25 = 56.25
            assert harmony[1] == pytest.approx(56.25)
        
        # Adjust discrete parameter 2
        with patch('pyharmonysearch.harmony_search.random.random', return_value=0.7):
            with patch('pyharmonysearch.harmony_search.random.randint', return_value=1):
                hs._pitch_adjustment(harmony, 2)
                assert harmony[2] == 40  # Index 2 + 1 = 3, value is 40
    
    def test_pitch_adjustment_modifies_harmony_in_place(self):
        """Test that pitch adjustment modifies the harmony list in place."""
        mock_obj = create_mock_objective_function(num_parameters=1, all_continuous=True, mpap=0.5)
        
        hs = HarmonySearch(mock_obj)
        harmony = [5.0]
        original_list_id = id(harmony)
        
        with patch('pyharmonysearch.harmony_search.random.random', side_effect=[0.7, 0.6]):
            hs._pitch_adjustment(harmony, 0)
        
        # Should modify the same list object
        assert id(harmony) == original_list_id
        # Value should have changed
        assert harmony[0] != 5.0
