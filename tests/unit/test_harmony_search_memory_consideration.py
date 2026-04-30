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
Unit tests for HarmonySearch._memory_consideration() method.

The _memory_consideration() method randomly selects a harmony from the harmony memory
and extracts the i-th parameter value from it, appending it to the harmony being built.

Method signature:
    def _memory_consideration(self, harmony: Harmony, i: int) -> None:
        memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
        harmony.append(self._harmony_memory[memory_index][0][i])
"""

import random
from unittest.mock import Mock, patch
import pytest

from pyharmonysearch import HarmonySearch
from tests.fixtures.mock_objective_functions import QuadraticObjectiveFunction


class TestMemoryConsiderationBasicBehavior:
    """Test basic behavior of _memory_consideration()."""
    
    def test_appends_value_to_harmony_list(self):
        """Test that _memory_consideration() appends exactly one value to the harmony list."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=3, hms=5)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory with known values
        hs._harmony_memory = [
            ([1.0, 2.0, 3.0], 14.0),
            ([4.0, 5.0, 6.0], 77.0),
            ([7.0, 8.0, 9.0], 194.0),
            ([10.0, 11.0, 12.0], 365.0),
            ([13.0, 14.0, 15.0], 590.0),
        ]
        
        harmony = []
        
        # Call _memory_consideration
        with patch('random.randint', return_value=0):
            hs._memory_consideration(harmony, 0)
        
        # Verify exactly one value was appended
        assert len(harmony) == 1
    
    def test_extracts_correct_parameter_index(self):
        """Test that _memory_consideration() extracts the i-th parameter from selected harmony."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=3, hms=5)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory with known values
        hs._harmony_memory = [
            ([1.0, 2.0, 3.0], 14.0),
            ([4.0, 5.0, 6.0], 77.0),
            ([7.0, 8.0, 9.0], 194.0),
            ([10.0, 11.0, 12.0], 365.0),
            ([13.0, 14.0, 15.0], 590.0),
        ]
        
        # Test parameter index 0
        harmony = []
        with patch('random.randint', return_value=2):  # Select harmony at index 2
            hs._memory_consideration(harmony, 0)
        assert harmony[0] == 7.0  # First parameter of third harmony
        
        # Test parameter index 1
        harmony = []
        with patch('random.randint', return_value=2):  # Select harmony at index 2
            hs._memory_consideration(harmony, 1)
        assert harmony[0] == 8.0  # Second parameter of third harmony
        
        # Test parameter index 2
        harmony = []
        with patch('random.randint', return_value=2):  # Select harmony at index 2
            hs._memory_consideration(harmony, 2)
        assert harmony[0] == 9.0  # Third parameter of third harmony
    
    def test_accesses_harmony_not_fitness_from_tuple(self):
        """Test that _memory_consideration() accesses harmony (index 0), not fitness (index 1)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory where harmony values differ from fitness values
        hs._harmony_memory = [
            ([10.0, 20.0], 500.0),   # harmony != fitness
            ([30.0, 40.0], 2500.0),  # harmony != fitness
            ([50.0, 60.0], 6100.0),  # harmony != fitness
        ]
        
        harmony = []
        with patch('random.randint', return_value=1):  # Select harmony at index 1
            hs._memory_consideration(harmony, 0)
        
        # Should get harmony value (30.0), not fitness (2500.0) or anything else
        assert harmony[0] == 30.0
    
    def test_preserves_existing_harmony_values(self):
        """Test that _memory_consideration() appends to existing harmony without modifying it."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=3, hms=2)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 2.0, 3.0], 14.0),
            ([4.0, 5.0, 6.0], 77.0),
        ]
        
        # Start with pre-existing values in harmony
        harmony = [100.0, 200.0]
        
        with patch('random.randint', return_value=0):
            hs._memory_consideration(harmony, 2)
        
        # Verify pre-existing values are unchanged and new value is appended
        assert len(harmony) == 3
        assert harmony[0] == 100.0
        assert harmony[1] == 200.0
        assert harmony[2] == 3.0


class TestMemoryConsiderationRandomSelection:
    """Test random selection behavior of _memory_consideration()."""
    
    def test_random_selection_from_harmony_memory(self):
        """Test that _memory_consideration() randomly selects from all harmonies."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([10.0, 20.0], 500.0),
            ([11.0, 21.0], 562.0),
            ([12.0, 22.0], 628.0),
            ([13.0, 23.0], 698.0),
            ([14.0, 24.0], 772.0),
        ]
        
        # Test with different random indices
        test_cases = [
            (0, 10.0),  # First harmony, first parameter
            (1, 11.0),  # Second harmony, first parameter
            (2, 12.0),  # Third harmony, first parameter
            (3, 13.0),  # Fourth harmony, first parameter
            (4, 14.0),  # Fifth harmony, first parameter
        ]
        
        for random_index, expected_value in test_cases:
            harmony = []
            with patch('random.randint', return_value=random_index):
                hs._memory_consideration(harmony, 0)
            assert harmony[0] == expected_value
    
    def test_uses_random_randint_with_correct_range(self):
        """Test that _memory_consideration() calls random.randint with correct range."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=7)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory
        hs._harmony_memory = [
            ([float(i), float(i+1)], float(i*10))
            for i in range(7)
        ]
        
        harmony = []
        
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 3
            hs._memory_consideration(harmony, 0)
            
            # Verify random.randint was called with (0, HMS-1)
            mock_randint.assert_called_once_with(0, 6)  # 0 to hms-1 (0 to 6)
    
    def test_actual_randomness_distribution(self):
        """Test that _memory_consideration() produces varied random selections over multiple calls."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        # Set random seed for reproducible test
        random.seed(42)
        
        hs._harmony_memory = [
            ([float(i*10), float(i*10+5)], float(i*100))
            for i in range(5)
        ]
        
        # Collect values from 100 selections
        selected_values = []
        for _ in range(100):
            harmony = []
            hs._memory_consideration(harmony, 0)
            selected_values.append(harmony[0])
        
        # Verify we got different values (not all the same)
        unique_values = set(selected_values)
        assert len(unique_values) > 1, "Should select from different harmonies"
        
        # Verify all selected values are from our harmony memory
        expected_values = {0.0, 10.0, 20.0, 30.0, 40.0}
        assert unique_values.issubset(expected_values)


class TestMemoryConsiderationEdgeCases:
    """Test edge cases for _memory_consideration()."""
    
    def test_hms_equals_one(self):
        """Test _memory_consideration() with HMS=1 (only one harmony in memory)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=3, hms=1)
        hs = HarmonySearch(obj_fun)
        
        # Only one harmony in memory
        hs._harmony_memory = [
            ([5.5, 6.6, 7.7], 123.45),
        ]
        
        # With HMS=1, random.randint(0, 0) should always return 0
        harmony = []
        hs._memory_consideration(harmony, 1)
        
        # Should always select from the only harmony available
        assert harmony[0] == 6.6
    
    def test_hms_equals_one_with_mock(self):
        """Test _memory_consideration() with HMS=1 using mock to verify randint call."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=1)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([10.0, 20.0], 500.0),
        ]
        
        harmony = []
        
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 0
            hs._memory_consideration(harmony, 0)
            
            # Verify random.randint was called with (0, 0) for HMS=1
            mock_randint.assert_called_once_with(0, 0)
        
        assert harmony[0] == 10.0
    
    def test_large_hms_value(self):
        """Test _memory_consideration() with large HMS value."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=100)
        hs = HarmonySearch(obj_fun)
        
        # Create large harmony memory
        hs._harmony_memory = [
            ([float(i), float(i+0.5)], float(i*i))
            for i in range(100)
        ]
        
        # Test selection from first and last harmonies
        harmony = []
        with patch('random.randint', return_value=0):
            hs._memory_consideration(harmony, 0)
        assert harmony[0] == 0.0
        
        harmony = []
        with patch('random.randint', return_value=99):
            hs._memory_consideration(harmony, 0)
        assert harmony[0] == 99.0
    
    def test_boundary_indices_hms(self):
        """Test _memory_consideration() with boundary indices (0 and HMS-1)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=10)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([float(i*100), float(i*100+50)], float(i))
            for i in range(10)
        ]
        
        # Test first index (0)
        harmony = []
        with patch('random.randint', return_value=0):
            hs._memory_consideration(harmony, 0)
        assert harmony[0] == 0.0
        
        # Test last index (HMS-1)
        harmony = []
        with patch('random.randint', return_value=9):
            hs._memory_consideration(harmony, 0)
        assert harmony[0] == 900.0
    
    def test_last_parameter_index(self):
        """Test _memory_consideration() with the last parameter index."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=5, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 2.0, 3.0, 4.0, 5.0], 55.0),
            ([6.0, 7.0, 8.0, 9.0, 10.0], 330.0),
            ([11.0, 12.0, 13.0, 14.0, 15.0], 855.0),
        ]
        
        # Test last parameter (index 4)
        harmony = []
        with patch('random.randint', return_value=1):
            hs._memory_consideration(harmony, 4)  # Last parameter
        assert harmony[0] == 10.0


class TestMemoryConsiderationWithDifferentHMS:
    """Test _memory_consideration() with various HMS values."""
    
    @pytest.mark.parametrize("hms_value", [1, 2, 5, 10, 20, 50])
    def test_various_hms_values(self, hms_value):
        """Test _memory_consideration() with different HMS values."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=hms_value)
        hs = HarmonySearch(obj_fun)
        
        # Create harmony memory with HMS harmonies
        hs._harmony_memory = [
            ([float(i), float(i+0.1)], float(i*10))
            for i in range(hms_value)
        ]
        
        # Test that random.randint is called with correct range
        harmony = []
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 0
            hs._memory_consideration(harmony, 0)
            
            # Verify correct range (0 to HMS-1)
            mock_randint.assert_called_once_with(0, hms_value - 1)
    
    @pytest.mark.parametrize("hms_value,memory_index", [
        (2, 0), (2, 1),
        (5, 0), (5, 2), (5, 4),
        (10, 0), (10, 5), (10, 9),
    ])
    def test_various_hms_and_index_combinations(self, hms_value, memory_index):
        """Test _memory_consideration() with various HMS values and memory indices."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=3, hms=hms_value)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([float(i*10), float(i*10+1), float(i*10+2)], float(i))
            for i in range(hms_value)
        ]
        
        harmony = []
        with patch('random.randint', return_value=memory_index):
            hs._memory_consideration(harmony, 1)  # Second parameter
        
        expected_value = float(memory_index * 10 + 1)
        assert harmony[0] == expected_value


class TestMemoryConsiderationIntegration:
    """Integration tests for _memory_consideration() with different objective functions."""
    
    def test_with_discrete_parameters(self):
        """Test _memory_consideration() with discrete parameter values."""
        from tests.fixtures.mock_objective_functions import DiscreteObjectiveFunction
        
        obj_fun = DiscreteObjectiveFunction(num_parameters=2, hms=4)
        hs = HarmonySearch(obj_fun)
        
        # Initialize with discrete values
        hs._harmony_memory = [
            ([-5, -3], 8),
            ([0, 0], 0),
            ([5, 3], 8),
            ([10, -10], 20),
        ]
        
        harmony = []
        with patch('random.randint', return_value=2):
            hs._memory_consideration(harmony, 0)
        
        assert harmony[0] == 5
        assert isinstance(harmony[0], int)
    
    def test_with_mixed_parameters(self):
        """Test _memory_consideration() with mixed continuous and discrete parameters."""
        from tests.fixtures.mock_objective_functions import MixedObjectiveFunction
        
        obj_fun = MixedObjectiveFunction(num_parameters=4, hms=3)
        hs = HarmonySearch(obj_fun)
        
        # Mixed: param 0 and 2 are continuous, 1 and 3 are discrete
        hs._harmony_memory = [
            ([1.5, -5, 2.5, 3], 100.0),
            ([3.7, 0, 4.2, -2], 50.0),
            ([5.1, 10, 6.8, 7], 200.0),
        ]
        
        # Test continuous parameter (index 0)
        harmony = []
        with patch('random.randint', return_value=1):
            hs._memory_consideration(harmony, 0)
        assert harmony[0] == 3.7
        assert isinstance(harmony[0], float)
        
        # Test discrete parameter (index 1)
        harmony = []
        with patch('random.randint', return_value=2):
            hs._memory_consideration(harmony, 1)
        assert harmony[0] == 10
        assert isinstance(harmony[0], int)
    
    def test_multiple_consecutive_calls(self):
        """Test multiple consecutive calls to _memory_consideration() build up harmony."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=4, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 2.0, 3.0, 4.0], 30.0),
            ([5.0, 6.0, 7.0, 8.0], 174.0),
            ([9.0, 10.0, 11.0, 12.0], 446.0),
        ]
        
        harmony = []
        
        # Build harmony parameter by parameter
        with patch('random.randint', return_value=0):
            hs._memory_consideration(harmony, 0)
        assert len(harmony) == 1
        assert harmony[0] == 1.0
        
        with patch('random.randint', return_value=1):
            hs._memory_consideration(harmony, 1)
        assert len(harmony) == 2
        assert harmony[1] == 6.0
        
        with patch('random.randint', return_value=2):
            hs._memory_consideration(harmony, 2)
        assert len(harmony) == 3
        assert harmony[2] == 11.0
        
        with patch('random.randint', return_value=0):
            hs._memory_consideration(harmony, 3)
        assert len(harmony) == 4
        assert harmony[3] == 4.0
        
        # Verify final harmony
        assert harmony == [1.0, 6.0, 11.0, 4.0]


class TestMemoryConsiderationMocking:
    """Test _memory_consideration() using various mocking strategies."""
    
    def test_mock_randint_return_values(self):
        """Test _memory_consideration() with controlled mock return values."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([i * 1.1, i * 2.2], float(i))
            for i in range(5)
        ]
        
        # Test sequence of different random values
        return_sequence = [0, 4, 2, 1, 3]
        expected_values = [0.0, 4.4, 4.4, 2.2, 6.6]
        
        for return_val, expected in zip(return_sequence, expected_values):
            harmony = []
            with patch('random.randint', return_value=return_val):
                hs._memory_consideration(harmony, 1)
            assert harmony[0] == expected
    
    def test_mock_randint_called_once_per_call(self):
        """Test that _memory_consideration() calls random.randint exactly once."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 2.0], 5.0),
            ([3.0, 4.0], 25.0),
            ([5.0, 6.0], 61.0),
        ]
        
        harmony = []
        
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 1
            hs._memory_consideration(harmony, 0)
            
            # Verify random.randint was called exactly once
            assert mock_randint.call_count == 1
    
    def test_mock_entire_random_module(self):
        """Test _memory_consideration() with entire random module mocked."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=4)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([10.0, 20.0], 500.0),
            ([30.0, 40.0], 2500.0),
            ([50.0, 60.0], 6100.0),
            ([70.0, 80.0], 13300.0),
        ]
        
        harmony = []
        
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 3
            hs._memory_consideration(harmony, 0)
            
            # Verify behavior
            assert harmony[0] == 70.0
            mock_randint.assert_called_with(0, 3)
