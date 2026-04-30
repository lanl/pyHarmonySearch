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
Unit tests for HarmonySearch._initialize() method.

Tests verify that _initialize():
1. With initial_harmonies: validates correct length
2. With initial_harmonies: raises ValueError on length mismatch
3. With initial_harmonies: validates each harmony has correct parameters
4. With initial_harmonies: raises ValueError on parameter count mismatch
5. With initial_harmonies: calculates fitness for each harmony
6. With initial_harmonies: stores as (harmony, fitness) tuples
7. Without initial_harmonies: creates HMS harmonies
8. Without initial_harmonies: each harmony has num_parameters values
9. Without initial_harmonies: uses _random_selection() method
10. Without initial_harmonies: calculates fitness for each
11. History initialization: adds first entry with gen=0
12. History initialization: contains deep copy of memory
13. Edge case: empty initial_harmonies list
14. Edge case: initial_harmonies with empty harmonies
15. Edge case: initial_harmonies with wrong dimensions
16. Edge case: HMS=0
17. Edge case: HMS=1 (minimal memory)
18. Edge case: num_parameters=0
19. Edge case: num_parameters=1
20. Error message verification: wrong HMS
21. Error message verification: wrong parameters
22. Deep copy verification: modifying history doesn't affect memory
"""

import pytest
import copy
from unittest.mock import Mock, patch, call
from pyharmonysearch import HarmonySearch
from tests.fixtures.test_helpers import (
    create_mock_objective_function,
    assert_valid_harmony_memory,
    assert_valid_harmony_history,
)


class TestInitializeWithInitialHarmonies:
    """Test _initialize() method when initial_harmonies are provided."""
    
    def test_accepts_initial_harmonies_with_correct_length(self):
        """Test that _initialize accepts initial_harmonies when length equals HMS."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=3)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        # Should not raise any exception
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        assert len(hs._harmony_memory) == 3
    
    def test_raises_valueerror_on_length_mismatch(self):
        """Test that _initialize raises ValueError when initial_harmonies length != HMS."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=5)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        with pytest.raises(ValueError) as excinfo:
            hs._initialize(initial_harmonies)
        
        assert "Number of initial harmonies does not equal to the harmony memory size" in str(excinfo.value)
    
    def test_validates_each_harmony_has_correct_parameters(self):
        """Test that _initialize validates each harmony has correct number of parameters."""
        mock_obj = create_mock_objective_function(num_parameters=3, hms=2)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        
        # Should not raise any exception
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        assert len(hs._harmony_memory) == 2
        assert len(hs._harmony_memory[0][0]) == 3
        assert len(hs._harmony_memory[1][0]) == 3
    
    def test_raises_valueerror_on_parameter_count_mismatch(self):
        """Test that _initialize raises ValueError when harmony has wrong parameter count."""
        mock_obj = create_mock_objective_function(num_parameters=3, hms=2)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],  # Wrong: only 2 parameters instead of 3
            [4.0, 5.0, 6.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        with pytest.raises(ValueError) as excinfo:
            hs._initialize(initial_harmonies)
        
        assert "Number of parameters in initial harmonies does not match that defined" in str(excinfo.value)
    
    def test_calculates_fitness_for_each_harmony(self):
        """Test that _initialize calculates fitness for each provided harmony."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=3)
        mock_obj.get_fitness = Mock(side_effect=[10.0, 20.0, 30.0])
        
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        # Verify get_fitness was called for each harmony
        assert mock_obj.get_fitness.call_count == 3
        mock_obj.get_fitness.assert_any_call([1.0, 2.0])
        mock_obj.get_fitness.assert_any_call([3.0, 4.0])
        mock_obj.get_fitness.assert_any_call([5.0, 6.0])
    
    def test_stores_harmonies_as_tuples(self):
        """Test that _initialize stores harmonies as (harmony, fitness) tuples."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=2)
        mock_obj.get_fitness = Mock(side_effect=[15.5, 25.7])
        
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        # Verify structure: list of (harmony, fitness) tuples
        assert len(hs._harmony_memory) == 2
        assert isinstance(hs._harmony_memory[0], tuple)
        assert isinstance(hs._harmony_memory[1], tuple)
        assert len(hs._harmony_memory[0]) == 2
        assert len(hs._harmony_memory[1]) == 2
        
        # Verify values
        assert hs._harmony_memory[0] == ([1.0, 2.0], 15.5)
        assert hs._harmony_memory[1] == ([3.0, 4.0], 25.7)


class TestInitializeWithoutInitialHarmonies:
    """Test _initialize() method when initial_harmonies are NOT provided (default behavior)."""
    
    def test_creates_hms_harmonies(self):
        """Test that _initialize creates HMS harmonies when none provided."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=7)
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        assert len(hs._harmony_memory) == 7
    
    def test_each_harmony_has_num_parameters_values(self):
        """Test that each generated harmony has num_parameters values."""
        mock_obj = create_mock_objective_function(num_parameters=4, hms=3)
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        for harmony, fitness in hs._harmony_memory:
            assert len(harmony) == 4
    
    def test_uses_random_selection_method(self):
        """Test that _initialize uses _random_selection() to generate harmonies."""
        mock_obj = create_mock_objective_function(num_parameters=3, hms=2)
        hs = HarmonySearch(mock_obj)
        
        # Mock _random_selection to track calls
        original_random_selection = hs._random_selection
        call_count = 0
        
        def mock_random_selection(harmony, i):
            nonlocal call_count
            call_count += 1
            original_random_selection(harmony, i)
        
        hs._random_selection = mock_random_selection
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        # Should call _random_selection HMS * num_parameters times
        # HMS=2, num_parameters=3, so 2*3=6 calls
        assert call_count == 6
    
    def test_calculates_fitness_for_each_generated_harmony(self):
        """Test that _initialize calculates fitness for each generated harmony."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=4)
        mock_obj.get_fitness = Mock(side_effect=[1.0, 2.0, 3.0, 4.0])
        
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        # Verify get_fitness was called HMS times (once per harmony)
        assert mock_obj.get_fitness.call_count == 4
    
    def test_stores_generated_harmonies_as_tuples(self):
        """Test that _initialize stores generated harmonies as (harmony, fitness) tuples."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=3)
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        # Verify structure
        for entry in hs._harmony_memory:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            harmony, fitness = entry
            assert isinstance(harmony, list)
            assert isinstance(fitness, (int, float))


class TestHistoryInitialization:
    """Test _initialize() method's handling of harmony_history."""
    
    def test_adds_first_entry_with_gen_zero(self):
        """Test that _initialize adds first history entry with gen=0."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=3)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        assert len(hs._harmony_history) == 1
        assert hs._harmony_history[0]['gen'] == 0
    
    def test_history_contains_deep_copy_of_memory(self):
        """Test that history contains a deep copy of harmony memory, not a reference."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=2)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        # Get the harmonies from history
        history_harmonies = hs._harmony_history[0]['harmonies']
        
        # Verify they match the current memory
        assert len(history_harmonies) == len(hs._harmony_memory)
        
        # Modify the memory
        hs._harmony_memory[0] = ([99.0, 99.0], 999.0)
        
        # Verify history was not affected (deep copy)
        assert history_harmonies[0] != ([99.0, 99.0], 999.0)
        assert history_harmonies[0] == ([1.0, 2.0], hs._harmony_memory[1][1])  # Original first harmony


class TestEdgeCases:
    """Test edge cases for _initialize() method."""
    
    def test_empty_initial_harmonies_list(self):
        """Test that _initialize raises ValueError with empty initial_harmonies list."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=3)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = []
        
        hs._harmony_memory = []
        hs._harmony_history = []
        with pytest.raises(ValueError) as excinfo:
            hs._initialize(initial_harmonies)
        
        assert "Number of initial harmonies does not equal to the harmony memory size" in str(excinfo.value)
    
    def test_initial_harmonies_with_empty_harmony(self):
        """Test that _initialize raises ValueError when a harmony is empty."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=2)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            []  # Empty harmony
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        with pytest.raises(ValueError) as excinfo:
            hs._initialize(initial_harmonies)
        
        assert "Number of parameters in initial harmonies does not match that defined" in str(excinfo.value)
    
    def test_initial_harmonies_with_wrong_dimensions(self):
        """Test that _initialize raises ValueError when harmonies have inconsistent dimensions."""
        mock_obj = create_mock_objective_function(num_parameters=3, hms=3)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0]  # Wrong: only 2 parameters
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        with pytest.raises(ValueError) as excinfo:
            hs._initialize(initial_harmonies)
        
        assert "Number of parameters in initial harmonies does not match that defined" in str(excinfo.value)
    
    def test_hms_zero(self):
        """Test _initialize with HMS=0 (edge case)."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=0)
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        # Should create empty memory and history with gen=0
        assert len(hs._harmony_memory) == 0
        assert len(hs._harmony_history) == 1
        assert hs._harmony_history[0]['gen'] == 0
        assert len(hs._harmony_history[0]['harmonies']) == 0
    
    def test_hms_one(self):
        """Test _initialize with HMS=1 (minimal harmony memory)."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=1)
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        assert len(hs._harmony_memory) == 1
        assert len(hs._harmony_memory[0][0]) == 2  # 2 parameters
    
    def test_num_parameters_zero(self):
        """Test _initialize with num_parameters=0 (edge case)."""
        mock_obj = create_mock_objective_function(num_parameters=0, hms=3)
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        # Should create HMS harmonies, each with 0 parameters
        assert len(hs._harmony_memory) == 3
        for harmony, fitness in hs._harmony_memory:
            assert len(harmony) == 0
    
    def test_num_parameters_one(self):
        """Test _initialize with num_parameters=1."""
        mock_obj = create_mock_objective_function(num_parameters=1, hms=2)
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        assert len(hs._harmony_memory) == 2
        for harmony, fitness in hs._harmony_memory:
            assert len(harmony) == 1


class TestErrorMessages:
    """Test that error messages are clear and informative."""
    
    def test_error_message_for_wrong_hms(self):
        """Test that error message is clear when HMS doesn't match."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=5)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        with pytest.raises(ValueError) as excinfo:
            hs._initialize(initial_harmonies)
        
        error_message = str(excinfo.value)
        assert "Number of initial harmonies" in error_message
        assert "harmony memory size" in error_message
    
    def test_error_message_for_wrong_parameters(self):
        """Test that error message is clear when parameter count doesn't match."""
        mock_obj = create_mock_objective_function(num_parameters=4, hms=2)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        with pytest.raises(ValueError) as excinfo:
            hs._initialize(initial_harmonies)
        
        error_message = str(excinfo.value)
        assert "Number of parameters" in error_message
        assert "initial harmonies" in error_message
        assert "does not match" in error_message


class TestDeepCopyVerification:
    """Test that deep copying works correctly in _initialize()."""
    
    def test_modifying_history_does_not_affect_memory(self):
        """Test that modifying history doesn't affect harmony memory."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=2)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        # Get reference to history harmonies
        history_harmonies = hs._harmony_history[0]['harmonies']
        
        # Modify history
        history_harmonies[0] = ([99.0, 99.0], 999.0)
        
        # Verify memory was not affected
        assert hs._harmony_memory[0] != ([99.0, 99.0], 999.0)
        assert hs._harmony_memory[0][0] == [1.0, 2.0]
    
    def test_modifying_memory_does_not_affect_history(self):
        """Test that modifying harmony memory doesn't affect history."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=2)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        # Store original history values
        original_history_harmony = copy.deepcopy(hs._harmony_history[0]['harmonies'][0])
        
        # Modify memory
        hs._harmony_memory[0] = ([77.0, 88.0], 777.0)
        
        # Verify history was not affected
        assert hs._harmony_history[0]['harmonies'][0] == original_history_harmony
        assert hs._harmony_history[0]['harmonies'][0] != ([77.0, 88.0], 777.0)
    
    def test_modifying_harmony_values_does_not_affect_history(self):
        """Test that modifying individual harmony values doesn't affect history."""
        mock_obj = create_mock_objective_function(num_parameters=2, hms=2)
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        # Store original history value
        original_value = hs._harmony_history[0]['harmonies'][0][0][0]
        
        # Modify harmony value in memory
        hs._harmony_memory[0][0][0] = 999.0
        
        # Verify history was not affected
        assert hs._harmony_history[0]['harmonies'][0][0][0] == original_value
        assert hs._harmony_history[0]['harmonies'][0][0][0] != 999.0


class TestInitializeIntegration:
    """Integration tests for _initialize() method."""
    
    def test_initialize_with_valid_initial_harmonies_full_workflow(self):
        """Test complete workflow with valid initial harmonies."""
        mock_obj = create_mock_objective_function(
            num_parameters=3,
            hms=4,
            maximize=True
        )
        hs = HarmonySearch(mock_obj)
        
        initial_harmonies = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [-1.0, -2.0, -3.0]
        ]
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(initial_harmonies)
        
        # Verify memory is properly structured
        assert_valid_harmony_memory(hs._harmony_memory, mock_obj)
        
        # Verify history is properly structured
        assert_valid_harmony_history(hs._harmony_history, mock_obj)
        
        # Verify all harmonies are present
        assert len(hs._harmony_memory) == 4
        
        # Verify harmonies match input
        memory_harmonies = [h for h, f in hs._harmony_memory]
        assert [1.0, 2.0, 3.0] in memory_harmonies
        assert [4.0, 5.0, 6.0] in memory_harmonies
        assert [7.0, 8.0, 9.0] in memory_harmonies
        assert [-1.0, -2.0, -3.0] in memory_harmonies
    
    def test_initialize_without_initial_harmonies_full_workflow(self):
        """Test complete workflow without initial harmonies (random generation)."""
        mock_obj = create_mock_objective_function(
            num_parameters=2,
            hms=5,
            maximize=False
        )
        hs = HarmonySearch(mock_obj)
        
        hs._harmony_memory = []
        hs._harmony_history = []
        hs._initialize(None)
        
        # Verify memory is properly structured
        assert_valid_harmony_memory(hs._harmony_memory, mock_obj)
        
        # Verify history is properly structured
        assert_valid_harmony_history(hs._harmony_history, mock_obj)
        
        # Verify correct number of harmonies
        assert len(hs._harmony_memory) == 5
        
        # Verify each harmony has correct structure
        for harmony, fitness in hs._harmony_memory:
            assert len(harmony) == 2
            assert isinstance(fitness, (int, float))
