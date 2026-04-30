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
Unit tests for HarmonySearch._update_harmony_memory() method.

Tests cover:
- Duplicate prevention
- Worst harmony selection for maximization/minimization
- Replacement logic based on fitness comparison
- Edge cases (memory size = 1, equal fitness values)
- Index replacement verification
- Various fitness range scenarios
"""

import pytest
from pyharmonysearch import HarmonySearch
from tests.fixtures.test_helpers import create_mock_objective_function


class TestUpdateHarmonyMemoryDuplicatePrevention:
    """Tests for duplicate harmony prevention."""

    def test_exact_duplicate_not_added_maximization(self):
        """Test that an exact duplicate harmony is not added to memory (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=3)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory manually
        hs._harmony_memory = [
            ([1.0, 2.0], 10.0),
            ([3.0, 4.0], 20.0),
            ([5.0, 6.0], 5.0),
        ]
        
        # Try to add exact duplicate
        hs._update_harmony_memory([1.0, 2.0], 10.0)
        
        # Memory should remain unchanged
        assert len(hs._harmony_memory) == 3
        assert hs._harmony_memory[0] == ([1.0, 2.0], 10.0)
        assert hs._harmony_memory[1] == ([3.0, 4.0], 20.0)
        assert hs._harmony_memory[2] == ([5.0, 6.0], 5.0)

    def test_exact_duplicate_not_added_minimization(self):
        """Test that an exact duplicate harmony is not added to memory (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=3)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory manually
        hs._harmony_memory = [
            ([1.0, 2.0], 10.0),
            ([3.0, 4.0], 20.0),
            ([5.0, 6.0], 5.0),
        ]
        
        # Try to add exact duplicate
        hs._update_harmony_memory([5.0, 6.0], 5.0)
        
        # Memory should remain unchanged
        assert len(hs._harmony_memory) == 3
        assert ([5.0, 6.0], 5.0) in hs._harmony_memory

    def test_duplicate_harmony_different_fitness_not_added(self):
        """Test that duplicate harmony with different fitness is not added."""
        obj_fun = create_mock_objective_function(maximize=True, hms=3)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory manually
        hs._harmony_memory = [
            ([1.0, 2.0], 10.0),
            ([3.0, 4.0], 20.0),
            ([5.0, 6.0], 5.0),
        ]
        
        # Try to add same harmony with different fitness
        hs._update_harmony_memory([1.0, 2.0], 15.0)
        
        # Memory should remain unchanged (duplicate check is based on (harmony, fitness) tuple)
        assert len(hs._harmony_memory) == 3
        assert ([1.0, 2.0], 10.0) in hs._harmony_memory
        assert ([1.0, 2.0], 15.0) not in hs._harmony_memory


class TestUpdateHarmonyMemoryWorstSelection:
    """Tests for correct worst harmony selection."""

    def test_worst_harmony_selection_maximization(self):
        """Test that the worst harmony (lowest fitness) is identified for maximization."""
        obj_fun = create_mock_objective_function(maximize=True, hms=5)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory with clear worst
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 25.0),
            ([3.0, 3.0], 15.0),
            ([4.0, 4.0], 5.0),   # Worst for maximization
            ([5.0, 5.0], 20.0),
        ]
        
        # Add a better harmony that should replace the worst
        hs._update_harmony_memory([6.0, 6.0], 30.0)
        
        # The worst harmony ([4.0, 4.0], 5.0) should be replaced
        assert ([4.0, 4.0], 5.0) not in hs._harmony_memory
        assert ([6.0, 6.0], 30.0) in hs._harmony_memory
        assert len(hs._harmony_memory) == 5

    def test_worst_harmony_selection_minimization(self):
        """Test that the worst harmony (highest fitness) is identified for minimization."""
        obj_fun = create_mock_objective_function(maximize=False, hms=5)
        hs = HarmonySearch(obj_fun)
        
        # Initialize harmony memory with clear worst
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 25.0),  # Worst for minimization
            ([3.0, 3.0], 15.0),
            ([4.0, 4.0], 5.0),
            ([5.0, 5.0], 20.0),
        ]
        
        # Add a better harmony that should replace the worst
        hs._update_harmony_memory([6.0, 6.0], 3.0)
        
        # The worst harmony ([2.0, 2.0], 25.0) should be replaced
        assert ([2.0, 2.0], 25.0) not in hs._harmony_memory
        assert ([6.0, 6.0], 3.0) in hs._harmony_memory
        assert len(hs._harmony_memory) == 5


class TestUpdateHarmonyMemoryReplacement:
    """Tests for replacement logic when new harmony is better/worse."""

    def test_replacement_when_new_harmony_better_maximization(self):
        """Test replacement occurs when new harmony is better than worst (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 20.0),
            ([3.0, 3.0], 5.0),   # Worst
        ]
        
        # New harmony better than worst
        hs._update_harmony_memory([4.0, 4.0], 15.0)
        
        assert ([3.0, 3.0], 5.0) not in hs._harmony_memory
        assert ([4.0, 4.0], 15.0) in hs._harmony_memory

    def test_replacement_when_new_harmony_better_minimization(self):
        """Test replacement occurs when new harmony is better than worst (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 5.0),
            ([3.0, 3.0], 20.0),  # Worst
        ]
        
        # New harmony better than worst
        hs._update_harmony_memory([4.0, 4.0], 8.0)
        
        assert ([3.0, 3.0], 20.0) not in hs._harmony_memory
        assert ([4.0, 4.0], 8.0) in hs._harmony_memory

    def test_no_replacement_when_new_harmony_worse_maximization(self):
        """Test no replacement when new harmony is worse than worst (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 20.0),
            ([3.0, 3.0], 5.0),   # Worst
        ]
        
        # New harmony worse than worst
        hs._update_harmony_memory([4.0, 4.0], 3.0)
        
        # Memory should remain unchanged
        assert len(hs._harmony_memory) == 3
        assert ([4.0, 4.0], 3.0) not in hs._harmony_memory
        assert ([3.0, 3.0], 5.0) in hs._harmony_memory

    def test_no_replacement_when_new_harmony_worse_minimization(self):
        """Test no replacement when new harmony is worse than worst (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 5.0),
            ([3.0, 3.0], 20.0),  # Worst
        ]
        
        # New harmony worse than worst
        hs._update_harmony_memory([4.0, 4.0], 25.0)
        
        # Memory should remain unchanged
        assert len(hs._harmony_memory) == 3
        assert ([4.0, 4.0], 25.0) not in hs._harmony_memory
        assert ([3.0, 3.0], 20.0) in hs._harmony_memory

    def test_no_replacement_for_equal_fitness_maximization(self):
        """Test no replacement when new harmony has equal fitness to worst (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 20.0),
            ([3.0, 3.0], 5.0),   # Worst
        ]
        
        # New harmony with same fitness as worst
        hs._update_harmony_memory([4.0, 4.0], 5.0)
        
        # Memory should remain unchanged (not strictly better)
        assert len(hs._harmony_memory) == 3
        assert ([4.0, 4.0], 5.0) not in hs._harmony_memory
        assert ([3.0, 3.0], 5.0) in hs._harmony_memory

    def test_no_replacement_for_equal_fitness_minimization(self):
        """Test no replacement when new harmony has equal fitness to worst (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 5.0),
            ([3.0, 3.0], 20.0),  # Worst
        ]
        
        # New harmony with same fitness as worst
        hs._update_harmony_memory([4.0, 4.0], 20.0)
        
        # Memory should remain unchanged (not strictly better)
        assert len(hs._harmony_memory) == 3
        assert ([4.0, 4.0], 20.0) not in hs._harmony_memory
        assert ([3.0, 3.0], 20.0) in hs._harmony_memory


class TestUpdateHarmonyMemoryEdgeCases:
    """Tests for edge cases."""

    def test_memory_size_one_replacement(self):
        """Test replacement when harmony memory size is 1."""
        obj_fun = create_mock_objective_function(maximize=True, hms=1)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [([1.0, 1.0], 10.0)]
        
        # Add better harmony
        hs._update_harmony_memory([2.0, 2.0], 15.0)
        
        assert len(hs._harmony_memory) == 1
        assert hs._harmony_memory[0] == ([2.0, 2.0], 15.0)

    def test_memory_size_one_no_replacement(self):
        """Test no replacement when harmony memory size is 1 and new is worse."""
        obj_fun = create_mock_objective_function(maximize=True, hms=1)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [([1.0, 1.0], 10.0)]
        
        # Add worse harmony
        hs._update_harmony_memory([2.0, 2.0], 5.0)
        
        assert len(hs._harmony_memory) == 1
        assert hs._harmony_memory[0] == ([1.0, 1.0], 10.0)

    def test_all_harmonies_same_fitness_maximization(self):
        """Test with all harmonies having same fitness (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=4)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 10.0),
            ([3.0, 3.0], 10.0),
            ([4.0, 4.0], 10.0),
        ]
        
        # Add better harmony
        hs._update_harmony_memory([5.0, 5.0], 15.0)
        
        # Should replace one of them (the first one found as worst)
        assert ([5.0, 5.0], 15.0) in hs._harmony_memory
        assert len(hs._harmony_memory) == 4

    def test_all_harmonies_same_fitness_minimization(self):
        """Test with all harmonies having same fitness (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=4)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 10.0),
            ([3.0, 3.0], 10.0),
            ([4.0, 4.0], 10.0),
        ]
        
        # Add better harmony
        hs._update_harmony_memory([5.0, 5.0], 5.0)
        
        # Should replace one of them (the first one found as worst)
        assert ([5.0, 5.0], 5.0) in hs._harmony_memory
        assert len(hs._harmony_memory) == 4


class TestUpdateHarmonyMemoryIndexReplacement:
    """Tests to verify correct index replacement."""

    def test_correct_index_replaced_maximization(self):
        """Test that the correct index is replaced (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=5)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 15.0),  # Index 0
            ([2.0, 2.0], 25.0),  # Index 1
            ([3.0, 3.0], 10.0),  # Index 2 - worst
            ([4.0, 4.0], 30.0),  # Index 3
            ([5.0, 5.0], 20.0),  # Index 4
        ]
        
        original_memory = [entry for entry in hs._harmony_memory]
        
        # Add harmony better than worst
        hs._update_harmony_memory([6.0, 6.0], 18.0)
        
        # Verify index 2 was replaced
        assert hs._harmony_memory[0] == original_memory[0]
        assert hs._harmony_memory[1] == original_memory[1]
        assert hs._harmony_memory[2] == ([6.0, 6.0], 18.0)
        assert hs._harmony_memory[3] == original_memory[3]
        assert hs._harmony_memory[4] == original_memory[4]

    def test_correct_index_replaced_minimization(self):
        """Test that the correct index is replaced (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=5)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 15.0),  # Index 0
            ([2.0, 2.0], 25.0),  # Index 1 - worst
            ([3.0, 3.0], 10.0),  # Index 2
            ([4.0, 4.0], 5.0),   # Index 3
            ([5.0, 5.0], 20.0),  # Index 4
        ]
        
        original_memory = [entry for entry in hs._harmony_memory]
        
        # Add harmony better than worst
        hs._update_harmony_memory([6.0, 6.0], 12.0)
        
        # Verify index 1 was replaced
        assert hs._harmony_memory[0] == original_memory[0]
        assert hs._harmony_memory[1] == ([6.0, 6.0], 12.0)
        assert hs._harmony_memory[2] == original_memory[2]
        assert hs._harmony_memory[3] == original_memory[3]
        assert hs._harmony_memory[4] == original_memory[4]


class TestUpdateHarmonyMemoryFitnessRanges:
    """Tests with various fitness ranges."""

    def test_new_harmony_better_than_all_maximization(self):
        """Test when new harmony is better than all existing harmonies (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 15.0),
            ([3.0, 3.0], 5.0),
        ]
        
        # Add harmony better than all
        hs._update_harmony_memory([4.0, 4.0], 25.0)
        
        assert ([4.0, 4.0], 25.0) in hs._harmony_memory
        assert ([3.0, 3.0], 5.0) not in hs._harmony_memory
        assert len(hs._harmony_memory) == 3

    def test_new_harmony_better_than_all_minimization(self):
        """Test when new harmony is better than all existing harmonies (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 15.0),
            ([3.0, 3.0], 5.0),
        ]
        
        # Add harmony better than all
        hs._update_harmony_memory([4.0, 4.0], 2.0)
        
        assert ([4.0, 4.0], 2.0) in hs._harmony_memory
        assert ([2.0, 2.0], 15.0) not in hs._harmony_memory
        assert len(hs._harmony_memory) == 3

    def test_new_harmony_worse_than_all_maximization(self):
        """Test when new harmony is worse than all existing harmonies (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),
            ([2.0, 2.0], 15.0),
            ([3.0, 3.0], 20.0),
        ]
        
        # Add harmony worse than all
        hs._update_harmony_memory([4.0, 4.0], 5.0)
        
        assert ([4.0, 4.0], 5.0) not in hs._harmony_memory
        assert len(hs._harmony_memory) == 3

    def test_new_harmony_worse_than_all_minimization(self):
        """Test when new harmony is worse than all existing harmonies (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=3)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 5.0),
            ([2.0, 2.0], 10.0),
            ([3.0, 3.0], 15.0),
        ]
        
        # Add harmony worse than all
        hs._update_harmony_memory([4.0, 4.0], 20.0)
        
        assert ([4.0, 4.0], 20.0) not in hs._harmony_memory
        assert len(hs._harmony_memory) == 3

    def test_new_harmony_in_middle_range_maximization(self):
        """Test when new harmony fitness is in middle range (maximization)."""
        obj_fun = create_mock_objective_function(maximize=True, hms=5)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 30.0),  # Best
            ([2.0, 2.0], 25.0),
            ([3.0, 3.0], 20.0),
            ([4.0, 4.0], 15.0),
            ([5.0, 5.0], 10.0),  # Worst
        ]
        
        # Add harmony in middle range
        hs._update_harmony_memory([6.0, 6.0], 22.0)
        
        assert ([6.0, 6.0], 22.0) in hs._harmony_memory
        assert ([5.0, 5.0], 10.0) not in hs._harmony_memory
        assert len(hs._harmony_memory) == 5

    def test_new_harmony_in_middle_range_minimization(self):
        """Test when new harmony fitness is in middle range (minimization)."""
        obj_fun = create_mock_objective_function(maximize=False, hms=5)
        hs = HarmonySearch(obj_fun)
        
        hs._harmony_memory = [
            ([1.0, 1.0], 10.0),  # Best
            ([2.0, 2.0], 15.0),
            ([3.0, 3.0], 20.0),
            ([4.0, 4.0], 25.0),
            ([5.0, 5.0], 30.0),  # Worst
        ]
        
        # Add harmony in middle range
        hs._update_harmony_memory([6.0, 6.0], 22.0)
        
        assert ([6.0, 6.0], 22.0) in hs._harmony_memory
        assert ([5.0, 5.0], 30.0) not in hs._harmony_memory
        assert len(hs._harmony_memory) == 5
