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
from unittest.mock import Mock, MagicMock, patch, call
from pyharmonysearch import HarmonySearch, ObjectiveFunctionInterface
from tests.fixtures.mock_objective_functions import (
    QuadraticObjectiveFunction,
    DiscreteObjectiveFunction,
    MixedObjectiveFunction,
)


class TestHarmonySearchRunInitialization:
    """Test suite for initialization phase of HarmonySearch.run()."""
    
    def test_random_seed_set_when_use_random_seed_true(self):
        """Test that random seed is set when use_random_seed() returns True."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch('random.seed') as mock_seed:
            hs.run()
            mock_seed.assert_called_once_with(42)
    
    def test_random_seed_not_set_when_use_random_seed_false(self):
        """Test that random seed is NOT set when use_random_seed() returns False."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, random_seed=None)
        hs = HarmonySearch(obj_fun)
        
        with patch('random.seed') as mock_seed:
            hs.run()
            mock_seed.assert_not_called()
    
    def test_harmony_memory_initialized_to_empty_list(self):
        """Test that harmony memory is initialized to empty list before _initialize()."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10)
        hs = HarmonySearch(obj_fun)
        
        # Mock _initialize to capture state before it runs
        original_initialize = hs._initialize
        memory_at_start = None
        
        def capture_and_run(initial_harmonies=None):
            nonlocal memory_at_start
            memory_at_start = hs._harmony_memory
            original_initialize(initial_harmonies)
        
        with patch.object(hs, '_initialize', side_effect=capture_and_run):
            hs.run()
        
        assert memory_at_start == []
    
    def test_harmony_history_initialized_to_empty_list(self):
        """Test that harmony history is initialized to empty list before _initialize()."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10)
        hs = HarmonySearch(obj_fun)
        
        # Mock _initialize to capture state before it runs
        original_initialize = hs._initialize
        history_at_start = None
        
        def capture_and_run(initial_harmonies=None):
            nonlocal history_at_start
            history_at_start = hs._harmony_history
            original_initialize(initial_harmonies)
        
        with patch.object(hs, '_initialize', side_effect=capture_and_run):
            hs.run()
        
        assert history_at_start == []
    
    def test_initialize_called_with_initial_harmonies(self):
        """Test that _initialize() is called with initial_harmonies parameter."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5)
        hs = HarmonySearch(obj_fun)
        
        initial_harmonies = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        
        with patch.object(hs, '_initialize') as mock_initialize:
            hs.run(initial_harmonies=initial_harmonies)
            mock_initialize.assert_called_once_with(initial_harmonies)


class TestHarmonySearchRunMainLoop:
    """Test suite for main loop of HarmonySearch.run()."""
    
    def test_loop_runs_exactly_max_imp_times(self):
        """Test that the main loop runs exactly get_max_imp() times."""
        max_imp = 25
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=max_imp, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        # Count fitness evaluations (one per improvisation)
        fitness_count = 0
        original_get_fitness = obj_fun.get_fitness
        
        def counting_get_fitness(vector):
            nonlocal fitness_count
            fitness_count += 1
            return original_get_fitness(vector)
        
        obj_fun.get_fitness = counting_get_fitness
        hs.run()
        
        # Should have max_imp + hms calls (hms from initialization, max_imp from main loop)
        assert fitness_count == max_imp + obj_fun.get_hms()
    
    def test_each_iteration_calls_get_num_parameters_times(self):
        """Test that each iteration generates harmony with get_num_parameters() values."""
        num_params = 3
        max_imp = 10
        obj_fun = QuadraticObjectiveFunction(num_parameters=num_params, hms=5, max_imp=max_imp, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        # Mock the methods to count calls during main loop only
        with patch.object(hs, '_memory_consideration') as mock_memory, \
             patch.object(hs, '_random_selection') as mock_random, \
             patch.object(hs, '_initialize'):
            
            hs.run()
            
            # Total calls should be num_params * max_imp
            total_calls = mock_memory.call_count + mock_random.call_count
            assert total_calls == num_params * max_imp
    
    def test_hmcr_memory_consideration_called_when_random_less_than_hmcr(self):
        """Test that memory_consideration is called when random() < hmcr."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=1, hms=5, max_imp=1, hmcr=0.75, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        # Set random to return value less than hmcr
        with patch('random.random', side_effect=[0.5]), \
             patch.object(hs, '_memory_consideration') as mock_memory, \
             patch.object(hs, '_random_selection') as mock_random, \
             patch.object(hs, '_initialize'):
            
            # Need to mock random for PAR check too
            with patch('random.random', return_value=0.9):
                hs.run()
            
            # Memory consideration should be called
            assert mock_memory.call_count == 1
            assert mock_random.call_count == 0
    
    def test_hmcr_random_selection_called_when_random_gte_hmcr(self):
        """Test that random_selection is called when random() >= hmcr."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=1, hms=5, max_imp=1, hmcr=0.75, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch.object(hs, '_initialize'):
            # Set random to return value >= hmcr
            with patch('random.random', return_value=0.85), \
                 patch.object(hs, '_memory_consideration') as mock_memory, \
                 patch.object(hs, '_random_selection') as mock_random:
                
                hs.run()
                
                # Random selection should be called
                assert mock_random.call_count == 1
                assert mock_memory.call_count == 0
    
    def test_par_pitch_adjustment_called_when_memory_and_random_less_than_par(self):
        """Test that pitch_adjustment is called when memory consideration AND random() < par."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=1, hms=5, max_imp=1, hmcr=0.75, par=0.5, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch.object(hs, '_initialize'):
            # First random for HMCR (< 0.75), second for PAR (< 0.5)
            with patch('random.random', side_effect=[0.5, 0.3]), \
                 patch.object(hs, '_memory_consideration') as mock_memory, \
                 patch.object(hs, '_pitch_adjustment') as mock_pitch:
                
                hs.run()
                
                # Both should be called
                assert mock_memory.call_count == 1
                assert mock_pitch.call_count == 1
    
    def test_par_no_pitch_adjustment_when_random_gte_par(self):
        """Test that pitch_adjustment is NOT called when random() >= par."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=1, hms=5, max_imp=1, hmcr=0.75, par=0.5, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch.object(hs, '_initialize'):
            # First random for HMCR (< 0.75), second for PAR (>= 0.5)
            with patch('random.random', side_effect=[0.5, 0.8]), \
                 patch.object(hs, '_memory_consideration') as mock_memory, \
                 patch.object(hs, '_pitch_adjustment') as mock_pitch:
                
                hs.run()
                
                # Memory consideration called, pitch adjustment not called
                assert mock_memory.call_count == 1
                assert mock_pitch.call_count == 0
    
    def test_fitness_evaluated_for_each_new_harmony(self):
        """Test that fitness is evaluated for each new harmony created."""
        max_imp = 15
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=max_imp, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        fitness_calls = []
        original_get_fitness = obj_fun.get_fitness
        
        def tracking_get_fitness(vector):
            fitness_calls.append(vector[:])  # Store a copy
            return original_get_fitness(vector)
        
        obj_fun.get_fitness = tracking_get_fitness
        hs.run()
        
        # Should have max_imp fitness calls in main loop + hms from initialization
        assert len(fitness_calls) == max_imp + obj_fun.get_hms()
    
    def test_harmony_memory_updated_each_iteration(self):
        """Test that harmony memory is updated after each iteration."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch.object(hs, '_update_harmony_memory') as mock_update, \
             patch.object(hs, '_initialize'):
            
            hs.run()
            
            # Should be called max_imp times (once per iteration)
            assert mock_update.call_count == 10


class TestHarmonySearchRunGenerationTracking:
    """Test suite for generation tracking in HarmonySearch.run()."""
    
    def test_generation_increments_correctly(self):
        """Test that generation number increments correctly."""
        hms = 5
        max_imp = 20  # 4 complete generations
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=hms, max_imp=max_imp, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Should have 5 generations (0 from init, then 1, 2, 3, 4 from loop)
        assert len(history) == 5
        assert history[0]['gen'] == 0
        assert history[1]['gen'] == 1
        assert history[2]['gen'] == 2
        assert history[3]['gen'] == 3
        assert history[4]['gen'] == 4
    
    def test_history_saved_every_hms_iterations(self):
        """Test that history is saved every get_hms() iterations."""
        hms = 3
        max_imp = 12  # Exactly 4 generations
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=hms, max_imp=max_imp, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Should have 5 entries: initial (gen 0) + 4 generations
        assert len(history) == 5
        
        # Verify each entry has correct generation number
        for i, entry in enumerate(history):
            assert entry['gen'] == i
    
    def test_deep_copy_of_harmony_memory_in_history(self):
        """Test that harmony_memory is deep copied into history (not referenced)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=6, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Verify that modifying a harmony in one generation doesn't affect others
        # Get harmonies from generation 0 and 1
        gen0_first_harmony = history[0]['harmonies'][0][0]
        gen1_first_harmony = history[1]['harmonies'][0][0]
        
        # Modify one
        gen0_first_harmony[0] = 999.0
        
        # Verify the other is not affected
        assert gen1_first_harmony[0] != 999.0


class TestHarmonySearchRunBestSolution:
    """Test suite for best solution selection in HarmonySearch.run()."""
    
    def test_correct_best_for_maximization(self):
        """Test that the best harmony is correctly selected for maximization problems."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=20, maximize=True, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Verify that best_fitness is the maximum in harmony_memory
        all_fitnesses = [fitness for harmony, fitness in memory]
        assert best_fitness == max(all_fitnesses)
        
        # Verify the harmony corresponds to the fitness
        for harmony, fitness in memory:
            if fitness == best_fitness:
                assert harmony == best_harmony
                break
    
    def test_correct_best_for_minimization(self):
        """Test that the best harmony is correctly selected for minimization problems."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=20, maximize=False, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Verify that best_fitness is the minimum in harmony_memory
        all_fitnesses = [fitness for harmony, fitness in memory]
        assert best_fitness == min(all_fitnesses)
        
        # Verify the harmony corresponds to the fitness
        for harmony, fitness in memory:
            if fitness == best_fitness:
                assert harmony == best_harmony
                break


class TestHarmonySearchRunReturnValue:
    """Test suite for return value of HarmonySearch.run()."""
    
    def test_return_tuple_format(self):
        """Test that run() returns a tuple with correct format: (harmony, fitness, memory, history)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        result = hs.run()
        
        # Should be a tuple with 4 elements
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        best_harmony, best_fitness, memory, history = result
        
        # Verify types
        assert isinstance(best_harmony, list)
        assert isinstance(best_fitness, (int, float))
        assert isinstance(memory, list)
        assert isinstance(history, list)
        
        # Verify best_harmony has correct length
        assert len(best_harmony) == obj_fun.get_num_parameters()
        
        # Verify memory structure
        for harmony, fitness in memory:
            assert isinstance(harmony, list)
            assert isinstance(fitness, (int, float))
            assert len(harmony) == obj_fun.get_num_parameters()
        
        # Verify history structure
        for entry in history:
            assert 'gen' in entry
            assert 'harmonies' in entry
            assert isinstance(entry['gen'], int)
            assert isinstance(entry['harmonies'], list)


class TestHarmonySearchRunEdgeCases:
    """Test suite for edge cases in HarmonySearch.run()."""
    
    def test_max_imp_zero(self):
        """Test that max_imp = 0 results in no main loop iterations."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=0, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Should only have initial generation
        assert len(history) == 1
        assert history[0]['gen'] == 0
        
        # Memory should have hms harmonies from initialization
        assert len(memory) == 5
    
    def test_max_imp_one(self):
        """Test that max_imp = 1 results in single iteration."""
        hms = 5
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=hms, max_imp=1, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        fitness_count = 0
        original_get_fitness = obj_fun.get_fitness
        
        def counting_get_fitness(vector):
            nonlocal fitness_count
            fitness_count += 1
            return original_get_fitness(vector)
        
        obj_fun.get_fitness = counting_get_fitness
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Should have hms + 1 fitness evaluations (hms from init, 1 from loop)
        assert fitness_count == hms + 1
    
    def test_max_imp_not_divisible_by_hms(self):
        """Test that max_imp not divisible by hms works correctly."""
        hms = 5
        max_imp = 13  # Not divisible by 5
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=hms, max_imp=max_imp, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Should save history at iterations 5, 10 (plus initial gen 0)
        # Total: 1 (init) + 2 (at 5 and 10) = 3
        assert len(history) == 3
        assert history[0]['gen'] == 0
        assert history[1]['gen'] == 1
        assert history[2]['gen'] == 2
    
    def test_max_imp_equals_hms(self):
        """Test that max_imp == hms results in exactly one generation."""
        hms = 10
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=hms, max_imp=hms, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Should have 2 entries: initial (gen 0) + one generation (gen 1)
        assert len(history) == 2
        assert history[0]['gen'] == 0
        assert history[1]['gen'] == 1
    
    def test_empty_harmony_memory_handling(self):
        """Test handling when harmony memory is somehow empty (edge case)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        # This shouldn't happen in normal operation, but test the best selection logic
        # by mocking _initialize to leave memory empty
        with patch.object(hs, '_initialize'):
            # Manually set empty memory
            hs._harmony_memory = []
            hs._harmony_history = [{'gen': 0, 'harmonies': []}]
            
            best_harmony, best_fitness, memory, history = hs.run()
            
            # With empty memory, best should be None and fitness should be -inf or +inf
            # depending on maximize
            assert best_harmony is None


class TestHarmonySearchRunRandomBehavior:
    """Test suite for random behavior in HarmonySearch.run()."""
    
    def test_different_results_without_seed(self):
        """Test that multiple runs without seed produce different results."""
        obj_fun1 = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=50, random_seed=None)
        obj_fun2 = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=50, random_seed=None)
        
        hs1 = HarmonySearch(obj_fun1)
        hs2 = HarmonySearch(obj_fun2)
        
        best1, fitness1, memory1, history1 = hs1.run()
        best2, fitness2, memory2, history2 = hs2.run()
        
        # With high probability, results should be different
        # (not guaranteed but extremely likely with 50 iterations)
        assert best1 != best2 or fitness1 != fitness2
    
    def test_same_results_with_seed(self):
        """Test that multiple runs with same seed produce identical results."""
        obj_fun1 = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=30, random_seed=12345)
        obj_fun2 = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=30, random_seed=12345)
        
        hs1 = HarmonySearch(obj_fun1)
        hs2 = HarmonySearch(obj_fun2)
        
        best1, fitness1, memory1, history1 = hs1.run()
        best2, fitness2, memory2, history2 = hs2.run()
        
        # Results should be identical
        assert best1 == best2
        assert fitness1 == fitness2
        assert len(memory1) == len(memory2)
        assert len(history1) == len(history2)
    
    def test_hmcr_zero_always_random_selection(self):
        """Test that HMCR = 0 always uses random selection."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, hmcr=0.0, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch.object(hs, '_memory_consideration') as mock_memory, \
             patch.object(hs, '_random_selection', wraps=hs._random_selection) as mock_random, \
             patch.object(hs, '_initialize'):
            
            hs.run()
            
            # Memory consideration should never be called
            assert mock_memory.call_count == 0
            # Random selection should be called num_parameters * max_imp times
            assert mock_random.call_count == obj_fun.get_num_parameters() * obj_fun.get_max_imp()
    
    def test_hmcr_one_always_memory_consideration(self):
        """Test that HMCR = 1 always uses memory consideration."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, hmcr=1.0, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch.object(hs, '_memory_consideration', wraps=hs._memory_consideration) as mock_memory, \
             patch.object(hs, '_random_selection') as mock_random, \
             patch.object(hs, '_initialize'):
            
            hs.run()
            
            # Memory consideration should be called num_parameters * max_imp times
            assert mock_memory.call_count == obj_fun.get_num_parameters() * obj_fun.get_max_imp()
            # Random selection should never be called in main loop
            assert mock_random.call_count == 0
    
    def test_par_zero_no_pitch_adjustment(self):
        """Test that PAR = 0 never performs pitch adjustment."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, hmcr=1.0, par=0.0, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch.object(hs, '_pitch_adjustment') as mock_pitch, \
             patch.object(hs, '_initialize'):
            
            hs.run()
            
            # Pitch adjustment should never be called
            assert mock_pitch.call_count == 0
    
    def test_par_one_always_pitch_adjustment(self):
        """Test that PAR = 1 always performs pitch adjustment after memory consideration."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, hmcr=1.0, par=1.0, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        with patch.object(hs, '_pitch_adjustment', wraps=hs._pitch_adjustment) as mock_pitch, \
             patch.object(hs, '_initialize'):
            
            hs.run()
            
            # Pitch adjustment should be called num_parameters * max_imp times
            # (since HMCR = 1.0, every parameter uses memory consideration)
            assert mock_pitch.call_count == obj_fun.get_num_parameters() * obj_fun.get_max_imp()


class TestHarmonySearchRunWithInitialHarmonies:
    """Test suite for HarmonySearch.run() with initial harmonies."""
    
    def test_run_with_valid_initial_harmonies(self):
        """Test that run() works correctly with valid initial harmonies."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=10, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        initial_harmonies = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        
        best_harmony, best_fitness, memory, history = hs.run(initial_harmonies=initial_harmonies)
        
        # Should complete successfully
        assert best_harmony is not None
        assert len(memory) == 3
        assert len(history) > 0
    
    def test_run_with_none_initial_harmonies(self):
        """Test that run() works correctly with None initial harmonies (random init)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run(initial_harmonies=None)
        
        # Should complete successfully with random initialization
        assert best_harmony is not None
        assert len(memory) == 5
        assert len(history) > 0


class TestHarmonySearchRunIntegration:
    """Integration tests for complete HarmonySearch.run() execution."""
    
    def test_complete_run_continuous_parameters(self):
        """Test a complete run with continuous parameters."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=3, hms=10, max_imp=50, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Verify basic properties
        assert len(best_harmony) == 3
        assert isinstance(best_fitness, (int, float))
        assert len(memory) == 10
        assert len(history) == 6  # gen 0 + 5 generations (50 / 10)
        
        # Verify all harmonies in memory are valid
        for harmony, fitness in memory:
            assert len(harmony) == 3
            for value in harmony:
                assert -100 <= value <= 100  # Within bounds
    
    def test_complete_run_discrete_parameters(self):
        """Test a complete run with discrete parameters."""
        obj_fun = DiscreteObjectiveFunction(num_parameters=2, hms=8, max_imp=40, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Verify basic properties
        assert len(best_harmony) == 2
        assert isinstance(best_fitness, (int, float))
        assert len(memory) == 8
        assert len(history) == 6  # gen 0 + 5 generations (40 / 8)
    
    def test_complete_run_mixed_parameters(self):
        """Test a complete run with mixed continuous and discrete parameters."""
        obj_fun = MixedObjectiveFunction(num_parameters=3, hms=5, max_imp=25, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Verify basic properties
        assert len(best_harmony) == 3
        assert isinstance(best_fitness, (int, float))
        assert len(memory) == 5
        assert len(history) == 6  # gen 0 + 5 generations (25 / 5)
    
    def test_run_converges_toward_optimum(self):
        """Test that algorithm converges toward optimum over generations."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=10, max_imp=100, maximize=False, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Get best fitness from each generation
        gen_fitnesses = []
        for entry in history:
            if entry['harmonies']:
                fitnesses = [fitness for harmony, fitness in entry['harmonies']]
                gen_fitnesses.append(min(fitnesses))  # Min for minimization
        
        # Check that later generations have better or equal fitness
        # (monotonic improvement is not guaranteed but should trend better)
        first_gen_fitness = gen_fitnesses[0]
        last_gen_fitness = gen_fitnesses[-1]
        
        # Last generation should be better than or equal to first
        assert last_gen_fitness <= first_gen_fitness
    
    def test_harmony_memory_size_maintained(self):
        """Test that harmony memory size is maintained throughout execution."""
        hms = 7
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=hms, max_imp=30, random_seed=42)
        hs = HarmonySearch(obj_fun)
        
        best_harmony, best_fitness, memory, history = hs.run()
        
        # Final memory should have exactly hms entries
        assert len(memory) == hms
        
        # Each generation in history should have exactly hms entries
        for entry in history:
            assert len(entry['harmonies']) == hms
