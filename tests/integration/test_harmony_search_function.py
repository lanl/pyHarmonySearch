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
Integration tests for harmony_search() function with actual multiprocessing.

These tests verify that harmony_search() correctly orchestrates multiple worker processes,
aggregates results, and returns the best solution across all iterations.
"""

import pytest
from datetime import timedelta
from unittest.mock import patch, MagicMock

from pyharmonysearch import harmony_search, harmony_search_serial
from tests.fixtures.test_helpers import (
    assert_harmony_in_bounds,
    assert_valid_harmony_memory,
    assert_valid_harmony_history,
    verify_results_structure,
    get_best_from_memory,
    create_mock_objective_function,
)


class TestHarmonySearchBasicConfigurations:
    """Tests for various (num_processes, num_iterations) combinations."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_2_processes_2_iterations(self, simple_objective_function):
        """Test with 2 processes and 2 iterations."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=2)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 2
        assert len(results.harmony_histories) == 2
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_4_processes_4_iterations(self, simple_objective_function):
        """Test with 4 processes and 4 iterations."""
        results = harmony_search(simple_objective_function, num_processes=4, num_iterations=4)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 4
        assert len(results.harmony_histories) == 4
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_2_processes_8_iterations(self, simple_objective_function):
        """Test with 2 processes and 8 iterations."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=8)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 8
        assert len(results.harmony_histories) == 8
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_8_processes_2_iterations(self, simple_objective_function):
        """Test with 8 processes and 2 iterations."""
        results = harmony_search(simple_objective_function, num_processes=8, num_iterations=2)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 2
        assert len(results.harmony_histories) == 2
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_3_processes_7_iterations(self, simple_objective_function):
        """Test with 3 processes and 7 iterations (non-divisible)."""
        results = harmony_search(simple_objective_function, num_processes=3, num_iterations=7)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 7
        assert len(results.harmony_histories) == 7
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)


class TestHarmonySearchProcessIterationRatios:
    """Tests for edge cases with process/iteration ratios."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_single_process_matches_serial(self, simple_objective_function):
        """Test that num_processes=1 produces same structure as serial."""
        results_parallel = harmony_search(simple_objective_function, num_processes=1, num_iterations=3)
        results_serial = harmony_search_serial(simple_objective_function, num_iterations=3)

        # Both should have same structure
        verify_results_structure(results_parallel)
        verify_results_structure(results_serial)

        assert len(results_parallel.harmony_memories) == len(results_serial.harmony_memories)
        assert len(results_parallel.harmony_histories) == len(results_serial.harmony_histories)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_multiple_processes_single_iteration_each(self, simple_objective_function):
        """Test multiple processes with single iteration per process."""
        num_processes = 4
        results = harmony_search(simple_objective_function, num_processes=num_processes, num_iterations=1)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 1
        assert len(results.harmony_histories) == 1
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_multiple_processes_multiple_iterations(self, simple_objective_function):
        """Test multiple processes with multiple iterations."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=6)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 6
        assert len(results.harmony_histories) == 6
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_more_processes_than_iterations(self, simple_objective_function):
        """Test when num_processes > num_iterations (some processes idle)."""
        results = harmony_search(simple_objective_function, num_processes=8, num_iterations=3)

        verify_results_structure(results)
        # Should still get results from all iterations
        assert len(results.harmony_memories) == 3
        assert len(results.harmony_histories) == 3
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_more_iterations_than_processes(self, simple_objective_function):
        """Test when num_iterations > num_processes (processes reused)."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=10)

        verify_results_structure(results)
        # Should get results from all iterations
        assert len(results.harmony_memories) == 10
        assert len(results.harmony_histories) == 10
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)


class TestHarmonySearchOptimizationDirection:
    """Tests for maximization and minimization problems."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_maximization_problem(self):
        """Test that maximization correctly selects highest fitness."""
        obj_fun = create_mock_objective_function(
            maximize=True,
            max_imp=50,
            hms=5,
            use_seed=True,
            random_seed=42
        )

        results = harmony_search(obj_fun, num_processes=2, num_iterations=3)

        verify_results_structure(results)

        # For maximization, best_fitness should be the maximum across all iterations
        all_fitnesses = []
        for memory in results.harmony_memories:
            for _, fitness in memory:
                all_fitnesses.append(fitness)

        assert results.best_fitness == max(all_fitnesses)
        assert results.best_fitness >= max(f for _, f in results.harmony_memories[0])
        assert results.best_fitness >= max(f for _, f in results.harmony_memories[1])

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_minimization_problem(self):
        """Test that minimization correctly selects lowest fitness."""
        obj_fun = create_mock_objective_function(
            maximize=False,
            max_imp=50,
            hms=5,
            use_seed=True,
            random_seed=42
        )

        results = harmony_search(obj_fun, num_processes=2, num_iterations=3)

        verify_results_structure(results)

        # For minimization, best_fitness should be the minimum across all iterations
        all_fitnesses = []
        for memory in results.harmony_memories:
            for _, fitness in memory:
                all_fitnesses.append(fitness)

        assert results.best_fitness == min(all_fitnesses)
        assert results.best_fitness <= min(f for _, f in results.harmony_memories[0])
        assert results.best_fitness <= min(f for _, f in results.harmony_memories[1])


class TestHarmonySearchInitialHarmonies:
    """Tests for initial_harmonies parameter."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_with_initial_harmonies(self, simple_objective_function):
        """Test harmony_search with provided initial harmonies."""
        # Create initial harmonies
        hms = simple_objective_function.get_hms()
        initial_harmonies = []
        for i in range(hms):
            harmony = [float(i), float(-i)]
            initial_harmonies.append(harmony)

        results = harmony_search(
            simple_objective_function,
            num_processes=2,
            num_iterations=2,
            initial_harmonies=initial_harmonies
        )

        verify_results_structure(results)
        assert len(results.harmony_memories) == 2
        assert len(results.harmony_histories) == 2

        # Each iteration should have used the initial harmonies
        for memory in results.harmony_memories:
            assert_valid_harmony_memory(memory, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_without_initial_harmonies(self, simple_objective_function):
        """Test harmony_search without initial harmonies (default behavior)."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=2)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 2
        assert len(results.harmony_histories) == 2

        # Should have randomly initialized harmonies
        for memory in results.harmony_memories:
            assert_valid_harmony_memory(memory, simple_objective_function)


class TestHarmonySearchResultsStructure:
    """Tests for the structure and validity of results."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_results_structure_validation(self, simple_objective_function):
        """Test that results have correct structure with all required fields."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=3)

        # Check namedtuple fields exist
        assert hasattr(results, 'elapsed_time')
        assert hasattr(results, 'best_harmony')
        assert hasattr(results, 'best_fitness')
        assert hasattr(results, 'harmony_memories')
        assert hasattr(results, 'harmony_histories')

        # Check types
        assert isinstance(results.elapsed_time, timedelta)
        assert isinstance(results.best_harmony, list)
        assert isinstance(results.best_fitness, (int, float))
        assert isinstance(results.harmony_memories, list)
        assert isinstance(results.harmony_histories, list)

        # Check that best_harmony is valid
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_best_solution_selected_across_all_processes(self, simple_objective_function):
        """Test that the global best is correctly selected from all processes."""
        results = harmony_search(simple_objective_function, num_processes=4, num_iterations=8)

        verify_results_structure(results)

        # Extract best from each iteration's memory
        all_best_fitnesses = []
        for memory in results.harmony_memories:
            best_harmony, best_fitness = get_best_from_memory(
                memory,
                maximize=simple_objective_function.maximize()
            )
            all_best_fitnesses.append(best_fitness)

        # The global best should be better than or equal to all individual bests
        if simple_objective_function.maximize():
            assert results.best_fitness >= max(all_best_fitnesses)
        else:
            assert results.best_fitness <= min(all_best_fitnesses)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_all_harmony_memories_collected(self, simple_objective_function):
        """Test that harmony memories from all iterations are collected."""
        num_iterations = 5
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=num_iterations)

        verify_results_structure(results)

        # Should have one harmony memory per iteration
        assert len(results.harmony_memories) == num_iterations

        # Each harmony memory should be valid
        for memory in results.harmony_memories:
            assert_valid_harmony_memory(memory, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_all_harmony_histories_collected(self, simple_objective_function):
        """Test that harmony histories from all iterations are collected."""
        num_iterations = 5
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=num_iterations)

        verify_results_structure(results)

        # Should have one harmony history per iteration
        assert len(results.harmony_histories) == num_iterations

        # Each harmony history should be valid
        for history in results.harmony_histories:
            assert_valid_harmony_history(history, simple_objective_function)


class TestHarmonySearchTiming:
    """Tests for time measurement."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_time_measurement(self, simple_objective_function):
        """Test that elapsed time is measured and reasonable."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=2)

        verify_results_structure(results)

        # Elapsed time should be positive
        assert results.elapsed_time.total_seconds() > 0

        # Should be less than some reasonable timeout (e.g., 30 seconds for 2 iterations)
        assert results.elapsed_time.total_seconds() < 30


class TestHarmonySearchDeterminism:
    """Tests for deterministic behavior with random seed."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_deterministic_with_random_seed(self):
        """Test that same seed produces same results across runs."""
        obj_fun1 = create_mock_objective_function(
            use_seed=True,
            random_seed=12345,
            max_imp=50,
            hms=5
        )

        obj_fun2 = create_mock_objective_function(
            use_seed=True,
            random_seed=12345,
            max_imp=50,
            hms=5
        )

        # Run with same seed
        results1 = harmony_search(obj_fun1, num_processes=1, num_iterations=2)
        results2 = harmony_search(obj_fun2, num_processes=1, num_iterations=2)

        # Results should be identical with same seed and single process
        assert results1.best_fitness == results2.best_fitness
        assert results1.best_harmony == results2.best_harmony

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        obj_fun1 = create_mock_objective_function(
            use_seed=True,
            random_seed=111,
            max_imp=100,
            hms=10
        )

        obj_fun2 = create_mock_objective_function(
            use_seed=True,
            random_seed=999,
            max_imp=100,
            hms=10
        )

        # Run with different seeds
        results1 = harmony_search(obj_fun1, num_processes=1, num_iterations=2)
        results2 = harmony_search(obj_fun2, num_processes=1, num_iterations=2)

        # Results should likely be different (though not guaranteed)
        # At least the harmony memories should differ
        assert results1.harmony_memories != results2.harmony_memories


class TestHarmonySearchPoolLifecycle:
    """Tests for multiprocessing.Pool lifecycle management."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_pool_close_and_join_called(self, simple_objective_function):
        """Test that pool.close() and pool.join() are called properly."""
        with patch('pyharmonysearch.harmony_search.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            # Mock apply_async to return mock results
            mock_result = MagicMock()
            mock_result.get.return_value = ([0.0, 0.0], 0.0, [([0.0, 0.0], 0.0)], [{'gen': 0, 'harmonies': []}])
            mock_pool.apply_async.return_value = mock_result

            harmony_search(simple_objective_function, num_processes=2, num_iterations=2)

            # Verify pool was created with correct number of processes
            mock_pool_class.assert_called_once_with(2)

            # Verify close and join were called
            mock_pool.close.assert_called_once()
            mock_pool.join.assert_called_once()

            # Verify close was called before join
            close_call_order = mock_pool.method_calls.index(('close', (), {}))
            join_call_order = mock_pool.method_calls.index(('join', (), {}))
            assert close_call_order < join_call_order

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_pool_terminate_on_keyboard_interrupt(self, simple_objective_function):
        """Test that pool.terminate() is called on KeyboardInterrupt."""
        with patch('pyharmonysearch.harmony_search.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            # Mock join to raise KeyboardInterrupt
            mock_pool.join.side_effect = KeyboardInterrupt()

            with pytest.raises(KeyboardInterrupt):
                harmony_search(simple_objective_function, num_processes=2, num_iterations=2)

            # Verify terminate was called
            mock_pool.terminate.assert_called_once()


class TestHarmonySearchRealObjectiveFunctions:
    """Integration tests with real objective functions."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_integration_with_continuous_objective_function(self, simple_objective_function):
        """Test full integration with continuous parameters."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=4)

        verify_results_structure(results)

        # Should find reasonable solution for quadratic function
        # Maximum is 4 at (0, -1)
        assert results.best_fitness <= 4.0  # Should not exceed theoretical maximum
        assert results.best_fitness > -100  # Should find something reasonable

        # All harmonies should be in bounds
        for memory in results.harmony_memories:
            for harmony, _ in memory:
                assert_harmony_in_bounds(harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_integration_with_discrete_objective_function(self, discrete_objective_function):
        """Test full integration with discrete parameters."""
        results = harmony_search(discrete_objective_function, num_processes=2, num_iterations=4)

        verify_results_structure(results)

        # Should find reasonable solution
        assert results.best_fitness <= 4.0  # Should not exceed theoretical maximum
        assert results.best_fitness > -100

        # All harmonies should contain valid discrete values
        for memory in results.harmony_memories:
            for harmony, _ in memory:
                assert_harmony_in_bounds(harmony, discrete_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_integration_with_mixed_objective_function(self, mixed_objective_function):
        """Test full integration with mixed continuous and discrete parameters."""
        results = harmony_search(mixed_objective_function, num_processes=2, num_iterations=4)

        verify_results_structure(results)

        # Should find reasonable solution
        assert results.best_fitness <= 4.0  # Should not exceed theoretical maximum
        assert results.best_fitness > -100

        # All harmonies should be valid
        for memory in results.harmony_memories:
            for harmony, _ in memory:
                assert_harmony_in_bounds(harmony, mixed_objective_function)


class TestHarmonySearchEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_single_iteration_single_process(self, simple_objective_function):
        """Test minimal case: 1 process, 1 iteration."""
        results = harmony_search(simple_objective_function, num_processes=1, num_iterations=1)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 1
        assert len(results.harmony_histories) == 1
        assert_harmony_in_bounds(results.best_harmony, simple_objective_function)

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_many_processes_many_iterations(self, simple_objective_function):
        """Test with larger number of processes and iterations."""
        # Reduce max_imp to speed up test
        obj_fun = create_mock_objective_function(
            max_imp=20,
            hms=5,
            use_seed=True,
            random_seed=42
        )

        results = harmony_search(obj_fun, num_processes=4, num_iterations=12)

        verify_results_structure(results)
        assert len(results.harmony_memories) == 12
        assert len(results.harmony_histories) == 12

    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_best_harmony_corresponds_to_best_fitness(self, simple_objective_function):
        """Test that best_harmony fitness matches best_fitness."""
        results = harmony_search(simple_objective_function, num_processes=2, num_iterations=4)

        verify_results_structure(results)

        # Compute fitness of best_harmony
        computed_fitness = simple_objective_function.get_fitness(results.best_harmony)

        # Should match the reported best_fitness
        assert abs(computed_fitness - results.best_fitness) < 1e-10
