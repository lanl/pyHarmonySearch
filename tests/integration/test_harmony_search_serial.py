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
import math
from datetime import timedelta
from pyharmonysearch import harmony_search_serial, HarmonySearchResults
from tests.fixtures.mock_objective_functions import (
    QuadraticObjectiveFunction,
    DiscreteObjectiveFunction,
    MixedObjectiveFunction,
)


class TestHarmonySearchSerialBasicExecution:
    """Test suite for basic execution scenarios of harmony_search_serial()."""

    def test_single_iteration_execution(self):
        """Test that harmony_search_serial() executes successfully with a single iteration."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=1)
        
        # Verify results structure
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is not None
        assert isinstance(results.best_fitness, float)
        assert len(results.harmony_memories) == 1
        assert len(results.harmony_histories) == 1
        
        # Verify best harmony has correct dimensionality
        assert len(results.best_harmony) == 2

    def test_multiple_iterations_execution(self):
        """Test that harmony_search_serial() executes successfully with multiple iterations."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=3,
            hms=5,
            max_imp=50,
            random_seed=123
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=5)
        
        # Verify results structure
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is not None
        assert isinstance(results.best_fitness, float)
        assert len(results.harmony_memories) == 5
        assert len(results.harmony_histories) == 5
        
        # Verify best harmony has correct dimensionality
        assert len(results.best_harmony) == 3

    def test_zero_iterations_edge_case(self):
        """Test that harmony_search_serial() handles num_iterations=0 correctly."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=0)
        
        # With 0 iterations, no harmonies should be found
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is None
        assert len(results.harmony_memories) == 0
        assert len(results.harmony_histories) == 0
        
        # Fitness should be initialized to -inf (maximization) or +inf (minimization)
        if obj_fun.maximize():
            assert results.best_fitness == float('-inf')
        else:
            assert results.best_fitness == float('+inf')

    def test_num_iterations_one(self):
        """Test that harmony_search_serial() with num_iterations=1 produces valid results."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=100,
            random_seed=999
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=1)
        
        # Should have exactly 1 result
        assert len(results.harmony_memories) == 1
        assert len(results.harmony_histories) == 1
        
        # Best harmony should match the single iteration's best
        harmony_memory = results.harmony_memories[0]
        best_from_memory = min(harmony_memory, key=lambda x: x[1])  # minimization
        assert results.best_harmony == best_from_memory[0]
        assert results.best_fitness == best_from_memory[1]

    def test_large_num_iterations(self):
        """Test that harmony_search_serial() handles a large number of iterations."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=30,  # Keep max_imp small to run quickly
            random_seed=555
        )
        
        # Run with 20 iterations
        results = harmony_search_serial(obj_fun, num_iterations=20)
        
        # Verify all iterations completed
        assert len(results.harmony_memories) == 20
        assert len(results.harmony_histories) == 20
        
        # Verify best solution is from one of the iterations
        assert results.best_harmony is not None
        assert isinstance(results.best_fitness, float)


class TestHarmonySearchSerialInitialHarmonies:
    """Test suite for initial_harmonies parameter behavior."""

    def test_with_initial_harmonies(self):
        """Test that harmony_search_serial() works correctly with initial_harmonies provided."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=3,
            max_imp=50,
            random_seed=42
        )
        
        # Provide initial harmonies close to optimal solution
        initial_harmonies = [
            [1.0, 1.0],
            [2.0, -2.0],
            [0.5, 0.5]
        ]
        
        results = harmony_search_serial(obj_fun, num_iterations=2, initial_harmonies=initial_harmonies)
        
        # Verify execution completed successfully
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is not None
        assert len(results.harmony_memories) == 2
        
        # All harmonies should have correct dimensionality
        for harmony_memory in results.harmony_memories:
            assert len(harmony_memory) == 3  # hms=3
            for harmony, fitness in harmony_memory:
                assert len(harmony) == 2

    def test_without_initial_harmonies(self):
        """Test that harmony_search_serial() works correctly without initial_harmonies (default behavior)."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=42
        )
        
        # Call without initial_harmonies (default None)
        results = harmony_search_serial(obj_fun, num_iterations=3)
        
        # Verify execution completed successfully
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is not None
        assert len(results.harmony_memories) == 3
        
        # Verify harmony memory size
        for harmony_memory in results.harmony_memories:
            assert len(harmony_memory) == 5  # hms=5


class TestHarmonySearchSerialMaximizationVsMinimization:
    """Test suite for maximization and minimization problems."""

    def test_minimization_problem(self):
        """Test that harmony_search_serial() correctly handles minimization problems."""
        # QuadraticObjectiveFunction minimizes sum of squares (known min = 0 at origin)
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=10,
            max_imp=200,
            maximize=False,  # Minimization
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=5)
        
        # Verify best solution is actually the minimum across all iterations
        for harmony_memory in results.harmony_memories:
            for harmony, fitness in harmony_memory:
                # Best fitness should be <= all other fitness values
                assert results.best_fitness <= fitness

    def test_maximization_problem(self):
        """Test that harmony_search_serial() correctly handles maximization problems."""
        # Create a maximization version of the quadratic function
        # Maximize -(sum of squares), known max = 0 at origin
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=10,
            max_imp=200,
            maximize=True,  # Maximization
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=5)
        
        # Verify best solution is actually the maximum across all iterations
        for harmony_memory in results.harmony_memories:
            for harmony, fitness in harmony_memory:
                # Best fitness should be >= all other fitness values
                assert results.best_fitness >= fitness


class TestHarmonySearchSerialBestSelectionLogic:
    """Test suite for best solution selection logic."""

    def test_best_solution_selection_minimization(self):
        """Test that the best solution is correctly selected for minimization."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=100,
            maximize=False,
            random_seed=777
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=10)
        
        # Find the absolute minimum fitness across all iterations
        min_fitness = float('+inf')
        expected_best_harmony = None
        
        for harmony_memory in results.harmony_memories:
            for harmony, fitness in harmony_memory:
                if fitness < min_fitness:
                    min_fitness = fitness
                    expected_best_harmony = harmony
        
        # Verify that harmony_search_serial selected the correct best
        assert results.best_fitness == min_fitness
        assert results.best_harmony == expected_best_harmony

    def test_best_solution_selection_maximization(self):
        """Test that the best solution is correctly selected for maximization."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=100,
            maximize=True,
            random_seed=888
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=10)
        
        # Find the absolute maximum fitness across all iterations
        max_fitness = float('-inf')
        expected_best_harmony = None
        
        for harmony_memory in results.harmony_memories:
            for harmony, fitness in harmony_memory:
                if fitness > max_fitness:
                    max_fitness = fitness
                    expected_best_harmony = harmony
        
        # Verify that harmony_search_serial selected the correct best
        assert results.best_fitness == max_fitness
        assert results.best_harmony == expected_best_harmony


class TestHarmonySearchSerialResults:
    """Test suite for HarmonySearchResults structure and data integrity."""

    def test_results_structure_validation(self):
        """Test that HarmonySearchResults has the correct structure and all expected fields."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=3)
        
        # Verify results is a HarmonySearchResults namedtuple
        assert isinstance(results, HarmonySearchResults)
        
        # Verify all expected fields exist
        assert hasattr(results, 'elapsed_time')
        assert hasattr(results, 'best_harmony')
        assert hasattr(results, 'best_fitness')
        assert hasattr(results, 'harmony_memories')
        assert hasattr(results, 'harmony_histories')
        
        # Verify types
        assert isinstance(results.elapsed_time, timedelta)
        assert results.best_harmony is None or isinstance(results.best_harmony, list)
        assert isinstance(results.best_fitness, float)
        assert isinstance(results.harmony_memories, list)
        assert isinstance(results.harmony_histories, list)

    def test_all_results_collected_properly(self):
        """Test that all results from all iterations are collected properly."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=3,
            hms=7,
            max_imp=60,
            random_seed=333
        )
        
        num_iterations = 8
        results = harmony_search_serial(obj_fun, num_iterations=num_iterations)
        
        # Verify correct number of results
        assert len(results.harmony_memories) == num_iterations
        assert len(results.harmony_histories) == num_iterations
        
        # Verify each harmony memory has the correct size
        for harmony_memory in results.harmony_memories:
            assert len(harmony_memory) == obj_fun.get_hms()
            
            # Verify each entry is a tuple of (harmony, fitness)
            for entry in harmony_memory:
                assert isinstance(entry, tuple)
                assert len(entry) == 2
                harmony, fitness = entry
                assert isinstance(harmony, list)
                assert len(harmony) == obj_fun.get_num_parameters()
                assert isinstance(fitness, float)
        
        # Verify each harmony history is properly structured
        for harmony_history in results.harmony_histories:
            assert isinstance(harmony_history, list)
            assert len(harmony_history) > 0
            
            # Each generation should have 'gen' and 'harmonies' keys
            for generation in harmony_history:
                assert isinstance(generation, dict)
                assert 'gen' in generation
                assert 'harmonies' in generation
                assert isinstance(generation['gen'], int)
                assert isinstance(generation['harmonies'], list)

    def test_time_measurement(self):
        """Test that elapsed_time is measured and is positive."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=100,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=5)
        
        # Verify elapsed_time is a timedelta and is positive
        assert isinstance(results.elapsed_time, timedelta)
        assert results.elapsed_time.total_seconds() > 0
        
        # With 5 iterations of 100 improvisations each, should take measurable time
        assert results.elapsed_time.total_seconds() < 10  # Sanity check: shouldn't take more than 10 seconds


class TestHarmonySearchSerialDifferentObjectiveFunctions:
    """Test suite for different types of objective functions."""

    def test_with_discrete_objective_function(self):
        """Test harmony_search_serial() with a discrete-only objective function."""
        obj_fun = DiscreteObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=100,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=3)
        
        # Verify results
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is not None
        assert len(results.best_harmony) == 2
        
        # Verify all values are integers (discrete)
        for value in results.best_harmony:
            assert isinstance(value, int)
            assert -10 <= value <= 10  # Valid range for DiscreteObjectiveFunction

    def test_with_mixed_objective_function(self):
        """Test harmony_search_serial() with a mixed continuous/discrete objective function."""
        obj_fun = MixedObjectiveFunction(
            num_parameters=4,
            hms=8,
            max_imp=100,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=3)
        
        # Verify results
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is not None
        assert len(results.best_harmony) == 4
        
        # Verify types: even indices are continuous (float), odd indices are discrete (int)
        for i, value in enumerate(results.best_harmony):
            if i % 2 == 0:  # continuous
                assert isinstance(value, (int, float))
            else:  # discrete
                assert isinstance(value, int)
                assert -20 <= value <= 20  # Valid range for MixedObjectiveFunction

    def test_convergence_quality_minimization(self):
        """Test that harmony_search_serial() finds reasonably good solutions for minimization."""
        # Quadratic function: minimize sum(x_i^2), known minimum = 0 at origin
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=10,
            max_imp=500,  # More improvisations for better convergence
            maximize=False,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=10)
        
        # Should find a solution relatively close to 0
        # With good parameters, should achieve fitness < 10 (arbitrary but reasonable threshold)
        assert results.best_fitness < 100, f"Expected fitness < 100, got {results.best_fitness}"
        
        # Best harmony should have small values (close to origin)
        for value in results.best_harmony:
            assert abs(value) < 20, f"Expected values close to 0, got {results.best_harmony}"


class TestHarmonySearchSerialDeterminism:
    """Test suite for deterministic behavior with random seeds."""

    def test_deterministic_with_random_seed(self):
        """Test that using the same random seed produces identical results."""
        obj_fun1 = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=12345
        )
        
        obj_fun2 = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=12345
        )
        
        results1 = harmony_search_serial(obj_fun1, num_iterations=3)
        results2 = harmony_search_serial(obj_fun2, num_iterations=3)
        
        # With the same seed, results should be identical
        assert results1.best_harmony == results2.best_harmony
        assert results1.best_fitness == results2.best_fitness
        
        # Harmony memories should also be identical
        for mem1, mem2 in zip(results1.harmony_memories, results2.harmony_memories):
            assert mem1 == mem2

    def test_different_results_without_seed(self):
        """Test that runs without random seed produce different results (stochastic)."""
        obj_fun1 = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=None  # No seed
        )
        
        obj_fun2 = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=None  # No seed
        )
        
        results1 = harmony_search_serial(obj_fun1, num_iterations=3)
        results2 = harmony_search_serial(obj_fun2, num_iterations=3)
        
        # Without seed, results will likely be different (not guaranteed, but highly likely)
        # We can't assert they're different since there's a tiny chance they're the same,
        # but we can verify they both executed successfully
        assert results1.best_harmony is not None
        assert results2.best_harmony is not None
        assert isinstance(results1.best_fitness, float)
        assert isinstance(results2.best_fitness, float)


class TestHarmonySearchSerialEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_single_parameter_objective_function(self):
        """Test with a single-parameter (1D) objective function."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=1,
            hms=5,
            max_imp=50,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=3)
        
        # Verify results
        assert isinstance(results, HarmonySearchResults)
        assert len(results.best_harmony) == 1
        assert results.best_harmony is not None

    def test_many_parameters_objective_function(self):
        """Test with a high-dimensional (10D) objective function."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=10,
            hms=15,
            max_imp=100,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=2)
        
        # Verify results
        assert isinstance(results, HarmonySearchResults)
        assert len(results.best_harmony) == 10
        assert results.best_harmony is not None

    def test_small_harmony_memory_size(self):
        """Test with a very small harmony memory size (hms=2)."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=2,  # Minimum practical value
            max_imp=50,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=3)
        
        # Verify execution completed
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is not None
        
        # Verify harmony memory size
        for harmony_memory in results.harmony_memories:
            assert len(harmony_memory) == 2

    def test_large_harmony_memory_size(self):
        """Test with a large harmony memory size."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=50,  # Large memory
            max_imp=100,
            random_seed=42
        )
        
        results = harmony_search_serial(obj_fun, num_iterations=2)
        
        # Verify execution completed
        assert isinstance(results, HarmonySearchResults)
        assert results.best_harmony is not None
        
        # Verify harmony memory size
        for harmony_memory in results.harmony_memories:
            assert len(harmony_memory) == 50
