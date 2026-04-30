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
import pickle
import os
import multiprocessing
import sys
import signal
from unittest.mock import patch

from pyharmonysearch import harmony_search, harmony_search_serial
from tests.fixtures.mock_objective_functions import (
    QuadraticObjectiveFunction,
    DiscreteObjectiveFunction,
    MixedObjectiveFunction,
    FixedParameterObjectiveFunction,
)


# Mark all tests as multiprocessing and slow
pytestmark = [pytest.mark.multiprocessing, pytest.mark.slow]


class TestObjectiveFunctionPicklability:
    """Test that all objective function types are picklable for multiprocessing."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_quadratic_objective_function_picklable(self):
        """Test that QuadraticObjectiveFunction can be pickled/unpickled."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=3,
            hms=5,
            max_imp=10,
            random_seed=42
        )
        
        # Pickle and unpickle
        pickled = pickle.dumps(obj_fun)
        unpickled = pickle.loads(pickled)
        
        # Test that unpickled function works correctly
        test_vector = [1.0, 2.0, 3.0]
        assert obj_fun.get_fitness(test_vector) == unpickled.get_fitness(test_vector)
        assert obj_fun.get_num_parameters() == unpickled.get_num_parameters()
        assert obj_fun.get_hms() == unpickled.get_hms()
        assert obj_fun.get_max_imp() == unpickled.get_max_imp()
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_discrete_objective_function_picklable(self):
        """Test that DiscreteObjectiveFunction can be pickled/unpickled."""
        obj_fun = DiscreteObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=10,
            random_seed=42
        )
        
        # Pickle and unpickle
        pickled = pickle.dumps(obj_fun)
        unpickled = pickle.loads(pickled)
        
        # Test that unpickled function works correctly
        test_vector = [5, -3]
        assert obj_fun.get_fitness(test_vector) == unpickled.get_fitness(test_vector)
        assert obj_fun.get_num_discrete_values(0) == unpickled.get_num_discrete_values(0)
        assert obj_fun.get_index(0, 5) == unpickled.get_index(0, 5)
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_mixed_objective_function_picklable(self):
        """Test that MixedObjectiveFunction can be pickled/unpickled."""
        obj_fun = MixedObjectiveFunction(
            num_parameters=4,
            hms=5,
            max_imp=10,
            random_seed=42
        )
        
        # Pickle and unpickle
        pickled = pickle.dumps(obj_fun)
        unpickled = pickle.loads(pickled)
        
        # Test that unpickled function works correctly
        test_vector = [1.5, 3, -2.5, -7]
        assert obj_fun.get_fitness(test_vector) == unpickled.get_fitness(test_vector)
        assert obj_fun.is_discrete(0) == unpickled.is_discrete(0)
        assert obj_fun.is_discrete(1) == unpickled.is_discrete(1)
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_fixed_parameter_objective_function_picklable(self):
        """Test that FixedParameterObjectiveFunction can be pickled/unpickled."""
        obj_fun = FixedParameterObjectiveFunction(
            num_parameters=4,
            hms=5,
            max_imp=10,
            random_seed=42
        )
        
        # Pickle and unpickle
        pickled = pickle.dumps(obj_fun)
        unpickled = pickle.loads(pickled)
        
        # Test that unpickled function works correctly
        test_vector = [1.0, 0.0, 2.0, 0.0]
        assert obj_fun.get_fitness(test_vector) == unpickled.get_fitness(test_vector)
        assert obj_fun.is_variable(0) == unpickled.is_variable(0)
        assert obj_fun.is_variable(1) == unpickled.is_variable(1)


class TestSingleProcessMatchesSerial:
    """Test that single-process multiprocessing produces same results as serial version."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_single_process_matches_serial_quadratic(self):
        """Test single-process multiprocessing matches serial for quadratic function."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=42
        )
        
        # Run serial version
        serial_results = harmony_search_serial(obj_fun, num_iterations=1)
        
        # Run multiprocessing version with 1 process
        mp_results = harmony_search(obj_fun, num_processes=1, num_iterations=1)
        
        # Results should match exactly with same seed
        assert serial_results.best_fitness == mp_results.best_fitness
        assert serial_results.best_harmony == mp_results.best_harmony
        assert len(serial_results.harmony_memories) == len(mp_results.harmony_memories)
        assert len(serial_results.harmony_histories) == len(mp_results.harmony_histories)
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_single_process_matches_serial_discrete(self):
        """Test single-process multiprocessing matches serial for discrete function."""
        obj_fun = DiscreteObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=99
        )
        
        # Run serial version
        serial_results = harmony_search_serial(obj_fun, num_iterations=1)
        
        # Run multiprocessing version with 1 process
        mp_results = harmony_search(obj_fun, num_processes=1, num_iterations=1)
        
        # Results should match exactly with same seed
        assert serial_results.best_fitness == mp_results.best_fitness
        assert serial_results.best_harmony == mp_results.best_harmony
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_single_process_matches_serial_mixed(self):
        """Test single-process multiprocessing matches serial for mixed function."""
        obj_fun = MixedObjectiveFunction(
            num_parameters=4,
            hms=5,
            max_imp=20,
            random_seed=123
        )
        
        # Run serial version
        serial_results = harmony_search_serial(obj_fun, num_iterations=1)
        
        # Run multiprocessing version with 1 process
        mp_results = harmony_search(obj_fun, num_processes=1, num_iterations=1)
        
        # Results should match exactly with same seed
        assert serial_results.best_fitness == mp_results.best_fitness
        assert serial_results.best_harmony == mp_results.best_harmony


class TestMultipleProcesses:
    """Test actual multiprocessing with different process counts."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_two_processes(self):
        """Test with 2 processes."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=30,
            random_seed=None  # Different results per process
        )
        
        results = harmony_search(obj_fun, num_processes=2, num_iterations=4)
        
        # Should have results from 4 iterations
        assert len(results.harmony_memories) == 4
        assert len(results.harmony_histories) == 4
        assert results.best_harmony is not None
        assert isinstance(results.best_fitness, float)
        
        # Best fitness should be reasonable (close to optimal of 0)
        assert results.best_fitness >= 0  # Sum of squares is non-negative
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_four_processes(self):
        """Test with 4 processes."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=3,
            hms=8,
            max_imp=40,
            random_seed=None
        )
        
        results = harmony_search(obj_fun, num_processes=4, num_iterations=8)
        
        # Should have results from 8 iterations
        assert len(results.harmony_memories) == 8
        assert len(results.harmony_histories) == 8
        assert results.best_harmony is not None
        assert len(results.best_harmony) == 3
        
        # Verify elapsed time is tracked
        assert results.elapsed_time.total_seconds() > 0


class TestIterationsVsProcesses:
    """Test work distribution with different iteration/process ratios."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_more_iterations_than_processes(self):
        """Test with more iterations than processes (work queuing)."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=None
        )
        
        # 10 iterations with 2 processes
        results = harmony_search(obj_fun, num_processes=2, num_iterations=10)
        
        # Should complete all 10 iterations
        assert len(results.harmony_memories) == 10
        assert len(results.harmony_histories) == 10
        assert results.best_harmony is not None
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_fewer_iterations_than_processes(self):
        """Test with fewer iterations than processes (some processes idle)."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=None
        )
        
        # 2 iterations with 4 processes (2 processes will be idle)
        results = harmony_search(obj_fun, num_processes=4, num_iterations=2)
        
        # Should complete exactly 2 iterations
        assert len(results.harmony_memories) == 2
        assert len(results.harmony_histories) == 2
        assert results.best_harmony is not None
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_equal_iterations_and_processes(self):
        """Test with equal number of iterations and processes."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=None
        )
        
        # 3 iterations with 3 processes
        results = harmony_search(obj_fun, num_processes=3, num_iterations=3)
        
        # Should complete exactly 3 iterations
        assert len(results.harmony_memories) == 3
        assert len(results.harmony_histories) == 3
        assert results.best_harmony is not None


class TestCPUCountProcesses:
    """Test with cpu_count() processes."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_cpu_count_processes(self):
        """Test with number of processes equal to CPU count."""
        cpu_count = multiprocessing.cpu_count()
        
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=None
        )
        
        # Use cpu_count processes
        results = harmony_search(obj_fun, num_processes=cpu_count, num_iterations=cpu_count * 2)
        
        # Should complete all iterations
        assert len(results.harmony_memories) == cpu_count * 2
        assert len(results.harmony_histories) == cpu_count * 2
        assert results.best_harmony is not None
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_half_cpu_count_processes(self):
        """Test with half the CPU count."""
        cpu_count = max(1, multiprocessing.cpu_count() // 2)
        
        obj_fun = DiscreteObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=None
        )
        
        results = harmony_search(obj_fun, num_processes=cpu_count, num_iterations=cpu_count * 3)
        
        assert len(results.harmony_memories) == cpu_count * 3
        assert results.best_harmony is not None


class TestResultsAggregation:
    """Test that results are properly aggregated across processes."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_best_fitness_selected_from_all_processes(self):
        """Test that the best fitness across all processes is selected."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=50,
            random_seed=None
        )
        
        # Run multiple iterations
        results = harmony_search(obj_fun, num_processes=3, num_iterations=9)
        
        # Verify best fitness is actually the best among all iterations
        for harmony_memory in results.harmony_memories:
            for harmony, fitness in harmony_memory:
                # Best fitness should be <= all other fitnesses (minimization)
                assert results.best_fitness <= fitness
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_all_harmony_memories_collected(self):
        """Test that harmony memories from all processes are collected."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=6,
            max_imp=30,
            random_seed=None
        )
        
        num_iterations = 5
        results = harmony_search(obj_fun, num_processes=2, num_iterations=num_iterations)
        
        # Should have one harmony memory per iteration
        assert len(results.harmony_memories) == num_iterations
        
        # Each harmony memory should have hms entries
        for harmony_memory in results.harmony_memories:
            assert len(harmony_memory) == obj_fun.get_hms()
            
            # Each entry should be a (harmony, fitness) tuple
            for harmony, fitness in harmony_memory:
                assert isinstance(harmony, list)
                assert isinstance(fitness, (int, float))
                assert len(harmony) == obj_fun.get_num_parameters()
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_all_harmony_histories_collected(self):
        """Test that harmony histories from all processes are collected."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=30,
            random_seed=None
        )
        
        num_iterations = 4
        results = harmony_search(obj_fun, num_processes=2, num_iterations=num_iterations)
        
        # Should have one harmony history per iteration
        assert len(results.harmony_histories) == num_iterations
        
        # Each history should contain generation snapshots
        for harmony_history in results.harmony_histories:
            assert isinstance(harmony_history, list)
            assert len(harmony_history) > 0  # Should have at least initial generation
            
            # Each generation should have 'gen' and 'harmonies' keys
            for generation in harmony_history:
                assert 'gen' in generation
                assert 'harmonies' in generation
                assert isinstance(generation['gen'], int)
                assert isinstance(generation['harmonies'], list)


class TestLargeResults:
    """Test that large results don't cause deadlocks or memory issues."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_large_harmony_memory(self):
        """Test with large harmony memory size."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=10,
            hms=50,  # Large harmony memory
            max_imp=100,
            random_seed=42
        )
        
        results = harmony_search(obj_fun, num_processes=2, num_iterations=4)
        
        # Should complete without deadlock
        assert len(results.harmony_memories) == 4
        for harmony_memory in results.harmony_memories:
            assert len(harmony_memory) == 50
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_many_parameters(self):
        """Test with many parameters (large harmony vectors)."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=100,  # Many parameters
            hms=10,
            max_imp=50,
            random_seed=42
        )
        
        results = harmony_search(obj_fun, num_processes=2, num_iterations=3)
        
        # Should complete without deadlock
        assert len(results.harmony_memories) == 3
        assert len(results.best_harmony) == 100
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_many_iterations(self):
        """Test with many iterations (large history)."""
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=10,
            max_imp=200,  # Many improvisations (large history)
            random_seed=42
        )
        
        results = harmony_search(obj_fun, num_processes=2, num_iterations=6)
        
        # Should complete without deadlock
        assert len(results.harmony_histories) == 6
        for harmony_history in results.harmony_histories:
            # Should have multiple generations
            assert len(harmony_history) > 1


class TestKeyboardInterrupt:
    """Test KeyboardInterrupt handling in multiprocessing."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_keyboard_interrupt_propagates(self):
        """Test that KeyboardInterrupt is properly caught and re-raised.
        
        Note: We can't easily simulate a real KeyboardInterrupt during pool execution
        in a test environment, but we verify that the exception handling code path exists.
        """
        # Import to ensure the exception handling code is present in harmony_search
        from pyharmonysearch.harmony_search import harmony_search as hs_func
        import inspect
        
        # Verify that the harmony_search function has KeyboardInterrupt handling
        source = inspect.getsource(hs_func)
        
        # Check that it has except KeyboardInterrupt clause
        assert 'except KeyboardInterrupt' in source
        assert 'pool.terminate()' in source
        assert 'raise' in source
        
        # Also verify the function can run normally without interruption
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=42
        )
        
        # Should complete successfully when not interrupted
        results = harmony_search(obj_fun, num_processes=2, num_iterations=2)
        assert results.best_harmony is not None


class TestTerminatingEvent:
    """Test the global terminating Event state management."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_terminating_event_cleared_before_run(self):
        """Test that terminating event is cleared before each run."""
        # The conftest.py reset_global_state fixture should handle this
        import sys
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Event should be clear at start
        assert not hs_module.terminating.is_set()
        
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=42
        )
        
        results = harmony_search(obj_fun, num_processes=2, num_iterations=2)
        
        # Event should still be clear after successful run
        assert not hs_module.terminating.is_set()
        assert results.best_harmony is not None
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_terminating_event_isolation_between_tests(self):
        """Test that terminating event doesn't leak between tests."""
        import sys
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Verify event is clear
        assert not hs_module.terminating.is_set()
        
        # Run a search
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=42
        )
        
        results = harmony_search(obj_fun, num_processes=2, num_iterations=2)
        
        # Event should still be clear
        assert not hs_module.terminating.is_set()


class TestWorkerRespectsTerminatingSignal:
    """Test that workers check and respect the terminating signal."""
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_worker_checks_terminating_event(self):
        """Test that worker function checks terminating event before starting work."""
        import sys
        from pyharmonysearch.harmony_search import worker
        
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Set terminating event
        hs_module.terminating.set()
        
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=42
        )
        
        # Worker should return None when terminating is set
        result = worker(obj_fun)
        assert result is None
        
        # Clear for cleanup
        hs_module.terminating.clear()
    
    @pytest.mark.multiprocessing
    @pytest.mark.slow
    def test_worker_runs_when_not_terminating(self):
        """Test that worker function runs normally when terminating is not set."""
        import sys
        from pyharmonysearch.harmony_search import worker
        
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Ensure terminating event is clear
        hs_module.terminating.clear()
        
        obj_fun = QuadraticObjectiveFunction(
            num_parameters=2,
            hms=5,
            max_imp=20,
            random_seed=42
        )
        
        # Worker should complete successfully
        result = worker(obj_fun)
        assert result is not None
        
        # Result should be a tuple: (harmony, fitness, harmony_memory, harmony_history)
        assert len(result) == 4
        harmony, fitness, harmony_memory, harmony_history = result
        assert isinstance(harmony, list)
        assert isinstance(fitness, float)
        assert isinstance(harmony_memory, list)
        assert isinstance(harmony_history, list)
