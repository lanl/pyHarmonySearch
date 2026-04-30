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
import sys
from unittest.mock import Mock, MagicMock, patch, call
from multiprocessing import Event

from pyharmonysearch.harmony_search import worker, HarmonySearch
from tests.fixtures.mock_objective_functions import QuadraticObjectiveFunction


class TestWorkerFunctionNormalExecution:
    """Test suite for normal execution scenarios of worker() function."""
    
    def test_worker_returns_tuple_of_four_elements(self):
        """Test that worker() returns a tuple with 4 elements: (harmony, fitness, harmony_memory, harmony_history)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        result = worker(obj_fun, initial_harmonies=None)
        
        # Should be a tuple with 4 elements
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        harmony, fitness, memory, history = result
        assert isinstance(harmony, list)
        assert isinstance(fitness, (int, float))
        assert isinstance(memory, list)
        assert isinstance(history, list)
    
    def test_worker_normal_execution_terminating_not_set(self):
        """Test worker() executes normally when terminating Event is not set."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        # Ensure terminating is not set (autouse fixture handles this)
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        assert not hs_module.terminating.is_set()
        
        result = worker(obj_fun, initial_harmonies=None)
        
        # Should execute successfully and return results
        assert result is not None
        harmony, fitness, memory, history = result
        assert len(harmony) == 2
        assert len(memory) == 3  # hms
    
    def test_worker_creates_harmony_search_instance(self):
        """Test that worker() creates a HarmonySearch instance with the objective function."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.return_value = ([1.0, 2.0], 5.0, [], [])
            MockHS.return_value = mock_hs_instance
            
            worker(obj_fun, initial_harmonies=None)
            
            # Verify HarmonySearch was instantiated with objective function
            MockHS.assert_called_once_with(obj_fun)
    
    def test_worker_calls_harmony_search_run_method(self):
        """Test that worker() calls the run() method on HarmonySearch instance."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.return_value = ([1.0, 2.0], 5.0, [], [])
            MockHS.return_value = mock_hs_instance
            
            worker(obj_fun, initial_harmonies=None)
            
            # Verify run() was called once
            mock_hs_instance.run.assert_called_once()
    
    def test_worker_returns_harmony_search_run_result(self):
        """Test that worker() returns the result from HarmonySearch.run()."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        expected_result = ([1.5, 2.5], 10.0, [([1.5, 2.5], 10.0)], [{'gen': 0, 'harmonies': []}])
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.return_value = expected_result
            MockHS.return_value = mock_hs_instance
            
            result = worker(obj_fun, initial_harmonies=None)
            
            # Verify worker returns the same result as HarmonySearch.run()
            assert result == expected_result


class TestWorkerFunctionWithInitialHarmonies:
    """Test suite for worker() function with initial_harmonies parameter."""
    
    def test_worker_with_initial_harmonies_provided(self):
        """Test that worker() passes initial_harmonies to HarmonySearch.run()."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        initial_harmonies = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.return_value = ([1.0, 2.0], 5.0, [], [])
            MockHS.return_value = mock_hs_instance
            
            worker(obj_fun, initial_harmonies=initial_harmonies)
            
            # Verify run() was called with initial_harmonies
            mock_hs_instance.run.assert_called_once_with(initial_harmonies=initial_harmonies)
    
    def test_worker_without_initial_harmonies(self):
        """Test that worker() passes None for initial_harmonies when not provided."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.return_value = ([1.0, 2.0], 5.0, [], [])
            MockHS.return_value = mock_hs_instance
            
            worker(obj_fun, initial_harmonies=None)
            
            # Verify run() was called with initial_harmonies=None
            mock_hs_instance.run.assert_called_once_with(initial_harmonies=None)
    
    def test_worker_default_initial_harmonies_parameter(self):
        """Test that worker() has default initial_harmonies=None parameter."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.return_value = ([1.0, 2.0], 5.0, [], [])
            MockHS.return_value = mock_hs_instance
            
            # Call without initial_harmonies parameter
            worker(obj_fun)
            
            # Verify run() was called with initial_harmonies=None
            mock_hs_instance.run.assert_called_once_with(initial_harmonies=None)


class TestWorkerFunctionTerminatingEvent:
    """Test suite for worker() function behavior with terminating Event."""
    
    def test_worker_checks_terminating_event_before_execution(self):
        """Test that worker() checks if terminating Event is set before executing."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        with patch.object(hs_module.terminating, 'is_set', return_value=False) as mock_is_set:
            with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
                mock_hs_instance = MagicMock()
                mock_hs_instance.run.return_value = ([1.0, 2.0], 5.0, [], [])
                MockHS.return_value = mock_hs_instance
                
                worker(obj_fun)
                
                # Verify is_set() was called
                mock_is_set.assert_called_once()
    
    def test_worker_executes_when_terminating_not_set(self):
        """Test that worker() executes HarmonySearch when terminating is not set."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Ensure terminating is not set
        hs_module.terminating.clear()
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.return_value = ([1.0, 2.0], 5.0, [], [])
            MockHS.return_value = mock_hs_instance
            
            result = worker(obj_fun)
            
            # Verify HarmonySearch was created and run
            MockHS.assert_called_once()
            mock_hs_instance.run.assert_called_once()
            assert result is not None
    
    def test_worker_does_not_execute_when_terminating_already_set(self):
        """Test that worker() does not execute HarmonySearch when terminating is already set."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Set terminating before calling worker
        hs_module.terminating.set()
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.return_value = ([1.0, 2.0], 5.0, [], [])
            MockHS.return_value = mock_hs_instance
            
            result = worker(obj_fun)
            
            # Verify HarmonySearch was NOT created
            MockHS.assert_not_called()
            mock_hs_instance.run.assert_not_called()
            # Result should be None (implicit return when condition is False)
            assert result is None
    
    def test_worker_returns_none_when_terminating_set(self):
        """Test that worker() returns None when terminating Event is already set."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Set terminating Event
        hs_module.terminating.set()
        
        result = worker(obj_fun)
        
        # Should return None
        assert result is None


class TestWorkerFunctionKeyboardInterrupt:
    """Test suite for worker() function handling of KeyboardInterrupt."""
    
    def test_worker_catches_keyboard_interrupt(self):
        """Test that worker() catches KeyboardInterrupt raised during execution."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            # Simulate KeyboardInterrupt during run()
            mock_hs_instance.run.side_effect = KeyboardInterrupt()
            MockHS.return_value = mock_hs_instance
            
            # Should raise KeyboardInterrupt after handling
            with pytest.raises(KeyboardInterrupt):
                worker(obj_fun)
    
    def test_worker_sets_terminating_event_on_keyboard_interrupt(self):
        """Test that worker() sets terminating Event when KeyboardInterrupt is raised."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Ensure terminating is not set initially
        hs_module.terminating.clear()
        assert not hs_module.terminating.is_set()
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.side_effect = KeyboardInterrupt()
            MockHS.return_value = mock_hs_instance
            
            try:
                worker(obj_fun)
            except KeyboardInterrupt:
                pass
            
            # Verify terminating Event was set
            assert hs_module.terminating.is_set()
    
    def test_worker_calls_terminating_set_on_keyboard_interrupt(self):
        """Test that worker() explicitly calls terminating.set() when KeyboardInterrupt occurs."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        with patch.object(hs_module.terminating, 'set') as mock_set:
            with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
                mock_hs_instance = MagicMock()
                mock_hs_instance.run.side_effect = KeyboardInterrupt()
                MockHS.return_value = mock_hs_instance
                
                try:
                    worker(obj_fun)
                except KeyboardInterrupt:
                    pass
                
                # Verify set() was called on terminating Event
                mock_set.assert_called_once()
    
    def test_worker_propagates_keyboard_interrupt_after_setting_event(self):
        """Test that worker() re-raises KeyboardInterrupt after setting terminating Event."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.side_effect = KeyboardInterrupt()
            MockHS.return_value = mock_hs_instance
            
            # Should raise KeyboardInterrupt
            with pytest.raises(KeyboardInterrupt):
                worker(obj_fun)
    
    def test_worker_keyboard_interrupt_prevents_other_workers(self):
        """Test that setting terminating Event on KeyboardInterrupt prevents other workers from starting."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        
        # Ensure terminating is not set initially
        hs_module.terminating.clear()
        
        # First worker raises KeyboardInterrupt
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS:
            mock_hs_instance = MagicMock()
            mock_hs_instance.run.side_effect = KeyboardInterrupt()
            MockHS.return_value = mock_hs_instance
            
            try:
                worker(obj_fun)
            except KeyboardInterrupt:
                pass
        
        # Now terminating should be set
        assert hs_module.terminating.is_set()
        
        # Second worker should not execute
        with patch('pyharmonysearch.harmony_search.HarmonySearch') as MockHS2:
            mock_hs_instance2 = MagicMock()
            MockHS2.return_value = mock_hs_instance2
            
            result = worker(obj_fun)
            
            # Should return None without executing
            assert result is None
            MockHS2.assert_not_called()


class TestWorkerFunctionIntegration:
    """Integration tests for worker() function with real HarmonySearch execution."""
    
    def test_worker_complete_execution_with_real_harmony_search(self):
        """Test worker() with actual HarmonySearch execution (no mocks)."""
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        harmony, fitness, memory, history = worker(obj_fun)
        
        # Verify structure of results
        assert len(harmony) == 2
        assert isinstance(fitness, (int, float))
        assert len(memory) == 3  # hms
        assert len(history) > 0  # At least initial generation
        
        # Verify harmony memory structure
        for h, f in memory:
            assert len(h) == 2
            assert isinstance(f, (int, float))
        
        # Verify harmony history structure
        for entry in history:
            assert 'gen' in entry
            assert 'harmonies' in entry
    
    def test_worker_with_different_initial_harmonies(self):
        """Test worker() produces different results with and without initial harmonies."""
        obj_fun1 = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        obj_fun2 = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=5, random_seed=42)
        
        # Run without initial harmonies
        harmony1, fitness1, memory1, history1 = worker(obj_fun1, initial_harmonies=None)
        
        # Run with initial harmonies
        initial_harmonies = [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]
        harmony2, fitness2, memory2, history2 = worker(obj_fun2, initial_harmonies=initial_harmonies)
        
        # Results should be different (with high probability)
        # Note: Due to random seed, results might be same, but initial harmonies in history should differ
        assert history1[0]['harmonies'] != history2[0]['harmonies']
    
    def test_worker_respects_objective_function_parameters(self):
        """Test that worker() respects the objective function's configuration."""
        hms = 5
        max_imp = 10
        obj_fun = QuadraticObjectiveFunction(num_parameters=3, hms=hms, max_imp=max_imp, random_seed=42)
        
        harmony, fitness, memory, history = worker(obj_fun)
        
        # Verify parameters
        assert len(harmony) == 3  # num_parameters
        assert len(memory) == hms  # harmony memory size
        # Number of generations should be based on max_imp / hms
        expected_generations = 1 + (max_imp // hms)  # Initial + complete generations
        assert len(history) == expected_generations
