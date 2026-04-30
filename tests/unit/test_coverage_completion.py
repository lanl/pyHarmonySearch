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
Tests specifically for achieving 100% coverage of missing lines.
"""

import pytest
from pyharmonysearch import HarmonySearch, harmony_search
from pyharmonysearch.harmony_search import worker
from tests.fixtures.mock_objective_functions import QuadraticObjectiveFunction


class TestInitializeErrorHandling:
    """Test error handling in HarmonySearch._initialize() for 100% coverage."""
    
    def test_initialize_raises_valueerror_on_wrong_hms(self):
        """Test that _initialize raises ValueError when initial_harmonies length != HMS.
        
        This test covers line 209 in harmony_search.py.
        """
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10)
        hs = HarmonySearch(obj_fun)
        
        # Initialize internal state
        hs._harmony_memory = []
        hs._harmony_history = []
        
        # Provide initial harmonies with wrong count (3 instead of 5)
        wrong_initial_harmonies = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]  # Only 3 harmonies but HMS=5
        ]
        
        with pytest.raises(ValueError, match="Number of initial harmonies does not equal to the harmony memory size"):
            hs._initialize(wrong_initial_harmonies)
    
    def test_initialize_raises_valueerror_on_wrong_parameters(self):
        """Test that _initialize raises ValueError when harmony has wrong number of parameters.
        
        This test covers line 215 in harmony_search.py.
        """
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=3, max_imp=10)
        hs = HarmonySearch(obj_fun)
        
        # Initialize internal state
        hs._harmony_memory = []
        hs._harmony_history = []
        
        # Provide initial harmonies with wrong parameter count
        wrong_initial_harmonies = [
            [1.0, 2.0],      # Correct: 2 parameters
            [3.0, 4.0, 5.0], # Wrong: 3 parameters instead of 2
            [6.0, 7.0]       # Correct: 2 parameters
        ]
        
        with pytest.raises(ValueError, match="Number of parameters in initial harmonies does not match that defined"):
            hs._initialize(wrong_initial_harmonies)


class TestKeyboardInterruptHandling:
    """Test KeyboardInterrupt handling for 100% coverage."""
    
    def test_harmony_search_keyboard_interrupt_handler(self):
        """Test that harmony_search catches KeyboardInterrupt and terminates pool.
        
        This test covers lines 83-85 in harmony_search.py.
        """
        from unittest.mock import Mock, patch
        
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10)
        
        with patch('pyharmonysearch.harmony_search.Pool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool
            
            # Make pool.join() raise KeyboardInterrupt
            mock_pool.join.side_effect = KeyboardInterrupt()
            
            # harmony_search should catch the exception, call terminate, and re-raise
            with pytest.raises(KeyboardInterrupt):
                harmony_search(obj_fun, num_processes=2, num_iterations=1)
            
            # Verify that terminate was called
            mock_pool.terminate.assert_called_once()
    
    def test_worker_keyboard_interrupt_sets_terminating_event(self):
        """Test that worker sets terminating Event on KeyboardInterrupt.
        
        This test covers lines 124-126 in harmony_search.py (worker function).
        """
        from unittest.mock import patch
        from pyharmonysearch.harmony_search import terminating
        
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10)
        
        # Clear terminating event before test
        terminating.clear()
        
        try:
            # Mock HarmonySearch.run() to raise KeyboardInterrupt
            with patch('pyharmonysearch.harmony_search.HarmonySearch.run', side_effect=KeyboardInterrupt()):
                with pytest.raises(KeyboardInterrupt):
                    worker(obj_fun, initial_harmonies=None)
                
                # Verify that terminating event was set
                assert terminating.is_set()
        finally:
            # Clean up: clear the event for other tests
            terminating.clear()
    
    def test_worker_returns_none_when_terminating_is_set(self):
        """Test that worker returns None when terminating Event is already set.
        
        This test covers the 121->exit branch (implicit return when terminating is set).
        """
        from pyharmonysearch.harmony_search import terminating
        
        obj_fun = QuadraticObjectiveFunction(num_parameters=2, hms=5, max_imp=10)
        
        # Set terminating event before calling worker
        terminating.set()
        
        try:
            result = worker(obj_fun, initial_harmonies=None)
            
            # Worker should return None when terminating is set
            assert result is None
        finally:
            # Clean up: clear the event for other tests
            terminating.clear()


class TestPitchAdjustmentNonVariableBranch:
    """Test non-variable parameter branch in _pitch_adjustment for 100% coverage."""
    
    def test_pitch_adjustment_returns_early_for_non_variable_parameter(self):
        """Test that _pitch_adjustment returns early (no-op) for non-variable parameters.
        
        This test covers the 258->exit branch (implicit return for non-variable parameters).
        """
        from tests.fixtures.mock_objective_functions import FixedParameterObjectiveFunction
        
        # FixedParameterObjectiveFunction has non-variable parameters
        obj_fun = FixedParameterObjectiveFunction(num_parameters=4, hms=5, max_imp=10)
        hs = HarmonySearch(obj_fun)
        
        # Create a harmony to adjust
        harmony = [1.0, 2.0, 3.0, 4.0]
        
        # Get the indices of non-variable parameters (even indices are variable, odd are fixed)
        # So parameter 1 is non-variable (fixed)
        non_variable_index = 1
        original_value = harmony[non_variable_index]
        
        # Call pitch adjustment on a non-variable parameter
        hs._pitch_adjustment(harmony, non_variable_index)
        
        # Value should be unchanged (implicit early return, no adjustment)
        assert harmony[non_variable_index] == original_value
