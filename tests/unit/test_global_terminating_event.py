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
from multiprocessing import Event


class TestGlobalTerminatingEvent:
    """Test the global terminating Event used for multiprocessing control."""

    def test_terminating_event_exists(self):
        """Test that the terminating Event is defined in the module."""
        # Need to use sys.modules to get the actual module, not the function
        from pyharmonysearch import harmony_search  # Import to ensure module is loaded
        hs_module = sys.modules['pyharmonysearch.harmony_search']
        assert hasattr(hs_module, 'terminating')
        assert isinstance(hs_module.terminating, type(Event()))

    def test_terminating_event_initial_state(self):
        """Test that terminating Event starts in unset state."""
        from pyharmonysearch.harmony_search import terminating
        
        # Clear it first to ensure clean state
        terminating.clear()
        
        # Should be unset (False) initially
        assert not terminating.is_set()

    def test_setting_terminating_event(self):
        """Test setting the terminating Event."""
        from pyharmonysearch.harmony_search import terminating
        
        # Clear it first
        terminating.clear()
        assert not terminating.is_set()
        
        # Set the event
        terminating.set()
        assert terminating.is_set()
        
        # Clean up - clear it for other tests
        terminating.clear()

    def test_clearing_terminating_event(self):
        """Test clearing the terminating Event."""
        from pyharmonysearch.harmony_search import terminating
        
        # Set it first
        terminating.set()
        assert terminating.is_set()
        
        # Clear the event
        terminating.clear()
        assert not terminating.is_set()

    def test_event_isolation_between_tests(self):
        """Test that Event state can be managed independently in tests."""
        from pyharmonysearch.harmony_search import terminating
        
        # Start with clear state
        terminating.clear()
        initial_state = terminating.is_set()
        
        # Change state
        terminating.set()
        changed_state = terminating.is_set()
        
        # Restore state
        terminating.clear()
        final_state = terminating.is_set()
        
        assert not initial_state
        assert changed_state
        assert not final_state

    def test_event_is_shared_across_imports(self):
        """Test that the terminating Event is the same object across imports."""
        from pyharmonysearch.harmony_search import terminating as terminating1
        from pyharmonysearch.harmony_search import terminating as terminating2
        
        # Should be the exact same object
        assert terminating1 is terminating2
        
        # Changes to one should affect the other
        terminating1.clear()
        assert not terminating2.is_set()
        
        terminating1.set()
        assert terminating2.is_set()
        
        # Clean up
        terminating1.clear()

    def test_event_type_is_multiprocessing_event(self):
        """Test that terminating is a multiprocessing.Event."""
        from pyharmonysearch.harmony_search import terminating
        
        # Check that it has Event methods
        assert hasattr(terminating, 'set')
        assert hasattr(terminating, 'clear')
        assert hasattr(terminating, 'is_set')
        assert hasattr(terminating, 'wait')
        
        assert callable(terminating.set)
        assert callable(terminating.clear)
        assert callable(terminating.is_set)
        assert callable(terminating.wait)

    def teardown_method(self):
        """Ensure terminating Event is cleared after each test."""
        from pyharmonysearch.harmony_search import terminating
        terminating.clear()
