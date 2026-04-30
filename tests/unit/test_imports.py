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


class TestImports:
    """Test that all expected symbols can be imported from pyharmonysearch package."""

    def test_harmony_search_importable(self):
        """Test that harmony_search function is importable from package root."""
        from pyharmonysearch import harmony_search
        assert callable(harmony_search)

    def test_harmony_search_serial_importable(self):
        """Test that harmony_search_serial function is importable from package root."""
        from pyharmonysearch import harmony_search_serial
        assert callable(harmony_search_serial)

    def test_harmony_search_class_importable(self):
        """Test that HarmonySearch class is importable from package root."""
        from pyharmonysearch import HarmonySearch
        assert isinstance(HarmonySearch, type)

    def test_objective_function_interface_importable(self):
        """Test that ObjectiveFunctionInterface is importable from package root."""
        from pyharmonysearch import ObjectiveFunctionInterface
        assert isinstance(ObjectiveFunctionInterface, type)

    def test_harmony_search_results_importable(self):
        """Test that HarmonySearchResults is importable from package root."""
        from pyharmonysearch import HarmonySearchResults
        # HarmonySearchResults is a namedtuple, so it's a class
        assert isinstance(HarmonySearchResults, type)

    def test_all_exports(self):
        """Test that __all__ contains all expected exports."""
        import pyharmonysearch
        expected_exports = [
            'harmony_search',
            'harmony_search_serial',
            'HarmonySearch',
            'HarmonySearchResults',
            'ObjectiveFunctionInterface'
        ]
        assert hasattr(pyharmonysearch, '__all__')
        assert set(pyharmonysearch.__all__) == set(expected_exports)

    def test_all_expected_symbols_accessible(self):
        """Test that all expected symbols are accessible from package namespace."""
        import pyharmonysearch
        
        # Test that all symbols in __all__ are actually accessible
        for symbol_name in pyharmonysearch.__all__:
            assert hasattr(pyharmonysearch, symbol_name), f"Symbol '{symbol_name}' not found in package"
            symbol = getattr(pyharmonysearch, symbol_name)
            assert symbol is not None, f"Symbol '{symbol_name}' is None"

    def test_direct_symbol_access(self):
        """Test that all symbols can be accessed directly from package."""
        import pyharmonysearch
        
        # Test functions
        assert callable(pyharmonysearch.harmony_search)
        assert callable(pyharmonysearch.harmony_search_serial)
        
        # Test classes
        assert isinstance(pyharmonysearch.HarmonySearch, type)
        assert isinstance(pyharmonysearch.ObjectiveFunctionInterface, type)
        assert isinstance(pyharmonysearch.HarmonySearchResults, type)

    def test_from_import_all(self):
        """Test that 'from pyharmonysearch import *' works correctly."""
        import pyharmonysearch
        
        # Test that __all__ is defined
        assert hasattr(pyharmonysearch, '__all__')
        
        # Test that all symbols in __all__ can be imported
        expected_symbols = {
            'harmony_search',
            'harmony_search_serial',
            'HarmonySearch',
            'HarmonySearchResults',
            'ObjectiveFunctionInterface'
        }
        
        # Verify each symbol exists and is accessible
        for symbol in expected_symbols:
            assert symbol in pyharmonysearch.__all__
            assert hasattr(pyharmonysearch, symbol)

    def test_submodule_imports(self):
        """Test that submodules can be imported directly."""
        # Test importing from submodules
        from pyharmonysearch.harmony_search import harmony_search, harmony_search_serial, HarmonySearch, HarmonySearchResults
        from pyharmonysearch.objective_function_interface import ObjectiveFunctionInterface
        
        assert callable(harmony_search)
        assert callable(harmony_search_serial)
        assert isinstance(HarmonySearch, type)
        assert isinstance(HarmonySearchResults, type)
        assert isinstance(ObjectiveFunctionInterface, type)

    def test_no_unexpected_exports(self):
        """Test that only expected symbols are exported via __all__."""
        import pyharmonysearch
        
        # Get all public symbols (not starting with _)
        public_symbols = [name for name in dir(pyharmonysearch) if not name.startswith('_')]
        
        # All public symbols should be in __all__
        for symbol in pyharmonysearch.__all__:
            assert symbol in public_symbols

    def test_harmony_search_results_namedtuple_fields(self):
        """Test that HarmonySearchResults has the expected namedtuple fields."""
        from pyharmonysearch import HarmonySearchResults
        
        expected_fields = [
            'elapsed_time',
            'best_harmony',
            'best_fitness',
            'harmony_memories',
            'harmony_histories'
        ]
        
        assert hasattr(HarmonySearchResults, '_fields')
        assert set(HarmonySearchResults._fields) == set(expected_fields)
