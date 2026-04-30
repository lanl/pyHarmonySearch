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
Unit tests for HarmonySearchResults namedtuple.
"""

import pytest
from datetime import timedelta
from pyharmonysearch.harmony_search import HarmonySearchResults


class TestHarmonySearchResultsCreation:
    """Test creating HarmonySearchResults with all fields."""

    def test_create_with_all_fields(self):
        """Test creating HarmonySearchResults with all valid fields."""
        elapsed_time = timedelta(seconds=10)
        best_harmony = [1.0, 2.0]
        best_fitness = 42.5
        harmony_memories = [[([1.0, 2.0], 42.5), ([3.0, 4.0], 30.0)]]
        harmony_histories = [[{'gen': 0, 'harmonies': [([1.0, 2.0], 42.5)]}]]

        result = HarmonySearchResults(
            elapsed_time=elapsed_time,
            best_harmony=best_harmony,
            best_fitness=best_fitness,
            harmony_memories=harmony_memories,
            harmony_histories=harmony_histories
        )

        assert result.elapsed_time == elapsed_time
        assert result.best_harmony == best_harmony
        assert result.best_fitness == best_fitness
        assert result.harmony_memories == harmony_memories
        assert result.harmony_histories == harmony_histories

    def test_create_with_positional_arguments(self):
        """Test creating HarmonySearchResults with positional arguments."""
        elapsed_time = timedelta(seconds=5)
        best_harmony = [0.5, 1.5]
        best_fitness = 10.0
        harmony_memories = []
        harmony_histories = []

        result = HarmonySearchResults(
            elapsed_time,
            best_harmony,
            best_fitness,
            harmony_memories,
            harmony_histories
        )

        assert result.elapsed_time == elapsed_time
        assert result.best_harmony == best_harmony
        assert result.best_fitness == best_fitness
        assert result.harmony_memories == harmony_memories
        assert result.harmony_histories == harmony_histories

    def test_create_with_empty_collections(self):
        """Test creating HarmonySearchResults with empty lists."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=[],
            best_fitness=0.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result.best_harmony == []
        assert result.harmony_memories == []
        assert result.harmony_histories == []

    def test_create_with_none_harmony(self):
        """Test creating HarmonySearchResults with None best_harmony."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=None,
            best_fitness=0.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result.best_harmony is None


class TestHarmonySearchResultsFieldAccess:
    """Test accessing HarmonySearchResults fields by name and index."""

    def test_access_by_name(self):
        """Test accessing all fields by their names."""
        elapsed_time = timedelta(seconds=15)
        best_harmony = [1.0, 2.0, 3.0]
        best_fitness = 99.9
        harmony_memories = [[([1.0], 1.0)]]
        harmony_histories = [[{'gen': 0}]]

        result = HarmonySearchResults(
            elapsed_time=elapsed_time,
            best_harmony=best_harmony,
            best_fitness=best_fitness,
            harmony_memories=harmony_memories,
            harmony_histories=harmony_histories
        )

        assert result.elapsed_time == elapsed_time
        assert result.best_harmony == best_harmony
        assert result.best_fitness == best_fitness
        assert result.harmony_memories == harmony_memories
        assert result.harmony_histories == harmony_histories

    def test_access_by_index(self):
        """Test accessing fields by numeric index."""
        elapsed_time = timedelta(seconds=20)
        best_harmony = [5.0, 6.0]
        best_fitness = -10.5
        harmony_memories = []
        harmony_histories = []

        result = HarmonySearchResults(
            elapsed_time,
            best_harmony,
            best_fitness,
            harmony_memories,
            harmony_histories
        )

        # Access by index (0-4 for 5 fields)
        assert result[0] == elapsed_time
        assert result[1] == best_harmony
        assert result[2] == best_fitness
        assert result[3] == harmony_memories
        assert result[4] == harmony_histories

    def test_unpacking(self):
        """Test unpacking HarmonySearchResults like a tuple."""
        elapsed_time = timedelta(seconds=7)
        best_harmony = [2.0, 4.0]
        best_fitness = 33.3
        harmony_memories = [[([1.0, 2.0], 10.0)]]
        harmony_histories = [[{'gen': 1}]]

        result = HarmonySearchResults(
            elapsed_time,
            best_harmony,
            best_fitness,
            harmony_memories,
            harmony_histories
        )

        et, bh, bf, hm, hh = result

        assert et == elapsed_time
        assert bh == best_harmony
        assert bf == best_fitness
        assert hm == harmony_memories
        assert hh == harmony_histories

    def test_iteration(self):
        """Test iterating over HarmonySearchResults fields."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=3),
            best_harmony=[1.0],
            best_fitness=5.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        fields = list(result)
        assert len(fields) == 5
        assert fields[0] == timedelta(seconds=3)
        assert fields[1] == [1.0]
        assert fields[2] == 5.0
        assert fields[3] == []
        assert fields[4] == []


class TestHarmonySearchResultsImmutability:
    """Test that HarmonySearchResults is immutable."""

    def test_cannot_modify_field_by_name(self):
        """Test that fields cannot be modified by name."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=10),
            best_harmony=[1.0, 2.0],
            best_fitness=50.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        with pytest.raises(AttributeError):
            result.best_fitness = 100.0

    def test_cannot_modify_field_by_index(self):
        """Test that fields cannot be modified by index."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=10),
            best_harmony=[1.0, 2.0],
            best_fitness=50.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        with pytest.raises(TypeError):
            result[2] = 100.0

    def test_cannot_add_new_field(self):
        """Test that new fields cannot be added."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=10),
            best_harmony=[1.0, 2.0],
            best_fitness=50.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        with pytest.raises(AttributeError):
            result.new_field = "not allowed"

    def test_cannot_delete_field(self):
        """Test that fields cannot be deleted."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=10),
            best_harmony=[1.0, 2.0],
            best_fitness=50.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        with pytest.raises(AttributeError):
            del result.best_fitness


class TestHarmonySearchResultsMetadata:
    """Test field names, count, and namedtuple metadata."""

    def test_field_names(self):
        """Test that _fields attribute contains correct field names."""
        expected_fields = ('elapsed_time', 'best_harmony', 'best_fitness', 
                          'harmony_memories', 'harmony_histories')
        assert HarmonySearchResults._fields == expected_fields

    def test_field_count(self):
        """Test that the namedtuple has exactly 5 fields."""
        assert len(HarmonySearchResults._fields) == 5

    def test_has_correct_typename(self):
        """Test that the namedtuple has the correct type name."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=[1.0],
            best_fitness=1.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert type(result).__name__ == 'HarmonySearchResults'

    def test_is_instance_of_tuple(self):
        """Test that HarmonySearchResults instances are tuples."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=[1.0],
            best_fitness=1.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert isinstance(result, tuple)

    def test_asdict(self):
        """Test converting to dictionary using _asdict()."""
        elapsed_time = timedelta(seconds=12)
        best_harmony = [3.0, 4.0]
        best_fitness = 77.7
        harmony_memories = [[([1.0], 1.0)]]
        harmony_histories = [[{'gen': 2}]]

        result = HarmonySearchResults(
            elapsed_time=elapsed_time,
            best_harmony=best_harmony,
            best_fitness=best_fitness,
            harmony_memories=harmony_memories,
            harmony_histories=harmony_histories
        )

        result_dict = result._asdict()

        assert result_dict == {
            'elapsed_time': elapsed_time,
            'best_harmony': best_harmony,
            'best_fitness': best_fitness,
            'harmony_memories': harmony_memories,
            'harmony_histories': harmony_histories
        }


class TestHarmonySearchResultsWithVariousDataTypes:
    """Test HarmonySearchResults with different data types."""

    def test_with_integer_harmony(self):
        """Test with integer values in harmony."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=5),
            best_harmony=[1, 2, 3],
            best_fitness=10,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result.best_harmony == [1, 2, 3]
        assert result.best_fitness == 10

    def test_with_mixed_numeric_types(self):
        """Test with mixed int/float values."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=5),
            best_harmony=[1, 2.5, 3],
            best_fitness=10.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result.best_harmony == [1, 2.5, 3]
        assert isinstance(result.best_harmony[0], int)
        assert isinstance(result.best_harmony[1], float)

    def test_with_negative_fitness(self):
        """Test with negative fitness value."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=5),
            best_harmony=[1.0, 2.0],
            best_fitness=-42.5,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result.best_fitness == -42.5

    def test_with_zero_timedelta(self):
        """Test with zero elapsed time."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=0),
            best_harmony=[1.0],
            best_fitness=1.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result.elapsed_time == timedelta(seconds=0)
        assert result.elapsed_time.total_seconds() == 0

    def test_with_large_elapsed_time(self):
        """Test with large elapsed time."""
        elapsed_time = timedelta(hours=2, minutes=30, seconds=45, microseconds=123456)
        result = HarmonySearchResults(
            elapsed_time=elapsed_time,
            best_harmony=[1.0],
            best_fitness=1.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result.elapsed_time == elapsed_time
        assert result.elapsed_time.total_seconds() == 9045.123456

    def test_with_complex_harmony_memories(self):
        """Test with complex harmony memories structure."""
        harmony_memories = [
            [([1.0, 2.0], 10.0), ([3.0, 4.0], 20.0)],
            [([5.0, 6.0], 15.0), ([7.0, 8.0], 25.0)],
            [([9.0, 10.0], 30.0)]
        ]

        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=5),
            best_harmony=[9.0, 10.0],
            best_fitness=30.0,
            harmony_memories=harmony_memories,
            harmony_histories=[]
        )

        assert len(result.harmony_memories) == 3
        assert len(result.harmony_memories[0]) == 2
        assert result.harmony_memories[2][0] == ([9.0, 10.0], 30.0)

    def test_with_complex_harmony_histories(self):
        """Test with complex harmony histories structure."""
        harmony_histories = [
            [
                {'gen': 0, 'harmonies': [([1.0, 2.0], 10.0)]},
                {'gen': 1, 'harmonies': [([3.0, 4.0], 20.0)]}
            ],
            [
                {'gen': 0, 'harmonies': [([5.0, 6.0], 15.0)]}
            ]
        ]

        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=5),
            best_harmony=[5.0, 6.0],
            best_fitness=15.0,
            harmony_memories=[],
            harmony_histories=harmony_histories
        )

        assert len(result.harmony_histories) == 2
        assert result.harmony_histories[0][1]['gen'] == 1
        assert result.harmony_histories[1][0]['harmonies'][0] == ([5.0, 6.0], 15.0)

    def test_with_extreme_fitness_values(self):
        """Test with extreme fitness values (inf, -inf)."""
        result_inf = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=[1.0],
            best_fitness=float('inf'),
            harmony_memories=[],
            harmony_histories=[]
        )

        result_neg_inf = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=[2.0],
            best_fitness=float('-inf'),
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result_inf.best_fitness == float('inf')
        assert result_neg_inf.best_fitness == float('-inf')


class TestHarmonySearchResultsRepresentation:
    """Test repr and str representations."""

    def test_repr(self):
        """Test __repr__ output."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=10),
            best_harmony=[1.0, 2.0],
            best_fitness=42.5,
            harmony_memories=[],
            harmony_histories=[]
        )

        repr_str = repr(result)

        # Should contain type name and field values
        assert 'HarmonySearchResults' in repr_str
        assert 'elapsed_time' in repr_str
        assert 'best_harmony' in repr_str
        assert 'best_fitness' in repr_str
        assert 'harmony_memories' in repr_str
        assert 'harmony_histories' in repr_str
        assert '42.5' in repr_str

    def test_str(self):
        """Test __str__ output (same as __repr__ for namedtuples)."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=5),
            best_harmony=[3.0],
            best_fitness=100.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        str_repr = str(result)

        # For namedtuples, str() should be same as repr()
        assert str_repr == repr(result)

    def test_repr_with_none_values(self):
        """Test __repr__ with None values."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=0),
            best_harmony=None,
            best_fitness=0.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        repr_str = repr(result)

        assert 'None' in repr_str
        assert 'HarmonySearchResults' in repr_str


class TestHarmonySearchResultsEquality:
    """Test equality and comparison operations."""

    def test_equality_same_values(self):
        """Test that two results with same values are equal."""
        elapsed_time = timedelta(seconds=10)
        best_harmony = [1.0, 2.0]
        best_fitness = 42.5
        harmony_memories = [[([1.0, 2.0], 42.5)]]
        harmony_histories = [[{'gen': 0}]]

        result1 = HarmonySearchResults(
            elapsed_time=elapsed_time,
            best_harmony=best_harmony,
            best_fitness=best_fitness,
            harmony_memories=harmony_memories,
            harmony_histories=harmony_histories
        )

        result2 = HarmonySearchResults(
            elapsed_time=elapsed_time,
            best_harmony=best_harmony,
            best_fitness=best_fitness,
            harmony_memories=harmony_memories,
            harmony_histories=harmony_histories
        )

        assert result1 == result2

    def test_inequality_different_values(self):
        """Test that results with different values are not equal."""
        result1 = HarmonySearchResults(
            elapsed_time=timedelta(seconds=10),
            best_harmony=[1.0, 2.0],
            best_fitness=42.5,
            harmony_memories=[],
            harmony_histories=[]
        )

        result2 = HarmonySearchResults(
            elapsed_time=timedelta(seconds=10),
            best_harmony=[1.0, 2.0],
            best_fitness=100.0,  # Different fitness
            harmony_memories=[],
            harmony_histories=[]
        )

        assert result1 != result2

    def test_hash_consistency(self):
        """Test that hashable namedtuples with same values have same hash."""
        elapsed_time = timedelta(seconds=10)
        best_harmony = (1.0, 2.0)  # Use tuple for hashability
        best_fitness = 42.5

        # Note: Lists are not hashable, so we can't hash HarmonySearchResults
        # with list fields, but we can test that identical objects would have
        # same hash if their fields were hashable
        result1 = HarmonySearchResults(
            elapsed_time=elapsed_time,
            best_harmony=best_harmony,
            best_fitness=best_fitness,
            harmony_memories=(),  # Use tuples instead of lists
            harmony_histories=()
        )

        result2 = HarmonySearchResults(
            elapsed_time=elapsed_time,
            best_harmony=best_harmony,
            best_fitness=best_fitness,
            harmony_memories=(),
            harmony_histories=()
        )

        # With hashable fields, hash should be consistent
        assert hash(result1) == hash(result2)


class TestHarmonySearchResultsEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_required_field(self):
        """Test that creating without required field raises TypeError."""
        with pytest.raises(TypeError):
            HarmonySearchResults(
                elapsed_time=timedelta(seconds=1),
                best_harmony=[1.0],
                best_fitness=1.0,
                harmony_memories=[]
                # Missing harmony_histories
            )

    def test_extra_field(self):
        """Test that providing extra field raises TypeError."""
        with pytest.raises(TypeError):
            HarmonySearchResults(
                elapsed_time=timedelta(seconds=1),
                best_harmony=[1.0],
                best_fitness=1.0,
                harmony_memories=[],
                harmony_histories=[],
                extra_field="not allowed"
            )

    def test_index_out_of_range(self):
        """Test that accessing invalid index raises IndexError."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=[1.0],
            best_fitness=1.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        with pytest.raises(IndexError):
            _ = result[5]  # Only 5 fields (indices 0-4)

    def test_invalid_field_name(self):
        """Test that accessing invalid field name raises AttributeError."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=[1.0],
            best_fitness=1.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        with pytest.raises(AttributeError):
            _ = result.invalid_field

    def test_length(self):
        """Test that len() returns correct number of fields."""
        result = HarmonySearchResults(
            elapsed_time=timedelta(seconds=1),
            best_harmony=[1.0],
            best_fitness=1.0,
            harmony_memories=[],
            harmony_histories=[]
        )

        assert len(result) == 5
