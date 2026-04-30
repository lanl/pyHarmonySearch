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
Test to validate that all fixtures are working correctly.
"""

from tests.fixtures.test_helpers import (
    assert_harmony_in_bounds,
    assert_valid_harmony_memory,
    create_mock_objective_function,
    get_best_from_memory,
)


def test_simple_objective_function_fixture(simple_objective_function):
    """Test that simple_objective_function fixture works."""
    obj = simple_objective_function
    
    assert obj.get_num_parameters() == 2
    assert obj.get_hms() == 10
    assert obj.maximize() is True
    assert obj.get_max_imp() == 100
    
    # Test fitness at known optimum
    fitness = obj.get_fitness([0, -1])
    assert fitness == 4.0


def test_discrete_objective_function_fixture(discrete_objective_function):
    """Test that discrete_objective_function fixture works."""
    obj = discrete_objective_function
    
    assert obj.get_num_parameters() == 2
    assert obj.is_discrete(0) is True
    assert obj.is_discrete(1) is True
    assert obj.get_num_discrete_values(0) == 21  # -10 to 10 inclusive


def test_mixed_objective_function_fixture(mixed_objective_function):
    """Test that mixed_objective_function fixture works."""
    obj = mixed_objective_function
    
    assert obj.get_num_parameters() == 2
    assert obj.is_discrete(0) is True
    assert obj.is_discrete(1) is False


def test_harmony_search_instance_fixture(harmony_search_instance):
    """Test that harmony_search_instance fixture works."""
    from pyharmonysearch import HarmonySearch
    
    assert isinstance(harmony_search_instance, HarmonySearch)


def test_sample_harmony_fixture(sample_harmony, simple_objective_function):
    """Test that sample_harmony fixture works."""
    assert isinstance(sample_harmony, list)
    assert len(sample_harmony) == 2
    
    # Verify it's within bounds
    assert_harmony_in_bounds(sample_harmony, simple_objective_function)


def test_sample_harmony_memory_fixture(sample_harmony_memory, simple_objective_function):
    """Test that sample_harmony_memory fixture works."""
    assert isinstance(sample_harmony_memory, list)
    assert len(sample_harmony_memory) == 10  # Should match HMS
    
    # Verify structure
    assert_valid_harmony_memory(sample_harmony_memory, simple_objective_function)
    
    # Check that best harmony is correct
    best_h, best_f = get_best_from_memory(sample_harmony_memory, maximize=True)
    assert best_h == [0.0, -1.0]
    assert best_f == 4.0


def test_sample_harmony_history_fixture(sample_harmony_history):
    """Test that sample_harmony_history fixture works."""
    assert isinstance(sample_harmony_history, list)
    assert len(sample_harmony_history) == 3  # 3 generations
    
    # Verify structure
    for i, gen_data in enumerate(sample_harmony_history):
        assert 'gen' in gen_data
        assert 'harmonies' in gen_data
        assert gen_data['gen'] == i


def test_seed_random_fixture(seed_random):
    """Test that seed_random fixture works."""
    import random
    
    # Seed and generate a value
    seed_random(12345)
    val1 = random.random()
    
    # Seed with same value and verify reproducibility
    seed_random(12345)
    val2 = random.random()
    
    assert val1 == val2


def test_create_mock_objective_function_helper():
    """Test the create_mock_objective_function helper."""
    mock_obj = create_mock_objective_function(
        num_parameters=3,
        maximize=False,
        all_continuous=True,
        use_seed=True
    )
    
    assert mock_obj.get_num_parameters() == 3
    assert mock_obj.maximize() is False
    assert all(not mock_obj.is_discrete(i) for i in range(3))


def test_assert_harmony_in_bounds_helper(simple_objective_function):
    """Test the assert_harmony_in_bounds helper."""
    # Valid harmony
    valid_harmony = [0.0, -1.0]
    assert_harmony_in_bounds(valid_harmony, simple_objective_function)
    
    # Out of bounds harmony should raise assertion
    try:
        invalid_harmony = [100.0, -1.0]  # Out of bounds
        assert_harmony_in_bounds(invalid_harmony, simple_objective_function)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass  # Expected


def test_get_best_from_memory_helper():
    """Test the get_best_from_memory helper."""
    memory = [
        ([1.0, 2.0], 10.0),
        ([3.0, 4.0], 20.0),
        ([5.0, 6.0], 5.0),
    ]
    
    # Test maximization
    best_h, best_f = get_best_from_memory(memory, maximize=True)
    assert best_f == 20.0
    assert best_h == [3.0, 4.0]
    
    # Test minimization
    best_h, best_f = get_best_from_memory(memory, maximize=False)
    assert best_f == 5.0
    assert best_h == [5.0, 6.0]
