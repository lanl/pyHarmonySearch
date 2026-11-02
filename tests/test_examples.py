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
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import importlib.util
from pathlib import Path

import pytest

from pyharmonysearch.harmony_search import harmony_search_serial

EXAMPLES = sorted((Path(__file__).parent.parent / 'examples').glob('*.py'))


def load_example(path):
    # The file names aren't valid module names. Loading under a different name means __name__ != '__main__', so the
    # example's own search doesn't run.
    spec = importlib.util.spec_from_file_location(f'example_{path.stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('path', EXAMPLES, ids=[path.name for path in EXAMPLES])
def test_example_runs(path):
    objective = load_example(path).ObjectiveFunction()
    objective._max_imp = objective.get_hms() * 2  # the real budgets take seconds to minutes

    results = harmony_search_serial(objective, 1)

    assert results.best_fitness == objective.get_fitness(results.best_harmony)
    for harmony, fitness in [(results.best_harmony, results.best_fitness), *results.harmony_memories[0]]:
        assert len(harmony) == objective.get_num_parameters()
        assert fitness == objective.get_fitness(harmony)
        for i, value in enumerate(harmony):
            if not objective.is_variable(i):
                assert value == objective.get_value(i)
            elif objective.is_discrete(i):
                assert objective.get_value(i, objective.get_index(i, value)) == value
            else:
                assert objective.get_lower_bound(i) <= value <= objective.get_upper_bound(i)


@pytest.mark.parametrize('path', EXAMPLES, ids=[path.name for path in EXAMPLES])
def test_example_discrete_values_round_trip(path):
    # A short search rarely reaches index 0, so check every index directly (this catches `if j:` in get_value()).
    objective = load_example(path).ObjectiveFunction()
    for i in range(objective.get_num_parameters()):
        if objective.is_variable(i) and objective.is_discrete(i):
            for j in range(objective.get_num_discrete_values(i)):
                assert objective.get_index(i, objective.get_value(i, j)) == j
