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

import importlib
from types import SimpleNamespace
from unittest import mock

import pytest
from helpers import SumObjective

from pyharmonysearch import HarmonySearch

# pyharmonysearch.harmony_search is the re-exported function, not the submodule, so import the module explicitly.
hs_module = importlib.import_module('pyharmonysearch.harmony_search')


@pytest.fixture
def fake_random(monkeypatch):
    """
        Replace the module's reference to random with scripted values. Only the code under test is affected, not the
        objective function's own random calls. Returns the list of (a, b) bounds passed to randint().
    """
    def install(randoms=(), randints=()):
        randoms, randints = iter(randoms), iter(randints)
        randint_bounds = []

        def randint(a, b):
            randint_bounds.append((a, b))
            return next(randints)

        monkeypatch.setattr(hs_module, 'random', SimpleNamespace(random=lambda: next(randoms), randint=randint))
        return randint_bounds
    return install


def make_search(objective, memory=()):
    hs = HarmonySearch(objective)
    hs._harmony_memory = list(memory)
    return hs


# run()

# random.random() is in [0, 1), so rates of 0 and 1 make the hmcr/par branches deterministic without patching.
@pytest.mark.parametrize(('hmcr', 'par', 'expected_random', 'expected_memory', 'expected_pitch'), [
    (0, 0, 3 * 2 + 6 * 2, 0, 0),
    (1, 0, 3 * 2, 6 * 2, 0),
    (1, 1, 3 * 2, 6 * 2, 6 * 2),
])
def test_run_improvisation_branches(hmcr, par, expected_random, expected_memory, expected_pitch):
    hs = HarmonySearch(SumObjective(('continuous', 'discrete'), hms=3, max_imp=6, hmcr=hmcr, par=par))
    spies = {name: mock.Mock(wraps=getattr(hs, name))
             for name in ('_random_selection', '_memory_consideration', '_pitch_adjustment')}
    vars(hs).update(spies)  # instance attributes shadow the methods, so nothing needs undoing

    hs.run()

    assert [spy.call_count for spy in spies.values()] == [expected_random, expected_memory, expected_pitch]


@pytest.mark.parametrize(('maximize', 'expected'), [(True, 5), (False, 1)])
def test_run_returns_best_harmony(maximize, expected):
    initial_harmonies = [[1], [5], [3]]
    objective = SumObjective(maximize=maximize, hms=3, max_imp=0)

    best_harmony, best_fitness, memory, history = HarmonySearch(objective).run(initial_harmonies=initial_harmonies)

    assert (best_harmony, best_fitness) == ([expected], expected)
    assert memory == [([1], 1), ([5], 5), ([3], 3)]
    assert history == [{'gen': 0, 'harmonies': memory}]


# Fitness ranges from 0 + 0 + 5 = 5 to 10 + 4 + 5 = 19.
@pytest.mark.parametrize(('maximize', 'near_optimal', 'best_discrete'), [
    (True, lambda f: f > 18, 4),
    (False, lambda f: f < 6, 0),
], ids=['maximize', 'minimize'])
def test_run_optimizes_within_bounds(maximize, near_optimal, best_discrete):
    objective = SumObjective(('continuous', 'discrete', 'fixed'), maximize=maximize, hms=5, max_imp=2000, seed=0)

    best_harmony, best_fitness, memory, _ = HarmonySearch(objective).run()

    assert near_optimal(best_fitness)
    assert best_harmony[1] == best_discrete
    for harmony, _ in memory:
        assert objective.lower <= harmony[0] <= objective.upper
        assert harmony[1] in objective.values
        assert harmony[2] == objective.fixed


@pytest.mark.parametrize(('hms', 'max_imp', 'expected_generations'), [
    (3, 6, [0, 1, 2]),
    (3, 7, [0, 1, 2]),
    (3, 2, [0]),
])
def test_run_records_history_every_hms_improvisations(hms, max_imp, expected_generations):
    _, _, _, history = HarmonySearch(SumObjective(hms=hms, max_imp=max_imp)).run()

    assert [snapshot['gen'] for snapshot in history] == expected_generations
    assert all(len(snapshot['harmonies']) == hms for snapshot in history)


def test_run_history_snapshots_are_copies():
    # Regression test for #16: history snapshots must not alias the live memory. Every random improvisation (>= 0)
    # beats these initial harmonies, so the memory is fully replaced.
    initial_harmonies = [[-100], [-101], [-102]]
    objective = SumObjective(hms=3, max_imp=6, hmcr=0)

    _, _, memory, history = HarmonySearch(objective).run(initial_harmonies=initial_harmonies)

    assert history[0]['harmonies'] == [([-100], -100), ([-101], -101), ([-102], -102)]
    assert all(fitness >= 0 for _, fitness in memory)


def test_run_with_seed_is_reproducible():
    def run(seed):
        objective = SumObjective(('continuous', 'discrete'), hms=4, max_imp=50, seed=seed)
        return HarmonySearch(objective).run()

    assert run(42) == run(42)
    assert run(42) != run(43)


@pytest.mark.parametrize(('initial_harmonies', 'message'), [
    ([[1], [2]], 'does not equal to the harmony memory size'),
    ([[1], [2], [3, 4]], 'Number of parameters in initial harmonies'),
])
def test_run_rejects_malformed_initial_harmonies(initial_harmonies, message):
    with pytest.raises(ValueError, match=message):
        HarmonySearch(SumObjective(hms=3)).run(initial_harmonies=initial_harmonies)


# _memory_consideration()

def test_memory_consideration_takes_note_from_random_memory_slot(fake_random):
    hs = make_search(SumObjective(('continuous', 'continuous'), hms=3), [([0, 1], 1), ([2, 3], 5), ([4, 5], 9)])
    randint_bounds = fake_random(randints=[2])
    harmony = [0]

    hs._memory_consideration(harmony, 1)

    assert harmony == [0, 5]
    assert randint_bounds == [(0, 2)]


# _pitch_adjustment()

def test_pitch_adjustment_leaves_fixed_parameter_alone(fake_random):
    fake_random()  # any random draw would raise StopIteration
    harmony = [5.0]

    make_search(SumObjective(('fixed',)))._pitch_adjustment(harmony, 0)

    assert harmony == [5.0]


DOWN, UP = 0.1, 0.9


# The step drawn is always the largest allowed, so the expected bounds show the step is clamped to the value range.
@pytest.mark.parametrize(('start', 'direction', 'expected_bounds', 'expected'), [
    (2, DOWN, (0, 2), 0),
    (1, DOWN, (0, 1), 0),
    (0, DOWN, (0, 0), 0),
    (2, UP, (0, 2), 4),
    (3, UP, (0, 1), 4),
    (4, UP, (0, 0), 4),
])
def test_discrete_pitch_adjustment_stays_within_values(fake_random, start, direction, expected_bounds, expected):
    objective = SumObjective(('discrete',), values=[10, 20, 30, 40, 50], mpai=2)
    randint_bounds = fake_random(randoms=[direction], randints=[expected_bounds[1]])
    harmony = [objective.values[start]]

    make_search(objective)._pitch_adjustment(harmony, 0)

    assert randint_bounds == [expected_bounds]
    assert harmony == [objective.values[expected]]


@pytest.mark.parametrize(('direction', 'r', 'mpap', 'expected'), [
    (DOWN, 0.5, 0.5, 3.0),  # 4 - (4 - 0) * 0.5 * 0.5
    (UP, 0.5, 0.5, 5.5),  # 4 + (10 - 4) * 0.5 * 0.5
    (DOWN, 0.0, 0.5, 4.0),
    (DOWN, 0.9999, 1.0, 0.0004),
    (UP, 0.9999, 1.0, 9.9994),
])
def test_continuous_pitch_adjustment_moves_toward_bound(fake_random, direction, r, mpap, expected):
    fake_random(randoms=[direction, r])
    harmony = [4.0]

    make_search(SumObjective(lower=0.0, upper=10.0, mpap=mpap))._pitch_adjustment(harmony, 0)

    assert harmony == [pytest.approx(expected)]


# _update_harmony_memory()

MEMORY = [([1], 1), ([5], 5), ([3], 3)]


@pytest.mark.parametrize(('maximize', 'candidate', 'expected'), [
    (True, ([4], 4), [([4], 4), ([5], 5), ([3], 3)]),  # better than the worst replaces the worst
    (False, ([2], 2), [([1], 1), ([2], 2), ([3], 3)]),
    (True, ([0], 0), MEMORY),  # worse than the worst is discarded
    (False, ([9], 9), MEMORY),
    (True, ([5], 5), MEMORY),  # duplicates are discarded even though they beat the worst
    (False, ([1], 1), MEMORY),
    (True, ([2], 1), MEMORY),  # ties with the worst are discarded
    (False, ([2], 5), MEMORY),
])
def test_update_harmony_memory(maximize, candidate, expected):
    hs = make_search(SumObjective(maximize=maximize), MEMORY)

    hs._update_harmony_memory(*candidate)

    assert hs._harmony_memory == expected
