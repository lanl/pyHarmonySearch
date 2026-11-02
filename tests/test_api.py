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
import multiprocessing
from datetime import timedelta

import pytest
from helpers import SumObjective, set_terminating

import pyharmonysearch

# pyharmonysearch.harmony_search is the re-exported function, not the submodule, so import the module explicitly.
hs_module = importlib.import_module('pyharmonysearch.harmony_search')


@pytest.fixture(autouse=True)
def reset_terminating():
    hs_module.terminating.clear()
    yield
    hs_module.terminating.clear()


class FakeHarmonySearch:

    """Stands in for HarmonySearch, returning scripted (harmony, fitness, memory, history) results in order."""

    def __init__(self, results, calls, objective_function):
        self.results = results
        self.calls = calls
        self.objective_function = objective_function

    def run(self, initial_harmonies=None):
        self.calls.append((self.objective_function, initial_harmonies))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def fake_search(monkeypatch):
    """
        Patch HarmonySearch with FakeHarmonySearch. Returns a function that scripts results (fitness values, or an
        exception to raise) and returns the list of (objective_function, initial_harmonies) each run() received.
    """
    results, calls = [], []
    monkeypatch.setattr(hs_module, 'HarmonySearch', lambda objective: FakeHarmonySearch(results, calls, objective))

    def script(*outcomes):
        results[:] = [outcome if isinstance(outcome, BaseException)
                      else ([f'harmony{i}'], outcome, f'memory{i}', f'history{i}')
                      for i, outcome in enumerate(outcomes)]
        return calls
    return script


class FakePool:

    """Runs tasks synchronously in-process, so worker() is covered and behaviour is deterministic."""

    def __init__(self, num_processes, initializer, initargs, raise_in=None):
        self.num_processes = num_processes
        self.initializer = initializer
        self.initargs = initargs
        self.raise_in = raise_in
        self.events = []
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.terminate()

    def apply_async(self, func, args):
        if self.raise_in == 'apply_async':
            raise KeyboardInterrupt
        try:
            result, error = func(*args), None
        except Exception as e:
            result, error = None, e
        return FakeAsyncResult(result, error)

    def close(self):
        self.events.append('close')

    def join(self):
        if self.raise_in == 'join':
            raise KeyboardInterrupt
        self.events.append('join')

    def terminate(self):
        self.events.append('terminate')


class FakeAsyncResult:

    """Like multiprocessing's AsyncResult, re-raises the task's exception from get() rather than when submitted."""

    def __init__(self, result, error):
        self.result = result
        self.error = error

    def get(self):
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def fake_pool(monkeypatch):
    """Patch Pool and return a function that installs FakePool (optionally raising) and returns the pools created."""
    pools = []

    def install(raise_in=None):
        def factory(num_processes, initializer, initargs):
            pools.append(FakePool(num_processes, initializer, initargs, raise_in))
            return pools[-1]
        monkeypatch.setattr(hs_module, 'Pool', factory)
        return pools
    return install


@pytest.fixture(params=['harmony_search', 'harmony_search_serial'])
def search(request, fake_pool):
    """Every aggregation test runs against both entry points, so each is checked to combine results correctly."""
    if request.param == 'harmony_search':
        fake_pool()
        return lambda objective, num_iterations, initial_harmonies=None: hs_module.harmony_search(
            objective, 2, num_iterations, initial_harmonies)
    return hs_module.harmony_search_serial


# harmony_search() and harmony_search_serial()

@pytest.mark.parametrize(('maximize', 'expected_index'), [(True, 1), (False, 2)])
def test_search_returns_best_across_iterations(search, fake_search, maximize, expected_index):
    fake_search(5, 7, 3)

    results = search(SumObjective(maximize=maximize), 3)

    assert results.best_harmony == [f'harmony{expected_index}']
    assert results.best_fitness == [5, 7, 3][expected_index]


def test_search_keeps_first_of_tied_best(search, fake_search):
    fake_search(5, 5)

    assert search(SumObjective(), 2).best_harmony == ['harmony0']


def test_search_collects_every_iteration(search, fake_search):
    calls = fake_search(5, 7, 3)
    initial_harmonies = [[1], [2], [3]]
    objective = SumObjective()

    results = search(objective, 3, initial_harmonies)

    assert results.harmony_memories == ['memory0', 'memory1', 'memory2']
    assert results.harmony_histories == ['history0', 'history1', 'history2']
    assert isinstance(results.elapsed_time, timedelta)
    assert calls == [(objective, initial_harmonies)] * 3


@pytest.mark.parametrize(('maximize', 'expected_fitness'), [(True, float('-inf')), (False, float('inf'))])
def test_search_with_zero_iterations(search, maximize, expected_fitness):
    results = search(SumObjective(maximize=maximize), 0)

    assert results.best_harmony is None
    assert results.best_fitness == expected_fitness
    assert results.harmony_memories == results.harmony_histories == []


def test_search_recovers_after_earlier_keyboard_interrupt(search, fake_search):
    # Regression test: a Ctrl+C in an earlier search (e.g. in a REPL) used to leave every later search broken.
    hs_module.terminating.set()
    fake_search(5)

    assert search(SumObjective(), 1).best_fitness == 5


def test_harmony_search_pool_lifecycle(fake_pool, fake_search):
    pools = fake_pool()
    fake_search(1)

    hs_module.harmony_search(SumObjective(), 4, 1)

    assert pools[0].num_processes == 4
    assert pools[0].events == ['close', 'join', 'terminate']
    # Regression test: workers must share the parent's terminating Event (see test_init_worker_shares_event_*).
    assert pools[0].initializer is hs_module._init_worker
    assert pools[0].initargs == (hs_module.terminating,)


@pytest.mark.parametrize('raise_in', ['apply_async', 'join'])
def test_harmony_search_terminates_pool_on_keyboard_interrupt(fake_pool, raise_in):
    pools = fake_pool(raise_in=raise_in)

    with pytest.raises(KeyboardInterrupt):
        hs_module.harmony_search(SumObjective(), 2, 1)

    assert pools[0].events[-1] == 'terminate'


def test_harmony_search_terminates_pool_on_other_errors(fake_pool, fake_search):
    # Regression test: the pool used to be left running on anything but a KeyboardInterrupt.
    pools = fake_pool()
    fake_search(ValueError('boom'))

    with pytest.raises(ValueError, match='boom'):
        hs_module.harmony_search(SumObjective(), 2, 1)

    assert pools[0].events == ['close', 'join', 'terminate']


@pytest.mark.parametrize('start_method', ['spawn', 'forkserver'])
def test_init_worker_shares_event_across_start_methods(start_method):
    # Regression test: spawn/forkserver workers re-import the module, so each used to get its own, unshared Event.
    event = multiprocessing.get_context(start_method).Event()

    with multiprocessing.get_context(start_method).Pool(1, initializer=hs_module._init_worker,
                                                        initargs=(event,)) as pool:
        pool.apply(set_terminating)

    assert event.is_set()


def test_harmony_search_with_real_pool():
    # Exercises pickling the objective function and results across processes (on Linux: fork, or forkserver on
    # Python 3.14+).
    objective = SumObjective(('continuous', 'discrete'), hms=3, max_imp=20, seed=1)

    results = hs_module.harmony_search(objective, 2, 2)

    assert len(results.harmony_memories) == len(results.harmony_histories) == 2
    assert results.harmony_memories[0] == results.harmony_memories[1]  # seeded, so every iteration is identical
    assert results.best_fitness == max(fitness for _, fitness in results.harmony_memories[0])


# worker()

def test_worker_runs_search(fake_search):
    calls = fake_search(5)
    objective = SumObjective()

    assert hs_module.worker(objective, [[1]]) == (['harmony0'], 5, 'memory0', 'history0')
    assert calls == [(objective, [[1]])]


def test_worker_does_nothing_once_terminating(fake_search):
    calls = fake_search(5)
    hs_module.terminating.set()

    assert hs_module.worker(SumObjective()) is None
    assert calls == []


def test_worker_signals_other_workers_on_keyboard_interrupt(fake_search):
    fake_search(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        hs_module.worker(SumObjective())

    assert hs_module.terminating.is_set()


def test_worker_does_not_signal_on_other_errors(fake_search):
    fake_search(ValueError('boom'))

    with pytest.raises(ValueError, match='boom'):
        hs_module.worker(SumObjective())

    assert not hs_module.terminating.is_set()


# Public API

def test_package_exports():
    assert pyharmonysearch.harmony_search is hs_module.harmony_search
    assert pyharmonysearch.HarmonySearch is hs_module.HarmonySearch
    assert set(pyharmonysearch.__all__) == {'HarmonySearch', 'ObjectiveFunctionInterface', 'harmony_search'}

