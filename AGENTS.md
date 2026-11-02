# AGENTS.md

Pure-Python harmony search optimizer. The package uses only the standard library and needs Python >=3.11 (`pyproject.toml`). Keep it that way: no NumPy or other runtime dependencies (pytest/ruff are dev-only extras).

## Layout
- `pyharmonysearch/harmony_search.py`: all of the algorithm.
  - `harmony_search()` runs `num_iterations` independent HS runs on a `multiprocessing.Pool`.
  - `harmony_search_serial()` does the same without a pool. It is **not** exported from `__init__.py`, so import it from `pyharmonysearch.harmony_search`.
  - `worker()` is the module-level function the pool pickles.
- `pyharmonysearch/objective_function_interface.py`: an abstract-by-convention base class. Its methods raise `NotImplementedError`; it does not use `abc`. Users subclass it to define the problem and **all** tuning params (`hms`, `hmcr`, `par`, `mpap`, `mpai`, `max_imp`, seed, `maximize`). `harmony_search()` takes no parameters beyond the process/iteration counts and `initial_harmonies`.
- `examples/*.py` and `README.md`: the usage docs. When you add an example, add a row to the matching table in the README's EXAMPLES section, plus a REFERENCES entry if it comes from the literature.

## Gotchas
- Objective functions must be picklable (top-level classes), because they are sent to pool workers. Calls that create a pool must sit under `if __name__ == '__main__':`.
- `HarmonySearchResults` is a namedtuple. Adding a field means changing the namedtuple, the 4-tuple returned by `HarmonySearch.run()`, `_combine_results()` (shared by `harmony_search` and `harmony_search_serial`), and the field list in `README.md`.
- Every best/worst comparison branches on `maximize()`. Keep both directions correct when you edit.
- Harmony memory holds `(harmony, fitness)` tuples. The history is deep-copied every `hms` improvisations; this fixed a shallow-copy bug in 1.4.3, so don't undo it.
- The global `terminating = Event()` and the `KeyboardInterrupt` handling in `worker()` are intentional. Don't remove them.
  - Pool workers get the parent's Event through the `_init_worker` initializer. Without it, spawn and forkserver workers would each get their own unshared Event.
  - Both entry points call `terminating.clear()` first, so an earlier Ctrl+C doesn't break later searches.
  - The pool is a `with` block, so it is terminated on any exception.
- In `get_value(i, j)`, `j` can be 0. Check `j is not None`, never `if j:`.
- The docstrings in the interface file use Python 2 `print x` syntax. They are illustrative, not runnable doctests.

## Testing / verification
- Setup: `pip install -e ".[dev]"`. Then `ruff check` and `pytest` from the repo root (CI runs both). pytest enforces 100% **branch** coverage via `fail_under`, so any new branch needs a test. `raise NotImplementedError` lines are excluded from coverage. Use `.venv/bin/` if a local venv exists. Single test: `pytest --no-cov tests/test_api.py::test_worker_runs_search` (without `--no-cov`, the coverage threshold fails any partial run).
- Test helpers that must cross process boundaries live in `tests/helpers.py` and are imported as `from helpers import ...`. This works because pytest's `pythonpath = ["tests"]` puts `tests/` on `sys.path` (`--import-mode=importlib` doesn't add it). Classes defined inside test files can't be pickled by spawn/forkserver workers (forkserver is the Linux default from Python 3.14).
- `pyharmonysearch.harmony_search` resolves to the re-exported *function*, so tests get the module with `importlib.import_module('pyharmonysearch.harmony_search')` and monkeypatch `Pool`, `HarmonySearch`, and `random` on it.
- Tests replace the module's `random` reference with scripted values (`fake_random` fixture) rather than patching `random.random` globally. `hmcr`/`par` of 0 or 1 make those branches deterministic without patching.
- The global `terminating` Event persists across calls in a process; the autouse fixture in `tests/test_api.py` clears it.
- Ruff config (rules, 120 cols, single quotes) is in `pyproject.toml`. There's no `ruff format` check.
- CI (`.github/workflows/tests.yml`) runs on Ubuntu only, against every maintained CPython.
- `tests/test_examples.py` imports every `examples/*.py` without running `__main__` and runs it briefly. Each example must define a top-level `ObjectiveFunction` with a no-argument constructor and a `_max_imp` attribute (the test shrinks it), and keep its search under `if __name__ == '__main__':`.
- To check real output, run the examples from `examples/` with the repo root on `PYTHONPATH`. Use `PYTHONPATH=.. python 2-D_continuous_seed.py`. That one is fast and seeded, with 1 process and 1 iteration. Expect roughly 4.0 best fitness. Several of the other examples use every CPU and take tens of seconds to minutes.

## Releases
- When you bump the version, update `version` in `pyproject.toml` and add an entry to `CHANGES.md`. If you add or drop a supported Python version, also update `requires-python`, the classifiers, the CI matrix, and the REQUIREMENTS line in `README.md`.
- Every `.py` file starts with the Triad National Security BSD license header (after the shebang in `examples/`). Keep it on new files.
