# Changes

## 1.4.2 (2020-01-13)
* Fixing inexplicable tab/space issue in PyPI release (#15).

## 1.4.1 (2018-07-11)
* Fixing build/deploy issue.

## 1.4.0 (2018-07-11)
* The initial harmonies can be specified by passing a list of harmonies to `harmony_search()`. (courtesy @szhan)
* `HarmonySearchResults` now also stores the history of harmonies (stored every `n`th fitness evaluations, where `n` is the number of individual solutions) of each run. (courtesy @szhan)

## 1.3.3 (2015-12-05)
* `HarmonySearchResults` now also stores the last complete harmony memory from each run.

## 1.3.2 (2014-11-03)
* For clarity, the harmony memory now remembers `(harmony, fitness)` tuples instead of a single array with the fitness appended to the end.

## 1.3.1 (2014-11-03)
* `HarmonySearchResults` is now a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) instead of a class.

## 1.3 (2014-10-30)
* An empty struct-like class, `HarmonySearchResults`, is now used to return search results for more flexibility/clarity.

## 1.2 (2013-10-09)
* pyHarmonySearch is now fully Python 3+ compliant.
* `KeyboardInterrupt` is now properly handled.

## 1.1 (2013-10-04)
* Moving to setuptools (using [ez_setup.py](https://bitbucket.org/pypa/setuptools/downloads/ez_setup.py) to manage it).
* `ObjectiveFunctionInterface` methods now output their name (dynamically generated using [inspect](http://docs.python.org/2/library/inspect.html)) if a `NotImplementedError` is raised.

## 1.0.1 (2013-06-30)
* Fixed a bug in the pitch adjustment step.
* Added a new, more complicated, example that solves a 5-D linear system of equations.

## 1.0 (2013-06-30)
* Refactored code so that HS is now implemented as a class called `HarmonySearch`. A non-class method `harmony_search` is used to call `HarmonySearch` with multiple processes. This now means we can do `from pyharmonysearch import ObjectiveFunctionInterface, harmony_search` when implementing an objective function.
* Added `setup.py` so that pyHS can be installed and imported anywhere.
* Added `CHANGES.md` to track changes. Although development has gone on since 2013-04-17, I will only henceforth track changes.
* Split pyHS code and examples into separate modules.
* Changes to readme to reflect refactoring.
