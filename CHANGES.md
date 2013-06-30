# Changes

## 1.0 (2013-06-30)
* Refactored code so that HS is now implemented as a class called `HarmonySearch`. A non-class method `harmony_search` is used to call `HarmonySearch` with multiple processes. This now means we can do `from pyharmonysearch import ObjectiveFunctionInterface, harmony_search` when implementing an objective function.
* Added `setup.py` so that pyHS can be installed and imported anywhere.
* Added `CHANGES.md` to track changes. Although development has gone on since 2013-04-17, I will only henceforth track changes.
* Split pyHS code and examples into separate modules.
* Changes to readme to reflect refactoring.