# pyHarmonySearch

[![Tests](https://github.com/lanl/pyHarmonySearch/actions/workflows/tests.yml/badge.svg)](https://github.com/lanl/pyHarmonySearch/actions/workflows/tests.yml)

## LICENSE
This software is licensed under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause). Please refer to the separate [LICENSE](LICENSE) file for the exact text of the license. You are obligated to give attribution if you use this code.

## ABOUT
pyHarmonySearch is a pure Python implementation of the harmony search (HS) global optimization algorithm. HS is a metaheuristic search algorithm that, similar to simulated annealing, tabu, and evolutionary searches, is based on real-world phenomena. Specifically, HS mimics a jazz band improvising together. Courtesy [Wikipedia](https://en.wikipedia.org/wiki/Harmony_search):

> In the HS algorithm, each musician (= decision variable) plays (= generates) a note (= a value) for finding a best harmony (= global optimum) all together.

pyHarmonySearch supports both continuous and discrete variables and can take advantage of parallel processing using [Python's multiprocessing module](https://docs.python.org/3/library/multiprocessing.html).

## REQUIREMENTS
This code does not rely on any third-party software. It only requires Python 3.11 or higher.

## INSTALL
pyHarmonySearch is available on PyPI at https://pypi.org/project/pyHarmonySearch/.

Install using [pip](https://pip.pypa.io/):

```bash
pip install pyHarmonySearch
```

Install from source:

```bash
pip install .
```

## USING THIS CODE
There are five examples included. The first three examples ([2-D_continuous_seed.py](examples/2-D_continuous_seed.py), [2-D_discrete_x.py](examples/2-D_discrete_x.py), and [2-D_continuous_fixed_x.py](examples/2-D_continuous_fixed_x.py)) are variations on each other that find the global maximum of a simple 2-D function. The fourth example ([5-D_linear_system.py](examples/5-D_linear_system.py)) stochastically solves a 5-D linear system of equations. The fifth example ([2-D_continuous_specified_initial_harmonies.py](examples/2-D_continuous_specified_initial_harmonies.py)) shows how to specify the initial harmonies. Read the documentation in each example for more information. Run them from the `examples/` directory after installing:

```console
> ./2-D_continuous_fixed_x.py
Elapsed time: 0:00:23.396807
Best harmony: [0.5, -1.0001700408343779]
Best fitness: 3.7499999710861145
> ./2-D_continuous_seed.py
Elapsed time: 0:00:03.119576
Best harmony: [0.027035817109283933, -0.9950109785422445]
Best fitness: 3.9992441742581275
> ./2-D_discrete_x.py
Elapsed time: 0:00:30.877592
Best harmony: [0, -0.9987024210158837]
Best fitness: 3.9999983162887798
> ./5-D_linear_system.py
Elapsed time: 0:02:29.715337
Best harmony: [8.118886532259536, 2.0254098515892665, 0.6678692319283357, 2.906307072622585, 9.814436850217918]
Best fitness: 0.8690865628414204
> ./2-D_continuous_specified_initial_harmonies.py
Elapsed time: 0:00:03.487337
Best harmony: [-0.01738884537223441, -0.9510801626682784]
Best fitness: 3.9973044775720585
```

`HarmonySearchResults`, a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple), is returned. It has five fields:

* `elapsed_time`: a `datetime.timedelta`.
* `best_harmony` and `best_fitness`: the best harmony (a list with one value per parameter) found across all iterations, and its fitness.
* `harmony_memories`: one final harmony memory per iteration. Each is a list of `(harmony, fitness)` tuples.
* `harmony_histories`: one history per iteration. Each is a list of `{'gen': generation, 'harmonies': harmony_memory}` snapshots, taken every `hms` improvisations.

Note that like many similar optimization algorithms, HS is stochastic, so you will get a slightly different result every time you run it. Because of the stochasticity, I have added the ability to run multiple iterations of HS simultaneously using [Python's multiprocessing module](https://docs.python.org/3/library/multiprocessing.html); you simply specify the number of processes on which to run the specified number of iterations. The resulting solution is the best solution found from all iterations. Also, the user can specify the initial harmonies. An optional random seed can be used to generate reproducible results.

In general, you will make use of this code in three steps:

1. Implement your own objective function that inherits from `ObjectiveFunctionInterface`.
1. Tune the input parameters, which your objective function returns. These are problem-specific, and the numbers used in the example implementations might not be appropriate for your problem.
    * `hms` (harmony memory size): how many harmonies are kept in memory.
    * `hmcr` (harmony memory considering rate): the probability of taking each value from memory rather than choosing it at random.
    * `par` (pitch adjusting rate): the probability of adjusting a value taken from memory.
    * `mpap` (maximum pitch adjustment proportion): the most a continuous value can move, as a proportion of its distance to the bound.
    * `mpai` (maximum pitch adjustment index): the most a discrete value can move, in indices.
    * `max_imp` (maximum improvisations): how many new harmonies to generate. This is the stopping criterion.
    * `maximize()` chooses maximization or minimization, and an optional random seed makes runs reproducible.
1. Run HS with `harmony_search(objective_function, num_processes, num_iterations, initial_harmonies=None)`. If given, `initial_harmonies` must contain exactly `hms` harmonies, each with one value per parameter; otherwise a `ValueError` is raised.

Work is sent to a [multiprocessing pool](https://docs.python.org/3/library/multiprocessing.html), so your objective function must be picklable (define it at the top level of a module), and the call to `harmony_search()` must be under `if __name__ == '__main__':`. This is required under the spawn and forkserver start methods: spawn is the default on macOS and Windows, and forkserver is the default on Linux from Python 3.14.

It will look something like this:

```python
from multiprocessing import cpu_count

from pyharmonysearch import ObjectiveFunctionInterface, harmony_search


class ObjectiveFunction(ObjectiveFunctionInterface):
    ...  # implement every method of ObjectiveFunctionInterface


if __name__ == '__main__':
    obj_fun = ObjectiveFunction()
    num_processes = cpu_count()  # use number of logical CPUs
    num_iterations = num_processes * 5  # each process does 5 iterations
    results = harmony_search(obj_fun, num_processes, num_iterations)
    print(f'Elapsed time: {results.elapsed_time}\nBest harmony: {results.best_harmony}\nBest fitness: {results.best_fitness}')
```

More documentation is provided in [harmony_search.py](pyharmonysearch/harmony_search.py) and [objective_function_interface.py](pyharmonysearch/objective_function_interface.py) and in the examples.

If your objective function already uses multiprocessing itself (e.g., in `get_fitness()`), use `harmony_search_serial(objective_function, num_iterations, initial_harmonies=None)` instead, which runs the iterations one after another in the current process. It isn't exported from the package, so import it with `from pyharmonysearch.harmony_search import harmony_search_serial`.

## TESTING
Install the development dependencies, then run the linter and the tests (which require 100% branch coverage) from the repository root:

```bash
pip install -e ".[dev]"
ruff check
pytest
```

## REFERENCES
[Neat Algorithms - Harmony Search](https://web.archive.org/web/20210805035109/http://harry.me/blog/2011/07/05/neat-algorithms-harmony-search/) (archived) provides a simple introduction on how HS works. Also, see the [HS Wikipedia entry](https://en.wikipedia.org/wiki/Harmony_search).

HS was first introduced by Geem et al. in 2001:

Z. W. Geem, J. H. Kim, and G. V. Loganathan, "A New Heuristic Optimization Algorithm: Harmony Search," Simulation, vol. 76, no. 2, pp. 60–68, Feb. 2001. [doi:10.1177/003754970107600201](https://doi.org/10.1177/003754970107600201)

Some modifications and improvements have been suggested to the original algorithm. None of these are implemented here. This package only implements the original algorithm.

* Z. W. Geem, "Improved Harmony Search from Ensemble of Music Players", in Knowledge-Based Intelligent Information and Engineering Systems, B. Gabrys, R. Howlett, and L. Jain, Eds. Springer Berlin / Heidelberg, 2006, pp. 86-93.
* M. Mahdavi, M. Fesanghary, and E. Damangir, "An improved harmony search algorithm for solving optimization problems", Applied Mathematics and Computation, vol. 188, no. 2, pp. 1567-1579, May 2007.
* M. G. H. Omran and M. Mahdavi, "Global-best harmony search", Applied Mathematics and Computation, vol. 198, no. 2, pp. 643-656, May 2008.
* Z. W. Geem and K.-B. Sim, "Parameter-setting-free harmony search algorithm," Applied Mathematics and Computation, vol. 217, no. 8, pp. 3881–3889, Dec. 2010.

## DIFFERENCES
Python optimization packages that implement HS:

* [pyOpt](https://github.com/madebr/pyOpt)
* [pygmo](https://github.com/esa/pygmo2)

pyHarmonySearch differs from other optimization packages in that it relies on no third-party software. It is simple and easy-to-use standalone Python code. The primary reason I developed this was because I got tired of having to compile and maintain my own third-party packages on a machine running several-year-old software for which I have only user-level access. SciPy is particularly nasty to build, and many scientific packages rely on it. I skirted this issue by implementing HS myself from scratch.
