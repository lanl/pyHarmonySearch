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
You use this code in three steps:

1. Implement your own objective function that inherits from `ObjectiveFunctionInterface`. It defines the problem: the parameters, their bounds or discrete values, which are fixed, and the fitness function. See [objective_function_interface.py](pyharmonysearch/objective_function_interface.py) for every method and the [examples](#examples) for complete implementations.
1. Tune the input parameters, which your objective function returns. These are problem-specific, and the numbers used in the example implementations might not be appropriate for your problem.
    * `hms` (harmony memory size): how many harmonies are kept in memory.
    * `hmcr` (harmony memory considering rate): the probability of taking each value from memory rather than choosing it at random.
    * `par` (pitch adjusting rate): the probability of adjusting a value taken from memory.
    * `mpap` (maximum pitch adjustment proportion): the most a continuous value can move, as a proportion of its distance to the bound.
    * `mpai` (maximum pitch adjustment index): the most a discrete value can move, in indices.
    * `max_imp` (maximum improvisations): how many new harmonies to generate. This is the stopping criterion.
    * `maximize()` chooses maximization or minimization, and an optional random seed makes runs reproducible.
1. Run HS with `harmony_search(objective_function, num_processes, num_iterations, initial_harmonies=None)`. If given, `initial_harmonies` must contain exactly `hms` harmonies, each with one value per parameter; otherwise a `ValueError` is raised.

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

### Iterations and multiprocessing
Like many similar optimization algorithms, HS is stochastic, so you will get a slightly different result every time you run it. Because of this, `harmony_search()` runs `num_iterations` independent searches on `num_processes` processes using [Python's multiprocessing module](https://docs.python.org/3/library/multiprocessing.html), and returns the best solution found across all of them.

Work is sent to a [multiprocessing pool](https://docs.python.org/3/library/multiprocessing.html), so your objective function must be picklable (define it at the top level of a module), and the call to `harmony_search()` must be under `if __name__ == '__main__':`. This is required under the spawn and forkserver start methods: spawn is the default on macOS and Windows, and forkserver is the default on Linux from Python 3.14.

If your objective function already uses multiprocessing itself (e.g., in `get_fitness()`), use `harmony_search_serial(objective_function, num_iterations, initial_harmonies=None)` instead, which runs the iterations one after another in the current process. It isn't exported from the package, so import it with `from pyharmonysearch.harmony_search import harmony_search_serial`.

### Results
`HarmonySearchResults`, a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple), is returned. It has five fields:

* `elapsed_time`: a `datetime.timedelta`.
* `best_harmony` and `best_fitness`: the best harmony (a list with one value per parameter) found across all iterations, and its fitness.
* `harmony_memories`: one final harmony memory per iteration. Each is a list of `(harmony, fitness)` tuples.
* `harmony_histories`: one history per iteration. Each is a list of `{'gen': generation, 'harmonies': harmony_memory}` snapshots, taken every `hms` improvisations.

### Constraints
HS has no built-in notion of constraints. Handle them in `get_fitness()` with a penalty that makes infeasible harmonies score worse than feasible ones, as the pressure vessel and knapsack examples do.

## EXAMPLES
Each example is a standalone script that defines an `ObjectiveFunction`, runs HS, and prints the result. Each one's docstring explains the problem and its known optimum. Most use every CPU and take from seconds to a few minutes; 2-D_continuous_seed.py is the quickest. Run them from the `examples/` directory after installing:

```console
> ./2-D_continuous_seed.py
Elapsed time: 0:00:02.525815
Best harmony: [0.027035817109283933, -0.9950109785422445]
Best fitness: 3.9992441742581275
```

### Feature demonstrations
These examples use simple problems to show one feature of the API at a time. The first four are variations on each other that maximize -(x^2 + (y+1)^2) + 4, whose maximum is 4 at (0, -1).

| Example | Demonstrates |
| --- | --- |
| [2-D_continuous_seed.py](examples/2-D_continuous_seed.py) | Continuous variables and a random seed for reproducible results. |
| [2-D_discrete_x.py](examples/2-D_discrete_x.py) | A discrete variable (x is an integer from -100 to 100) alongside a continuous one. |
| [2-D_continuous_fixed_x.py](examples/2-D_continuous_fixed_x.py) | A fixed variable: x is held at 0.5, so the maximum becomes 3.75 at (0.5, -1). |
| [2-D_continuous_specified_initial_harmonies.py](examples/2-D_continuous_specified_initial_harmonies.py) | Specifying the initial harmonies. |
| [5-D_linear_system.py](examples/5-D_linear_system.py) | Minimization: stochastically solving a 5-D linear system of equations (fitness 0 at the exact solution). |

### Problems from the literature
These examples solve well-known optimization problems with known optima, so you can judge how close HS gets.

| Example | Problem | Known optimum | Demonstrates |
| --- | --- | --- | --- |
| [4-D_pressure_vessel.py](examples/4-D_pressure_vessel.py) | Pressure vessel design: choose a cylindrical vessel's plate thicknesses, radius, and length to minimize its manufacturing cost. | About 6059.71 (best known) | Mixed discrete and continuous variables, and inequality constraints handled with a penalty. |
| [N-D_rastrigin.py](examples/N-D_rastrigin.py) | The Rastrigin function: a bowl covered in many local minima, a standard benchmark for global optimizers. | 0 at the origin | A configurable number of dimensions, tuning to avoid stalling in a local minimum, and printing convergence from `harmony_histories`. |
| [15-D_knapsack.py](examples/15-D_knapsack.py) | 0/1 knapsack: pick the items with the largest total profit that fit within a weight capacity. | 1458 (weight 749) | A purely discrete (binary) problem with a capacity constraint handled with a penalty. |

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

The problems in the [literature examples](#problems-from-the-literature) come from:

* E. Sandgren, "Nonlinear Integer and Discrete Programming in Mechanical Design Optimization," Journal of Mechanical Design, vol. 112, no. 2, pp. 223–229, Jun. 1990. [doi:10.1115/1.2912596](https://doi.org/10.1115/1.2912596) (pressure vessel)
* K. S. Lee and Z. W. Geem, "A new meta-heuristic algorithm for continuous engineering optimization: harmony search theory and practice," Computer Methods in Applied Mechanics and Engineering, vol. 194, no. 36–38, pp. 3902–3933, Sep. 2005. [doi:10.1016/j.cma.2004.09.007](https://doi.org/10.1016/j.cma.2004.09.007) (pressure vessel solved with HS; the best known cost of about 6059.71 comes from later work)
* H. Mühlenbein, M. Schomisch, and J. Born, "The parallel genetic algorithm as function optimizer," Parallel Computing, vol. 17, no. 6–7, pp. 619–632, Sep. 1991. [doi:10.1016/S0167-8191(05)80052-3](https://doi.org/10.1016/S0167-8191(05)80052-3) (generalized Rastrigin function)
* J. Burkardt, [KNAPSACK_01: Data for the 01 Knapsack Problem](https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html) (instance P07, from Kreher and Stinson)

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
