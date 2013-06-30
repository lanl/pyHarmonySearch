# pyHarmonySearch

## AUTHOR
Geoffrey Fairchild
* [http://www.gfairchild.com/](http://www.gfairchild.com/)
* [https://github.com/gfairchild](https://github.com/gfairchild)
* [http://www.linkedin.com/in/gfairchild/](http://www.linkedin.com/in/gfairchild/)

## LICENSE
This software is licensed under the [BSD 3-Clause License](http://opensource.org/licenses/BSD-3-Clause). Please refer to the separate LICENSE.txt file for the exact text of the license. You are obligated to give attribution if you use this code.

## ABOUT
pyHarmonySearch is a pure Python implementation of the harmony search (HS) global optimization algorithm. HS is a metaheuristic search algorithm that, similar to simulated annealing, tabu, and evolutionary searches, is based on real world phenomena. Specifically, HS mimics a jazz band improvising together. Courtesy [Wikipedia](http://en.wikipedia.org/wiki/Harmony_search):

> In the HS algorithm, each musician (= decision variable) plays (= generates) a note (= a value) for finding a best harmony (= global optimum) all together.

pyHarmonySearch supports both continuous and discrete variables and can take advantage of parallel processing using [Python's multiprocessing module](http://docs.python.org/2/library/multiprocessing.html).

## REQUIREMENTS
This code does not rely on any 3rd party software. It only requires Python 2.7 or higher. It likely works under Python 3 using 2to3, but I haven't tested it.

## INSTALL
Using [pip](http://www.pip-installer.org/):

	pip install pyHarmonySearch

From source:

	python setup.py install

## USING THIS CODE
There are four examples included. The first three examples are variations on each other that find the global maximum of a simple 2-D function. The fourth example stochastically solves a 5-D linear system of equations. Read the documentation in each example for more information.

	> python 2-D_continuous_seed.py
	([0.00231032289834628, -0.8551231136684692], 3.9790053502149862)
	> python 2-D_discrete_x.py
	([0, -1.0018982245301231], 3.9999963967436334)
	> python 2-D_continuous_fixed_x.py
	([0.5, -1.0033134899758807], 3.74998902078418)
	> python 5-D_linear_system.py
	([4.052292922336895, 1.6898107367416735, 1.1055896068620388, 4.577893112908056, 6.746541898876046], 0.9010333766161225)
	
The output is a tuple, where the first element is the solution vector (e.g., `[0.002, -0.855]`), and the second element is the solution (e.g., `3.979`).

Note that like many similar optimization algorithms, HS is stochastic, so you will get a slightly different result every time you run it. Because of the stochasticity, I have added the ability to run multiple iterations of HS simultaneously using [Python's multiprocessing module](http://docs.python.org/2/library/multiprocessing.html); you simply specify the number of processes on which to run the specified number of iterations. The resulting solution is the best solution found from all iterations. An optional random seed is available to allow reproducible results.

In general, you will make use of this code in three steps:

1. Implement your own objective function that inherits from `ObjectiveFunctionInterface`.
1. Tune the various input parameters (e.g., `hms`, `hmcr`). These are problem-specific, and the numbers used in the example implementations might not be appropriate for your problem.
1. Run HS.

It will look something like this:
	
```python
from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
class ObjectiveFunction(ObjectiveFunctionInterface):
	#IMPLEMENT ME
if __name__ == '__main__':
	obj_fun = ObjectiveFunction()
	print harmony_search(obj_fun, num_processes, num_iterations)
```

More documentation is provided in **harmony_search.py** and **objective_function_interface.py** and in the examples.

## REFERENCES
[http://harry.me/2011/05/07/neat-algorithms---harmony-search/](http://harry.me/2011/05/07/neat-algorithms---harmony-search/) provides a simple introduction on how HS works. Also, see the [HS Wikipedia entry](http://en.wikipedia.org/wiki/Harmony_search).

HS was first introduced by Geem et al. in 2001:

Z. W. Geem, J. H. Kim, and G. V. Loganathan, "A New Heuristic Optimization Algorithm: Harmony Search," Simulation, vol. 76, no. 2, pp. 60–68, Feb. 2001.

Some modifications and improvements have been suggested to the original algorithm. None of these are implemented here. This package only implements the original algorithm.

* Z. W. Geem, "Improved Harmony Search from Ensemble of Music Players", in Knowledge-Based Intelligent Information and Engineering Systems, B. Gabrys, R. Howlett, and L. Jain, Eds. Springer Berlin / Heidelberg, 2006, pp. 86-93.
* M. Mahdavi, M. Fesanghary, and E. Damangir, "An improved harmony search algorithm for solving optimization problems", Applied Mathematics and Computation, vol. 188, no. 2, pp. 1567-1579, May 2007.
* M. G. H. Omran and M. Mahdavi, "Global-best harmony search", Applied Mathematics and Computation, vol. 198, no. 2, pp. 643-656, May 2008.
* Z. W. Geem and K.-B. Sim, "Parameter-setting-free harmony search algorithm," Applied Mathematics and Computation, vol. 217, no. 8, pp. 3881–3889, Dec. 2010.

## DIFFERENCES
Python optimization packages that implement HS:

* [pyOpt](http://www.pyopt.org/)
* [PyGMO](http://pagmo.sourceforge.net/pygmo/)

pyHarmonySearch differs from other optimization packages in that it relies on no 3rd party software. It is simple and easy-to-use standalone Python code. The primary reason I developed this was because I got tired of having to compile and maintain my own 3rd party packages on a machine running several-year-old software for which I have only user-level access. SciPy is particularly nasty to build, and many scientific packages rely on it. I skirted this issue by implementing HS myself from scratch.