# pyHarmonySearch

## AUTHORS
Geoffrey Fairchild
* [http://www.gfairchild.com/](http://www.gfairchild.com/)
* [https://github.com/gfairchild](https://github.com/gfairchild)
* [http://www.linkedin.com/in/gfairchild/](http://www.linkedin.com/in/gfairchild/)

## LICENSE
This software is licensed under the [BSD 3-Clause License](http://opensource.org/licenses/BSD-3-Clause). Please refer to the separate LICENSE.txt file for the exact text of the license. You are obligated to give attribution if you use this code.

## ABOUT
pyHarmonySearch implements the harmony search (HS) global optimization algorithm in Python. HS is a metaheuristic search algorithm that, similar to simulated annealing, tabu, and evolutionary searches, is based on real world phenomena. Specifically, HS mimics a jazz band improvising together. Courtesy [Wikipedia](http://en.wikipedia.org/wiki/Harmony_search):

> In the HS algorithm, each musician (= decision variable) plays (= generates) a note (= a value) for finding a best harmony (= global optimum) all together.

## REQUIREMENTS
This code does not rely on any 3rd party software. It only requires Python 2.7 or higher. It may run on earlier versions of Python and Python 3 via 2to3, but I haven't tested it.

## USING THIS CODE
To run the sample code, simply type the following:

	> python harmony_search.py test_continuous_params TestContinuousObjectiveFunction
	[0.00231032289834628, -0.8551231136684692] 3.97900535021
	> python harmony_search.py test_discrete_params TestDiscreteObjectiveFunction
	[0, -0.999840357461153] 3.99999997451
	> python harmony_search.py test_continuous_fixed_x_params TestContinuousFixedXObjectiveFunction
	[0.5, -0.9661388447184436] 3.74885342216
	
The output is the solution (e.g., `3.979`) appended to the solution vector (e.g., `[0.002, -0.855]`).

Note that like many similar optimization algorithms, HS is stochastic in nature. Thus, you will get a slightly different result every time you run it. An optional `random_seed` parameter is available to allow reproducible results.

This HS implementation allows both continuous variables (e.g., x is in the range `[-5, 5]`) and discrete variables (e.g., x is chosen from the set `(3, 4, 6, 8, 9)`).

The structure of this project is as follows:

* harmony_search.py - The Python implementation of the HS algorithm.
* objective_function_interface.py - The "interface" that your objective function needs to implement.
* test_continuous_params.py - Example objective function implementation where both variables are continuous. The test objective function maximizes the function `(-(x^2 + (y+1)^2) + 4)`. The answer is `4` at the point `(0, -1)`. This implementation uses `random_seed`, so the same result will be returned every run.
* test_discrete_params.py - Example objective function identical to test_continuous_params.py, only parameter `x` is discrete and `y` is continuous. `random_seed` is not set, so successive runs will yield different results.
* test_continuous_fixed_x_params.py - Example objective function identical to test_continuous_params.py, only parameter `x` is not variable and is fixed at `0.5`. `random_seed` is not set.

In general, you will make use of this code in two steps:

1. Implement your own params file, complete with objective function that inherits from `ObjectiveFunctionInterface`.
1. Tune the various input parameters (e.g., `hms`, `hmcr`). These are problem-specific, and the numbers used in the example implementations might not be appropriate for your problem.

More documentation is provided in harmony_search.py and objective_function_interface.py.

## REFERENCES
[http://harry.me/2011/05/07/neat-algorithms---harmony-search/](http://harry.me/2011/05/07/neat-algorithms---harmony-search/) provides a simple introduction on how HS works. Also, see the [HS Wikipedia entry](http://en.wikipedia.org/wiki/Harmony_search).

Harmony search was first introduced by Geem et al. in 2001:

Z. W. Geem, J. H. Kim, and G. V Loganathan, "A New Heuristic Optimization Algorithm: Harmony Search", Simulation, vol. 76, no. 2, pp. 60-68, Feb. 2001.
	
Some modifications and improvements have been suggested to the original algorithm. None of these are implemented here. This package only implements the original algorithm.

* Z. W. Geem, "Improved Harmony Search from Ensemble of Music Players", in Knowledge-Based Intelligent Information and Engineering Systems, B. Gabrys, R. Howlett, and L. Jain, Eds. Springer Berlin / Heidelberg, 2006, pp. 86-93.
* M. Mahdavi, M. Fesanghary, and E. Damangir, "An improved harmony search algorithm for solving optimization problems", Applied Mathematics and Computation, vol. 188, no. 2, pp. 1567-1579, May 2007.
* M. G. H. Omran and M. Mahdavi, "Global-best harmony search", Applied Mathematics and Computation, vol. 198, no. 2, pp. 643-656, May 2008.

## DIFFERENCES
Python optimization packages that implement HS:

* [pyOpt](http://www.pyopt.org/)
* [PyGMO](http://pagmo.sourceforge.net/pygmo/)

pyHarmonySearch differs from other optimization packages in that it relies on no 3rd party software. It is simple and easy-to-use standalone Python code. The primary reason I developed this was because I got tired of having to compile and maintain my own 3rd party packages on a machine running several-year-old software for which I have only user-level access. SciPy is particularly nasty to build, and many scientific packages rely on it. I skirted this issue by implementing HS myself from scratch.
