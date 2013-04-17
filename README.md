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
This code does not rely on any 3rd party software. It only requires Python 2.7 or higher.

## USING THIS CODE
To run the sample code, simply type the following:

	> python harmony_search.py
	[0.01672298894632301, -1.116479132293307] 3.9861529533808993
	
The output is the solution `(3.9862)` appended to the solution vector `[0.0167, -1.1165]`.

Note that like many similar optimization algorithms, HS is stochastic in nature. Thus, you will get a slightly different result every time you run it.

This HS implementation only works on continuous variables (e.g., x is in the range `[-5, 5]`). It may be possible to allow for discrete variables (e.g., x is chosen from the set `(3, 4, 6, 8, 9)`), but the pitch adjustment step will need to be modified.

The structure of the code is simple. There are three components:
		
* harmony_search.py - The Python code that implements the HS algorithm.
* objective_function_interface.py - The "interface" that you need to implement that defines the objective function.
* test_objective_function.py - Example objective function implementation. This test objective function maximizes the function `(-(x^2 + (y+1)^2) + 4)`. The answer is `4` at the point `(0, -1)`.

In general, you will make use of this code in three steps:

1. Implement your own objective function (inheriting from `ObjectiveFunctionInterface`).
1. Ensure that harmony_search.py calls your objective function. This occurs where `obj_fun` is assigned above `main()`.
1. Tune the various input parameters in harmony_search.py. These are problem-specific, and the defaults might not be appropriate for your problem.

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

pyHarmonySearch differs from other optimization packages in that it relies on no 3rd party software. It is simple and easy-to-use standalone Python code. The primary reason I developed this was because I got tired of trying to compile and maintain my own 3rd party packages. SciPy is particularly nasty to install from scratch without root access on a machine running several-year-old software, and many scientific packages rely on it. I skirted this issue by implementing HS myself from scratch.
