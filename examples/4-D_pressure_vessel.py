#!/usr/bin/env python

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
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import random
from bisect import bisect_left
from math import pi
from multiprocessing import cpu_count

from pyharmonysearch import ObjectiveFunctionInterface, harmony_search


class ObjectiveFunction(ObjectiveFunctionInterface):

    """
        The pressure vessel design problem: a cylindrical vessel is capped at both ends by hemispherical heads. Choose
        the shell thickness, head thickness, inner radius, and length of the cylindrical section to minimize the total
        cost of material, forming, and welding. The thicknesses are discrete because steel plate only comes in
        multiples of 0.0625 in. The constraints set minimum shell and head thicknesses relative to the radius (the ASME
        pressure vessel code limits), require a working volume of at least 1,296,000 in^3 (750 ft^3), and cap the
        length at 240 in.

        Goal:

            minimize f = 0.6224*x1*x3*x4 + 1.7781*x2*x3^2 + 3.1661*x1^2*x4 + 19.84*x1^2*x3

            subject to:

                g1 = -x1 + 0.0193*x3 <= 0
                g2 = -x2 + 0.00954*x3 <= 0
                g3 = -pi*x3^2*x4 - (4/3)*pi*x3^3 + 1296000 <= 0
                g4 = x4 - 240 <= 0  (never violated here because x4 <= 200, but kept for fidelity to the literature)

            where x1 = Ts (shell thickness) and x2 = Th (head thickness) are in {0.0625, 0.125, ..., 99 * 0.0625}, and
            x3 = R (inner radius) and x4 = L (length) are continuous with 10 <= R, L <= 200.

            The best known minimum is about 6059.714 at (0.8125, 0.4375, 42.098446, 176.636596). It lies exactly on the
            g1 and g3 boundaries, so rounded values of R and L may be slightly infeasible.

        This example demonstrates mixed discrete and continuous variables and inequality constraints. Harmony search
        has no built-in notion of constraints, so get_fitness() adds a static penalty to infeasible harmonies: a
        constant larger than the cost of any feasible design (so infeasible harmonies always lose to feasible ones),
        plus a term that grows with the violation to steer the search toward the feasible region. g3 is in cubic
        inches, so each violation is normalized before summing. Expect results somewhat above the best known value.

        Reference: E. Sandgren, Journal of Mechanical Design, 1990. Solved with HS in K. S. Lee and Z. W. Geem, Computer
        Methods in Applied Mechanics and Engineering, 2005 (they reported 7198.433; the best known value is from later
        work).
    """

    PENALTY = 1e7  # larger than the cost of any feasible design within the bounds

    def __init__(self):
        thicknesses = [0.0625 * k for k in range(1, 100)]  # sorted, so get_index() can use binary search
        self._lower_bounds = [None, None, 10, 10]
        self._upper_bounds = [None, None, 200, 200]
        self._variable = [True, True, True, True]
        self._discrete_values = [thicknesses, thicknesses, None, None]

        # define all input parameters
        self._maximize = False  # do we maximize or minimize?
        self._max_imp = 10000  # maximum number of improvisations
        self._hms = 10  # harmony memory size
        self._hmcr = 0.8  # harmony memory considering rate
        self._par = 0.4  # pitch adjusting rate
        self._mpap = 0.02  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only

    @staticmethod
    def cost(vector):
        ts, th, r, length = vector
        return 0.6224 * ts * r * length + 1.7781 * th * r ** 2 + 3.1661 * ts ** 2 * length + 19.84 * ts ** 2 * r

    @staticmethod
    def constraints(vector):
        """
            Return the constraint values g1-g4. The design is feasible if all of them are <= 0.
        """
        ts, th, r, length = vector
        return [
            -ts + 0.0193 * r,
            -th + 0.00954 * r,
            -pi * r ** 2 * length - (4 / 3) * pi * r ** 3 + 1296000,
            length - 240,
        ]

    @staticmethod
    def violation(vector):
        """
            Sum of the constraint violations, each normalized so that g3 doesn't dominate the others.
        """
        g1, g2, g3, g4 = ObjectiveFunction.constraints(vector)
        return max(0, g1) + max(0, g2) + max(0, g3) / 1296000 + max(0, g4) / 240

    def is_feasible(self, vector):
        return all(g <= 0 for g in self.constraints(vector))

    def get_fitness(self, vector):
        cost = self.cost(vector)
        if self.is_feasible(vector):
            return cost
        return cost + ObjectiveFunction.PENALTY * (1 + self.violation(vector))

    def get_value(self, i, j=None):
        if self.is_discrete(i):
            if j is not None:  # j can be 0, so don't just check truthiness
                return self._discrete_values[i][j]
            return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    def get_lower_bound(self, i):
        """
            This won't be called except for continuous variables, so we don't need to worry about returning None.
        """
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        """
            This won't be called except for continuous variables.
        """
        return self._upper_bounds[i]

    def get_num_discrete_values(self, i):
        if self.is_discrete(i):
            return len(self._discrete_values[i])
        return float('+inf')

    def get_index(self, i, v):
        """
            Because self.discrete_values is in sorted order, we can use binary search.
        """
        return ObjectiveFunction.binary_search(self._discrete_values[i], v)

    @staticmethod
    def binary_search(a, x):
        """
            Code courtesy Python bisect module: https://docs.python.org/3/library/bisect.html#searching-sorted-lists
        """
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        return self._discrete_values[i] is not None

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return hasattr(self, '_random_seed') and self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize

if __name__ == '__main__':
    obj_fun = ObjectiveFunction()
    num_processes = cpu_count()  # use number of logical CPUs
    num_iterations = num_processes * 8  # each process does 8 iterations
    results = harmony_search(obj_fun, num_processes, num_iterations)
    print(f'Elapsed time: {results.elapsed_time}\nBest harmony: {results.best_harmony}\nBest fitness: {results.best_fitness}')
    print(f'Feasible: {obj_fun.is_feasible(results.best_harmony)}')
    print('Best known: 6059.714 at (0.8125, 0.4375, 42.098446, 176.636596)')
