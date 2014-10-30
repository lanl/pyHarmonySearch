#!/usr/bin/env python

"""
    Copyright (c) 2013, Los Alamos National Security, LLC
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Los Alamos National Security, LLC nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
from math import pow
import random
from bisect import bisect_left
from multiprocessing import cpu_count


class ObjectiveFunction(ObjectiveFunctionInterface):

    """
        This is a toy objective function that contains a mixture of continuous and discrete variables.

        Goal:

            maximize -(x^2 + (y+1)^2) + 4
            The maximum is 4 at (0, -1).

        In this implementation, x is a discrete variable with choices ranging from -100 to 100 in increments of 1.
        y is still a continuous variable.

        Warning: Stochastically solving a linear system is dumb. This is just a toy example.
    """

    def __init__(self):
        self._lower_bounds = [None, -1000]
        self._upper_bounds = [None, 1000]
        self._variable = [True, True]
        self._discrete_values = [[x for x in range(-100, 101)], None]

        # define all input parameters
        self._maximize = True  # do we maximize or minimize?
        self._max_imp = 50000  # maximum number of improvisations
        self._hms = 100  # harmony memory size
        self._hmcr = 0.75  # harmony memory considering rate
        self._par = 0.5  # pitch adjusting rate
        self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 10  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only

    def get_fitness(self, vector):
        return -(pow(vector[0], 2) + pow(vector[1] + 1, 2)) + 4

    def get_value(self, i, j=None):
        if self.is_discrete(i):
            if j:
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
            Code courtesy Python bisect module: http://docs.python.org/2/library/bisect.html#searching-sorted-lists
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
    num_iterations = num_processes * 5  # each process does 5 iterations
    print(harmony_search(obj_fun, num_processes, num_iterations))
