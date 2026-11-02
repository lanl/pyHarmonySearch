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
from multiprocessing import cpu_count

from pyharmonysearch import ObjectiveFunctionInterface, harmony_search


class ObjectiveFunction(ObjectiveFunctionInterface):

    """
        This is the 0/1 knapsack problem. Given items that each have a weight and a profit (value), pick the subset of
        items with the largest total profit whose total weight doesn't exceed the knapsack's capacity. Each item is
        either taken whole or left out (hence 0/1), so each variable is a yes/no choice. It's a classic NP-hard
        combinatorial problem: n items have 2^n possible subsets. This instance has 15 items.

        Goal:

            maximize sum(p_i * x_i)
            subject to sum(w_i * x_i) <= 750, x_i in {0, 1}
            The maximum is 1458 (weight 749) at x = (1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1).

        This example demonstrates a purely discrete (binary) problem with a constraint. An overweight selection gets the
        negative of its excess weight as its fitness, so it always loses to any feasible selection (whose profit is >= 0)
        while still being pushed toward feasibility. With only 2^15 possible selections, the instance is small enough to
        brute-force, so the optimum above is known to be the true optimum.

        Reference: instance P07 (from Kreher and Stinson) in John Burkardt's KNAPSACK_01 test data
        (https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html).
    """

    def __init__(self):
        self._capacity = 750
        self._weights = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
        self._profits = [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]
        self._discrete_values = [0, 1]  # every item is either left out (0) or packed (1)

        # define all input parameters
        self._maximize = True  # do we maximize or minimize?
        self._max_imp = 10000  # maximum number of improvisations
        self._hms = 50  # harmony memory size
        self._hmcr = 0.9  # harmony memory considering rate
        self._par = 0.3  # pitch adjusting rate
        self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - unused: there are no continuous variables
        self._mpai = 1  # maximum pitch adjustment index (also defined in pitch_adjustment()) - with only 2 values, 1 is enough to flip the bit

    def get_fitness(self, vector):
        weight = self.get_weight(vector)
        if weight > self._capacity:
            return self._capacity - weight
        return sum(p * x for p, x in zip(self._profits, vector, strict=True))

    def get_weight(self, vector):
        return sum(w * x for w, x in zip(self._weights, vector, strict=True))

    def get_value(self, i, j=None):
        if j is not None:  # j can be 0, so don't just check truthiness
            return self._discrete_values[j]
        return random.choice(self._discrete_values)

    def get_num_discrete_values(self, i):
        return len(self._discrete_values)

    def get_index(self, i, v):
        """
            Each value is its own index.
        """
        return v

    def is_variable(self, i):
        return True

    def is_discrete(self, i):
        # all variables are discrete, so get_lower_bound() and get_upper_bound() are never called
        return True

    def get_num_parameters(self):
        return len(self._weights)

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
    num_iterations = num_processes * 2  # each process does 2 iterations
    results = harmony_search(obj_fun, num_processes, num_iterations)
    print(f'Elapsed time: {results.elapsed_time}\nBest harmony: {results.best_harmony}\nBest fitness: {results.best_fitness}')
    print(f'Selected items: {[i + 1 for i, x in enumerate(results.best_harmony) if x]}')
    print(f'Total weight: {obj_fun.get_weight(results.best_harmony)}')
