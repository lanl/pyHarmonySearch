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
from math import cos, pi
from multiprocessing import cpu_count

from pyharmonysearch import ObjectiveFunctionInterface, harmony_search


class ObjectiveFunction(ObjectiveFunctionInterface):

    """
        The Rastrigin function is a standard benchmark for global optimizers. It is a bowl (the x^2 term) overlaid with a
        cosine "egg-crate" ripple that puts a local minimum near every integer point, roughly 11^n of them within the
        bounds. Greedy or local methods get stuck in one of these, so a global method must escape them.

        Goal:

            minimize 10n + sum_i (x_i^2 - 10 cos(2 pi x_i)), where -5.12 <= x_i <= 5.12
            The minimum is 0 at x = (0, ..., 0).

        This example demonstrates a configurable number of dimensions n (a constructor argument, default 10), tuning for
        a multimodal problem, and printing convergence from the harmony history. Plain HS can stall in a neighboring
        local minimum (fitness near 1, with one coordinate near +/-1 instead of 0): once the memory agrees on that
        coordinate, small pitch adjustments can't climb out, and only random selection can. A lower hmcr makes random
        selection more frequent, and a larger max_imp gives it more chances. Note that since all variables are
        continuous, we don't actually need to implement get_index() and get_num_discrete_values().

        Reference: H. Mühlenbein, M. Schomisch, and J. Born, Parallel Computing, 1991.
    """

    def __init__(self, num_parameters=10):
        self._lower_bounds = [-5.12] * num_parameters
        self._upper_bounds = [5.12] * num_parameters
        self._variable = [True] * num_parameters

        # define all input parameters
        self._maximize = False  # minimize
        self._max_imp = 50000  # maximum number of improvisations
        self._hms = 30  # harmony memory size
        self._hmcr = 0.9  # harmony memory considering rate
        self._par = 0.3  # pitch adjusting rate
        self._mpap = 0.02  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only

    def get_fitness(self, vector):
        return 10 * len(vector) + sum(x ** 2 - 10 * cos(2 * pi * x) for x in vector)

    def get_value(self, i, j=None):
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        return False

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return False

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
    num_processes = max(1, cpu_count() - 1)  # leave one logical CPU free, but always use at least one
    num_iterations = num_processes  # each process does 1 iteration
    results = harmony_search(obj_fun, num_processes, num_iterations)
    print(f'Elapsed time: {results.elapsed_time}\nBest harmony: {results.best_harmony}\nBest fitness: {results.best_fitness}')

    # show convergence of the iteration that found the best harmony, using about every 10th of its history snapshots
    best_iteration = next(i for i, memory in enumerate(results.harmony_memories)
                          if (results.best_harmony, results.best_fitness) in memory)
    history = results.harmony_histories[best_iteration]
    print(f'Convergence of iteration {best_iteration}:')
    step = max(1, len(history) // 10)
    for snapshot in [history[k] for k in sorted({*range(0, len(history), step), len(history) - 1})]:
        print(f'  Generation {snapshot["gen"]}: best fitness {min(fitness for _, fitness in snapshot["harmonies"])}')
