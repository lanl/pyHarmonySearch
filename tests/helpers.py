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
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import importlib
import random

from pyharmonysearch import ObjectiveFunctionInterface

# This lives in its own importable module (not a test file or conftest.py) so it can be pickled and sent to
# multiprocessing workers under the spawn/forkserver start methods. pyproject.toml puts tests/ on sys.path.


def set_terminating():
    """Run in a pool worker: set whatever Event the worker's pyharmonysearch module holds."""
    importlib.import_module('pyharmonysearch.harmony_search').terminating.set()


class SumObjective(ObjectiveFunctionInterface):

    """
        Fitness is the sum of the harmony, which makes good and bad harmonies trivial to construct by hand.

        kinds holds one entry per parameter: 'continuous' (in [lower, upper]), 'discrete' (one of values), or 'fixed'.
    """

    def __init__(self, kinds=('continuous',), *, lower=0.0, upper=10.0, values=(0, 1, 2, 3, 4), fixed=5.0,
                 maximize=True, hms=3, max_imp=6, hmcr=0.75, par=0.5, mpap=0.5, mpai=2, seed=None):
        self.kinds = list(kinds)
        self.lower = lower
        self.upper = upper
        self.values = list(values)
        self.fixed = fixed
        self._maximize = maximize
        self.hms = hms
        self.max_imp = max_imp
        self.hmcr = hmcr
        self.par = par
        self.mpap = mpap
        self.mpai = mpai
        self.seed = seed

    def get_fitness(self, vector):
        return sum(vector)

    def get_value(self, i, j=None):
        kind = self.kinds[i]
        if kind == 'fixed':
            return self.fixed
        if kind == 'discrete':
            return self.values[j] if j is not None else random.choice(self.values)
        return random.uniform(self.lower, self.upper)

    def get_index(self, i, v):
        return self.values.index(v)

    def get_num_discrete_values(self, i):
        return len(self.values)

    def get_lower_bound(self, i):
        return self.lower

    def get_upper_bound(self, i):
        return self.upper

    def is_variable(self, i):
        return self.kinds[i] != 'fixed'

    def is_discrete(self, i):
        return self.kinds[i] == 'discrete'

    def get_num_parameters(self):
        return len(self.kinds)

    def use_random_seed(self):
        return self.seed is not None

    def get_random_seed(self):
        return self.seed

    def get_max_imp(self):
        return self.max_imp

    def get_hmcr(self):
        return self.hmcr

    def get_par(self):
        return self.par

    def get_hms(self):
        return self.hms

    def get_mpai(self):
        return self.mpai

    def get_mpap(self):
        return self.mpap

    def maximize(self):
        return self._maximize


class SphereObjective(ObjectiveFunctionInterface):

    """
        Minimize sum((x_i - c_i)^2) over [lower, upper]^n, so the optimum (fitness 0 at center) is inside the bounds.
        With maximize=True the fitness is negated, which puts the maximum at the same point.
    """

    def __init__(self, center=(1.0, -2.0, 3.0), *, lower=-10.0, upper=10.0, maximize=False, hms=10, max_imp=1000,
                 hmcr=0.9, par=0.3, mpap=0.1, seed=0):
        self.center = list(center)
        self.lower = lower
        self.upper = upper
        self._maximize = maximize
        self.hms = hms
        self.max_imp = max_imp
        self.hmcr = hmcr
        self.par = par
        self.mpap = mpap
        self.seed = seed

    def get_fitness(self, vector):
        distance = sum((x - c) ** 2 for x, c in zip(vector, self.center, strict=True))
        return -distance if self._maximize else distance

    def get_value(self, i, j=None):
        return random.uniform(self.lower, self.upper)

    def get_lower_bound(self, i):
        return self.lower

    def get_upper_bound(self, i):
        return self.upper

    def is_variable(self, i):
        return True

    def is_discrete(self, i):
        return False

    def get_num_parameters(self):
        return len(self.center)

    def use_random_seed(self):
        return True

    def get_random_seed(self):
        return self.seed

    def get_max_imp(self):
        return self.max_imp

    def get_hmcr(self):
        return self.hmcr

    def get_par(self):
        return self.par

    def get_hms(self):
        return self.hms

    def get_mpai(self):
        return 0

    def get_mpap(self):
        return self.mpap

    def maximize(self):
        return self._maximize
