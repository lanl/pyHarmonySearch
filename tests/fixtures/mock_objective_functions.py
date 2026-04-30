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
from pyharmonysearch import ObjectiveFunctionInterface


class QuadraticObjectiveFunction(ObjectiveFunctionInterface):
    """
    Simple continuous optimization problem with known global minimum.
    
    Objective: Minimize sum(x_i^2)
    Known minimum: 0 at [0, 0, ...]
    
    This implementation is fully configurable and picklable for multiprocessing tests.
    All parameters are continuous variables in the range [-100, 100].
    
    Args:
        num_parameters: Number of parameters in the objective function (default: 2)
        hms: Harmony memory size (default: 10)
        max_imp: Maximum number of improvisations (default: 1000)
        hmcr: Harmony memory considering rate (default: 0.75)
        par: Pitch adjusting rate (default: 0.5)
        mpap: Maximum pitch adjustment proportion for continuous vars (default: 0.25)
        mpai: Maximum pitch adjustment index for discrete vars (default: 2)
        maximize: True for maximization, False for minimization (default: False)
        random_seed: Random seed for deterministic behavior (default: None)
    """
    
    def __init__(self, num_parameters=2, hms=10, max_imp=1000, hmcr=0.75, par=0.5,
                 mpap=0.25, mpai=2, maximize=False, random_seed=None):
        self._num_parameters = num_parameters
        self._lower_bounds = [-100.0] * num_parameters
        self._upper_bounds = [100.0] * num_parameters
        self._hms = hms
        self._max_imp = max_imp
        self._hmcr = hmcr
        self._par = par
        self._mpap = mpap
        self._mpai = mpai
        self._maximize = maximize
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        """Compute sum of squares: sum(x_i^2)."""
        return sum(x**2 for x in vector)
    
    def get_value(self, i, j=None):
        """Return a random value uniformly distributed in [-100, 100]."""
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])
    
    def get_index(self, i, v):
        """Not needed for continuous variables."""
        raise NotImplementedError("get_index() not needed for continuous variables")
    
    def get_num_discrete_values(self, i):
        """Return inf for continuous variables."""
        return float('+inf')
    
    def get_lower_bound(self, i):
        """Return lower bound for parameter i."""
        return self._lower_bounds[i]
    
    def get_upper_bound(self, i):
        """Return upper bound for parameter i."""
        return self._upper_bounds[i]
    
    def is_variable(self, i):
        """All parameters are variable."""
        return True
    
    def is_discrete(self, i):
        """All parameters are continuous."""
        return False
    
    def get_num_parameters(self):
        """Return the number of parameters."""
        return self._num_parameters
    
    def use_random_seed(self):
        """Return whether a random seed should be used."""
        return self._random_seed is not None
    
    def get_random_seed(self):
        """Return the random seed."""
        return self._random_seed
    
    def get_max_imp(self):
        """Return the maximum number of improvisations."""
        return self._max_imp
    
    def get_hmcr(self):
        """Return the harmony memory considering rate."""
        return self._hmcr
    
    def get_par(self):
        """Return the pitch adjusting rate."""
        return self._par
    
    def get_hms(self):
        """Return the harmony memory size."""
        return self._hms
    
    def get_mpai(self):
        """Return the maximum pitch adjustment index."""
        return self._mpai
    
    def get_mpap(self):
        """Return the maximum pitch adjustment proportion."""
        return self._mpap
    
    def maximize(self):
        """Return whether this is a maximization problem."""
        return self._maximize


class DiscreteObjectiveFunction(ObjectiveFunctionInterface):
    """
    Discrete-only optimization problem.
    
    Objective: Minimize sum of absolute values: sum(|x_i|)
    All parameters are discrete, chosen from [-10, -9, ..., 9, 10].
    Known minimum: 0 at [0, 0, ...]
    
    This implementation tests discrete variable handling including get_index()
    and get_num_discrete_values() using binary search for efficiency.
    
    Args:
        num_parameters: Number of parameters in the objective function (default: 2)
        hms: Harmony memory size (default: 10)
        max_imp: Maximum number of improvisations (default: 1000)
        hmcr: Harmony memory considering rate (default: 0.75)
        par: Pitch adjusting rate (default: 0.5)
        mpap: Maximum pitch adjustment proportion for continuous vars (default: 0.25)
        mpai: Maximum pitch adjustment index for discrete vars (default: 3)
        maximize: True for maximization, False for minimization (default: False)
        random_seed: Random seed for deterministic behavior (default: None)
    """
    
    def __init__(self, num_parameters=2, hms=10, max_imp=1000, hmcr=0.75, par=0.5,
                 mpap=0.25, mpai=3, maximize=False, random_seed=None):
        self._num_parameters = num_parameters
        # Create sorted discrete value lists for efficient binary search
        self._discrete_values = [list(range(-10, 11)) for _ in range(num_parameters)]
        self._hms = hms
        self._max_imp = max_imp
        self._hmcr = hmcr
        self._par = par
        self._mpap = mpap
        self._mpai = mpai
        self._maximize = maximize
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        """Compute sum of absolute values: sum(|x_i|)."""
        return sum(abs(x) for x in vector)
    
    def get_value(self, i, j=None):
        """Return a discrete value. If j is provided, return the j-th value."""
        if j is not None:
            return self._discrete_values[i][j]
        return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
    
    def get_index(self, i, v):
        """Get the index of value v for parameter i using binary search."""
        idx = bisect_left(self._discrete_values[i], v)
        if idx != len(self._discrete_values[i]) and self._discrete_values[i][idx] == v:
            return idx
        raise ValueError(f"Value {v} not found in discrete values for parameter {i}")
    
    def get_num_discrete_values(self, i):
        """Return the number of discrete values for parameter i."""
        return len(self._discrete_values[i])
    
    def get_lower_bound(self, i):
        """Not needed for discrete variables."""
        raise NotImplementedError("get_lower_bound() not needed for discrete variables")
    
    def get_upper_bound(self, i):
        """Not needed for discrete variables."""
        raise NotImplementedError("get_upper_bound() not needed for discrete variables")
    
    def is_variable(self, i):
        """All parameters are variable."""
        return True
    
    def is_discrete(self, i):
        """All parameters are discrete."""
        return True
    
    def get_num_parameters(self):
        """Return the number of parameters."""
        return self._num_parameters
    
    def use_random_seed(self):
        """Return whether a random seed should be used."""
        return self._random_seed is not None
    
    def get_random_seed(self):
        """Return the random seed."""
        return self._random_seed
    
    def get_max_imp(self):
        """Return the maximum number of improvisations."""
        return self._max_imp
    
    def get_hmcr(self):
        """Return the harmony memory considering rate."""
        return self._hmcr
    
    def get_par(self):
        """Return the pitch adjusting rate."""
        return self._par
    
    def get_hms(self):
        """Return the harmony memory size."""
        return self._hms
    
    def get_mpai(self):
        """Return the maximum pitch adjustment index."""
        return self._mpai
    
    def get_mpap(self):
        """Return the maximum pitch adjustment proportion."""
        return self._mpap
    
    def maximize(self):
        """Return whether this is a maximization problem."""
        return self._maximize


class MixedObjectiveFunction(ObjectiveFunctionInterface):
    """
    Mixed continuous and discrete optimization problem.
    
    Objective: Minimize x0^2 + |x1| + x2^2 + |x3|
    Even-indexed parameters (0, 2) are continuous in [-50, 50].
    Odd-indexed parameters (1, 3) are discrete in [-20, -19, ..., 19, 20].
    
    This implementation tests both continuous and discrete code paths in the same
    objective function, ensuring proper handling of mixed variable types.
    
    Args:
        num_parameters: Number of parameters (must be even, default: 4)
        hms: Harmony memory size (default: 10)
        max_imp: Maximum number of improvisations (default: 1000)
        hmcr: Harmony memory considering rate (default: 0.75)
        par: Pitch adjusting rate (default: 0.5)
        mpap: Maximum pitch adjustment proportion for continuous vars (default: 0.25)
        mpai: Maximum pitch adjustment index for discrete vars (default: 3)
        maximize: True for maximization, False for minimization (default: False)
        random_seed: Random seed for deterministic behavior (default: None)
    """
    
    def __init__(self, num_parameters=4, hms=10, max_imp=1000, hmcr=0.75, par=0.5,
                 mpap=0.25, mpai=3, maximize=False, random_seed=None):
        if num_parameters % 2 != 0:
            raise ValueError("num_parameters must be even for MixedObjectiveFunction")
        
        self._num_parameters = num_parameters
        # Even indices are continuous, odd indices are discrete
        self._lower_bounds = []
        self._upper_bounds = []
        self._discrete_values = []
        
        for i in range(num_parameters):
            if i % 2 == 0:  # continuous
                self._lower_bounds.append(-50.0)
                self._upper_bounds.append(50.0)
                self._discrete_values.append(None)
            else:  # discrete
                self._lower_bounds.append(None)
                self._upper_bounds.append(None)
                self._discrete_values.append(list(range(-20, 21)))
        
        self._hms = hms
        self._max_imp = max_imp
        self._hmcr = hmcr
        self._par = par
        self._mpap = mpap
        self._mpai = mpai
        self._maximize = maximize
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        """
        Minimize x0^2 + |x1| + x2^2 + |x3| + ...
        Even indices squared, odd indices absolute value.
        """
        fitness = 0.0
        for i, x in enumerate(vector):
            if i % 2 == 0:  # continuous - squared
                fitness += x**2
            else:  # discrete - absolute value
                fitness += abs(x)
        return fitness
    
    def get_value(self, i, j=None):
        """Return a value for parameter i."""
        if self.is_discrete(i):
            if j is not None:
                return self._discrete_values[i][j]
            return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])
    
    def get_index(self, i, v):
        """Get the index of value v for discrete parameter i using binary search."""
        if not self.is_discrete(i):
            raise NotImplementedError("get_index() only for discrete variables")
        
        idx = bisect_left(self._discrete_values[i], v)
        if idx != len(self._discrete_values[i]) and self._discrete_values[i][idx] == v:
            return idx
        raise ValueError(f"Value {v} not found in discrete values for parameter {i}")
    
    def get_num_discrete_values(self, i):
        """Return the number of discrete values for parameter i."""
        if self.is_discrete(i):
            return len(self._discrete_values[i])
        return float('+inf')
    
    def get_lower_bound(self, i):
        """Return lower bound for continuous parameter i."""
        if self.is_discrete(i):
            raise NotImplementedError("get_lower_bound() only for continuous variables")
        return self._lower_bounds[i]
    
    def get_upper_bound(self, i):
        """Return upper bound for continuous parameter i."""
        if self.is_discrete(i):
            raise NotImplementedError("get_upper_bound() only for continuous variables")
        return self._upper_bounds[i]
    
    def is_variable(self, i):
        """All parameters are variable."""
        return True
    
    def is_discrete(self, i):
        """Return whether parameter i is discrete (odd indices)."""
        return self._discrete_values[i] is not None
    
    def get_num_parameters(self):
        """Return the number of parameters."""
        return self._num_parameters
    
    def use_random_seed(self):
        """Return whether a random seed should be used."""
        return self._random_seed is not None
    
    def get_random_seed(self):
        """Return the random seed."""
        return self._random_seed
    
    def get_max_imp(self):
        """Return the maximum number of improvisations."""
        return self._max_imp
    
    def get_hmcr(self):
        """Return the harmony memory considering rate."""
        return self._hmcr
    
    def get_par(self):
        """Return the pitch adjusting rate."""
        return self._par
    
    def get_hms(self):
        """Return the harmony memory size."""
        return self._hms
    
    def get_mpai(self):
        """Return the maximum pitch adjustment index."""
        return self._mpai
    
    def get_mpap(self):
        """Return the maximum pitch adjustment proportion."""
        return self._mpap
    
    def maximize(self):
        """Return whether this is a maximization problem."""
        return self._maximize


class FixedParameterObjectiveFunction(ObjectiveFunctionInterface):
    """
    Optimization problem with both variable and fixed parameters.
    
    Objective: Minimize x0^2 + x1^2
    Parameters x0 and x2 are variable (continuous in [-100, 100]).
    Parameters x1 and x3 are fixed at 0.0.
    
    This tests the is_variable() functionality, ensuring that fixed parameters
    are handled correctly and return constant values.
    
    Args:
        num_parameters: Number of parameters (default: 4)
        hms: Harmony memory size (default: 10)
        max_imp: Maximum number of improvisations (default: 1000)
        hmcr: Harmony memory considering rate (default: 0.75)
        par: Pitch adjusting rate (default: 0.5)
        mpap: Maximum pitch adjustment proportion for continuous vars (default: 0.25)
        mpai: Maximum pitch adjustment index for discrete vars (default: 2)
        maximize: True for maximization, False for minimization (default: False)
        random_seed: Random seed for deterministic behavior (default: None)
    """
    
    def __init__(self, num_parameters=4, hms=10, max_imp=1000, hmcr=0.75, par=0.5,
                 mpap=0.25, mpai=2, maximize=False, random_seed=None):
        self._num_parameters = num_parameters
        self._lower_bounds = [-100.0] * num_parameters
        self._upper_bounds = [100.0] * num_parameters
        # Even-indexed parameters are variable, odd-indexed are fixed
        self._variable = [i % 2 == 0 for i in range(num_parameters)]
        self._fixed_values = [0.0 if not self._variable[i] else None for i in range(num_parameters)]
        self._hms = hms
        self._max_imp = max_imp
        self._hmcr = hmcr
        self._par = par
        self._mpap = mpap
        self._mpai = mpai
        self._maximize = maximize
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        """Minimize sum of squares of variable parameters only."""
        fitness = 0.0
        for i, x in enumerate(vector):
            if self.is_variable(i):
                fitness += x**2
        return fitness
    
    def get_value(self, i, j=None):
        """Return a value for parameter i. Fixed parameters return constant value."""
        if not self.is_variable(i):
            return self._fixed_values[i]
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])
    
    def get_index(self, i, v):
        """Not needed for continuous variables."""
        raise NotImplementedError("get_index() not needed for continuous variables")
    
    def get_num_discrete_values(self, i):
        """Return inf for continuous variables."""
        return float('+inf')
    
    def get_lower_bound(self, i):
        """Return lower bound for parameter i."""
        return self._lower_bounds[i]
    
    def get_upper_bound(self, i):
        """Return upper bound for parameter i."""
        return self._upper_bounds[i]
    
    def is_variable(self, i):
        """Return whether parameter i is variable."""
        return self._variable[i]
    
    def is_discrete(self, i):
        """All parameters are continuous."""
        return False
    
    def get_num_parameters(self):
        """Return the number of parameters."""
        return self._num_parameters
    
    def use_random_seed(self):
        """Return whether a random seed should be used."""
        return self._random_seed is not None
    
    def get_random_seed(self):
        """Return the random seed."""
        return self._random_seed
    
    def get_max_imp(self):
        """Return the maximum number of improvisations."""
        return self._max_imp
    
    def get_hmcr(self):
        """Return the harmony memory considering rate."""
        return self._hmcr
    
    def get_par(self):
        """Return the pitch adjusting rate."""
        return self._par
    
    def get_hms(self):
        """Return the harmony memory size."""
        return self._hms
    
    def get_mpai(self):
        """Return the maximum pitch adjustment index."""
        return self._mpai
    
    def get_mpap(self):
        """Return the maximum pitch adjustment proportion."""
        return self._mpap
    
    def maximize(self):
        """Return whether this is a maximization problem."""
        return self._maximize


class PathologicalObjectiveFunction(ObjectiveFunctionInterface):
    """
    Pathological objective function for robustness testing.
    
    This function tests edge cases and unusual return values:
    - Returns NaN when any parameter equals 0
    - Returns +Inf when sum of squares > 1000
    - Returns -Inf when sum of squares < -1000
    - Otherwise returns normal sum of squares
    
    This tests the algorithm's robustness to:
    - Non-finite fitness values (NaN, Inf, -Inf)
    - Extreme values
    - Boundary conditions
    
    Args:
        num_parameters: Number of parameters (default: 2)
        hms: Harmony memory size (default: 10)
        max_imp: Maximum number of improvisations (default: 100)
        hmcr: Harmony memory considering rate (default: 0.75)
        par: Pitch adjusting rate (default: 0.5)
        mpap: Maximum pitch adjustment proportion for continuous vars (default: 0.25)
        mpai: Maximum pitch adjustment index for discrete vars (default: 2)
        maximize: True for maximization, False for minimization (default: False)
        random_seed: Random seed for deterministic behavior (default: None)
    """
    
    def __init__(self, num_parameters=2, hms=10, max_imp=100, hmcr=0.75, par=0.5,
                 mpap=0.25, mpai=2, maximize=False, random_seed=None):
        self._num_parameters = num_parameters
        self._lower_bounds = [-100.0] * num_parameters
        self._upper_bounds = [100.0] * num_parameters
        self._hms = hms
        self._max_imp = max_imp
        self._hmcr = hmcr
        self._par = par
        self._mpap = mpap
        self._mpai = mpai
        self._maximize = maximize
        self._random_seed = random_seed
    
    def get_fitness(self, vector):
        """
        Return pathological fitness values:
        - NaN if any parameter is exactly 0
        - +Inf if sum of squares > 1000
        - -Inf if sum is negative (shouldn't happen with squares, but for testing)
        - Otherwise sum of squares
        """
        # Check for zero (returns NaN)
        if any(abs(x) < 1e-10 for x in vector):
            return float('nan')
        
        # Compute sum of squares
        fitness = sum(x**2 for x in vector)
        
        # Return Inf for large values
        if fitness > 1000:
            return float('+inf')
        
        # Return -Inf for negative values (shouldn't happen with squares)
        if fitness < 0:
            return float('-inf')
        
        return fitness
    
    def get_value(self, i, j=None):
        """Return a random value uniformly distributed in [-100, 100]."""
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])
    
    def get_index(self, i, v):
        """Not needed for continuous variables."""
        raise NotImplementedError("get_index() not needed for continuous variables")
    
    def get_num_discrete_values(self, i):
        """Return inf for continuous variables."""
        return float('+inf')
    
    def get_lower_bound(self, i):
        """Return lower bound for parameter i."""
        return self._lower_bounds[i]
    
    def get_upper_bound(self, i):
        """Return upper bound for parameter i."""
        return self._upper_bounds[i]
    
    def is_variable(self, i):
        """All parameters are variable."""
        return True
    
    def is_discrete(self, i):
        """All parameters are continuous."""
        return False
    
    def get_num_parameters(self):
        """Return the number of parameters."""
        return self._num_parameters
    
    def use_random_seed(self):
        """Return whether a random seed should be used."""
        return self._random_seed is not None
    
    def get_random_seed(self):
        """Return the random seed."""
        return self._random_seed
    
    def get_max_imp(self):
        """Return the maximum number of improvisations."""
        return self._max_imp
    
    def get_hmcr(self):
        """Return the harmony memory considering rate."""
        return self._hmcr
    
    def get_par(self):
        """Return the pitch adjusting rate."""
        return self._par
    
    def get_hms(self):
        """Return the harmony memory size."""
        return self._hms
    
    def get_mpai(self):
        """Return the maximum pitch adjustment index."""
        return self._mpai
    
    def get_mpap(self):
        """Return the maximum pitch adjustment proportion."""
        return self._mpap
    
    def maximize(self):
        """Return whether this is a maximization problem."""
        return self._maximize
