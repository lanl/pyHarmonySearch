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

from objective_function_interface import ObjectiveFunctionInterface
from math import pow
import random
from bisect import bisect_left

#define all input parameters
maximize = True #do we maximize or minimize?
max_imp = 50000 #maximum number of improvisations
hms = 100 #harmony memory size
hmcr = 0.75 #harmony memory considering rate
par = 0.5 #pitch adjusting rate
mpap = 0.25 #maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
mpai = 2 #maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only

class TestDiscreteObjectiveFunction(ObjectiveFunctionInterface):
	"""
		This is a toy objective function that contains a mixture of continuous and discrete variables.

		Goal:

			maximize -(x^2 + (y+1)^2) + 4
			The maximum is 4 at (0, -1).

		In this implementation, x is a discrete variable with choices ranging from -100 to 100 in increments of 1.
		y is still a continuous variable.
	"""

	def __init__(self):
		self.lower_bounds = [None, -1000]
		self.upper_bounds = [None, 1000]
		self.variable = [True, True]
		self.discrete_values = [[x for x in xrange(-100, 101)], None]

	def fitness(self, vector):
		return -(pow(vector[0], 2) + pow(vector[1] + 1, 2)) + 4

	def get_value(self, i, j=None):
		if self.is_discrete(i):
			if j:
				return self.discrete_values[i][j]
			return self.discrete_values[i][random.randint(0, len(self.discrete_values[i]) - 1)]
		return random.uniform(self.lower_bounds[i], self.upper_bounds[i])

	def lower_bound(self, i):
		"""
			This won't be called except for continuous variables, so we don't need to worry about returning None.
		"""
		return self.lower_bounds[i]
	
	def upper_bound(self, i):
		"""
			This won't be called except for continuous variables.
		"""
		return self.upper_bounds[i]

	def get_num_discrete_values(self, i):
		if self.is_discrete(i):
			return len(self.discrete_values[i])
		return float('+inf')
	
	def get_index(self, i, v):
		"""
			Because self.discrete_values is in sorted order, we can use binary search.
		"""
		return self.binary_search(self.discrete_values[i], v)
	
	def is_variable(self, i):
		return self.variable[i]
	
	def is_discrete(self, i):
		return self.discrete_values[i] is not None
	
	def get_num_parameters(self):
		return len(self.lower_bounds)

	def binary_search(self, a, x, lo=0, hi=None):
		hi = hi if hi is not None else len(a)
		pos = bisect_left(a, x, lo, hi)
		if pos == hi or a[pos] != x:
			raise ValueError('value [%s] not found' % x)
		return pos
