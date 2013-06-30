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

class ObjectiveFunction(ObjectiveFunctionInterface):
	"""
		This is a toy objective function that contains only continuous variables.

		Goal:

			maximize -(x^2 + (y+1)^2) + 4
			The maximum is 4 at (0, -1).

		Note that since all variables are continuous, we don't actually need to implement get_index() and get_num_discrete_values().
	"""

	def __init__(self):
		self._lower_bounds = [-1000, -1000]
		self._upper_bounds = [1000, 1000]
		self._variable = [True, True]
		
		#define all input parameters
		self._maximize = True #do we maximize or minimize?
		self._max_imp = 50000 #maximum number of improvisations
		self._hms = 100 #harmony memory size
		self._hmcr = 0.75 #harmony memory considering rate
		self._par = 0.5 #pitch adjusting rate
		self._mpap = 0.25 #maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
		self._mpai = 2 #maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
		self._random_seed = 8675309 #optional random seed for reproducible results

	def get_fitness(self, vector):
		"""
			maximize -(x^2 + (y+1)^2) + 4
			The maximum is 4 at (0, -1).
		"""
		return -(pow(vector[0], 2) + pow(vector[1] + 1, 2)) + 4

	def get_value(self, i, index=None):
		"""
			Values are returned uniformly at random in their entire range. Since both parameters are continuous, index can be ignored.
		"""
		return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

	def get_lower_bound(self, i):
		return self._lower_bounds[i]
	
	def get_upper_bound(self, i):
		return self._upper_bounds[i]
	
	def is_variable(self, i):
		return self._variable[i]
	
	def is_discrete(self, i):
		#all variables are continuous
		return False
	
	def get_num_parameters(self):
		return len(self._lower_bounds)
	
	def use_random_seed(self):
		return hasattr(self, '_random_seed') and self._random_seed
	
	def get_random_seed(self):
		return self._random_seed
	
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
	num_processes = 1
	num_iterations = 1 #because random_seed is defined, there's no point in running this multiple times
	print harmony_search(obj_fun, num_processes, num_iterations)