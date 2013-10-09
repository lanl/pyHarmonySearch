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
import random
from bisect import bisect_left
from multiprocessing import cpu_count

class ObjectiveFunction(ObjectiveFunctionInterface):
	"""
		This is a 5-D system of equations that comes from http://answers.yahoo.com/question/index?qid=20120830142005AA5dTyg.
		This example is quite a bit more computationally expensive than the others.

		Find values of w, x, y, and z that satisfy the following system of equations:
		
			A + 2B + 3C + 2D = 19.968
			-B + C = -1.15
			2B - 3C + D = 4.624
			3B + C + 2D + E = 22.312
			2D + E = 15.882
		
		First, I transform the system:
		
			A + 2B + 3C + 2D - 19.968 = 0
			-B + C + 1.15 = 0
			2B - 3C + D - 4.624 = 0
			3B + C + 2D + E - 22.312 = 0
			2D + E - 15.882 = 0
		
		I treat this as a minimization problem:
		
			min(sum(abs(A + 2B + 3C + 2D - 19.968) +
					abs(-B + C + 1.15) +
					abs(2B - 3C + D - 4.624) +
					abs(3B + C + 2D + E - 22.312) +
					abs(2D + E - 15.882)))
		
		Thus, we've found the optimal solution if the fitness is zero.
		
		The solution is A = 7.805, B = 1.895, C = 0.745, D = 3.069, E = 9.744.
	"""

	def __init__(self):
		#all variables vary in the range [-100, 100]
		self._lower_bounds = [-100, -100, -100, -100, -100]
		self._upper_bounds = [100, 100, 100, 100, 100]
		self._variable = [True, True, True, True, True]
		
		#define all input parameters
		self._maximize = False #minimize
		self._max_imp = 500000 #maximum number of improvisations
		self._hms = 250 #harmony memory size
		self._hmcr = 0.75 #harmony memory considering rate
		self._par = 0.5 #pitch adjusting rate
		self._mpap = 0.5 #maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only

	def get_fitness(self, vector):
		return abs(vector[0] + 2*vector[1] + 3*vector[2] + 2*vector[3]             - 19.968) + \
			   abs(          -   vector[1] +   vector[2]                           + 1.15) + \
			   abs(            2*vector[1] - 3*vector[2] +   vector[3]             - 4.624) + \
			   abs(            3*vector[1] +   vector[2] + 2*vector[3] + vector[4] - 22.312) + \
			   abs(                                        2*vector[3] + vector[4] - 15.882)

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
	num_processes = cpu_count() - 1 #use number of logical CPUs - 1 so that I have one available for use
	num_iterations = num_processes #each process does 1 iterations
	print(harmony_search(obj_fun, num_processes, num_iterations))
