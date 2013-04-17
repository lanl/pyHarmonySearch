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

class ObjectiveFunctionInterface(object):
	"""
		This interface must be implemented by you and passed to initialize(). This defines the objective function HS optimizes.
	"""

	def fitness(self, vector):
		"""
			Return the objective function value given a solution vector containing each decision variable. In practice,
			vector should be a list of parameters.

			For example, say that the objective function is (-(x^2 + (y+1)^2) + 4). A possible call to fitness may look like this:

			>>> print obj_fun.fitness([4, 7])
			-76
		"""
		raise NotImplementedError
	
	def lower_bound(self, i):
		"""
			Return the lower bound of parameter i. Using the example for fitness(), the lower bound for y may be -1000.
			Seeing as y is the 2nd parameter (index 1 in a 0-indexed system), this call may look like the following:

			>>> print obj_fun.lower_bound(1)
			-1000

			This should be 0-indexed.
		"""
		raise NotImplementedError
	
	def upper_bound(self, i):
		"""
			Return the upper bound of parameter i.
		"""
		raise NotImplementedError

	def is_variable(self, i):
		"""
			Return whether or not the parameter at the specified index should be varied by HS. It may be the case that HS should
			only vary certain parameters to reduce the search space. In the fitness() example, perhaps HS should only vary x. This call
			may look like:

			>>> print obj_fun.is_variable(0)
			True
			>>> print obj_fun.is_variable(1)
			False

			Currently, if a parameter is not variable, it will be fixed at the bound mean: ((lower_bound + upper_bound) / 2).
		"""
		raise NotImplementedError

	def num_parameters(self):
		"""
			Return the number of parameters used by the objective function. Using the example in fitness(), this will be 2.
			A sample call may look like the following:

			>>> print obj_fun.num_parameters()
			2
		"""
		raise NotImplementedError
