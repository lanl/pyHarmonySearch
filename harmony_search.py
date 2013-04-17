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

import random
from test_objective_function import ObjectiveFunction

#define all input parameters
maximize = True #do we maximize or minimize?
max_imp = 50000 #maximum number of improvisations
hms = 100 #harmony memory size
hmcr = 0.75 #harmony memory considering rate
par = 0.5 #pitch adjusting rate
mpap = 0.25 #maximum pitch adjustment proportion (new parameter defined in pitch_adjustment())

#harmony_memory stores the best hms harmonies
harmony_memory = list()

#make sure your objective function class gets instantiated here
obj_fun = ObjectiveFunction()

def main():
	"""
		This is the main HS loop. It initializes the harmony memory and then continually generates new harmonies
		until the stopping criterion (max_imp iterations) is reached.
	"""
	initialize()

	#create max_imp improvisations
	num_imp = 0
	while(num_imp < max_imp):
		#generate new harmony
		solution_vector = list()
		for i in range(0, obj_fun.num_parameters()):
			if random.random() < hmcr:
				memory_consideration(solution_vector, i)
				if random.random() < par:
					pitch_adjustment(solution_vector, i)
			else:
				random_selection(solution_vector, i)
		solution_vector.append(obj_fun.fitness(solution_vector))
		
		update_harmony_memory(solution_vector)

		num_imp += 1
	
	#print out best harmony
	best_index = None
	best_fitness = float('-inf') if maximize else float('+inf')
	for i, harmony in enumerate(harmony_memory):
		if (maximize and harmony[-1] > best_fitness) or (not maximize and harmony[-1] < best_fitness):
			best_index = i
			best_fitness = harmony[-1]
	print harmony_memory[best_index][:-1], harmony_memory[best_index][-1]

def initialize():
	"""
		Initialize harmony_memory, the matrix (list of lists) containing the various harmonies (solution vectors). Note
		that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
		merely stores previous harmonies.
	"""
	for i in range(0, hms):
		solution_vector = list()
		for j in range(0, obj_fun.num_parameters()):
			random_selection(solution_vector, j)
		solution_vector.append(obj_fun.fitness(solution_vector))
		harmony_memory.append(solution_vector)

def random_selection(solution_vector, i):
	"""
		If variable, the next note is randomly selected from the bounds. If it's not variable, it arbitrarily gets assigned as
		the mean of the lower and upper bounds.
	"""
	if(obj_fun.is_variable(i)):
		solution_vector.append(random.uniform(obj_fun.lower_bound(i), obj_fun.upper_bound(i)))
	else:
		solution_vector.append((obj_fun.lower_bound(i) + obj_fun.upper_bound(i)) / 2.0)

def memory_consideration(solution_vector, i):
	"""
		Randomly choose a note previously played.
	"""
	memory_index = random.randint(0, hms - 1)
	solution_vector.append(harmony_memory[memory_index][i])

def pitch_adjustment(solution_vector, i):
	"""
		If variable, randomly adjust the pitch up or down by some amount. The amount is defined as:

		(upper_bound - current_value) * rand() * mpap
			or
		(current_value - lower_bound) * rand() * mpap

		The probability of adjusting the pitch either up or down is fixed at 0.5. The maximum pitch adjustment proportion
		determines the maximum amount the pitch may change. For example, suppose that it is decided via coin flip that
		the pitch will be adjusted down. Also suppose that mpap is set to 0.25. This means that the maximum value the pitch
		can be dropped will be 25% of the difference between the lower bound and the current pitch.
	"""
	if(obj_fun.is_variable(i)):
		if random.random() < 0.5:
			#adjust pitch down
			solution_vector[i] = (solution_vector[i] - obj_fun.lower_bound(i)) * random.random() * mpap
		else:
			#adjust pitch up
			solution_vector[i] = (obj_fun.upper_bound(i) - solution_vector[i]) * random.random() * mpap
	
def update_harmony_memory(solution_vector):
	"""
		Update the harmony memory if necessary with the given harmony. If the given harmony is better than the worst
		harmony in memory, replace it. This function doesn't allow duplicate harmonies in memory.
	"""
	if solution_vector not in harmony_memory:
		worst_index = None
		worst_fitness = float('+inf') if maximize else float('-inf')
		for i, harmony in enumerate(harmony_memory):
			if (maximize and harmony[-1] < worst_fitness) or (not maximize and harmony[-1] > worst_fitness):
				worst_index = i
				worst_fitness = harmony[-1]
		if (maximize and solution_vector[-1] > worst_fitness) or (not maximize and solution_vector[-1] < worst_fitness):
			harmony_memory[worst_index] = solution_vector

if __name__ == '__main__':
	main()
