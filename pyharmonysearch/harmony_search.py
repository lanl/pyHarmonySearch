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
from multiprocessing import Pool, Event
from datetime import datetime
from collections import namedtuple

# Note: We use a global multiprocessing.Event to deal with a KeyboardInterrupt. This idea comes from
# http://stackoverflow.com/questions/14579474/multiprocessing-pool-spawning-new-childern-after-terminate-on-linux-python2-7.
# This is not necessary when running under Python 3, but to keep 2.7 compatability, I'm leaving it in.
terminating = Event()

# HarmonySearchResults is a struct-like object that we'll use to attach the results of the search.
# namedtuples are lightweight and trivial to extend should more results be desired in the future. Right now, we're just
# keeping track of the total elapsed clock time, the best harmony found, the fitness for that harmony, and the harmony memory,
# which allows you to see the top harmonies.
HarmonySearchResults = namedtuple('HarmonySearchResults', ['elapsed_time', 'best_harmony', 'best_fitness', 'harmony_memories', 'harmony_histories'])


def harmony_search(objective_function, num_processes, num_iterations, initial_harmonies=None):
    """
        Here, we use multiprocessing.Pool to do multiple harmony searches simultaneously. Since HS is stochastic (unless random_seed is set),
        multiple runs can find different results. We run the specified number of iterations on the specified number of processes and return
        an instance of HarmonySearchResults.
    """
    pool = Pool(num_processes)
    try:
        start = datetime.now()
        pool_results = [pool.apply_async(worker, args=(objective_function, initial_harmonies,)) for i in range(num_iterations)]
        pool.close()  # no more tasks will be submitted to the pool
        pool.join()  # wait for all tasks to finish before moving on
        end = datetime.now()
        elapsed_time = end - start

        # find best harmony from all iterations
        best_harmony = None
        best_fitness = float('-inf') if objective_function.maximize() else float('+inf')
        harmony_memories = list()
        harmony_histories = list()
        for result in pool_results:
            harmony, fitness, harmony_memory, harmony_history = result.get()  # multiprocessing.pool.AsyncResult is returned for each process, so we need to call get() to pull out the value
            if (objective_function.maximize() and fitness > best_fitness) or (not objective_function.maximize() and fitness < best_fitness):
                best_harmony = harmony
                best_fitness = fitness
            harmony_memories.append(harmony_memory)
            harmony_histories.append(harmony_history)

        return HarmonySearchResults(elapsed_time=elapsed_time, best_harmony=best_harmony, best_fitness=best_fitness,\
                                    harmony_memories=harmony_memories, harmony_histories=harmony_histories)
    except KeyboardInterrupt:
        pool.terminate()


def worker(objective_function, initial_harmonies=None):
    """
        This is just a dummy function to make multiprocessing work with a class. It also checks/sets the global multiprocessing.Event to prevent
        new processes from starting work on a KeyboardInterrupt.
    """
    try:
        if not terminating.is_set():
            hs = HarmonySearch(objective_function)
            return hs.run(initial_harmonies=initial_harmonies)
    except KeyboardInterrupt:
        terminating.set()  # set the Event to true to prevent the other processes from doing any work


class HarmonySearch(object):

    """
        This class implements the harmony search (HS) global optimization algorithm. In general, what you'll do is this:

        1. Implement an objective function that inherits from ObjectiveFunctionInterface.
        2. Initialize HarmonySearch with this objective function (e.g., hs = HarmonySearch(objective_function)).
        3. Run HarmonySearch (e.g., results = hs.run()).
    """

    def __init__(self, objective_function):
        """
            Initialize HS with the specified objective function. Note that this objective function must implement ObjectiveFunctionInterface.
        """
        self._obj_fun = objective_function

    def run(self, initial_harmonies=None):
        """
            This is the main HS loop. It initializes the harmony memory and then continually generates new harmonies
            until the stopping criterion (max_imp iterations) is reached.
        """
        # set optional random seed
        if self._obj_fun.use_random_seed():
            random.seed(self._obj_fun.get_random_seed())

        # harmony_memory stores the best hms harmonies
        self._harmony_memory = list()

        # harmony_history stores all hms harmonies every nth improvisations (i.e., one 'generation')
        self._harmony_history = list()

        # fill harmony_memory using random parameter values by default, but with initial_harmonies if provided
	self._initialize(initial_harmonies)

        # create max_imp improvisations
        generation = 0
        num_imp = 0
        while(num_imp < self._obj_fun.get_max_imp()):
            # generate new harmony
            harmony = list()
            for i in range(0, self._obj_fun.get_num_parameters()):
                if random.random() < self._obj_fun.get_hmcr():
                    self._memory_consideration(harmony, i)
                    if random.random() < self._obj_fun.get_par():
                        self._pitch_adjustment(harmony, i)
                else:
                    self._random_selection(harmony, i)
            fitness = self._obj_fun.get_fitness(harmony)
            self._update_harmony_memory(harmony, fitness)
            num_imp += 1

            # save harmonies every nth improvisations (i.e., one 'generation')
            if num_imp % self._obj_fun.get_hms() == 0:
                generation += 1
                harmony_list = {'gen': generation, 'harmonies': self._harmony_memory}
                self._harmony_history.append(harmony_list)

        # return best harmony
        best_harmony = None
        best_fitness = float('-inf') if self._obj_fun.maximize() else float('+inf')
        for harmony, fitness in self._harmony_memory:
            if (self._obj_fun.maximize() and fitness > best_fitness) or (not self._obj_fun.maximize() and fitness < best_fitness):
                best_harmony = harmony
                best_fitness = fitness
        return best_harmony, best_fitness, self._harmony_memory, self._harmony_history

    def _initialize(self, initial_harmonies=None):
        """
            Initialize harmony_memory, the matrix (list of lists) containing the various harmonies (solution vectors). Note
            that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
            merely stores previous harmonies.

            If harmonies are provided, then use them instead of randomly initializing them.

            Populate harmony_history with initial harmony memory.
        """
        if initial_harmonies is not None:
            # verify that the initial harmonies are provided correctly

            if len(initial_harmonies) != self._obj_fun.get_hms():
                raise ValueError('Number of initial harmonies does not equal to the harmony memory size.')
            
            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_harmonies)):
                num_parameters_initial_harmonies = len(initial_harmonies[i])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial harmonies does not match that defined.')
        else:
            initial_harmonies = list()
            for i in range(0, self._obj_fun.get_hms()):
                harmony = list()
                for j in range(0, self._obj_fun.get_num_parameters()):
                    self._random_selection(harmony, j)
                initial_harmonies.append(harmony)

        for i in range(0, self._obj_fun.get_hms()):
            fitness = self._obj_fun.get_fitness(initial_harmonies[i])
            self._harmony_memory.append((initial_harmonies[i], fitness))

	harmony_list = {'gen': 0, 'harmonies': self._harmony_memory}
	self._harmony_history.append(harmony_list)

    def _random_selection(self, harmony, i):
        """
            Choose a note according to get_value(). Remember that even if a note is not variable, get_value() must still
            return a valid value.
        """
        harmony.append(self._obj_fun.get_value(i))

    def _memory_consideration(self, harmony, i):
        """
            Randomly choose a note previously played.
        """
        memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
        harmony.append(self._harmony_memory[memory_index][0][i])

    def _pitch_adjustment(self, harmony, i):
        """
            If variable, randomly adjust the pitch up or down by some amount. This is the only place in the algorithm where there
            is an explicit difference between continuous and discrete variables.

            The probability of adjusting the pitch either up or down is fixed at 0.5. The maximum pitch adjustment proportion (mpap)
            and maximum pitch adjustment index (mpai) determine the maximum amount the pitch may change for continuous and discrete
            variables, respectively.

            For example, suppose that it is decided via coin flip that the pitch will be adjusted down. Also suppose that mpap is set to 0.25.
            This means that the maximum value the pitch can be dropped will be 25% of the difference between the lower bound and the current
            pitch. mpai functions similarly, only it relies on indices of the possible values instead.
        """
        if(self._obj_fun.is_variable(i)):
            if self._obj_fun.is_discrete(i):
                current_index = self._obj_fun.get_index(i, harmony[i])
                # discrete variable
                if random.random() < 0.5:
                    # adjust pitch down
                    harmony[i] = self._obj_fun.get_value(i, current_index - random.randint(0, min(self._obj_fun.get_mpai(), current_index)))
                else:
                    # adjust pitch up
                    harmony[i] = self._obj_fun.get_value(i, current_index + random.randint(0, min(self._obj_fun.get_mpai(), self._obj_fun.get_num_discrete_values(i) - current_index - 1)))
            else:
                # continuous variable
                if random.random() < 0.5:
                    # adjust pitch down
                    harmony[i] -= (harmony[i] - self._obj_fun.get_lower_bound(i)) * random.random() * self._obj_fun.get_mpap()
                else:
                    # adjust pitch up
                    harmony[i] += (self._obj_fun.get_upper_bound(i) - harmony[i]) * random.random() * self._obj_fun.get_mpap()

    def _update_harmony_memory(self, considered_harmony, considered_fitness):
        """
            Update the harmony memory if necessary with the given harmony. If the given harmony is better than the worst
            harmony in memory, replace it. This function doesn't allow duplicate harmonies in memory.
        """
        if (considered_harmony, considered_fitness) not in self._harmony_memory:
            worst_index = None
            worst_fitness = float('+inf') if self._obj_fun.maximize() else float('-inf')
            for i, (harmony, fitness) in enumerate(self._harmony_memory):
                if (self._obj_fun.maximize() and fitness < worst_fitness) or (not self._obj_fun.maximize() and fitness > worst_fitness):
                    worst_index = i
                    worst_fitness = fitness
            if (self._obj_fun.maximize() and considered_fitness > worst_fitness) or (not self._obj_fun.maximize() and considered_fitness < worst_fitness):
                self._harmony_memory[worst_index] = (considered_harmony, considered_fitness)
