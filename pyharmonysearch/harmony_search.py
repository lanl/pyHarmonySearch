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


def harmony_search(objective_function, num_processes, num_iterations):
    """
        Here, we use multiprocessing.Pool to do multiple harmony searches simultaneously. Since HS is stochastic (unless random_seed is set),
        multiple runs can find different results. Here, we run the specified number of iterations on the specified number of processes
        and return a tuple: (solution_vector, fitness)
    """
    terminating_event = Event()
    pool = Pool(num_processes, initializer=initializer, initargs=(terminating_event,))
    try:
        results = [pool.apply_async(worker, args=(objective_function,)) for i in range(num_iterations)]
        pool.close()  # no more tasks will be submitted to the pool
        pool.join()  # wait for all tasks to finish before moving on

        # find best harmony from all iterations and output
        best_harmony = None
        best_fitness = float('-inf') if objective_function.maximize() else float('+inf')
        for result in results:
            harmony, fitness = result.get()  # multiprocessing.pool.AsyncResult is returned for each process, so we need to call get() to pull out the value
            if (objective_function.maximize() and fitness > best_fitness) or (not objective_function.maximize() and fitness < best_fitness):
                best_harmony = harmony
                best_fitness = fitness
        return best_harmony, best_fitness
    except KeyboardInterrupt:
        pool.terminate()


def initializer(terminating_event):
    """
        This guy uses a global multiprocessing.Event to prevent new processes from spawning after a KeyboardInterrupt is first caught. This idea comes
        from http://stackoverflow.com/questions/14579474/multiprocessing-pool-spawning-new-childern-after-terminate-on-linux-python2-7.
    """
    global terminating
    terminating = terminating_event


def worker(objective_function):
    """
        This is just a dummy function to make multiprocessing work with a class. It also checks/sets the global multiprocessing.Event to prevent
        new processes from starting work on a KeyboardInterrupt.
    """
    try:
        if not terminating.is_set():
            hs = HarmonySearch(objective_function)
            return hs.run()
    except KeyboardInterrupt:
        terminating.set()  # set the Event to true to prevent the other processes from doing any work
        return


class HarmonySearch(object):

    """
        This class implements the harmony search (HS) global optimization algorithm. In general, what you'll do is this:

        1. Implement an objective function that inherits from ObjectiveFunctionInterface.
        2. Initialize HarmonySearch with this objective function (e.g., hs = HarmonySearch(objective_function)).
        3. Run HarmonySearch (e.g., solution_vector, solution = hs.run()).
    """

    def __init__(self, objective_function):
        """
            Initialize HS with the specified objective function. Note that this objective function must implement ObjectiveFunctionInterface.
        """
        self._obj_fun = objective_function

    def run(self):
        """
            This is the main HS loop. It initializes the harmony memory and then continually generates new harmonies
            until the stopping criterion (max_imp iterations) is reached.
        """
        # set optional random seed
        if self._obj_fun.use_random_seed():
            random.seed(self._obj_fun.get_random_seed())

        # harmony_memory stores the best hms harmonies
        self._harmony_memory = list()

        # fill harmony_memory using random parameter values
        self._initialize()

        # create max_imp improvisations
        num_imp = 0
        while(num_imp < self._obj_fun.get_max_imp()):
            # generate new harmony
            solution_vector = list()
            for i in range(0, self._obj_fun.get_num_parameters()):
                if random.random() < self._obj_fun.get_hmcr():
                    self._memory_consideration(solution_vector, i)
                    if random.random() < self._obj_fun.get_par():
                        self._pitch_adjustment(solution_vector, i)
                else:
                    self._random_selection(solution_vector, i)
            solution_vector.append(self._obj_fun.get_fitness(solution_vector))

            self._update_harmony_memory(solution_vector)

            num_imp += 1

        # return best harmony
        best_harmony = None
        best_fitness = float('-inf') if self._obj_fun.maximize() else float('+inf')
        for harmony in self._harmony_memory:
            if (self._obj_fun.maximize() and harmony[-1] > best_fitness) or (not self._obj_fun.maximize() and harmony[-1] < best_fitness):
                best_harmony = harmony[:-1]
                best_fitness = harmony[-1]
        return best_harmony, best_fitness

    def _initialize(self):
        """
            Initialize harmony_memory, the matrix (list of lists) containing the various harmonies (solution vectors). Note
            that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
            merely stores previous harmonies.
        """
        for i in range(0, self._obj_fun.get_hms()):
            solution_vector = list()
            for j in range(0, self._obj_fun.get_num_parameters()):
                self._random_selection(solution_vector, j)
            solution_vector.append(self._obj_fun.get_fitness(solution_vector))
            self._harmony_memory.append(solution_vector)

    def _random_selection(self, solution_vector, i):
        """
            Choose a note according to get_value(). Remember that even if a note is not variable, get_value() must still
            return a valid value.
        """
        solution_vector.append(self._obj_fun.get_value(i))

    def _memory_consideration(self, solution_vector, i):
        """
            Randomly choose a note previously played.
        """
        memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
        solution_vector.append(self._harmony_memory[memory_index][i])

    def _pitch_adjustment(self, solution_vector, i):
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
                current_index = self._obj_fun.get_index(i, solution_vector[i])
                # discrete variable
                if random.random() < 0.5:
                    # adjust pitch down
                    solution_vector[i] = self._obj_fun.get_value(i, current_index - random.randint(0, min(self._obj_fun.get_mpai(), current_index)))
                else:
                    # adjust pitch up
                    solution_vector[i] = self._obj_fun.get_value(i, current_index + random.randint(0, min(self._obj_fun.get_mpai(), self._obj_fun.get_num_discrete_values(i) - current_index - 1)))
            else:
                # continuous variable
                if random.random() < 0.5:
                    # adjust pitch down
                    solution_vector[i] -= (solution_vector[i] - self._obj_fun.get_lower_bound(i)) * random.random() * self._obj_fun.get_mpap()
                else:
                    # adjust pitch up
                    solution_vector[i] += (self._obj_fun.get_upper_bound(i) - solution_vector[i]) * random.random() * self._obj_fun.get_mpap()

    def _update_harmony_memory(self, solution_vector):
        """
            Update the harmony memory if necessary with the given harmony. If the given harmony is better than the worst
            harmony in memory, replace it. This function doesn't allow duplicate harmonies in memory.
        """
        if solution_vector not in self._harmony_memory:
            worst_index = None
            worst_fitness = float('+inf') if self._obj_fun.maximize() else float('-inf')
            for i, harmony in enumerate(self._harmony_memory):
                if (self._obj_fun.maximize() and harmony[-1] < worst_fitness) or (not self._obj_fun.maximize() and harmony[-1] > worst_fitness):
                    worst_index = i
                    worst_fitness = harmony[-1]
            if (self._obj_fun.maximize() and solution_vector[-1] > worst_fitness) or (not self._obj_fun.maximize() and solution_vector[-1] < worst_fitness):
                self._harmony_memory[worst_index] = solution_vector
