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

import inspect


class ObjectiveFunctionInterface(object):

    """
        This interface must be implemented by you. This defines the objective function HS optimizes.
    """

    def get_fitness(self, vector):
        """
            Return the objective function value given a solution vector containing each decision variable. In practice,
            vector should be a list of parameters.

            For example, suppose the objective function is (-(x^2 + (y+1)^2) + 4). A possible call to fitness may look like this:

            >>> print obj_fun.fitness([4, 7])
            -76
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_value(self, i, j=None):
        """
            Get a valid value of parameter i. You can return values any way you like - uniformly at random, according to some
            distribution, etc.

            For example, suppose the x parameter in fitness() varies uniformly at random in the range [-1000, 1000]:

            >>> print obj_fun.get_value(0)
            763.406542555
            >>> print obj_fun.get_value(0)
            -80.8100680841

            j is used only for discrete parameters in the pitch adjustment step. j maps to some value the discrete
            parameter can take on. If parameter i is continuous, j should be ignored.

            For example, suppose that a variable z is discrete and can take on the values [-3, -1, 0, 3, 4.5, 6.3, 8, 9, 12]. Also
            suppose that z is the 3rd parameter in the objective function (i.e., i = 2).

            >>> print obj_fun.get_value(2, 1)
            -1
            >>> print obj_fun.get_value(2, 3)
            3
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_index(self, i, v):
        """
            Get the index of the value v of the specified parameter.

            As an example, consider the variable z from get_value() above:

            >>> print obj_fun.get_index(2, 6.3)
            5

            This will only be called for discrete variables in the pitch adjustment step. The behavior here isn't well-defined in the case
            where the possible values for a variable contain non-unique elements.

            For best performance, store discrete values in a sorted list that can be binary searched. Additionally, this list should not
            contain any duplicate values.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_num_discrete_values(self, i):
        """
            Get the number of values possible for the discrete parameter i.

            As an example, consider the variables z and x from get_value() above:

            >>> print get_num_discrete_values(2)
            9
            >>> print get_num_discrete_values(0)
            inf

            This will only be called for discrete variables in the pitch adjustment step. If i is a continuous variable, +inf
            can be returned, but this function might not be implemented for continuous variables, so this shouldn't be
            counted on.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_lower_bound(self, i):
        """
            Return the lower bound of parameter i. Using the example for fitness(), the lower bound for y may be -1000.
            Seeing as y is the 2nd parameter (index 1 in a 0-indexed system), this call may look like the following:

            >>> print obj_fun.lower_bound(1)
            -1000

            This will only be called for continuous variables in the pitch adjustment step.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_upper_bound(self, i):
        """
            Return the upper bound of parameter i.

            This will only be called for continuous variables in the pitch adjustment step.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def is_variable(self, i):
        """
            Return whether or not the parameter at the specified index should be varied by HS. It may be the case that HS should
            only vary certain parameters while others should remain fixed. In the fitness() example, perhaps HS should only vary x.
            This call may look like:

            >>> print obj_fun.is_variable(0)
            True
            >>> print obj_fun.is_variable(1)
            False

            Note that if a parameter is not variable, it should still return a valid value in get_value(). This value can be constant,
            but a valid value must be returned.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def is_discrete(self, i):
        """
            Return whether or not the parameter at the specified index is a discrete parameter. Not all parameters may be continuous.
            This only really matters in the pitch adjustment step of HS. Suppose that x is continuous (e.g., x varies in [-1000, 1000]),
            and y is discrete (e.g., y is only allowed to take on values [-5, 3, 6, 9, 12, 45]):

            >>> print obj_fun.is_discrete(0)
            False
            >>> print obj_fun.is_discrete(1)
            True
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_num_parameters(self):
        """
            Return the number of parameters used by the objective function. Using the example in fitness(), this will be 2.
            A sample call may look like the following:

            >>> print obj_fun.get_num_parameters()
            2
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def use_random_seed(self):
        """
            Return whether or not a random seed should be used. If a random seed is used, the same result will be generated each time (i.e., multiple
            HS iterations will return the same best solution).
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_random_seed(self):
        """
            Return an optional random seed. If use_random_seed() == False, this won't be called.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_max_imp(self):
        """
            Return the maximum number of improvisations. This represents the stopping criterion (i.e., the number of fitness evaluations HS
            performs until search stops).
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_hmcr(self):
        """
            Return the harmony memory considering rate. This represents the proportion of memory consideration calls vs. random selection calls.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_par(self):
        """
            Return the pitch adjusting rate. This represents how often pitch adjustment will occur if memory consideration has already been done.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_hms(self):
        """
            Return the harmony memory size. This represents the size of the vector that stores previously best harmonies.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_mpai(self):
        """
            Return the maximum pitch adjustment index. This determines the range from which pitch adjustment may occur for discrete variables. Also known as
            discrete bandwidth.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def get_mpap(self):
        """
            Return the maximum pitch adjustment proportion. This determines the range from which pitch adjustment may occur for continuous variables. Also known as
            continuous bandwidth.
        """
        raise NotImplementedError(inspect.stack()[0][3])

    def maximize(self):
        """
            Return True if this is a maximization problem, False if minimization problem.
        """
        raise NotImplementedError(inspect.stack()[0][3])
