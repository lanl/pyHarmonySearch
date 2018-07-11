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

from setuptools import setup

setup(
    name='pyHarmonySearch',
    version='1.4.1',
    description='pyHarmonySearch is a pure Python implementation of the harmony search (HS) global optimization algorithm.',
    long_description="""pyHarmonySearch is a pure Python implementation of the harmony search (HS) global optimization algorithm. HS is a metaheuristic search algorithm that, similar to simulated annealing, tabu, and evolutionary searches, is based on real world phenomena. Specifically, HS mimics a jazz band improvising together. Courtesy `Wikipedia <http://en.wikipedia.org/wiki/Harmony_search>`_:

    In the HS algorithm, each musician (= decision variable) plays (= generates) a note (= a value) for finding a best harmony (= global optimum) all together.

pyHarmonySearch supports both continuous and discrete variables and can take advantage of parallel processing using `Python's built-in multiprocessing module <http://docs.python.org/3.4/library/multiprocessing.html>`_.

For more information on pyHarmonySearch, visit the `GitHub project page <https://github.com/gfairchild/pyHarmonySearch>`_.""",
    author='Geoffrey Fairchild',
    author_email='mail@gfairchild.com',
    maintainer='Geoffrey Fairchild',
    maintainer_email='mail@gfairchild.com',
    url='https://github.com/gfairchild/pyHarmonySearch',
    license='BSD 3-Clause License',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=[
        'pyharmonysearch',
    ],
)
