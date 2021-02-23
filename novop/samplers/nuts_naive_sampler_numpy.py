#!/usr/env/python
# Copyright 2019 The Centre for Translational Data Science (CTDS) 
# at the University of Sydney. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""

Created by Created by Hadi Afshar. 
24/4/20
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# for extra info map:
DEPTH = 'J'
C_SIZE = 'C'
TRACED_STATES = 'B'  # though it is not B but every state traversed even if it is dropped from B


class NaiveNUTSSampler(object):
    MAX_CONSTRUCTED_DEPTH = 0   # this is just to see how large B grows in practice
    MAX_POSSIBLE_DEPTH = 12     # B is limited to 2 to power this value

    def __init__(self, model, epsilon, DELTA_max=1000):
        self.model = model
        self.epsilon = epsilon
        self.DELTA_max = DELTA_max
        self.extra_info = None

        self.extra_info = None

    def generate_sample(self, current_q):
        # p         --> r
        # theta     --> q
        # L(theta)  --> -U(q)
        # H=p^2/2 + U(q)  --> r^2/2 - L(theta)
        # p(q,p) = e^{-H}   --> p(theta, r) = e^{L(theta) - r^2/2}
        # s~Uniform(0,1) < |J|.e^{-(H-H_0)} = |J|.e^{-H}/e^{-H0}
        #   => s~Uniform(0,1).e^{-H0} < |J|.e^{-H} --> u~(0, e^{L(theta0) - r0^2/2}) < |J|. e^{L(theta) - r^2/2}
        self.extra_info = {TRACED_STATES: 0}  # reset

        p_init = np.array(np.random.normal(size=current_q.shape))  # r0

        # initial hamiltonian
        h0 = 0.5 * p_init.dot(p_init) + self.model.u(current_q)  # r^2/2 - L(theta)

        u = np.exp(-h0) * np.random.uniform()  # u ~ Uniform[0, e^{L(theta0 - r^2/2)}]

        # initialize:
        q_minus = current_q.copy()
        q_plus =  current_q.copy()
        p_minus = p_init.copy()
        p_plus =  p_init.copy()
        j = 0  # dept of the tree
        C = [(current_q, p_init)]
        s = 1

        while s == 1:
            # choose a direction v_j from uniformly [-1, 1]
            dir = int(2 * (np.random.uniform() < 0.5) - 1)
            if dir == -1:  # backward
                q_minus, p_minus, _, _, C_prim, s_prim = self.build_tree(q_minus, p_minus, u, dir, j)
            else:
                _, _, q_plus, p_plus, C_prim, s_prim = self.build_tree(q_plus, p_plus, u, dir, j)

            if s_prim == 1:
                C.extend(C_prim)

            s = s_prim * \
                ((q_plus - q_minus).dot(p_minus) >= 0) * \
                ((q_plus - q_minus).dot(p_plus) >= 0)

            j += 1

            if j > self.MAX_CONSTRUCTED_DEPTH:
                self.MAX_CONSTRUCTED_DEPTH = j
                print("MAX_J: ", self.MAX_CONSTRUCTED_DEPTH)

            if j > self.MAX_POSSIBLE_DEPTH:  # a sanity check so the tree does not grow ad infinit
                # print("Termination due to a very large j=", j)
                # print('q_minus: {}, q_plus: {}'.format(q_minus, q_plus))
                # print('C: ', C, "\n")
                s = 0


        # print('c/b:', len(C)/(2**j))

        # sample from C uniformly:
        self.extra_info[C_SIZE] = len(C)
        self.extra_info[DEPTH] = j

        q, p = C[np.random.randint(low=0, high=len(C))]
        return q

    def build_tree(self, q, p, u, dir, depth):
        # dir = v
        # depth = j
        if depth == 0:
            self.extra_info[TRACED_STATES] += 1

            q_prim, p_prim = self.leapfrog(q.copy(), p.copy(), dir * self.epsilon)

            h = 0.5 * p_prim.dot(p_prim) + self.model.u(q_prim)  # r'^2/2 - L(theta')
            if u <= np.exp(-h):  # exp{L(theta) - r^2/2}
                C_prim = [(q_prim, p_prim)]
            else:
                C_prim = []

            if -h + self.DELTA_max > np.log(u):  # if L(theta' - 1/2 r'.r' > log(u)-DELTA_max)
                s_prim = 1
            else:
                # print('DELTA_max condition!!!!')
                # print("p(q')", np.exp(-h))
                s_prim = 0
            return q_prim, p_prim, q_prim, p_prim, C_prim, s_prim
        else:
            # Recursively build the left and right sub-trees:
            q_minus, p_minus, q_plus, p_plus, C_prim, s_prim = self.build_tree(q, p, u, dir, depth - 1)
            if dir == -1:
                q_minus, p_minus, _, _, C_second, s_second = self.build_tree(q_minus, p_minus, u, dir, depth=depth - 1)
            else:
                _, _, q_plus, p_plus, C_second, s_second = self.build_tree(q_plus, p_plus, u, dir, depth=depth - 1)
                # todo nsend bio + pic to DARE
            delta_q = q_plus - q_minus
            s_prim = s_prim * s_second * (delta_q.dot(p_minus) >= 0) * (delta_q.dot(p_plus) >= 0)
            C_prim.extend(C_second)
            return q_minus, p_minus, q_plus, p_plus, C_prim, s_prim

    def leapfrog(self, q, p, time):
        # half-step:
        p -= 0.5 * time * self.model.partial(q=q)

        # full-step:
        q += time * p

        # half-step:
        p -= 0.5 * time * self.model.partial(q=q)

        return q, p

    @staticmethod
    def name():
        return 'Naive NUTS'



