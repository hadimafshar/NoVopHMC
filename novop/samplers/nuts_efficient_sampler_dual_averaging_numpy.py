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
from tqdm import tqdm


class EffectiveNUTSSamplerWithDualAveraging(object):
    MAX_POSSIBLE_DEPTH = 12
    MAX_J = 0

    def __init__(self, model,  # model.u() is -\mathbb{L}() in NUTS paper
                 init_q,  # starting point for adaptively selecting epsilon
                 num_adapt_iterations,  # M_adapt
                 target_mean_acceptance_prob_delta=0.5,  # delta
                 DELTA_max=1000):
        self.model = model

        assert 0 < target_mean_acceptance_prob_delta < 1
        self.delta = target_mean_acceptance_prob_delta

        self.DELTA_max = DELTA_max

        self.epsilon = self.find_reasonable_epsilon(init_q.copy())
        print("init reasonable epsilon:", self.epsilon)

        self.mu = np.log(10 * self.epsilon)
        self.epsilon_bar = 1.0
        self.H_bar = 0.0
        self.gamma = 0.05
        self.t0 = 10
        self.kappa = 0.75

        current_q = init_q.copy()
        for m in tqdm(range(1, num_adapt_iterations + 1)):
            current_q = self.generate_sample(current_q, adapt_phase_m=m)
            print("self.epsilon_bar", self.epsilon_bar)
        self.epsilon = self.epsilon_bar
        print("dual averaged final epsilon:", self.epsilon)

    def find_reasonable_epsilon(self, q):
        eps = 1.0
        p = np.array(np.random.normal(size=q.shape))
        (q_prim, p_prim) = self.leapfrog(q, p, eps)
        a = 1.0 if ((self.prob(q_prim, p_prim) / self.prob(q, p)) > 0.5) else -1.0
        while (self.prob(q_prim, p_prim) / self.prob(q, p)) ** a > 2.0 ** (-a):
            eps = 2.0 ** a * eps
            (q_prim, p_prim) = self.leapfrog(q, p, eps)
        return eps

    def prob(self, q, p):
        return np.exp(-0.5 * p.dot(p) - self.model.u(q))

    def generate_sample(self, current_q, adapt_phase_m=None):
        # p         --> r
        # theta     --> q
        # L(theta)  --> -U(q)
        # H=p^2/2 + U(q)  --> r^2/2 - L(theta)
        # p(q,p) = e^{-H}   --> p(theta, r) = e^{L(theta) - r^2/2}
        # s~Uniform(0,1) < |J|.e^{-(H-H_0)} = |J|.e^{-H}/e^{-H0}
        #   => s~Uniform(0,1).e^{-H0} < |J|.e^{-H} --> u~(0, e^{L(theta0) - r0^2/2}) < |J|. e^{L(theta) - r^2/2}

        p_init = np.array(np.random.normal(size=current_q.shape))  # r0

        # initial hamiltonian
        h0 = 0.5 * p_init.dot(p_init) + self.model.u(current_q)  # r^2/2 - L(theta)

        u = np.random.uniform(low=0, high=np.exp(-h0))  # u ~ Uniform[0, e^{L(theta0 - r^2/2)}]

        # initialize:
        q_minus = current_q.copy()
        q_plus = current_q.copy()
        p_minus = p_init.copy()
        p_plus = p_init.copy()
        j = 0  # dept of the tree
        q_m = current_q.copy()
        n = 1
        s = 1

        while s == 1:
            # choose a direction v_j from uniformly [-1, 1]
            dir = int(2 * (np.random.uniform() < 0.5) - 1)
            if dir == -1:  # backward
                q_minus, p_minus, _, _, q_prim, n_prim, s_prim, alpha, n_alpha = \
                    self.build_tree(q_minus, p_minus, u, dir, j, q_m, p_init)
            else:
                _, _, q_plus, p_plus, q_prim, n_prim, s_prim, alpha, n_alpha = \
                    self.build_tree(q_plus, p_plus, u, dir, j, q_m, p_init)

            if s_prim == 1:
                # with probability min{1, n'/n} set q_m <- q_prim:
                if np.random.uniform() < (n_prim / n):
                    q_m = q_prim

            n = n + n_prim

            s = s_prim * \
                ((q_plus - q_minus).dot(p_minus) >= 0) * \
                ((q_plus - q_minus).dot(p_plus) >= 0)

            j += 1
            if j > self.MAX_J:
                self.MAX_J = j
                print("MAX_J: ", self.MAX_J)

            if j > self.MAX_POSSIBLE_DEPTH:  # a sanity check so the tree does not grow ad infinit
                print("Termination due to a very large j=", self.MAX_POSSIBLE_DEPTH + 1)
                s = 0

        if adapt_phase_m is not None:
            # then we are in the adapt mode:
            self.H_bar = (1 - (1 / (adapt_phase_m + self.t0))) * self.H_bar + \
                         (1 / (adapt_phase_m + self.t0)) * (self.delta - alpha / n_alpha)
            self.epsilon = np.exp(self.mu - (np.sqrt(adapt_phase_m) / self.gamma) * self.H_bar)
            self.epsilon_bar = np.exp((adapt_phase_m ** (-self.kappa)) * np.log(self.epsilon) +
                                      (1 - adapt_phase_m ** (-self.kappa)) * np.log(self.epsilon_bar))
        # sample from C uniformly:
        # q, p = C[np.random.randint(low=0, high=len(C))]
        return q_m

    def build_tree(self, q, p, u, dir, depth, q0, p0):
        # dir = v
        # depth = j
        if depth == 0:
            q_prim, p_prim = self.leapfrog(q, p, dir * self.epsilon)

            h = 0.5 * p_prim.dot(p_prim) + self.model.u(q_prim)  # r'^2/2 - L(theta')
            if u <= np.exp(-h):  # exp{L(theta) - r^2/2}
                # C_prim = [(q_prim, p_prim)]
                n_prim = 1
            else:
                n_prim = 0
                # C_prim = []

            if -h > np.log(u) - self.DELTA_max:  # if L(theta' - 1/2 r'.r' > log(u)-DELTA_max)
                s_prim = 1
            else:
                s_prim = 0
            return q_prim, p_prim, q_prim, p_prim, q_prim, n_prim, s_prim, min(1,
                                                                               self.prob(q_prim, p_prim) / self.prob(q0,
                                                                                                                     p0)), 1.0
        else:
            # Recursively build the left and right sub-trees:
            q_minus, p_minus, q_plus, p_plus, q_prim, n_prim, s_prim, alpha_prim, n_alpha_prim = self.build_tree(q, p,
                                                                                                                 u, dir,
                                                                                                                 depth - 1,
                                                                                                                 q0, p0)
            if s_prim == 1:
                if dir == -1:
                    q_minus, p_minus, _, _, q_second, n_second, s_second, alpha_second, n_alpha_second = self.build_tree(
                        q_minus, p_minus, u, dir,
                        depth - 1, q0, p0)
                else:
                    _, _, q_plus, p_plus, q_second, n_second, s_second, alpha_second, n_alpha_second = self.build_tree(
                        q_plus, p_plus, u, dir,
                        depth - 1, q0, p0)

                # with probability n''/(n' + n''), set q' <- q'':
                # what if n' = n'' = 0 ?? # TODO check this
                if n_prim == n_second == 0:
                    if np.random.uniform() < 0.5:
                        q_prim = q_second
                else:
                    if np.random.uniform() < (n_second / (n_prim + n_second)):
                        q_prim = q_second

                alpha_prim += alpha_second
                n_alpha_prim += n_alpha_second

                delta_q = q_plus - q_minus
                s_prim = s_second * (delta_q.dot(p_minus) >= 0) * (
                        delta_q.dot(p_plus) >= 0)
                # C_prim.extend(C_second)
                n_prim = n_prim + n_second

            return q_minus, p_minus, q_plus, p_plus, q_prim, n_prim, s_prim, alpha_prim, n_alpha_prim

    def leapfrog(self, q, p, time):
        q = q.copy()
        p = p.copy()
        # half-step:
        p -= 0.5 * time * self.model.partial(q=q)

        # full-step:
        q += time * p

        # half-step:
        p -= 0.5 * time * self.model.partial(q=q)

        return q, p

    def name(self):
        return 'Effective NUTS with dual averaging'
