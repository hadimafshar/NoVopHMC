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

import jax.numpy as np
import jax
import numpy

# for extra info map:
DEPTH = 'J'
C_SIZE = 'C'
B_SIZE = 'B'  # though it is not B but every state traversed even if it is dropped from B


# todo find a nicer way...
def jax_numpy_to_vanilla_numpy(q):
    try:
        if type(q) == jax.interpreters.ad.JVPTracer:
            q_primal = q.primal
            if type(q_primal) == jax.interpreters.ad.JVPTracer:
                r = numpy.array(q_primal.primal)
            else:
                r = numpy.array(q.primal)  # puuuuuf!!!
        else:  # DeviceArray
            r = numpy.array(q)  # copy converts jax to numpy
    except Exception as e:
        raise (e)
    return r


class NoVoPNaiveNUTSSampler(object):
    MAX_CONSTRUCTED_DEPTH = 0  # this is just to see how large B grows in practice
    MAX_POSSIBLE_DEPTH = 12  # B is limited to 2 to power this value

    def __init__(self, model, epsilon):
        self.model = model
        self.epsilon = epsilon

        self._temp_q = None
        self._temp_p = None

        self.extra_info = None

    def first_discontinouity_x_tx_delta_u(self, q, p, t_max):
        x, tx, dim_x = self.model.first_discontinuity_x_tx_dimx(q, p)

        if x is None:
            return None, None, None, None
        if tx >= t_max:
            return None, None, None, None

        if (x == q).all():  # otherwise it will stick to a boundary
            return None, None, None, None

        tx_plus = tx + 0.000001
        tx_minus = tx - 0.000001

        x_plus = q + tx_plus * p
        x_minus = q + tx_minus * p

        u_plus = self.model.u(x_plus)
        u_minus = self.model.u(x_minus)

        return x, tx, u_plus - u_minus, dim_x  # todo dim_x should not be returned... just to test something...

    @staticmethod
    def qp_comb(q, p):
        return np.concatenate((q, p))

    @staticmethod
    def qp_comb_inverse(combed_qp):
        n = int(combed_qp.shape[0] / 2)
        return combed_qp[:n], combed_qp[n:]

    def _refractive_full_step(self, q, p):
        t0 = 0
        x, tx, delta_u, iii = self.first_discontinouity_x_tx_delta_u(q, p, self.epsilon - t0)
        while x is not None:  # while there is a reachable discontinuity
            q = x
            t0 = t0 + tx
            current_p_sqr = p.dot(p)
            new_mag_sqr = current_p_sqr - 2 * delta_u
            # p_info = 'p: ' + str(jax_numpy_to_vanilla_numpy(p))
            if new_mag_sqr > 0:
                # Reflect:
                direction = p / np.sqrt(current_p_sqr)  # unit vector in the direction of (current) p

                p = np.sqrt(new_mag_sqr) * direction
            else:
                p = -p

            x, tx, delta_u, iii = self.first_discontinouity_x_tx_delta_u(q, p, self.epsilon - t0)
        q += (self.epsilon - t0) * p
        return q, p

    def _combed_refractive_full_step_wrapper(self, combed_qp):
        """
        encapsulates full-step to compute Jacobian. The output is also recorded in temp variables to be used alongside
            the Jacobian
        :param combed_qp: combed (q, p) if the form (q1, p1, q2, p2, ...)
        :return: output of full-step in the combed form
        """
        q1, p1 = self.qp_comb_inverse(combed_qp=combed_qp)

        q2, p2 = self._refractive_full_step(q1, p1)

        # any output is passed this way, so that they can be accessed while taking Jacobian
        self._temp_q = q2
        self._temp_p = p2

        combed_qp2 = self.qp_comb(q2, p2)
        return combed_qp2

    def _full_step_with_jacobian_compute(self, q, p):
        combed_qp = self.qp_comb(q, p)
        jacob = jax.jacfwd(self._combed_refractive_full_step_wrapper)(combed_qp)
        jacob_det = np.linalg.det(jacob)

        # now the actual output that is set by self.combed_full_step:
        q2 = self._temp_q
        p2 = self._temp_p

        # just to avoid future bugs
        self._temp_q = None
        self._temp_p = None

        return q2, p2, jacob_det

    def generate_sample(self, current_q):
        self.extra_info = {B_SIZE: 0}  # reset

        # p         --> r
        # theta     --> q
        # L(theta)  --> -U(q)
        # H=p^2/2 + U(q)  --> r^2/2 - L(theta)
        # p(q,p) = e^{-H}   --> p(theta, r) = e^{L(theta) - r^2/2}
        # s~Uniform(0,1) < |J|.e^{-(H-H_0)} = |J|.e^{-H}/e^{-H0}
        #   => s~Uniform(0,1).e^{-H0} < |J|.e^{-H} --> u~(0, e^{L(theta0) - r0^2/2}) < |J|. e^{L(theta) - r^2/2}
        p_init = np.array(numpy.random.normal(size=current_q.shape))  # r0

        # initial hamiltonian
        h0 = 0.5 * p_init.dot(p_init) + self.model.u(current_q)  # r^2/2 - L(theta)

        u = numpy.random.uniform(low=0.0, high=np.exp(-h0))  # u ~ Uniform[0, e^{L(theta0 - r^2/2)}]

        # initialize:
        q_minus = self.copy_jax_array(current_q)
        q_plus = self.copy_jax_array(current_q)
        p_minus = self.copy_jax_array(p_init)
        p_plus = self.copy_jax_array(p_init)
        depth = 0  # j, dept of the tree
        C = [(current_q, p_init, 1.0)]
        s = 1
        jacob_minus = 1.0
        jacob_plus = 1.0

        while s == 1:
            # choose a direction v_j from uniformly [-1, 1]
            dir = int(2 * (numpy.random.uniform() < 0.5) - 1)
            if dir == -1:  # backward
                q_minus, p_minus, jacob_minus, _, _, _, C_prim, s_prim = self.build_tree(q_minus, p_minus, jacob_minus,
                                                                                         u, dir, depth)
            else:
                _, _, _, q_plus, p_plus, jacob_plus, C_prim, s_prim = self.build_tree(q_plus, p_plus, jacob_plus,
                                                                                      u, dir, depth)

            if s_prim == 1:
                C.extend(C_prim)

            s = s_prim * \
                ((q_plus - q_minus).dot(p_minus) >= 0) * \
                ((q_plus - q_minus).dot(p_plus) >= 0)

            depth += 1

            if depth > self.MAX_CONSTRUCTED_DEPTH:
                self.MAX_CONSTRUCTED_DEPTH = depth
                print("MAX_J: ", self.MAX_CONSTRUCTED_DEPTH)

            if depth > self.MAX_POSSIBLE_DEPTH:  # a sanity check so the tree does not grow ad infinit
                # print("Termination due to a very large j=", j)
                # print('q_minus: {}, q_plus: {}'.format(q_minus, q_plus))
                # print('C: ', C, "\n")
                s = 0

        # sample from C uniformly:
        self.extra_info[C_SIZE] = len(C)
        self.extra_info[DEPTH] = depth
        q, p, jacob = C[numpy.random.randint(low=0, high=len(C))]
        return q

    def build_tree(self, q, p, jacob, u, dir, depth):
        # dir = v
        # depth = j
        if depth == 0:
            self.extra_info[B_SIZE] += 1

            q_prim, p_prim, jacob_prim = self.leapfrog_with_jacob(q, p, time_direction=dir)

            h_prim = 0.5 * p_prim.dot(p_prim) + self.model.u(q_prim)  # r'^2/2 - L(theta')
            jacob_prim = jacob * jacob_prim
            if u <= np.exp(-h_prim) * np.abs(jacob_prim):  # exp{L(theta) - r^2/2}
                C_prim = [(q_prim, p_prim, jacob_prim)]
            else:
                C_prim = []

            s_prim = 1
            return q_prim, p_prim, jacob_prim, \
                   q_prim, p_prim, jacob_prim, C_prim, s_prim
        else:
            # Recursively build the left and right sub-trees:
            q_minus, p_minus, jacob_minus, \
            q_plus, p_plus, jacob_plus, C_prim, s_prim = self.build_tree(q=q, p=p,
                                                                         jacob=jacob,
                                                                         u=u, dir=dir,
                                                                         depth=depth - 1)
            if dir == -1:
                q_minus, p_minus, jacob_minus, \
                _, _, _, C_second, s_second = self.build_tree(q=q_minus, p=p_minus,
                                                              jacob=jacob_minus,
                                                              u=u, dir=dir,
                                                              depth=depth - 1)
            else:
                _, _, _, q_plus, p_plus, jacob_plus, \
                C_second, s_second = self.build_tree(q=q_plus, p=p_plus,
                                                     jacob=jacob_plus,
                                                     u=u, dir=dir, depth=depth - 1)
            delta_q = q_plus - q_minus
            s_prim = s_prim * s_second * (delta_q.dot(p_minus) >= 0) * (delta_q.dot(p_plus) >= 0)
            C_prim.extend(C_second)
            return q_minus, p_minus, jacob_minus, \
                   q_plus, p_plus, jacob_plus, C_prim, s_prim

    def leapfrog_with_jacob(self, q, p, time_direction):
        # half-step:
        p -= 0.5 * time_direction * self.epsilon * self.model.partial(q=q)

        # full-step:
        q, p, jacob = self.full_step_with_jacob(q, p, time_direction)

        # half-step:
        p -= 0.5 * time_direction * self.epsilon * self.model.partial(q=q)

        return q, p, jacob

    def full_step_with_jacob(self, q, p, time_direction):  # time_direction is either -1 or 1
        # todo (Another) ugly hack! this was not needed but just in case...?????
        q_numpy = jax_numpy_to_vanilla_numpy(q)
        p_numpy = jax_numpy_to_vanilla_numpy(p)
        q = np.array(q_numpy)
        p = np.array(p_numpy)

        # full-step:

        # NOTE: the underlying code assumes that time is positive, so (if negative) we flip momentum instead
        x, tx, delta_u, iii = self.first_discontinouity_x_tx_delta_u(q, time_direction * p, self.epsilon)
        if x is None:
            q += time_direction * self.epsilon * p
            jacob = 1.0
        else:
            p = p * time_direction
            q, p, jacob = self._full_step_with_jacobian_compute(q, p)
            p = p * time_direction

        # to avoid Jax errors!
        q_numpy = jax_numpy_to_vanilla_numpy(q)
        p_numpy = jax_numpy_to_vanilla_numpy(p)
        q = np.array(q_numpy)
        p = np.array(p_numpy)

        return q, p, jacob

    @staticmethod
    def copy_jax_array(arr):
        # NOTE: assignment in jax creates a copy
        arr2 = arr
        return arr2

    @staticmethod
    def name():
        return 'NoVoP.Naive.NUTS'


