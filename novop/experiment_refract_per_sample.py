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
2/12/20
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax

from experiment import *


############################
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


#########################################################
class NovopHmcSampler_Info(object):
    def __init__(self, model, l, epsilon):
        self.model = model
        self.l = l
        self.epsilon = epsilon

        self._temp_q = None
        self._temp_p = None

        self.info__reflect_count = 0
        self.info__refract_count = 0

    @staticmethod
    def qp_comb(q, p):
        return np.concatenate((q, p))

    @staticmethod
    def qp_comb_inverse(combed_qp):
        n = int(combed_qp.shape[0] / 2)
        return combed_qp[:n], combed_qp[n:]

    def _combed_full_step_wrapper(self, combed_qp):
        """
        encapsulates full-step to compute Jacobian. The output is also recorded in temp variables to be used alongside
            the Jacobian
        :param combed_qp: combed (q, p) if the form (q1, p1, q2, p2, ...)
        :return: output of full-step in the combed form
        """
        q1, p1 = self.qp_comb_inverse(combed_qp=combed_qp)

        q2, p2 = self.full_step(q1, p1)

        # any output is passed this way, so that they can be accessed while taking Jacobian
        self._temp_q = q2
        self._temp_p = p2

        combed_qp2 = self.qp_comb(q2, p2)
        return combed_qp2

    def full_step_with_jacobian(self, q, p):
        combed_qp = self.qp_comb(q, p)
        jacob = jax.jacfwd(self._combed_full_step_wrapper)(combed_qp)
        jacob_det = np.linalg.det(jacob)

        # now the actual output that is set by self.combed_full_step:
        q2 = self._temp_q
        p2 = self._temp_p

        # just to avoid future bugs
        self._temp_q = None
        self._temp_p = None

        return q2, p2, jacob_det

    # def full_step(self, q, p):
    #     q += self.epsilon * p
    #     return q, p
    def first_discontinouity_x_tx_delta_u(self, q, p, t_max):
        x, tx, dim_x = self.model.first_discontinuity_x_tx_dimx(q, p)

        if x is None:
            return None, None, None, None
        if tx >= t_max:
            return None, None, None, None

        if (x == q).all():  # otherwise it will stick to a boundary
            print("warning! sticking to a boundary!")
            return None, None, None, None

        tx_plus = tx + 0.000001
        tx_minus = tx - 0.000001

        x_plus = q + tx_plus * p
        x_minus = q + tx_minus * p

        u_plus = self.model.u(x_plus)
        u_minus = self.model.u(x_minus)

        return x, tx, u_plus - u_minus, dim_x  # todo dim_x should not be returned... just to test something...

    def full_step(self, q, p):
        t0 = 0
        x, tx, delta_u, iii = self.first_discontinouity_x_tx_delta_u(q, p, self.epsilon - t0)
        while x is not None:  # while there is a reachable discontinuity
            q = x
            t0 = t0 + tx
            current_p_sqr = p.dot(p)
            new_mag_sqr = current_p_sqr - 2 * delta_u
            # p_info = 'p: ' + str(jax_numpy_to_vanilla_numpy(p))
            if new_mag_sqr > 0:
                self.info__refract_count += 1
                direction = p / np.sqrt(current_p_sqr)  # unit vector in the direction of (current) p

                p = np.sqrt(new_mag_sqr) * direction
                # p_info += ' == R e f R a c t==> ' + str(jax_numpy_to_vanilla_numpy(p)) + ' DeltaU: ' + str(
                #     jax_numpy_to_vanilla_numpy(delta_u)) + ' iii: ' + str(iii)

                # print(' == R e f R a c t==> ' + str(jax_numpy_to_vanilla_numpy(p)) + ' DeltaU: ' + str(
                #     jax_numpy_to_vanilla_numpy(delta_u)) + ' iii: ' + str(iii))
            else:
                self.info__reflect_count += 1
                p = -p
                # p = jax.ops.index_update(p, iii, -p[iii])
                # p_info += ' <== Reflect == || ' + str(jax_numpy_to_vanilla_numpy(p)) + ' DeltaU: ' + str(
                #     jax_numpy_to_vanilla_numpy(delta_u))
                # print(' <== Reflect == || ' + str(jax_numpy_to_vanilla_numpy(p)) + ' DeltaU: ' + str(
                #     jax_numpy_to_vanilla_numpy(delta_u)))

            # self._traject.addPoint(q, 'simpl. border ' + p_info)
            # print('TRAJECT: ', self._traject)  # todo temp

            x, tx, delta_u, iii = self.first_discontinouity_x_tx_delta_u(q, p, self.epsilon - t0)
        q += (self.epsilon - t0) * p
        # self._traject.addPoint(q, 'end full-step')
        return q, p

    def generate_sample(self, current_q):
        self.info__reflect_count = 0
        self.info__refract_count = 0

        p_init = np.array(numpy.random.normal(size=current_q.shape))

        h0 = 0.5 * p_init.dot(p_init) + self.model.u(current_q)  # initial hamiltonian

        q, p, total_jacob_det = self.transform(current_q, p_init)

        h1 = 0.5 * p.dot(p) + self.model.u(q)
        delta_h = h1 - h0

        if numpy.random.uniform() < np.exp(-delta_h) * abs(total_jacob_det):
            # accept:
            return q
        else:
            # reject:
            return current_q

    @staticmethod
    def copy_jax_array(arr):
        # NOTE: assignment in jax creates a copy, todo if numpy is used instead, arr.copy() should be used
        arr2 = arr
        return arr2

    def transform(self, current_q, p_init):

        q = self.copy_jax_array(current_q)
        p = self.copy_jax_array(p_init)
        total_jacobian = 1.0

        _temp_momentum_change_half_step = None  # make sure for every new sample this is reset

        for _ in range(self.l):
            # half-step:
            if _temp_momentum_change_half_step is not None:
                p = p - _temp_momentum_change_half_step
            else:
                p = p - 0.5 * self.epsilon * self.model.partial(q=q)

            # todo (Another) ugly hack! this was not needed but just in case...?????
            q_numpy = jax_numpy_to_vanilla_numpy(q)
            p_numpy = jax_numpy_to_vanilla_numpy(p)
            q = np.array(q_numpy)
            p = np.array(p_numpy)

            # full-step:
            x, tx, delta_u, iii = self.first_discontinouity_x_tx_delta_u(q, p, self.epsilon)
            if x is None:
                q += self.epsilon * p
            else:
                q, p, jacob = self.full_step_with_jacobian(q, p)
                total_jacobian *= jacob

            # todo ugly hack! here i get jax trace.level error... how about disconnecting things here?
            q_numpy = jax_numpy_to_vanilla_numpy(q)
            p_numpy = jax_numpy_to_vanilla_numpy(p)
            q = np.array(q_numpy)
            p = np.array(p_numpy)

            # half-step:
            _temp_momentum_change_half_step = 0.5 * self.epsilon * self.model.partial(q=q)
            p = p - _temp_momentum_change_half_step

        return q, p, total_jacobian

    @staticmethod
    def name():
        return 'NovopHMC'


############################

def reflect_count(dimension, model_name='radial_bi', num_samples_per_chain=1000,
                  l=10, epsilon=0.1):
    model = make_model(model_name, dimension, None)

    # init point:
    sample = init_sample(model_name, dimension)

    sampler = NovopHmcSampler_Info(model=model, l=l, epsilon=epsilon)
    total_reflects = 0
    total_refracts = 0
    for sample_count in tqdm(range(1, num_samples_per_chain + 1)):  # num samples
        sample = sampler.generate_sample(current_q=sample)
        total_reflects += sampler.info__reflect_count
        total_refracts += sampler.info__refract_count
    print('dimension: ', dimension)
    print("total_refracts: ", total_refracts, "\t rate:", (total_refracts/num_samples_per_chain))
    print("total_reflects: ", total_reflects, "\t rate:", (total_reflects/num_samples_per_chain))

def main():
    reflect_count(dimension=5)


if __name__ == "__main__":
    main()
