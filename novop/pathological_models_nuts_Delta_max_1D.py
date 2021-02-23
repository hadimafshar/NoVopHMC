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
6/5/20
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax

from hmc_models import Model

import numpy as numpy
import jax.numpy as np

import matplotlib.pyplot as plt

from samplers.nuts_naive_sampler_numpy import NaiveNUTSSampler  # , NoVoPNaiveNUTSSampler
from tqdm import tqdm
import time


class Uniform2DModel(Model):
    dim = 2

    def u(self, q):
        assert q.shape[0] == 2, "q.shape: {}".format(q.shape)
        if -2 < q[0] < 2:
            return 0
        else:
            return np.inf

    def partial(self, q):
        return jax.jacfwd(self.u)(q)

    def first_discontinuity_x_tx_dimx(self, q, p):
        """
        :param q: start location
        :param p: velocity
        :return: (first_x, min_time_x, dim_x) where:
                first_x     is the nearest point where line q(t) = q + p.t reaches the first discontinuity
                min_time_x  is the time to reach the first discontinuity
                dim_x       assuming boundaries are in form q_i=b, the 'i' associated with the collision boundary.

                Note: if by t_max, a discontinuity is not reached, None, np.inf, None is returned.
        """
        raise Exception('not implemented')

    def get_dim(self):
        return self.dim

    def model_name(self):
        return ('Uniform.2D')


class Step1DModel(Model):
    b_neg = -2
    b_pos = 20
    dim = 1
    c = 5.0
    h = 3

    def u(self, q):
        if q.shape == ():  # this happens in MH where a number is returned
            q = np.array([q])
        assert q.shape[0] == 1, "q.shape: {}".format(q.shape)
        if self.b_neg < q[0] < 0:
            return self.c
        elif 0 <= q[0] < self.b_pos:
            return self.c + self.h
        else:
            return np.inf

    def normalization_constant(self):
        return np.abs(self.b_neg) * np.exp(-self.c) + self.b_pos * np.exp(-self.c - self.h)

    def partial(self, q):
        return 0  # jax.jacfwd(self.u)(q)

    def first_discontinuity_x_tx_dimx(self, q, p):
        raise Exception('not implemented')

    def get_dim(self):
        return self.dim

    def model_name(self):
        return ('Step.1D')


import scipy.stats as stats


class RandomWalkMetropolisHastingsSampler(object):
    def __init__(self, model, proposal_variance):
        self.model = model
        self.cov = proposal_variance * numpy.eye(model.get_dim())

    def generate_sample(self, current_q):
        proposal_density = stats.multivariate_normal(mean=current_q, cov=self.cov)
        proposed_q = proposal_density.rvs()
        log_g_prop_q = proposal_density.logpdf(proposed_q)  # log g(q_prop | q_curr)
        log_g_curr_q = stats.multivariate_normal(mean=proposed_q, cov=self.cov).logpdf(
            current_q)  # log g(q_curr | q_prop)
        log_p_prop_q = -self.model.u(proposed_q)
        log_p_curr_q = -self.model.u(current_q)

        log_accept = log_p_prop_q + log_g_curr_q - log_p_curr_q - log_g_prop_q
        if numpy.random.uniform(low=0, high=1) < numpy.exp(log_accept):
            return proposed_q
        else:
            return current_q

    def name(self):
        return "MH"


def analysis1D_role_of_DELTA_max():
    model = Step1DModel()

    rng = numpy.arange(model.b_neg - 1, model.b_pos + 1, 0.01)
    Z = model.normalization_constant()
    y = [np.exp(-model.u(np.array([i]))) / Z for i in rng]
    plt.plot([x for x in rng], [model.u(np.array([i])) for i in rng])
    plt.ylabel('a 1D piecewise constant potential function')
    plt.show()
    return
    print('sampling...')
    start = time.time()
    # sampling:
    # sampler = BaselineHmcSampler(model=model, l=10, epsilon=0.1)
    # sampler = NovopHmcSampler(model=model, l=10, epsilon=0.1)
    # sampler = NovopHmcSampler(model=model, l=10, epsilon=0.1)
    # sampler = RandomWalkMetropolisHastingsSampler(model=model, proposal_variance=0.1)
    sampler = NaiveNUTSSampler(model=model, epsilon=0.1, DELTA_max=1.0)
    # sampler = NoVoPNaiveNUTSSampler(model=model, epsilon=0.1)
    sample = np.array([-0.1])
    samples = numpy.empty((0, 1), dtype=float)
    for sample_count in tqdm(range(5000)):
        sample = sampler.generate_sample(current_q=sample)
        samples = numpy.vstack((samples, sample))

    n, bins, patches = plt.hist(samples, bins=int(model.b_pos + np.abs(model.b_neg)), density=True, facecolor='g',
                                alpha=0.75)

    duration = time.time() - start
    print("took {}".format(duration))

    # plt.ylim(top=np.exp(-10))
    plt.plot([x for x in rng], y)
    plt.show()


def main():
    analysis1D_role_of_DELTA_max()


if __name__ == "__main__":
    main()
