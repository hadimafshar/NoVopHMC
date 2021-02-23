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
4/2/20
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
import numpy as numpy
import hmc_models as models
from samplers.fast_hmc_samplers import BaselineHmcSampler, NovopHmcSampler, ReflectiveHmcSampler
from samplers.mh_samplers import AutoTuneMetropolisHastingsSampler
from samplers.nuts_efficient_sampler_dual_averaging_numpy import EffectiveNUTSSamplerWithDualAveraging
from samplers.novop_nuts_naive_sampler_jax import NoVoPNaiveNUTSSampler
from samplers.nuts_naive_sampler_numpy import NaiveNUTSSampler

from tqdm import tqdm
from sklearn.datasets import make_spd_matrix
import pickle
import time


class RadialRefModel(models.SphereLayerModel):
    def __init__(self, dim, bounds=np.array([3., 6.])):
        super().__init__(dim=dim, bounds=bounds)
        self.a = np.array(
            make_spd_matrix(n_dim=dim, random_state=2020))  # a random symmetric positive-definite matrix

    def value_in_layer(self, layer, q):
        o = q[None, :].dot(self.a).dot(q[:, None])
        o = o[0][0] ** 0.5

        if layer == 0:
            return o
        if layer == 1:
            return 1 + o
        if layer == 2:
            return 50 + o  # np.inf
        raise Exception('unexpected region {i}'.format(i=layer))


class RadialRefModelBiValueDiag(RadialRefModel):
    def __init__(self, dim, bounds=np.array([3., 6.])):
        super().__init__(dim=dim, bounds=bounds)
        a = [numpy.exp(-5.0) if numpy.random.uniform() < 0.5 else numpy.exp(5.0) for i in range(dim)]
        self.a = np.diag(np.array(a))


def test_radial_model():
    model = RadialRefModel(dim=50, bounds=np.array([3.0, 4.0]))
    q = np.zeros(50)
    p = np.ones_like(q)
    first_x, min_time_x, _ = model.first_discontinuity_x_tx_dimx(q, p)
    print('first_x: ', first_x)
    print('min_time_x: ', min_time_x)


def make_model(model_name, dimension):
    if model_name == 'cube':
        model = models.FastRefCubeModel(dim=dimension)
    elif model_name == 'radial':
        model = RadialRefModel(dim=dimension)
    elif model_name == 'radial_bi':
        model = RadialRefModelBiValueDiag(dim=dimension)
    else:
        raise Exception('unknown model {}'.format(model_name))
    return model


def init_sample(model_name, dimension):
    # to guarantee that the initial sample is not lost in an ambient space
    if model_name in ['radial', 'radial_bi']:
        sample = numpy.random.uniform(low=5.5 / (dimension ** 0.5), high=5.99 / (dimension ** 0.5), size=dimension)
    elif model_name == 'cube':
        sample = numpy.random.uniform(low=5.5, high=5.99, size=dimension)
    elif model_name == 'pref':
        sample = numpy.random.uniform(low=-1, high=1, size=dimension)  # (low=0.9, high=1, size=dimension)
    else:
        raise Exception('unknown model {}'.format(model_name))
    return sample


def make_sampler(sampler_name, model, l, epsilon, init_sample):
    if sampler_name == 'MH':
        sampler = AutoTuneMetropolisHastingsSampler(model=model, init_sample=init_sample,
                                                    return_extra_info=False)
    elif sampler_name == 'Novop':
        sampler = NovopHmcSampler(model=model, l=l, epsilon=epsilon)
    elif sampler_name == 'RHMC':
        sampler = ReflectiveHmcSampler(model=model, l=l, epsilon=epsilon)
    elif sampler_name == 'BaseHMC':
        sampler = BaselineHmcSampler(model=model, l=l, epsilon=epsilon)
    elif sampler_name == 'NovopNuts':
        sampler = NoVoPNaiveNUTSSampler(model=model, epsilon=epsilon)
    elif sampler_name == 'BaseNUTS':
        sampler = NaiveNUTSSampler(model=model, epsilon=epsilon)
    elif sampler_name == 'EffectiveNUTS_dual':
        sampler = EffectiveNUTSSamplerWithDualAveraging(model=model, init_q=init_sample,
                                                        num_adapt_iterations=NUTS_DUAL_AV_NUM_ADAPT_ITR)
    else:
        raise Exception('Unknown alg {}'.format(sampler_name))
    return sampler


def experiment1_convergence_vs_itr(model_name, dimension, sampler_name, num_samples_per_chain,
                                   mcmc_chain_ids, l=10, epsilon=0.1):
    model = make_model(model_name, dimension)

    for mcmc_chain in mcmc_chain_ids:
        print("\nMCMC Chain: ", mcmc_chain, "  model name: ", model_name, "  dim:", dimension, " sampler:",
              sampler_name)

        # init point:
        sample = init_sample(model_name, dimension)

        sampler = make_sampler(sampler_name, model=model, l=l, epsilon=epsilon, init_sample=sample)

        sum_samples = numpy.zeros(dimension)
        running_errors = []
        running_samples = []
        running_times = []
        running_C_set_sizes = []
        running_B_set_sizes = []
        start_time = time.time()
        for sample_count in tqdm(range(1, num_samples_per_chain + 1)):  # num samples
            sample = sampler.generate_sample(current_q=sample)
            if sampler_name in ['NovopNuts', 'BaseNUTS', 'EffectiveNUTS_dual']:
                running_C_set_sizes.append(sampler.extra_info['C'])
                running_B_set_sizes.append(2 ** sampler.extra_info['J'])

            sum_samples += sample
            # print('sample: ', sample)
            expected_value = sum_samples / sample_count
            worst_mean_abs_err = float(numpy.max(abs(expected_value)))
            running_errors.append(worst_mean_abs_err)
            running_samples.append(sample.copy())
            running_times.append(time.time() - start_time)

            # save every 1000 samples
            if sample_count % 1000 == 0:
                save_samples(running_samples=running_samples,
                             running_times=running_times,
                             running_errors=running_errors,
                             model_name='_tmp_' + model_name,
                             sampler_name=sampler_name,
                             dimension=dimension,
                             mcmc_chain=mcmc_chain)

        # saving the final model:
        save_samples(running_samples=running_samples,
                     running_times=running_times,
                     running_errors=running_errors,
                     model_name=model_name,
                     sampler_name=sampler_name,
                     dimension=dimension,
                     mcmc_chain=mcmc_chain)

        ####################
        # save C anc B sets:
        if len(running_C_set_sizes) > 0:
            with open(RESULT_PATH + '/C_setsize_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                                     a=sampler_name,
                                                                                                     d=dimension,
                                                                                                     c=mcmc_chain,
                                                                                                     n=len(
                                                                                                         running_C_set_sizes)),
                      'wb') as f:
                pickle.dump(running_C_set_sizes, f, pickle.HIGHEST_PROTOCOL)
            with open(RESULT_PATH + '/B_setsize_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                                     a=sampler_name,
                                                                                                     d=dimension,
                                                                                                     c=mcmc_chain,
                                                                                                     n=len(
                                                                                                         running_C_set_sizes)),
                      'wb') as f:
                pickle.dump(running_B_set_sizes, f, pickle.HIGHEST_PROTOCOL)


def save_samples(running_samples, running_times, running_errors,
                 model_name, sampler_name, dimension, mcmc_chain, n=None):
    if n is None:
        n = len(running_errors)  # for itr experiments it is None, for time experiments it is manually set

    # save error measure:
    with open(RESULT_PATH + '/{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                   a=sampler_name,
                                                                                   d=dimension,
                                                                                   c=mcmc_chain,
                                                                                   n=n), 'wb') as f:
        pickle.dump(running_errors, f, pickle.HIGHEST_PROTOCOL)

    # save samples:
    with open(RESULT_PATH + '/samples_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                           a=sampler_name,
                                                                                           d=dimension,
                                                                                           c=mcmc_chain,
                                                                                           n=n),
              'wb') as f:
        pickle.dump(running_samples, f, pickle.HIGHEST_PROTOCOL)

    # save times:
    with open(RESULT_PATH + '/times_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                         a=sampler_name,
                                                                                         d=dimension,
                                                                                         c=mcmc_chain,
                                                                                         n=n),
              'wb') as f:
        pickle.dump(running_times, f, pickle.HIGHEST_PROTOCOL)


#####################################
def experiment2_convergence_vs_time(model_name, dimension, sampler_name,
                                    desired_sampling_time_per_chain_in_seconds,  # like 1400
                                    max_num_samples_per_chain, mcmc_chain_ids):
    model = make_model(model_name, dimension)

    for mcmc_chain in mcmc_chain_ids:
        print("\nMCMC Chain: ", mcmc_chain, "  model name: ", model_name, "  dim:", dimension, " sampler:",
              sampler_name)

        # init point:
        sample = init_sample(model_name, dimension)

        sampler = make_sampler(sampler_name, model=model, l=10, epsilon=0.1, init_sample=sample)

        sample_record_period = desired_sampling_time_per_chain_in_seconds / max_num_samples_per_chain

        sum_samples = numpy.zeros(dimension)
        running_errors = []
        running_samples = []
        running_times = []
        start_time = time.time()
        sample_count = 0
        passed_time = 0
        pbar = tqdm(total=max_num_samples_per_chain)
        record_count = 0  # each time a sample is recorded, this is increased by 1
        while passed_time < desired_sampling_time_per_chain_in_seconds:
            sample_count += 1
            sample = sampler.generate_sample(current_q=sample)
            sum_samples += sample
            expected_value = sum_samples / sample_count
            worst_mean_abs_err = float(numpy.max(abs(expected_value)))

            passed_time = time.time() - start_time
            if passed_time >= sample_record_period * record_count:
                running_errors.append(worst_mean_abs_err)
                running_samples.append(sample.copy())
                running_times.append(time.time() - start_time)
                record_count += 1
                pbar.update(1)

        # saving the final model:
        save_samples(running_samples=running_samples,
                     running_times=running_times,
                     running_errors=running_errors,
                     model_name=model_name,
                     sampler_name='_FIXED_TIME_' + sampler_name,
                     dimension=dimension,
                     mcmc_chain=mcmc_chain,
                     n=desired_sampling_time_per_chain_in_seconds)
        pbar.close()



def main():
    experiment = 'itr'  # 'time'
    # model = 'cube'  #'radial_bi',  # 'pref', 'cube', 'radial',
    model = 'radial_bi'
    if experiment == 'itr':
        experiment1_convergence_vs_itr(model_name=model,
                                       dimension=5,
                                       sampler_name='Novop',
                                       # 'BaseNUTS',  #'NovopNuts', #'Novop', 'MH' 'RHMC',  # 'BaseHMC',
                                       num_samples_per_chain=200,
                                       mcmc_chain_ids=['L1', 'L2', 'L3', 'L4', 'L5'],
                                       )
    elif experiment == 'time':
        experiment2_convergence_vs_time(model_name=model,
                                        dimension=30,
                                        sampler_name='Novop',
                                        # 'BaseNUTS',  #'NovopNuts', #'Novop', 'MH' 'RHMC',  # 'BaseHMC',
                                        desired_sampling_time_per_chain_in_seconds=140,  # 1400,
                                        max_num_samples_per_chain=10000,
                                        mcmc_chain_ids=[
                                            'L1', 'L2', 'L3', 'L4', 'L5']
                                        )


NUTS_DUAL_AV_NUM_ADAPT_ITR = 100  # this is for adaptive parameter tuning in the NUTS dual parameter adaptation 1000
RESULT_PATH = './results'



if __name__ == "__main__":
    main()
