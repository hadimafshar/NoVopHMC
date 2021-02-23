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
15/10/20
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import matplotlib.pyplot as plt


def __rho_s_hat_f(mu_hat_f, sig2_hat_f, M, s, f_arr):
    result = 0.0
    for m in np.arange(s + 1, M + 1):
        result += (f_arr[m - 1] - mu_hat_f) * (f_arr[m - s - 1] - mu_hat_f)  # since indexes starts from 0 rather than 1
    return result / (sig2_hat_f * (M - s))


def effective_sample_size_given_mean_var(theta_array, f, mu_hat_f, sig2_hat_f, cutoff=0.05):
    """
    :param theta_array:  MxDim array where M is the sample size and Dim is the dimensionality of the distribution
    :param f: reduction function
    :param cutoff: parameter, see "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
Matthew" appendix
    :param mu_hat_f: Mean calculated from an independent large chain
    :param sig2_hat_f: variance calculated from an independent large chain
    :return: Effective sample size given mu_hat_f and sig2_hat_f
    """
    f_arr = np.apply_along_axis(f, 1, theta_array)
    M = len(theta_array)

    a = 0.0
    for s in np.arange(1, M):  # so its is up to M - 1
        rho_s_hat_f = __rho_s_hat_f(mu_hat_f, sig2_hat_f, M, s, f_arr)
        a += (1 - s / M) * rho_s_hat_f
        # print(s, '\t', a)
        if rho_s_hat_f < cutoff:
            # print("break")
            break

    return M / (1 + 2 * a)


def effective_sample_size(theta_array: np.ndarray, large_theta_array: np.ndarray, f=np.max, cutoff=0.05):
    """
    :param theta_array: MxDim array where M is the sample size and Dim is the dimensionality of the distribution
    :param large_theta_array: NxDim where ideally N >> M
    :param f: reduction function
    :param cutoff: parameter, see "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
Matthew" appendix
    :return: Effective sample size
    """
    assert theta_array.shape[1] == large_theta_array.shape[1]
    # f(large_theta_array):
    f_large = np.apply_along_axis(f, 1, large_theta_array)
    mu_hat_f = np.mean(f_large)
    sig2_hat_f = np.var(f_large)

    return effective_sample_size_given_mean_var(theta_array=theta_array, f=f,
                                                mu_hat_f=mu_hat_f, sig2_hat_f=sig2_hat_f, cutoff=cutoff)


def test_basic():
    n = 10000
    a = np.expand_dims(np.random.normal(loc=0, scale=1, size=n), axis=0)
    a2 = np.expand_dims(np.random.normal(loc=10, scale=2, size=n), axis=0)
    a = np.append(a, a2, axis=0)
    a = a.T
    print(a.shape)

    m = 1000
    b = np.expand_dims(np.random.normal(loc=9.8, scale=1, size=m), axis=0)
    b2 = np.expand_dims(np.random.normal(loc=10, scale=2, size=m), axis=0)
    b = np.append(b, b2, axis=0)
    b = b.T
    print(b.shape)

    print('ess: ', effective_sample_size(theta_array=b, large_theta_array=a))


def __fetch_samples_and_time(path, model, alg, dim, chain, entries, limit_entries=None):
    with open(path + 'samples_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model,
                                                                                   a=alg,
                                                                                   d=dim,
                                                                                   c=chain,
                                                                                   n=entries),
              'rb') as f:
        samples = pickle.load(f)

    if limit_entries:
        samples = samples[:limit_entries]

    np_arr = np.vstack([np.expand_dims(s, axis=0) for s in samples])

    with open(path + 'times_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model,
                                                                                 a=alg,
                                                                                 d=dim,
                                                                                 c=chain,
                                                                                 n=entries),
              'rb') as f:
        times = pickle.load(f)

    if limit_entries:
        times = times[:limit_entries]
    time = times[-1]
    return np_arr, time


def __fetch_C_B_set_sizes(path, model, alg, dim, chain, entries, limit_entries=None):
    with open(path + 'C_setsize_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model,
                                                                                     a=alg,
                                                                                     d=dim,
                                                                                     c=chain,
                                                                                     n=entries),
              'rb') as f:
        cs = pickle.load(f)
    with open(path + 'B_setsize_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model,
                                                                                     a=alg,
                                                                                     d=dim,
                                                                                     c=chain,
                                                                                     n=entries),
              'rb') as f:
        bs = pickle.load(f)


    if limit_entries:
        bs = bs[:limit_entries]

    sum_c = sum(cs)
    sum_b = sum(bs)
    return sum_c, sum_b


def __acceptance_rate(path, model, alg, dim, chain, entries):
    with open(path + 'samples_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model,
                                                                                   a=alg,
                                                                                   d=dim,
                                                                                   c=chain,
                                                                                   n=entries),
              'rb') as f:
        samples = pickle.load(f)
    assert len(samples) == entries
    acc = 0
    for i in range(1, entries):
        if (samples[i] == samples[i - 1]).all():
            acc += 1
    return acc / entries


def __concat_data(model_name, alg_str, dim, chains, entries, path):
    arr = np.empty([0, dim])
    for chain in chains:
        with open(path + 'samples_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                       a=alg_str,
                                                                                       d=dim,
                                                                                       c=chain,
                                                                                       n=entries),
                  'rb') as f:
            samples = pickle.load(f)
            # arr0 = np.empty([0, dim])
            arr0 = np.vstack([np.expand_dims(s, axis=0) for s in samples])
            # print(arr0.shape)
            arr = np.vstack([arr, arr0])
        print('arr.shape: ', arr.shape)

    return arr


def radial_bi_experiment():
    path = "./results/"
    dimension = 50
    model_name = 'radial_bi'
    func = np.sum  # symmetric reduction function so that we know that mean is 0
    big_arr = __concat_data(model_name=model_name, alg_str='Novop', dim=dimension,
                            chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                            entries=5000, path=path)

    big_arr = np.apply_along_axis(func, 1, big_arr)
    print("big_arr.mean:", big_arr.mean(), "... big_arr.var:", big_arr.var())
    big_arr_var = big_arr.var()
    big_arr_mean = 0.0  # by symmetry of model and reduction function

    for alg in ['BaseHMC', 'Novop', 'BaseNUTS', 'NovopNuts']:
        arr, time = __fetch_samples_and_time(path, model=model_name, alg=alg, dim=dimension,
                                             chain='LL1', entries=2000 if alg == 'BaseNUTS' else 5000,
                                             limit_entries=2000)
        ESS = effective_sample_size_given_mean_var(theta_array=arr, f=func,
                                                   mu_hat_f=big_arr_mean, sig2_hat_f=big_arr_var)
        print('alg:{a}:'.format(a=alg))
        print('ess: ', ESS)
        print('ess/sec', ESS / time)
        print("--------")


def radial_bi_config_plot_experiment():
    reference_arr_path = "./results_long_run/"
    path = "./results/"

    dimension = 50
    model_name = 'radial_bi'
    func = np.sum  # symmetric reduction function so that we know that mean is 0
    big_arr = __concat_data(model_name=model_name, alg_str='Novop', dim=dimension,
                            chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                            entries=5000, path=reference_arr_path)

    big_arr = np.apply_along_axis(func, 1, big_arr)
    print("big_arr.mean:", big_arr.mean(), "... big_arr.var:", big_arr.var())
    big_arr_var = big_arr.var()
    big_arr_mean = 0.0  # by symmetry of model and reduction function

    #####################################
    ALGS = ['Novop', 'BaseHMC']  # 'BaseNUTS', 'NovopNuts']
    COLORS = ['black', 'green']
    LABELS = ['Novop HMC', 'Baseline HMC']
    alg_to_ESSes_per_grad = dict()
    l = 40  # 40 #10 20 40

    EPSILONS = [0.001, 0.01,
                0.05, 0.1, 0.2]
    for alg in ALGS:
        alg_to_ESSes_per_grad[alg] = list()
        for eps in EPSILONS:
            arr, time = __fetch_samples_and_time(path, model=model_name, alg=alg, dim=dimension,
                                                 chain='comfig_L{l}.eps{e}'.format(l=l, e=eps), entries=500)
            ESS = effective_sample_size_given_mean_var(theta_array=arr, f=func,
                                                       mu_hat_f=big_arr_mean, sig2_hat_f=big_arr_var)
            alg_to_ESSes_per_grad[alg].append(ESS / l)
            print('L={l}\t.eps={e}'.format(l=l, e=eps))
            print('alg:{a}:'.format(a=alg))
            print('ess: ', ESS)
            print('ess/sec', ESS / time)
            print("--------")
    print(alg_to_ESSes_per_grad)

    for i in range(len(ALGS)):
        plt.plot([str(e) for e in EPSILONS], alg_to_ESSes_per_grad[ALGS[i]], COLORS[i], alpha=0.5, label=LABELS[i])
    plt.legend()
    plt.ylim(bottom=0)
    plt.ylim(top=2)
    plt.xlabel('$\epsilon$ ($L$=' + str(l) + ')')
    plt.ylabel('ESS per gradient')
    plt.show()


def main():
    # test_basic()
    # radial_bi_experiment()
    radial_bi_config_plot_experiment()


if __name__ == "__main__":
    main()
