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

import numpy as numpy
import scipy.stats as stats
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

FINAL_LATEX = False
if FINAL_LATEX:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


def load_means_ups_downs(model_name, alg_str, dim, chains_list, confidence, entries, path):
    # a list of list of errors (with size: no. MCMC chains X no. samples per chain)
    alg_chain_errors = []
    for chain in chains_list:
        with open(path + '{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                               a=alg_str,
                                                                               d=dim,
                                                                               c=chain,
                                                                               n=entries),
                  'rb') as f:
            running_errors = pickle.load(f)
            alg_chain_errors.append(running_errors)

    if len(alg_chain_errors) == 1:  # only one chain: no up and no down
        return alg_chain_errors[0], alg_chain_errors[0], alg_chain_errors[0]

    # inverse list of lists. (with size: no.samples per chain X no. MCMC chains)
    chain_errors_per_iter = zip(*alg_chain_errors)

    # with size: no. samples per chain X 3
    mean_up_down_per_itr = [mean_confidence_interval(data=chain_errors, confidence=confidence) for chain_errors in
                            chain_errors_per_iter]

    # with size: 3 X no.samples per chain
    means_ups_downs = list(zip(*mean_up_down_per_itr))

    return means_ups_downs[0], means_ups_downs[1], means_ups_downs[2]


def load_align_means_ups_downs(model_name, alg_str, dim, chains_list, confidence, entries, path, max_time):
    # a list of list of errors (with size: no. MCMC chains X no. samples per chain)
    alg_chain_errors = []
    alg_chain_times = []
    for chain in chains_list:
        with open(path + '{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                               a=alg_str,
                                                                               d=dim,
                                                                               c=chain,
                                                                               n=entries),
                  'rb') as f:
            running_errors = pickle.load(f)
            alg_chain_errors.append(running_errors)
        with open(path + 'times_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                     a=alg_str,
                                                                                     d=dim,
                                                                                     c=chain,
                                                                                     n=entries), 'rb') as f:
            sampling_times = pickle.load(f)
            alg_chain_times.append(sampling_times)

    reference_times = alg_chain_times[0]
    for i in range(len(reference_times)):
        if reference_times[i] > max_time:
            reference_times = reference_times[:i]
            break

    if len(alg_chain_errors) == 1:  # only one chain: no up and no down
        return alg_chain_errors[0][:len(reference_times)], \
               alg_chain_errors[0][:len(reference_times)], \
               alg_chain_errors[0][:len(reference_times)], reference_times

    # align:
    aligned_alg_chain_errors = []
    for i in range(len(alg_chain_errors)):
        errs = alg_chain_errors[i]
        times = alg_chain_times[i]
        aligned_errs = align_data(errs, times, reference_times)
        aligned_alg_chain_errors.append(aligned_errs)

    # inverse list of lists. (with size: no.samples per chain X no. MCMC chains)
    chain_errors_per_iter = zip(*aligned_alg_chain_errors)

    # with size: no. samples per chain X 3
    mean_up_down_per_itr = [mean_confidence_interval(data=chain_errors, confidence=confidence) for chain_errors in
                            chain_errors_per_iter]

    # with size: 3 X no.samples per chain
    means_ups_downs = list(zip(*mean_up_down_per_itr))

    return means_ups_downs[0], means_ups_downs[1], means_ups_downs[2], reference_times


def align_data(errs, times, reference_times):
    # note: times are reference_times should be sorted lists
    n = len(errs)
    assert n == len(times)
    aligned_errs = []

    running_ix = 0
    for ref_time in reference_times:

        # ideally times[running_ix] <= ref_time < times[running_ix + 1]
        while (running_ix < n - 1) and (ref_time >= times[running_ix + 1]):
            running_ix += 1

        if ref_time <= times[running_ix]:
            aligned_errs.append(errs[running_ix])
        elif running_ix == n - 1:
            aligned_errs.append(errs[running_ix])
        else:
            assert times[running_ix] < ref_time < times[running_ix + 1]
            d1 = (ref_time - times[running_ix])
            d2 = (times[running_ix + 1] - ref_time)
            aligned_errs.append((errs[running_ix] * d1 + errs[running_ix + 1] * d2) / (d1 + d2))
    return aligned_errs


def plot_data(model_name, alg_str, dim, chains, color, path, burn_in=0, entries=10000, label=None,
              plot_type='ErrVsItr'
              ):
    if plot_type == 'ErrVsItr':
        return plot_err_vs_itr(model_name, alg_str, dim, chains, color, burn_in, entries, label, path)
    elif plot_type == 'ErrVsTime':
        return plot_err_vs_time(model_name, alg_str, dim, chains, color, burn_in, entries, label, path)
    elif plot_type == 'SampleTraceVsItr':
        return plot_trace_vs_Itr(model_name=model_name, alg_str=alg_str, dim=dim, chains=chains,
                                 color=color, burn_in=burn_in, entries=entries, label=label, path=path)
    else:
        raise Exception('Unknown plot type')


def plot_err_vs_itr(model_name, alg_str, dim, chains, color, burn_in, entries, label, path):
    if label is None:
        label = alg_str

    means, ups, downs = load_means_ups_downs(model_name, alg_str=alg_str, dim=dim, chains_list=chains,
                                             confidence=0.95, entries=entries, path=path)
    iterations = [_ for _ in range(1, len(means) + 1)]

    assert len(means) == len(ups) == len(downs) == len(iterations)

    plt.fill_between(iterations[burn_in:], ups[burn_in:], downs[burn_in:], color=color[0], alpha=0.1)

    return plt.plot(iterations[burn_in:], means[burn_in:], color, alpha=0.5, label=label)


def plot_err_vs_time(model_name, alg_str, dim, chains, color, burn_in, entries, label, path, max_time=1400):
    if label is None:
        label = alg_str

    means, ups, downs, ref_times = load_align_means_ups_downs(model_name, alg_str=alg_str, dim=dim, chains_list=chains,
                                                              confidence=0.95, entries=entries, path=path,
                                                              max_time=max_time)

    assert len(means) == len(ups) == len(downs) == len(ref_times)
    plt.fill_between(ref_times[burn_in:], ups[burn_in:], downs[burn_in:], color=color[0], alpha=0.1)
    return plt.plot(ref_times[burn_in:], means[burn_in:], color, alpha=0.5, label=label)


def plot_trace_vs_Itr(model_name, alg_str, dim, chains, color, entries, label, path, burn_in=0,
                      # path=END_RESULTS_PATH_TO_PLOT,
                      axis=None):
    if label is None:
        label = alg_str

    chain = chains[0]  # just one chain
    with open(path + 'samples_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                   a=alg_str,
                                                                                   d=dim,
                                                                                   c=chain,
                                                                                   n=entries), 'rb') as f:
        samples = pickle.load(f)
        trace = numpy.array([numpy.max(s) for s in samples])
    n = min(len(trace), 1000)
    iterations = [_ for _ in range(1, n + 1)]
    if axis is None:
        return plt.plot(iterations[burn_in:n], trace[burn_in:n], color, alpha=0.5, label=label)
    else:
        return axis.plot(iterations[burn_in:n], trace[burn_in:n], color, alpha=0.5, label=label)


def plot_trace_vs_time(model_name, alg_str, dim, chains, color, entries, label, path,  # =END_RESULTS_PATH_TO_PLOT,
                       max_time=800, axis=None):
    if label is None:
        label = alg_str

    chain = chains[0]  # just one chain
    with open(path + 'samples_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                   a=alg_str,
                                                                                   d=dim,
                                                                                   c=chain,
                                                                                   n=entries), 'rb') as f:
        samples = pickle.load(f)
        trace = numpy.array([numpy.max(s) for s in samples])

    with open(path + 'times_{m}_{a}_dim_{d}_chain_{c}_entries_{n}.pickle'.format(m=model_name,
                                                                                 a=alg_str,
                                                                                 d=dim,
                                                                                 c=chain,
                                                                                 n=entries), 'rb') as f:
        sampling_times = pickle.load(f)

    for i in range(len(sampling_times)):
        if sampling_times[i] > max_time:
            sampling_times = sampling_times[:i]
            samples = samples[:i]
            break
    trace = numpy.array([numpy.max(s) for s in samples])
    if axis is None:
        return plt.plot(sampling_times, trace, color, alpha=0.5, label=label)
    else:
        return axis.plot(sampling_times, trace, color, alpha=0.5, label=label)


def experiment1_cube_plot(dimension, l, eps, path):
    model_name = 'cube'
    if dimension == 5:
        baseline = plot_data(model_name=model_name, alg_str='BaseHMC', dim=dimension,
                             chains=['1', '2', '3', '4', '5'], color='g', entries=5000,
                             path=path, label='Baseline HMC')
        simpline = plot_data(model_name=model_name, alg_str='SimpleRHMC', dim=dimension,
                             chains=['1', '2', '3', '4', '5'], color='k',
                             entries=5000, path=path, label='NoVoP HMC')
        reflline = plot_data(model_name=model_name, alg_str='RHMC', dim=dimension,
                             chains=['1', '2', '3', '4', '5'], color='r',
                             entries=5000, path=path)

        metropol = plot_data(model_name=model_name, alg_str='MH', dim=dimension,
                             chains=['1', '2', '3', '4', '5'], color='b', entries=5000,
                             label='RWMH',
                             path=path)
    elif dimension == 20:
        simpline = plot_data(model_name=model_name, alg_str='SimpleRHMC', dim=dimension,
                             chains=['1', '2', '3', '4', '5'], color='k',
                             entries=5000, path=path, label='NoVoP HMC')
        reflline = plot_data(model_name=model_name, alg_str='RHMC', dim=dimension,
                             chains=['1', '2', '3', '4', '5'], color='r',
                             entries=5000, path=path, label='RHMC')
        metropol = plot_data(model_name=model_name, alg_str='MH', dim=dimension,
                             chains=['1', '2', '3', '4', '5'],
                             color='b',
                             entries=5000, path=path, label='RWMH')
        baseline = plot_data(model_name=model_name, alg_str='BaseHMC', dim=dimension, label='Baseline HMC',
                             chains=['1', '2', '3', '4', '5'], color='g', entries=5000, path=path)

    else:
        raise Exception

    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel('Iteration (dim={d})'.format(d=dimension))
    plt.ylabel('Error (WMAE)')
    plt.show()


def experiment1_radial_plot_err_vs_itr(dimension, l, eps, path):
    model_name = 'radial_bi'

    if dimension == 5:
        baseline = plot_data(model_name=model_name, alg_str='BaseHMC', dim=dimension,
                             chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                             color='g', entries=5000, path=path)
        simpline = plot_data(model_name=model_name, alg_str='SimpleRHMC', dim=dimension,
                             chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                             color='k',
                             entries=5000, label='NoVoP HMC')
        metropol = plot_data(model_name=model_name, alg_str='MH', dim=dimension,
                             chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                             color='b', entries=5000, path=path)

    if dimension == 20:
        baseline = plot_data(model_name=model_name, alg_str='BaseHMC', dim=dimension,
                             chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                             color='g', entries=5000, path=path)
        simpline = plot_data(model_name=model_name, alg_str='SimpleRHMC', dim=dimension,
                             chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                             color='k',
                             entries=5000, label='NoVoP HMC')
        metropol = plot_data(model_name=model_name, alg_str='MH', dim=dimension,
                             chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                             color='b', entries=5000, path=path)

    if dimension == 50:
        metropol50 = plot_data(model_name=model_name, alg_str='MH', dim=dimension,
                               chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                               color='b', entries=5000, label='RWMH', path=path)
        base_hmc = plot_data(model_name=model_name, alg_str='BaseHMC', dim=dimension,
                             chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                             color='g--', entries=5000, label='Baseline HMC', path=path)
        novop_nuts = plot_data(model_name=model_name, alg_str='NovopNuts', dim=dimension,
                               chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                               color='m',
                               entries=5000, label='NoVoP NUTS', path=path)
        base_nuts = plot_data(model_name=model_name, alg_str='BaseNuts', dim=dimension,
                              chains=['L1'],
                              color='r-.',
                              entries=2000, label='Baseline NUTS', path=path)
        novop_hmc = plot_data(model_name=model_name, alg_str='Novop', dim=dimension,
                              chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'], color='k',
                              entries=5000, label='NoVoP HMC', path=path)

    plt.legend()
    plt.ylim(bottom=0)
    plt.ylim(top=3)
    plt.xlabel('Iteration (dim={d})'.format(d=dimension))
    plt.ylabel('Error (WMAE)')

    if FINAL_LATEX:
        plt.savefig(path + 'err_vs_itr.pgf')
    else:
        plt.show()


def experiment1_radial_plot_err_vs_time(dimension, l, eps, path):
    model_name = 'radial_bi'

    if dimension == 50:
        base_nuts = plot_data(model_name=model_name, alg_str='BaseNuts', dim=dimension,
                              chains=['L1'],
                              color='r-.',
                              entries=2000, label='Baseline NUTS', plot_type='ErrVsTime', path=path)
        metropol50 = plot_data(model_name=model_name, alg_str='_FIXED_TIME_MH', dim=dimension,
                               chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                               color='b', entries=1400, label='RWMH', plot_type='ErrVsTime', path=path)
        novop_nuts = plot_data(model_name=model_name, alg_str='NovopNuts', dim=dimension,
                               chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                               color='m',
                               entries=5000, label='NoVoP NUTS', plot_type='ErrVsTime', path=path)
        base_hmc = plot_data(model_name=model_name, alg_str='_FIXED_TIME_BaseHMC', dim=dimension,
                             chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                             color='g--', entries=1400, label='Baseline HMC', plot_type='ErrVsTime', path=path)
        novop_hmc = plot_data(model_name=model_name, alg_str='Novop', dim=dimension,
                              chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'], color='k',
                              entries=5000, label='NoVoP HMC', plot_type='ErrVsTime', path=path)

    plt.legend()
    plt.ylim(bottom=0)
    plt.ylim(top=3)
    plt.xlabel('Time (sec)')  # (dim={d})'.format(d=dimension))
    plt.ylabel('Error (WMAE)')

    if FINAL_LATEX:
        plt.savefig(path + 'err_vs_time.pgf')
    else:
        plt.show()


def experiment1_radial_sample_trace_vs_itr(dimension, l, eps, path):
    model_name = 'radial_bi'

    if dimension == 50:
        y_low = 0.5
        y_high = 3.6
        entries = 5000

        fig, (ax_mh, ax_bhmc, ax_nhmc, ax_bn, ax_nn) = plt.subplots(5, 1, sharex=True)
        plot_trace_vs_Itr(model_name=model_name, alg_str='MH', dim=dimension,
                          chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                          color='b', entries=entries, label='RWMH', axis=ax_mh, path=path)
        ax_mh.set_ylim([y_low, y_high])
        ax_mh.legend(loc='lower right')

        plot_trace_vs_Itr(model_name=model_name, alg_str='BaseHMC', dim=dimension,
                          chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                          color='g', entries=entries, label='Baseline HMC', axis=ax_bhmc, path=path)
        ax_bhmc.set_ylim([y_low, y_high])
        ax_bhmc.legend(loc='lower right')

        plot_trace_vs_Itr(model_name=model_name, alg_str='Novop', dim=dimension,
                          chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                          color='k', entries=entries, label='NoVoP HMC', axis=ax_nhmc, path=path)
        ax_nhmc.set_ylim([y_low, y_high])
        ax_nhmc.legend(loc='lower right')

        plot_trace_vs_Itr(model_name=model_name, alg_str='BaseNuts', dim=dimension,
                          chains=['L1'],
                          color='r',
                          entries=2000, label='Baseline NUTS', axis=ax_bn, path=path)
        ax_bn.set_ylim([y_low, y_high])
        ax_bn.legend(loc='lower right')

        plot_trace_vs_Itr(model_name=model_name, alg_str='NovopNuts', dim=dimension,
                          chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                          color='m', entries=entries, label='NoVoP NUTS', axis=ax_nn, path=path)
        ax_nn.set_ylim([y_low, y_high])
        ax_nn.legend(loc='lower right')

    plt.xlabel('Trace of samples versus Iteration')  # (dim={d})'.format(d=dimension))
    if FINAL_LATEX:
        plt.savefig(path + 'trace_vs_itr.pgf')
    else:
        plt.show()


def experiment1_radial_sample_trace_vs_time(dimension, l, eps, path):
    model_name = 'radial_bi'

    if dimension == 50:
        y_low = 0.5
        y_high = 3.6
        entries = 5000

        fig, (ax_mh, ax_bhmc, ax_nhmc, ax_bn, ax_nn) = plt.subplots(5, 1, sharex=True)
        plot_trace_vs_time(model_name=model_name, alg_str='_FIXED_TIME_MH', dim=dimension,
                           chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                           color='b', entries=1400, label='RWMH', axis=ax_mh, path=path)
        ax_mh.set_ylim([y_low, y_high])
        ax_mh.legend(loc='lower right')

        plot_trace_vs_time(model_name=model_name, alg_str='_FIXED_TIME_BaseHMC', dim=dimension,
                           chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                           color='g', entries=1400, label='Baseline HMC', axis=ax_bhmc, path=path)
        ax_bhmc.set_ylim([y_low, y_high])
        ax_bhmc.legend(loc='upper right')

        plot_trace_vs_time(model_name=model_name, alg_str='Novop', dim=dimension,
                           chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                           color='k', entries=entries, label='NoVoP HMC', axis=ax_nhmc, path=path)
        ax_nhmc.set_ylim([y_low, y_high])
        ax_nhmc.legend(loc='lower right')

        plot_trace_vs_time(model_name=model_name, alg_str='BaseNuts', dim=dimension,
                           chains=['L1'],
                           color='r',
                           entries=2000, label='Baseline NUTS', axis=ax_bn, path=path)
        ax_bn.set_ylim([y_low, y_high])
        ax_bn.legend(loc='lower middle')

        plot_trace_vs_time(model_name=model_name, alg_str='NovopNuts', dim=dimension,
                           chains=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
                           color='m', entries=entries, label='NoVoP NUTS', axis=ax_nn, path=path)
        ax_nn.set_ylim([y_low, y_high])
        ax_nn.legend(loc='lower middle')

    plt.xlabel('Trace of samples versus time (sec)')
    if FINAL_LATEX:
        plt.savefig(path + 'trace_vs_time.pgf')
    else:
        plt.show()




def mean_confidence_interval(data, confidence):
    """
    :param data: sample data
    :param confidence: confidence
    :return: confidence interval from sample data
    """
    a = 1.0 * numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def main():
    END_RESULTS_PATH_TO_PLOT = "./END_RESULTS/"

    # model_name = 'cube'  # 'cube'  # 'radial'
    model_name = 'radial_bi'
    if model_name == 'radial_bi':
        EXPERIMENT_TYPE = 'err_vs_itr'  # 'err_vs_time' #'trace_vs_itr'  # 'trace_vs_time'
        if EXPERIMENT_TYPE == 'err_vs_itr':
            experiment1_radial_plot_err_vs_itr(dimension=50, l=10, eps=0.1, path=END_RESULTS_PATH_TO_PLOT)
        elif EXPERIMENT_TYPE == 'err_vs_time':
            experiment1_radial_plot_err_vs_time(dimension=50, l=10, eps=0.1, path=END_RESULTS_PATH_TO_PLOT)
        elif EXPERIMENT_TYPE == 'trace_vs_itr':
            experiment1_radial_sample_trace_vs_itr(dimension=50, l=10, eps=0.1, path=END_RESULTS_PATH_TO_PLOT)
        elif EXPERIMENT_TYPE == 'trace_vs_time':
            experiment1_radial_sample_trace_vs_time(dimension=50, l=10, eps=0.1, path=END_RESULTS_PATH_TO_PLOT)
        else:
            raise Exception('Unknown type')
    elif model_name == 'cube':  # error vs iterations where A is randomly generated
        experiment1_cube_plot(dimension=5, l=10, eps=0.1, path=END_RESULTS_PATH_TO_PLOT)


if __name__ == "__main__":
    main()
