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
4/4/20
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import scipy.stats as stats


class VanillaMetropolisHastingsSampler(object):
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
            return proposed_q, True  # accept -- NOTE: second value is boolean rather than trajectory
        else:
            return current_q, False  # reject


class AutoTuneMetropolisHastingsSampler(object):
    """
    MH sampler that tunes its proposal variance parameter by testing ,say 100, equi-distance variances in (0,max_variance]
    and taking the one with acceptance rate closest to 0.24
    """

    def __init__(self, model, init_sample, max_variance=3.0, num_variances_to_be_tested=10,
                 num_samples_per_variance=100,
                 return_extra_info=True):
        self.return_extra_info = return_extra_info
        best_variance = None
        best_acceptance_rate = numpy.inf

        for i in range(1, num_variances_to_be_tested + 1):
            var = i * max_variance / num_variances_to_be_tested
            inner_sampler = VanillaMetropolisHastingsSampler(model=model, proposal_variance=var)

            sample = init_sample.copy()
            accepted_count = 0
            for _ in range(1, num_samples_per_variance + 1):
                sample, accepted = inner_sampler.generate_sample(sample)
                if accepted:
                    accepted_count += 1
            acceptance_rate = accepted_count / num_samples_per_variance
            print('for var: ', var, ' acceptance_rate = ', acceptance_rate)
            if abs(acceptance_rate - 0.24) < abs(best_acceptance_rate - 0.24):
                best_acceptance_rate = acceptance_rate
                best_variance = var

        print('best var: ', best_variance)
        self.inner_sampler = VanillaMetropolisHastingsSampler(model=model, proposal_variance=best_variance)

    def generate_sample(self, current_q):
        sample, info = self.inner_sampler.generate_sample(current_q)
        if self.return_extra_info:
            return sample, info
        else:
            return sample
