import copy
import sys
import os
from scipy.stats import mode
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import beta as beta_distr
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from itertools import combinations
from functools import reduce
from operator import iconcat
from math import comb
from natsort import index_natsorted
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from numpy import linalg as la
import numpy as np
import pandas as pd
import scipy.linalg
import random
import string
import dendropy
import nexus # pip3 install python-nexus
import warnings
# np.set_printoptions(suppress = True, precision = 3)
np.set_printoptions(threshold = sys.maxsize)
from collections.abc import Iterable
#from .extract_properties import *
SMALL_NUMBER = 1e-10

from .utilities import *


class FossilSimulator():
    def __init__(self,
                 range_q=[0.5, 5.],
                 range_alpha=None, # list of range e.g. [1.0, 5.0],
                 poi_shifts=0,
                 fixed_shift_times=[],
                 fixed_q=[],
                 q_loguniform=False,
                 alpha_loguniform=False,
                 age_effect=None, # list with alpha and beta of the beta distribution
                 age_effect_mass_extinction=None, # list with alpha and beta of the beta distribution
                 cat_trait_effect=None, # dict of lists for each categorical trait e.g. for first trait with three states {'1': [0.5, 1.0, 2.0]}. For two traits {'1': [0.5, 1.0, 2.0], '2': [1.0, 1.0, 3.0]}
                 cont_trait_effect=None, # dict of lists for each continuous trait, growth rate, min, and max e.g. for first trait {'1': [2.0, 0.2, 2.0]}. For two traits {'1': [2.0, 0.2, 2.0], '2': [4.5, 0.5, 1.5]}
                 paleoenv=None,
                 paleoenv_effect=None,
                 qtt_res=1,  # time resolution for the sampling rate through time
                 seed = -1):
        self.range_q = range_q
        self.range_alpha = range_alpha
        self.poi_shifts = poi_shifts
        self.fixed_shift_times = fixed_shift_times
        self.fixed_q = fixed_q
        self.q_loguniform = q_loguniform
        self.alpha_loguniform = alpha_loguniform
        self.age_effect = age_effect
        self.age_effect_mass_extinction = age_effect_mass_extinction
        self.cat_trait_effect = cat_trait_effect
        self.cont_trait_effect = cont_trait_effect
        self.paleoenv = paleoenv
        self.paleoenv_effect = paleoenv_effect
        self.qtt_res = qtt_res
        self.seed = seed
        self._init_seed()


    def _init_seed(self):
        if self.seed == -1:
            np.random.seed()
        else:
            np.random.seed(self.seed)


    def get_duration(self, sp_x, upper, lower):
        ts = np.copy(sp_x[:,0])
        te = np.copy(sp_x[:, 1])
        ts[ts > upper] = upper
        te[te < lower] = lower
        d = ts - te
        d[d < 0.0] = 0.0

        return d, ts, te


    def get_is_alive(self, sp_x):
        return sp_x[:,1] == 0.0


    def make_sampling_heterogeneity(self, sp_x):
        if self.range_alpha is None:
            alpha = np.array([-999])
            h = np.ones(self.n_taxa)
        else:
            if self.alpha_loguniform:
                log_range_alpha = np.log(self.range_alpha)
                log_alpha = np.random.uniform(np.min(log_range_alpha), np.max(log_range_alpha), 1)
                alpha = np.exp(log_alpha)
            else:
                alpha = np.random.uniform(np.min(self.range_alpha), np.max(self.range_alpha), 1)
            h = np.random.gamma(alpha, 1.0 / alpha, self.n_taxa)
        return alpha, h


    def make_sampling_rate(self, sp_x):
        root_age = np.max(sp_x)
        death_age = np.min(sp_x)
        shift_time_q = self.fixed_shift_times
        nS = len(self.fixed_shift_times)
        if nS == 0:
            # Number of rate shifts expected according to a Poisson distribution
            nS = np.random.poisson(self.poi_shifts)
            shift_time_q = np.random.uniform(0, root_age, nS) # Also works with nS = 0

        if len(self.fixed_q) > 0:
            q = np.array(self.fixed_q)
            if self.q_loguniform:
                q = np.exp(q)
        elif self.q_loguniform:
            log_range_q = np.log(self.range_q)
            log_q = np.random.uniform(np.min(log_range_q), np.max(log_range_q), nS + 1)
            q = np.exp(log_q)
        else:
            q = np.random.uniform(np.min(self.range_q), np.max(self.range_q), nS + 1)

        shift_time_q_lowres = np.concatenate((np.array(root_age), shift_time_q, np.zeros(1)), axis=None)
        shift_time_q_lowres = np.sort(shift_time_q_lowres)[::-1]
        shift_time_q_highres = np.arange(0.0, np.ceil(root_age), self.qtt_res)[::-1]
        shift_time_q_highres = shift_time_q_highres[shift_time_q_highres <= root_age]
        shift_time_q_highres = shift_time_q_highres[shift_time_q_highres >= death_age]
        shift_time_q = np.concatenate((shift_time_q_lowres, shift_time_q_highres), axis=None)
        shift_time_q = np.sort(np.unique(shift_time_q))[::-1]
        d = np.digitize(shift_time_q[1:], shift_time_q_lowres[1:-1], right=False)
        q = q[d]

        self.write_me_trait = False

        return q, shift_time_q, shift_time_q_lowres


    def get_fossil_occurrences(self, res_bd, q, shift_time_q, is_alive):
        sp_x = res_bd['ts_te']
        alpha, sampl_hetero = self.make_sampling_heterogeneity(sp_x)
        n_taxa = len(sp_x)
        occ = [np.array([])] * n_taxa
        qtt_taxa = np.full((self.n_taxa, len(q)), np.nan)
        qmtt_taxa = np.full((self.n_taxa, len(q)), np.nan)
        len_q = len(q)
        self.make_cont_trait_multiplier(res_bd)
        self.make_taxon_age_multipliers(res_bd, len_q, shift_time_q)
        self.make_paleoenv_multipliers(shift_time_q)

        for i in range(len_q):
            dur, ts, te = self.get_duration(sp_x, upper=shift_time_q[i], lower=shift_time_q[i + 1])
            dur = dur.flatten()
            cat_trait_multiplier = self.make_cat_trait_multiplier(res_bd,
                                                                  upper=shift_time_q[i + 1],
                                                                  lower=shift_time_q[i])
            cont_trait_multipliers = self.get_cont_traits_multipliers_time_slice(res_bd,
                                                                                 upper=shift_time_q[i + 1],
                                                                                 lower=shift_time_q[i])
            age_multipliers = self.age_multipliers[:, i].reshape(-1)
            paleo_multipliers = self.paleoenv_multipliers[i]
            q_multipliers = cat_trait_multiplier * cont_trait_multipliers * age_multipliers * paleo_multipliers
            poi_rate_occ = q[i] * q_multipliers * sampl_hetero * dur
            exp_occ = np.round(np.random.poisson(poi_rate_occ))
            non_zero_branch_length = dur > 0.0
            qtt_taxa[non_zero_branch_length, i] = poi_rate_occ[non_zero_branch_length] / dur[non_zero_branch_length]
            qmtt_taxa[non_zero_branch_length, i] = q_multipliers[non_zero_branch_length]

            for y in range(n_taxa):
                if exp_occ[y] != 0:
                    occ_y = np.random.uniform(te[y], ts[y], exp_occ[y])
                else:
                    occ_y = np.array([])
                present = np.array([])
                if is_alive[y] and i == (len_q - 1): # Alive and most recent sampling strata
                    present = np.zeros(1, dtype='float')
                occ[y] = np.concatenate((occ[y], occ_y, present))

        lineages_sampled = []
        occ2 = []
        for i in range(n_taxa):
            O = occ[i]
            O = O[O != 0.0] # Do not count single occurrence at the present
            if len(O) > 0:
                lineages_sampled.append(i)
                occ2.append(occ[i])
        lineages_sampled = np.array(lineages_sampled)

        lineages_sampled = lineages_sampled.astype(int)
        qtt_taxa = qtt_taxa[lineages_sampled, :]
        qmtt_taxa = qmtt_taxa[lineages_sampled, :]

        return occ2, lineages_sampled, alpha, qtt_taxa, qmtt_taxa


    def harmonic_mean_q_through_time(self, q_rates, shift_time_q):
        qtt = np.full(q_rates.shape[1], np.nan)
        not_all_nan = np.sum(np.isnan(q_rates), axis=0) < q_rates.shape[0]
        qtt[not_all_nan] = 1 / np.nanmean(1 / q_rates[:, not_all_nan], axis=0)

        qtt = np.concatenate((qtt, qtt[-1]), axis=None)
        qtt = np.c_[shift_time_q, qtt]
        qtt = pd.DataFrame(qtt, columns=['time', 'q'])

        return qtt


    def baseline_q_through_time(self, q, shift_time_q):
        q = np.concatenate((q, q[-1]), axis=None)
        baseline_qtt = np.c_[shift_time_q, q]
        baseline_qtt = pd.DataFrame(baseline_qtt, columns=['time', 'q'])

        return baseline_qtt


    def get_taxon_names(self, lineages_sampled):
        num_taxa = len(lineages_sampled)
        taxon_names = []
        for i in range(num_taxa):
            taxon_names.append('T%s' % lineages_sampled[i])

        return taxon_names


    def make_cat_trait_multiplier(self, res_bd, upper=np.inf, lower=-np.inf):
        multipliers = np.ones(self.n_taxa)
        if not self.cat_trait_effect is None and res_bd['cat_traits'].size > 0:
            cat_traits = get_majority_cat_trait_per_taxon(res_bd, sim_fossil=None, upper=upper, lower=lower)
            num_cat_traits = cat_traits.shape[1]
            for i in range(num_cat_traits):
                if str(i + 1) in self.cat_trait_effect.keys():
                    multipliers *= np.array(self.cat_trait_effect[str(i + 1)])[cat_traits[:, i]]
        return multipliers


    def get_cont_traits_multipliers_time_slice(self, res_bd, upper=-np.inf, lower=np.inf):
        time = res_bd['true_rates_through_time']['time']
        # larger time value: more distant past; negative value: future
        time = np.concatenate((-np.inf, time, np.inf), axis=None)
        trait_idx = np.logical_and(time < lower, time >= upper)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category = RuntimeWarning)
            mean_multiplier = np.nanmean(self.cont_trait_multipliers[trait_idx, :, :], axis=0).reshape(-1)
        mean_multiplier[np.isnan(mean_multiplier)] = 1.0

        return mean_multiplier


    def trans_to_mean_1_and_min_max(self, x, m, M, tol=1e-6):

        def get_deviation(alpha):
            x_pow = x ** alpha
            #x_scaled = m + (x_pow - np.min(x_pow)) * (M - m) / (np.max(x_pow) - np.min(x_pow))
            x_scaled = m + x_pow * (M - m)
            x_final = x_scaled / np.nanmean(x_scaled)
            err_min = (np.nanmin(x_final) - m) ** 2
            err_max = (np.nanmax(x_final) - M) ** 2
            return err_min + err_max

        result = minimize_scalar(get_deviation, bounds=(0.01, 10), method='bounded', options={'xatol': tol})
        alpha_opt = result.x
        # Apply transformation with optimal alpha
        x **= alpha_opt
        x_scaled = m + (x - np.nanmin(x)) * (M - m) / (np.nanmax(x) - np.nanmin(x))
        x_scaled /= np.nanmean(x_scaled)
        return x_scaled


    # def make_cont_trait_multiplier(self, res_bd, upper=np.inf, lower=-np.inf):
    #     # n_lineages = res_bd['ts_te'].shape[0]
    #     self.cont_trait_multiplier = np.ones(self.n_taxa)
    #     if not self.cont_trait_effect is None and res_bd['cont_traits'].size > 0:
    #         cont_trait = self.get_cont_traits_time_slice(res_bd, upper, lower)
    #         # sd = np.diag(res_bd['expected_sd_cont_traits'])
    #         sd = np.std(np.nanmean(res_bd['cont_traits'], axis=0), axis=-1)
    #         num_cat_traits = len(sd)
    #         for i in range(num_cat_traits):
    #             if str(i + 1) in self.cont_trait_effect.keys():
    #                 k = self.cont_trait_effect[str(i + 1)][0] # growth rate
    #                 m = self.cont_trait_effect[str(i + 1)][1] # min of effect
    #                 M = self.cont_trait_effect[str(i + 1)][2] # max of effect
    #                 x = cont_trait[i, :]
    #                 # scale to [-1, 1]; with '3', 99.7% are within [-1, 1]
    #                 # don't perform a min max scaling because this would be done for each qshift
    #                 # and the sd will be low in the earliest bin
    #                 x /= (3 * sd[i])
    #                 self.cont_trait_multiplier *= (M - m) / (1 + np.exp(-k * x)) + m
    #                 y = 1 / (1 + np.exp(-k * x))
    #                 print('x:', np.mean(x), np.min(x), np.max(x))
    #                 print('y:', np.mean(y), np.min(y), np.max(y))
    #                 bla = self.trans_to_mean_1_and_min_max(y, m, M)
    #                 print('trans y', np.mean(bla), np.min(bla), np.max(bla))

    def make_cont_trait_multiplier(self, res_bd):
        cont_traits = res_bd['cont_traits']
        self.cont_trait_multipliers = np.ones_like(cont_traits)
        if not self.cont_trait_effect is None and res_bd['cont_traits'].size > 0:
            # sd = np.diag(res_bd['expected_sd_cont_traits'])
            sd = np.std(np.nanmean(cont_traits, axis=0), axis=-1)
            num_cat_traits = len(sd)
            for i in range(num_cat_traits):
                if str(i + 1) in self.cont_trait_effect.keys():
                    k = self.cont_trait_effect[str(i + 1)][0]  # growth rate
                    m = self.cont_trait_effect[str(i + 1)][1]  # min of effect
                    M = self.cont_trait_effect[str(i + 1)][2]  # max of effect
                    x = cont_traits[:, i, :] / (3 * sd[i])
                    multipliers = 1 / (1 + np.exp(-k * x))
                    # print('x:', np.nanmean(x), np.nanmin(x), np.nanmax(x))
                    # print('y:', np.nanmean(multipliers), np.nanmin(multipliers), np.nanmax(multipliers))
                    self.cont_trait_multipliers[:, i, :] = self.trans_to_mean_1_and_min_max(multipliers, m, M)
                    # print('trans y', np.nanmean(multipliers), np.nanmin(multipliers), np.nanmax(multipliers))


    def make_taxon_age_multipliers(self, res_bd, len_q, shift_time_q):
        self.age_multipliers = np.ones((self.n_taxa, len_q))

        if not self.age_effect is None:
            ts = res_bd['ts_te'][:, 0]
            te = res_bd['ts_te'][:, 1]

            me_vict = res_bd['mass_ext_victim']
            if not self.age_effect_mass_extinction is None and np.any(me_vict == 1):
                self.write_me_trait = True

            for i in range(self.n_taxa):
                taxon_bins = np.unique(np.concatenate((ts[i], shift_time_q, te[i]), axis=None))[::-1]
                taxon_bins = taxon_bins[taxon_bins <= ts[i]]
                taxon_bins = taxon_bins[taxon_bins >= te[i]]

                ts_equals_shift = np.isin(ts[i], shift_time_q) and ts[i] < shift_time_q[0]
                if ts_equals_shift:
                    taxon_bins = np.concatenate((taxon_bins[0] + self.qtt_res / 100.0, taxon_bins), axis=None)
                m_idx = np.digitize(taxon_bins, shift_time_q[1:])

                not_dupl = np.unique(m_idx, return_index=True)[1]
                m_idx = m_idx[not_dupl]

                # scale [0, 1] for beta distribution
                if len(m_idx) > 1:
                    M_bin = np.max(taxon_bins)
                    m_bin = np.min(taxon_bins)
                    taxon_bins = (taxon_bins - m_bin) / (M_bin - m_bin)

                    for j in range(len(taxon_bins) - 1):
                        x = np.linspace(taxon_bins[j], taxon_bins[j + 1], 100)
                        # Why do I need to swap alpha and beta?
                        b = beta_distr.pdf(x, self.age_effect[1], self.age_effect[0])
                        if not self.age_effect_mass_extinction is None and me_vict[i] == 1:
                            b = beta_distr.pdf(x, self.age_effect_mass_extinction[1], self.age_effect_mass_extinction[0])
                        self.age_multipliers[i, m_idx[j]] = np.mean(b)


    def bin_env(self, shift_time_q):
        self.binned_env = np.zeros(len(shift_time_q) - 1)
        self.paleoenv = self.paleoenv[np.argsort(self.paleoenv[:, 0]), :]
        temp_res_q = np.min(np.diff(shift_time_q[::-1]))
        if temp_res_q <= np.min(np.diff(self.paleoenv[:, 0])):
            # subsample when temporal resolution of qShifts is smaller than of paleoenv
            highres_time_paleoenv = np.sort(np.unique(np.concatenate((np.arange(0, np.max(self.paleoenv[:, 0]), temp_res_q), self.paleoenv[:, 0]), axis=None)))[
                ::-1]
            rep_ind = np.searchsorted(self.paleoenv[:, 0], highres_time_paleoenv, side='right') - 1
            rep_ind = rep_ind[::-1]
            self.paleoenv = self.paleoenv[rep_ind, :]
            self.paleoenv[:, 0] = highres_time_paleoenv[::-1]
        for i in range(len(shift_time_q) - 1):
            idx = np.logical_and(self.paleoenv[:, 0] < shift_time_q[i], self.paleoenv[:, 0] >= shift_time_q[i + 1])
            self.binned_env[i] = np.mean(self.paleoenv[idx, 1])


    def make_paleoenv_multipliers(self, shift_time_q):
        self.paleoenv_multipliers = np.ones(len(shift_time_q) - 1)
        if not self.paleoenv_effect is None:
            self.bin_env(shift_time_q)
            sd = np.std(self.binned_env)
            k = self.paleoenv_effect[0]  # growth rate
            m = self.paleoenv_effect[1]  # min of effect
            M = self.paleoenv_effect[2]  # max of effect
            x = self.binned_env / (3 * sd)
            multipliers = 1 / (1 + np.exp(-k * x))
            self.paleoenv_multipliers = self.trans_to_mean_1_and_min_max(multipliers, m, M)


    def run_simulation(self, res_bd):
        sp_x = res_bd['ts_te']
        self.n_taxa = sp_x.shape[0]
        is_alive = self.get_is_alive(sp_x)

        q, shift_time_q, shift_time_q_write = self.make_sampling_rate(sp_x)
        fossil_occ, taxa_sampled, alpha, qtt_taxa, qmtt_taxa = self.get_fossil_occurrences(res_bd, q, shift_time_q, is_alive)

        taxon_names = self.get_taxon_names(taxa_sampled)
        qtt = self.harmonic_mean_q_through_time(qtt_taxa, shift_time_q)
        q_baseline = self.baseline_q_through_time(q, shift_time_q)
        shift_time_q_write = shift_time_q_write[1:-1]
        qtt_taxa = pd.DataFrame(qtt_taxa, columns=shift_time_q[1:].tolist(), index=taxon_names)
        qmtt_taxa = pd.DataFrame(qmtt_taxa, columns=shift_time_q[1:].tolist(), index=taxon_names)

        d = {'fossil_occurrences': fossil_occ,
             'taxon_names': taxon_names,
             'taxa_sampled': taxa_sampled,
             'q': q,
             'shift_time': shift_time_q_write,
             'alpha': alpha,
             'q_baseline': q_baseline,
             'qtt': qtt,
             'qtt_taxa': qtt_taxa,
             'qmultitt_taxa': qmtt_taxa,
             'write_me_trait': self.write_me_trait}

        return d
