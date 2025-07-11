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


class bdnn_simulator():
    def __init__(self,
                 s_species = 1,  # number of starting species
                 rangeSP = [100, 1000],  # min/max size data set
                 minEX_SP = 0,  # minimum number of extinct lineages allowed
                 minExtant_SP = 0, # minimum number of extant lineages
                 maxExtant_SP = np.inf,  # maximum number of extant lineages
                 timewindow_rangeSP = None, # time window for which the min/max size of the data set should be checked e.g. np.array([20., 10.])
                 root_r = [30., 100],  # range root ages
                 rangeL = [0.2, 0.5], # range speciation rate
                 rangeM = [0.2, 0.5], # range extinction rate
                 scale = 100., # root * scale = steps for the simulation
                 mass_extinction_prob = 0.0,
                 mass_extinction_magnitude = [0.0], # single value or range
                 mass_extinction_times = [-1.0], # array of ages with mass extinction events, or random
                 mass_extinction_trait_dependent = False, # should mass extinction magnitude be trait/env specific?
                 poiL = 0, # Number of rate shifts expected according to a Poisson distribution
                 poiM = 0, # Number of rate shifts expected according to a Poisson distribution
                 range_linL = None, # None or a range (e.g. [-0.2, 0.2])
                 range_linM = None, # None or a range (e.g. [-0.2, 0.2])
                 # fix speciation rate through time
                 # numpy 2D array with time in the 1st column and rate in the 2nd
                 # skyline trajectory as in Silvestro et al., 2019 Paleobiology:
                 # np.array([[35., 0.4], [20.001, 0.4], [20., 0.1], [10.001, 0.1], [10., 0.01], [0.0, 0.01]])
                 # decline until 20 Ma and then constant
                 # np.array([[35., 0.4], [20., 0.1], [0.0, 0.1]])
                 # overwrittes poiL and range_linL
                 fixed_Ltt = None,
                 fixed_Mtt = None, # fix extinction rate through time (see fixed Ltt)
                 n_cont_traits = [0, 0], # number of continuous traits
                 cont_traits_sigma_clado = [0.0, 0.0], # evolutionary rates for continuous traits at speciation event vv
                 cont_traits_sigma = [0.1, 0.5], # evolutionary rates for continuous traits
                 cont_traits_cor = [-1, 1], # evolutionary correlation between continuous traits
                 cont_traits_Theta1 = [0, 0],  # morphological optima; 0 is no directional change from the ancestral values
                 cont_traits_alpha = [0, 0],  # strength of attraction towards Theta1; 0 is pure Brownian motion; [0.5, 2.0] is sensible
                 cont_traits_effect_sp = np.array([[[[SMALL_NUMBER, SMALL_NUMBER]]]]), # 4D array; range of effect of continuous traits on speciation (0 is no effect)
                 cont_traits_effect_ex = np.array([[[[SMALL_NUMBER, SMALL_NUMBER]]]]), # 4D array; range of effect of continuous traits on extinction (0 is no effect)
                 cont_traits_effect_bellu_sp = np.array([[[[1, -1]]]]), # 4D array; whether the effect causes a bell-shape (1) or a u-shape (-1) over the trait range
                 cont_traits_effect_bellu_ex = np.array([[[[1, -1]]]]), # 4D array; whether the effect causes a bell-shape (1) or a u-shape (-1) over the trait range
                 cont_traits_effect_optimum_sp = np.array([[[[0., 0.]]]]), # 3D array
                 cont_traits_effect_optimum_ex = np.array([[[[0., 0.]]]]), # 3D array
                 cont_traits_effect_shift_sp = None, # 1D numpy array with shift times
                 cont_traits_effect_shift_ex = None,  # 1D numpy array with shift times
                 n_cat_traits = [0, 0], # range of the number of categorical traits
                 n_cat_traits_states = [2, 5], # range number of states for categorical trait, can be set to [0,0] to avid any trait
                 cat_traits_ordinal = [False, False], # Is categorical trait ordinal or discrete? Random sampling of one of these values
                 cat_traits_dir = 2.0, # concentration parameter dirichlet distribution for transition probabilities between categorical states
                 cat_traits_diag = None, # fix diagonal of categorical transition matrix to this value (overwrites cat_traits_dir), probability of no state change
                 # range of effect of categorical traits on speciation (1st row) and extinction (2nd row) (1 is no effect)
                 # effects can be fixed with e.g. np.array([[2.3., 2.3.],[1.5.,1.5.]]) and cat_traits_effect_decr_incr = np.array([[True, True],[False, False]])
                 # effect of n_cat_traits_states > 2 can be fixed with n_cat_traits_states = [3, 3] AND np.array([[1.5., 2.3.],[0.2.,1.5.]]) (no need for cat_traits_effect_decr_incr)
                 # or in case of 4 states with n_cat_traits_states = [4, 4] AND np.array([[1.5., 2.3., 0.6],[0.2.,1.5., 1.9]])
                 cat_traits_effect = np.array([[1., 1.],
                                               [1., 1.]]),
                 cat_traits_effect_decr_incr = np.array([[True, False],
                                                         [True, False]]), # should categorical effect cause a decrease (True) or increase (False) in speciation (1st row) and extinction (2nd row)?
                 cat_traits_min_freq = [0], # list of length 1 or equal to n_cat_traits, minimum frequency of state
                 n_areas = [0, 0], # number of biogeographic areas (minimum of 2)
                 dispersal = [0.1, 0.3], # range for the rate of area expansion
                 extirpation = [0.05, 0.1], # range for the rate of area loss
                 sp_env_file = None, # Path to environmental file influencing speciation
                 sp_env_eff = [0.00001, 0.00001],  # range environmental effect on speciation rate
                 ex_env_file = None,  # Path to environmental file influencing speciation
                 ex_env_eff = [0.00001, 0.00001],  # range environmental effect on speciation rate
                 # List with multiplier for the states of the first categorical traits for environmental effect
                 # e.g. for three states [[0.0, 1.0, -0.5], [None]]
                 # state0 no environmental effect, state1 1 * sp_env_eff, state2 -0.5 * sp_env_eff
                 # No modification of environmental effect on extinction
                 env_effect_cat_trait = [None, None],
                 K_lam = None, # carrying capacity K
                 K_mu = None, # carrying capacity K
                 # fix carrying capacity K through time
                 # numpy 2D array with time in the 1st column and rate in the 2nd
                 # np.array([[35., 200.], [20.001, 200.], [20., 110.], [10.001, 110.], [10., 80.], [0.0, 80.]])
                 # decline until 20 Ma and then constant
                 # np.array([[35., 100.], [20., 50.], [0.0, 50.]])
                 # overwrittes K_lam
                 fixed_K_lam = None,
                 fixed_K_mu = None,
                 seed = None):
        self.s_species = s_species
        self.rangeSP = rangeSP
        self.minSP = np.min(rangeSP)
        self.maxSP = np.max(rangeSP)
        self.minEX_SP = minEX_SP
        self.minExtant_SP = minExtant_SP
        self.maxExtant_SP = maxExtant_SP
        self.timewindow_rangeSP = timewindow_rangeSP
        self.minSP_timewindow = copy.deepcopy(self.minSP)
        self.maxSP_timewindow = copy.deepcopy(self.maxSP)
        self.root_r = root_r
        self.rangeL = rangeL
        self.rangeM = rangeM
        self.scale = scale
        self.mass_extinction_prob = mass_extinction_prob
        self.mass_extinction_magnitude = np.sort(mass_extinction_magnitude)
        self.mass_extinction_times = np.array(mass_extinction_times)
        self.mass_extinction_trait_dependent = mass_extinction_trait_dependent
        self.poiL = poiL
        self.poiM = poiM
        self.range_linL = range_linL
        self.range_linM = range_linM
        self.fixed_Ltt = fixed_Ltt
        self.fixed_Mtt = fixed_Mtt
        self.n_cont_traits = n_cont_traits
        self.cont_traits_sigma_clado = cont_traits_sigma_clado
        self.cont_traits_sigma = cont_traits_sigma
        self.cont_traits_cor = cont_traits_cor
        self.cont_traits_Theta1 = cont_traits_Theta1
        self.cont_traits_alpha = cont_traits_alpha
        self.cont_traits_effect_sp = cont_traits_effect_sp
        self.cont_traits_effect_ex = cont_traits_effect_ex
        self.cont_traits_effect_bellu_sp = cont_traits_effect_bellu_sp
        self.cont_traits_effect_bellu_ex = cont_traits_effect_bellu_ex
        self.cont_traits_effect_optimum_sp = cont_traits_effect_optimum_sp
        self.cont_traits_effect_optimum_ex = cont_traits_effect_optimum_ex
        self.cont_traits_effect_shift_sp = cont_traits_effect_shift_sp
        self.cont_traits_effect_shift_ex = cont_traits_effect_shift_ex
        self.n_cat_traits = n_cat_traits
        self.n_cat_traits_states = n_cat_traits_states
        self.cat_traits_ordinal = cat_traits_ordinal
        self.cat_traits_dir = cat_traits_dir
        self.cat_traits_diag = cat_traits_diag
        self.cat_traits_effect = cat_traits_effect
        self.cat_traits_effect_decr_incr = cat_traits_effect_decr_incr
        self.cat_traits_min_freq = cat_traits_min_freq
        self.n_areas = n_areas
        self.dispersal = dispersal
        self.extirpation = extirpation
        self.sp_env_file = sp_env_file
        self.sp_env_eff = sp_env_eff
        self.ex_env_file = ex_env_file
        self.ex_env_eff = ex_env_eff
        self.env_effect_cat_trait = env_effect_cat_trait
        self.K_lam = K_lam,
        self.K_mu = K_mu,
        self.fixed_K_lam = fixed_K_lam,
        self.fixed_K_mu = fixed_K_mu,
        self.seed = seed
        self._init_seed()


    def _init_seed(self):
        if self.seed is None:
            np.random.seed()
        else:
            np.random.seed(self.seed)

    def simulate(self, L, M, root, dT,
                 n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_varcov_clado,
                 cont_trait_effect_sp, cont_trait_effect_ex, expected_sd_cont_traits,
                 cont_traits_effect_shift_sp, cont_traits_effect_shift_ex,
                 n_cat_traits, cat_states, cat_traits_Q, cat_trait_effect,
                 n_areas, dispersal, extirpation, env_eff_sp, env_eff_ex):
        ts = list()
        te = list()
        nwk = list()

        root = int(root * self.scale)
        # Trace ancestor descendant relationship
        # First entry: ancestor (for the seeding species, this is an index of themselfs)
        # Following entries: descendants
        anc_desc = []

        # Track time of origin/extinction (already in ts and te) and rates per lineage
        lineage_rates = []

        # Track if lineage is a victim of mass extinction
        lineage_victim_mass_extinction = []

        for i in range(self.s_species):
            ts.append(root)
            te.append(-0.0)
            anc_desc.append(str(i))
            lineage_rates_tmp = np.zeros(5 + 3 * n_cont_traits + 2 * n_cat_traits)
            lineage_rates_tmp[:] = np.nan
            lineage_rates_tmp[:5] = np.array([root, -0.0, L[root], M[root], 0.0])
            lineage_rates.append(lineage_rates_tmp)
            lineage_victim_mass_extinction.append(0)

        # init newick string (not working with > 1 self.s_species)
        nwk_str = 'No newick string with more than one starting species'
        if self.s_species == 1:
            nwk.append('T0:%s' % self.format_age_for_newick(0.0))
            nwk_str = nwk[0] + ';'

            # init continuous traits (if there are any to simulate)
        root_plus_1 = np.abs(root) + 2

        # init categorical traits
        cat_traits = np.empty((root_plus_1, n_cat_traits, self.s_species))
        cat_traits[:] = np.nan
        # init continuous traits
        cont_traits = np.empty((root_plus_1, n_cont_traits, self.s_species))
        cont_traits[:] = np.nan
        # init lineage-specific rates through time
        lineage_rates_through_time = np.empty((root_plus_1, 2, self.s_species))
        lineage_rates_through_time[:] = np.nan

        for i in range(self.s_species):
            cat_trait_yi = 0
            if n_cat_traits > 0:
                for y in range(n_cat_traits):
                    if self.cat_traits_diag is None:
                        cat_traits_Q[y] = dT * cat_traits_Q[y] * 0.01  # Only for anagenetic evolution of categorical traits
                    pi = self.get_stationary_distribution(cat_traits_Q[y])
                    cat_trait_yi = int(np.random.choice(cat_states[y], 1, p = pi))
                    cat_traits[-1, y, i] = cat_trait_yi
                    lineage_rates[i][2] = lineage_rates[i][2] * cat_trait_effect[y][0, cat_trait_yi]
                    lineage_rates[i][3] = lineage_rates[i][3] * cat_trait_effect[y][1, cat_trait_yi]
                    lineage_rates[i][(5 + 3 * n_cont_traits + y):(6 + 3 * n_cont_traits + y)] = cat_trait_yi
                    # lineage_rates[i][2] = L[root] * cat_trait_effect[y][0, int(cat_trait_yi)]
                    # lineage_rates[i][3] = M[root] * cat_trait_effect[y][1, int(cat_trait_yi)]
            if n_cont_traits > 0:
                Theta0 = np.zeros(n_cont_traits)
                cont_traits_i = self.evolve_cont_traits(Theta0, n_cont_traits, cont_traits_alpha, cont_traits_Theta1, cont_traits_varcov) # from past to present
                cont_traits[-1, :, i] = cont_traits_i
                lineage_rates[i][5:(5 + n_cont_traits)] = cont_traits_i
                # print('lineage_rates[i]: ', lineage_rates[i])
                # print('current state: ', lineage_rates[i][5 + n_cont_traits])
                lineage_rates[i][2] = self.get_rate_by_cont_trait_transformation(lineage_rates[i][2],
                                                                                 cont_traits_i,
                                                                                 cont_trait_effect_sp[0, :, cat_trait_yi, :],
                                                                                 expected_sd_cont_traits,
                                                                                 n_cont_traits)
                lineage_rates[i][3] = self.get_rate_by_cont_trait_transformation(lineage_rates[i][3],
                                                                                 cont_traits_i,
                                                                                 cont_trait_effect_ex[0, :, cat_trait_yi, :],
                                                                                 expected_sd_cont_traits,
                                                                                 n_cont_traits)

        # init biogeography
        biogeo = np.empty((root_plus_1, 1, self.s_species))
        biogeo[:] = np.nan
        biogeo[-1,:,:] = np.random.choice(np.arange(n_areas + 1), self.s_species)
        areas_comb = []
        if n_areas > 1:
            biogeo_states = np.arange(2**n_areas - 1)
            DEC_Q, areas_comb = self.make_anagenetic_DEC_matrix(n_areas, dT * dispersal, dT * extirpation)
            # DEC_clado_weight = self.get_DEC_clado_weight(n_areas)

        mass_ext_time = []
        mass_ext_mag = []

        # Trait dependent rates through time
        lineage_weighted_lambda_tt = np.zeros(np.abs(root))
        lineage_weighted_mu_tt = np.zeros(np.abs(root))

        me_prob = self.mass_extinction_prob / self.scale
        me_times = self.mass_extinction_times * self.scale

        # evolution (from the past to the present)
        exceeded_diversity = False
        for t in range(root, 0):
            # time i.e. integers self.root * self.scale
            # t = 0 not simulated!
            t_abs = abs(t)
            # print('t: ', t)
            l = L[t]
            m = M[t]

            TE = len(te)
            if self.timewindow_rangeSP is None:
                if TE > self.maxSP:
                    exceeded_diversity = True
                    break
            elif (self.timewindow_rangeSP[1] - t_abs) == 0.0:
                rangeSP_OK_in_timewindow, diversity_in_window = self.check_diversity_in_timewindow(-np.array(ts) / self.scale, -np.array(te) / self.scale)
                if rangeSP_OK_in_timewindow is False:
                    exceeded_diversity = True
                    break
            elif TE > self.maxSP_timewindow and t_abs <= self.timewindow_rangeSP[0] and t_abs >= self.timewindow_rangeSP[1]:
                exceeded_diversity = True
                break

            ran_vec = np.random.random(TE)
            te_extant = np.where(np.array(te) == 0)[0]
            ran_vec_cat_trait = np.random.random(TE * n_cat_traits).reshape((TE, n_cat_traits))
            ran_vec_biogeo = np.random.random(TE)

            no = np.random.uniform(0, 1)  # draw a random number
            no_extant_lineages = len(te_extant)  # the number of currently extant species
            mass_ext = False
            if (no < me_prob and no_extant_lineages > 10) or np.isin(t_abs, me_times):  # mass extinction condition
                mass_ext = True
                # print("Mass extinction", t_abs / self.scale, mass_ext)
                # increased loss of species: increased ext probability for this time bin
                m_me = np.random.uniform(self.mass_extinction_magnitude[0], self.mass_extinction_magnitude[-1])
                mass_ext_time.append(t_abs)
                mass_ext_mag.append(m_me)

            # Trait dependent rates for all extant lineages
            lineage_lambda = np.zeros(TE)
            lineage_lambda[:] = np.nan
            lineage_mu = np.zeros(TE)
            lineage_mu[:] = np.nan

            num_sp_events_at_t = 0

            for j in te_extant:  # extant lineages
                l_j = l + 0.
                m_j = m + 0.

                # environmental effect
                if self.sp_env_file is not None:
                    eff_sp = env_eff_sp
                    cat_trait_j = 0
                    if n_cat_traits > 0 and self.env_effect_cat_trait[0] is not None:
                        cat_trait_j = int(cat_traits[t_abs + 1, 0, j])
                    #     eff_sp = eff_sp * self.env_effect_cat_trait[0][cat_trait_j]
                    # eff_sp = np.exp(eff_sp * env_sp[t_abs])
                    # l_j = float(l_j * eff_sp)
                    l_j = self.get_rate_by_env_transformation(l_j, t_abs, eff_sp, rate_type = 'l', cate_state = cat_trait_j)
                if self.ex_env_file is not None:
                    eff_ex = env_eff_ex
                    cat_trait_j = 0
                    if n_cat_traits > 0 and self.env_effect_cat_trait[1] is not None:
                        cat_trait_j = int(cat_traits[t_abs + 1, 0, j])
                    #     eff_ex = env_eff_ex * self.env_effect_cat_trait[1][cat_trait_j]
                    # eff_ex = np.exp(eff_ex * env_ex[t_abs])
                    # m_j = float(m_j * eff_ex)
                    m_j = self.get_rate_by_env_transformation(m_j, t_abs, eff_ex, rate_type = 'm', cate_state = cat_trait_j)

                # categorical trait evolution
                cat_trait_j = 0
                if n_cat_traits > 0:
                    for y in range(n_cat_traits):
                        if self.cat_traits_diag is None:
                            cat_trait_j = self.evolve_cat_traits_ana(cat_traits_Q[y], cat_traits[t_abs + 1, y, j],
                                                                     ran_vec_cat_trait[j, y], cat_states[y])
                        else:
                            cat_trait_j = cat_traits[t_abs + 1, y, j]  # No change along branches
                        cat_trait_j = int(cat_trait_j)
                        cat_traits[t_abs, y, j] = cat_trait_j
                        l_j = l_j * cat_trait_effect[y][0, cat_trait_j]
                        m_j = m_j * cat_trait_effect[y][1, cat_trait_j]

                # continuous trait evolution
                if n_cont_traits > 0:
                    cont_trait_j = self.evolve_cont_traits(cont_traits[t_abs + 1, :, j], n_cont_traits, cont_traits_alpha, cont_traits_Theta1, cont_traits_varcov)
                    cont_traits[t_abs, :, j] = cont_trait_j
                    cont_traits_bin = cont_traits_effect_shift_sp[t_abs]
                    l_j = self.get_rate_by_cont_trait_transformation(l_j,
                                                                     cont_trait_j,
                                                                     cont_trait_effect_sp[cont_traits_bin, :, cat_trait_j, :],
                                                                     expected_sd_cont_traits,
                                                                     n_cont_traits)
                    cont_traits_bin = cont_traits_effect_shift_ex[t_abs]
                    m_j = self.get_rate_by_cont_trait_transformation(m_j,
                                                                     cont_trait_j,
                                                                     cont_trait_effect_ex[cont_traits_bin, :, cat_trait_j, :],
                                                                     expected_sd_cont_traits,
                                                                     n_cont_traits)
                if self.K_lam[0] is not None or self.fixed_K_lam[0] is not None:
                    l_j = self.get_divdep_lam(l_j, no_extant_lineages, t_abs)
                if self.K_mu[0] is not None or self.fixed_K_mu[0] is not None:
                    m_j = self.get_divdep_mu(m_j, no_extant_lineages, t_abs)

                lineage_lambda[j] = l_j
                lineage_mu[j] = m_j

                # range evolution
                if n_areas > 1:
                    biogeo_j = self.evolve_cat_traits_ana(DEC_Q, biogeo[t_abs + 1, 0, j], ran_vec_biogeo[j], biogeo_states)
                    biogeo[t_abs, 0, j] = biogeo_j
                    # if biogeo_j > n_areas:
                    #     m_j = m_j * DEC_clado_weight

                ran = ran_vec[j]

                # speciation
                if ran < l_j:
                    num_sp_events_at_t += 1
                    te.append(-0.0)  # add species
                    ts.append(t)  # sp time
                    anc_desc.append(str(len(ts) - 1) + '_' +  str(j))

                    lineage_rates_tmp = np.zeros(5 + 3 * n_cont_traits + 2 * n_cat_traits)
                    l_new = l + 0.0
                    m_new = m + 0.0

                    # Inherit traits
                    cat_trait_new = 0
                    if n_cat_traits > 0:
                        cat_traits_new_species = self.empty_traits(root_plus_1, n_cat_traits)
                        if self.cat_traits_diag is None:
                            cat_traits_new_species[t_abs, :] = cat_traits[t_abs, :, j] # inherit state at speciation
                        else:
                            for y in range(n_cat_traits):
                                # Change of categorical trait at speciation
                                ancestral_cat_trait = cat_traits[t_abs, y, j]
                                cat_trait_new = self.evolve_cat_traits_clado(cat_traits_Q[y], ancestral_cat_trait, cat_states[y])
                                cat_trait_new = int(cat_trait_new)
                                cat_traits_new_species[t_abs, y] = cat_trait_new
                                # trait state for the just originated lineage
                                lineage_rates_tmp[(5 + 3 * n_cont_traits + y):(6 + 3 * n_cont_traits + y)] = cat_trait_new
                                # trait state of the ancestral lineage
                                lineage_rates_tmp[(5 + 3 * n_cont_traits + y + n_cat_traits):(6 + 3 * n_cont_traits + y + n_cat_traits)] = ancestral_cat_trait
                                l_new = l_new * cat_trait_effect[y][0, cat_trait_new]
                                m_new = m_new * cat_trait_effect[y][1, cat_trait_new]
                        cat_traits = np.dstack((cat_traits, cat_traits_new_species))
                    if n_cont_traits > 0:
                        cont_traits_new_species = self.empty_traits(root_plus_1, n_cont_traits)
                        # cont_traits_at_origin = cont_traits[t_abs, :, j]
                        cont_traits_at_origin = self.evolve_cont_traits(cont_traits[t_abs, :, j],
                                                                        n_cont_traits,
                                                                        cont_traits_alpha,
                                                                        cont_traits_Theta1,
                                                                        cont_traits_varcov_clado)
                        cont_traits_new_species[t_abs,:] = cont_traits_at_origin
                        cont_traits = np.dstack((cont_traits, cont_traits_new_species))
                        lineage_rates_tmp[5:(5 + n_cont_traits)] = cont_traits[t_abs, :, j]
                        lineage_rates_tmp[(5 + n_cont_traits):(5 + 2 * n_cont_traits)] = cont_traits_at_origin
                        cont_traits_bin = cont_traits_effect_shift_sp[t_abs]
                        l_new = self.get_rate_by_cont_trait_transformation(l_new,
                                                                           cont_traits_at_origin,
                                                                           cont_trait_effect_sp[cont_traits_bin, :, cat_trait_new, :],
                                                                           expected_sd_cont_traits,
                                                                           n_cont_traits)
                        cont_traits_bin = cont_traits_effect_shift_ex[t_abs]
                        m_new = self.get_rate_by_cont_trait_transformation(m_new,
                                                                           cont_traits_at_origin,
                                                                           cont_trait_effect_ex[cont_traits_bin, :, cat_trait_new, :],
                                                                           expected_sd_cont_traits, n_cont_traits)
                        # environmental effect
                        if self.sp_env_file is not None:
                            eff_sp = env_eff_sp
                            cat_trait_j = 0
                            if n_cat_traits > 0 and self.env_effect_cat_trait[0] is not None:
                                cat_trait_j = cat_trait_new
                            l_new = self.get_rate_by_env_transformation(l_new, t_abs, eff_sp, rate_type = 'l',
                                                                        cate_state = cat_trait_j)
                        if self.ex_env_file is not None:
                            eff_ex = env_eff_ex
                            cat_trait_j = 0
                            if n_cat_traits > 0 and self.env_effect_cat_trait[1] is not None:
                                cat_trait_j = cat_trait_new
                            m_new = self.get_rate_by_env_transformation(m_new, t_abs, eff_ex, rate_type = 'm',
                                                                        cate_state = cat_trait_j)
                    if n_areas > 1:
                        biogeo_new_species = self.empty_traits(root_plus_1, 1)
                        # biogeo_at_origin = biogeo[t_abs, :, j]
                        # biogeo_new_species[t_abs, :] = biogeo_at_origin
                        biogeo_ancestor, biogeo_descendant = self.evolve_biogeo_clado(n_areas, areas_comb, biogeo[t_abs, :, j])
                        biogeo_new_species[t_abs, :] = biogeo_descendant
                        biogeo[t_abs, :, j] = biogeo_ancestor
                        biogeo = np.dstack((biogeo, biogeo_new_species))

                    lineage_rates_tmp[:5] = np.array([t, -0.0, l_new, m_new, l_j])
                    lineage_rates.append(lineage_rates_tmp)
                    lineage_victim_mass_extinction.append(0)
                    # add new species to lineage-specific rates through time
                    lineage_rates_through_time_new_species = self.empty_traits(root_plus_1, 2)
                    lineage_rates_through_time_new_species[t_abs, 0] = l_new
                    lineage_rates_through_time_new_species[t_abs, 1] = m_new
                    lineage_rates_through_time = np.dstack((lineage_rates_through_time,
                                                            lineage_rates_through_time_new_species))
                    # modify nwk list
                    if self.s_species == 1:
                        nwk_j_before = nwk[j]
                        nwk[j] = self.update_nwk(nwk_j_before, anagenetic = False)
                        nwk.append('T' + str(len(ts) - 1) + ':' + self.format_age_for_newick(0.0))
                        nwk_str = self.add_speciation_to_nwk_str(nwk_str, nwk_j_before, str(len(ts) - 1))

                # mass extinction
                elif mass_ext:
                    if self.mass_extinction_trait_dependent:
                        # TODO: m_j will be a very small value
                        m_me = m_me * m_j
                    if ran < m_me:
                        te[j] = t
                        lineage_rates[j][1] = t
                        lineage_victim_mass_extinction[j] = 1

                # extinction
                elif (ran > l_j and ran < (l_j + m_j) ) or t == -1:
                    if t != -1:
                        te[j] = t
                        lineage_rates[j][1] = t
                    else:
                        lineage_rates[j][3] = m_j # Extinction rate at extinction time (or present for extant species)
                        # print('m_j at extinction or present: ', t_abs / self.scale, j, m_j * self.scale)
                        if n_cont_traits > 0:
                            lineage_rates[j][(5 + 2 * n_cont_traits):(5 + 3 * n_cont_traits)] = cont_trait_j

                # no speciation or extinction
                else:
                    if self.s_species == 1:
                        nwk_j_before = nwk[j]
                        nwk_j_after = self.update_nwk(nwk_j_before)
                        nwk[j] = nwk_j_after
                        nwk_str_new = nwk_str.replace(nwk_j_before, nwk_j_after)
                        del nwk_str
                        nwk_str = nwk_str_new
                        #print(nwk_str)

                # trace lineage-specific rates through time
                lineage_rates_through_time[t_abs, 0, j] = l_j
                lineage_rates_through_time[t_abs, 1, j] = m_j

            if t != -1:
                lineage_weighted_lambda_tt[t_abs-1] = self.get_harmonic_mean(lineage_lambda)
                lineage_weighted_mu_tt[t_abs-1] = self.get_harmonic_mean(lineage_mu)

        lineage_rates = np.array(lineage_rates)
        lineage_rates[:, 0] = -lineage_rates[:, 0] / self.scale # Why is it not working? lineage_rates[:,:2] = -lineage_rates[:,:2] / self.scale
        lineage_rates[:, 1] = -lineage_rates[:, 1] / self.scale
        lineage_rates[:, 2] = lineage_rates[:, 2] * self.scale
        lineage_rates[:, 3] = lineage_rates[:, 3] * self.scale
        lineage_rates[:, 4] = lineage_rates[:, 4] * self.scale

        return -np.array(ts) / self.scale, -np.array(te) / self.scale, anc_desc, cont_traits, cat_traits, mass_ext_time, mass_ext_mag, np.array(lineage_victim_mass_extinction), lineage_weighted_lambda_tt, lineage_weighted_mu_tt, lineage_rates, biogeo, areas_comb, lineage_rates_through_time, nwk_str, exceeded_diversity


    def get_random_settings(self, root, sp_env_ts, ex_env_ts, verbose):
        root = np.abs(root)
        root_scaled = int(root * self.scale)
        dT = root / root_scaled

        if self.fixed_Ltt is None:
            L_shifts, L, timesL = self.make_shifts_birth_death(root_scaled, self.poiL, self.rangeL)
        else:
            L_shifts, L, timesL = self.make_fixed_bd_through_time(root_scaled, self.fixed_Ltt)

        if self.fixed_Mtt is None:
            M_shifts, M, timesM = self.make_shifts_birth_death(root_scaled, self.poiM, self.rangeM)
        else:
            M_shifts, M, timesM = self.make_fixed_bd_through_time(root_scaled, self.fixed_Mtt)

        L_shifts, linL = self.add_linear_time_effect(L_shifts, self.range_linL, self.fixed_Ltt)
        M_shifts, linM = self.add_linear_time_effect(M_shifts, self.range_linM, self.fixed_Mtt)

        # categorical traits
        n_cat_traits = np.random.choice(np.arange(min(self.n_cat_traits), max(self.n_cat_traits) + 1), 1)
        n_cat_traits = int(n_cat_traits)
        cat_traits_Q = []
        n_cat_traits_states = np.zeros(n_cat_traits, dtype = int)
        cat_states = []
        cat_trait_effect = []
        if n_cat_traits > 0:
            for i in range(n_cat_traits):
                n_cat_traits_states[i] = np.random.choice(np.arange(min(self.n_cat_traits_states),
                                                                    max(self.n_cat_traits_states) + 1),
                                                          1)
                Qi = self.make_cat_traits_Q(n_cat_traits_states[i])
                cat_traits_Q.append(Qi)
                cat_states_i = np.arange(n_cat_traits_states[i])
                cat_states.append(cat_states_i)
                cat_trait_effect_i = self.make_cat_trait_effect(n_cat_traits_states[i])
                cat_trait_effect.append(cat_trait_effect_i)

        # continuous traits
        n_cont_traits = np.random.choice(np.arange(min(self.n_cont_traits), max(self.n_cont_traits) + 1), 1)
        n_cont_traits = int(n_cont_traits)
        cont_traits_varcov = []
        cont_traits_Theta1 = []
        cont_traits_alpha = []
        cont_traits_varcov_clado = []
        cont_traits_effect_sp = []
        cont_traits_effect_ex = []
        expected_sd_cont_traits = []
        cont_traits_effect_shift_sp = []
        cont_traits_effect_shift_ex = []
        if n_cont_traits > 0:
            cont_traits_cor = self.make_cor_cont_traits(n_cont_traits)
            cont_traits_varcov = self.make_cont_traits_varcov(n_cont_traits, self.cont_traits_sigma, cont_traits_cor, self.scale)
            cont_traits_Theta1 = self.make_cont_traits_Theta1(n_cont_traits)
            cont_traits_alpha = self.make_cont_traits_alpha(n_cont_traits, root_scaled)
            cont_traits_varcov_clado = self.make_cont_traits_varcov(n_cont_traits, self.cont_traits_sigma_clado, cont_traits_cor, 1.0)
            n_cat_states_sp = 1
            n_cat_states_ex = 1
            if n_cat_traits > 0:
                if n_cat_traits > 1 and verbose:
                    print('State-dependent effect of continuous traits set to the states of the first categorical trait')
                n_cat_states_sp = len(cat_states[0])
                n_cat_states_ex = len(cat_states[0])
            cont_traits_effect_shift_sp = self.make_cont_trait_effect_time_vec(root_scaled, self.cont_traits_effect_shift_sp)
            # print('cont_traits_effect_shift_sp: ', cont_traits_effect_shift_sp)
            # print(np.unique(cont_traits_effect_shift_sp, return_counts = True) )
            cont_traits_effect_shift_ex = self.make_cont_trait_effect_time_vec(root_scaled, self.cont_traits_effect_shift_ex)
            n_time_bins_sp = len(np.unique(cont_traits_effect_shift_sp))
            n_time_bins_ex = len(np.unique(cont_traits_effect_shift_ex))
            # if self.cont_traits_effect_optimum_sp is None:
            #     cont_traits_effect_optimum_sp = np.array([[np.zeros(2)]])
            # if self.cont_traits_effect_optimum_ex is None:
            #     cont_traits_effect_optimum_ex = np.array([[np.zeros(2)]])
            cont_traits_effect_sp, expected_sd_cont_traits = self.get_cont_trait_effect_parameters(root,
                                                                                                   cont_traits_varcov,
                                                                                                   n_time_bins_sp,
                                                                                                   n_cont_traits,
                                                                                                   n_cat_states_sp,
                                                                                                   self.cont_traits_effect_sp,
                                                                                                   self.cont_traits_effect_bellu_sp,
                                                                                                   self.cont_traits_effect_optimum_sp,
                                                                                                   verbose)
            cont_traits_effect_ex, _ = self.get_cont_trait_effect_parameters(root,
                                                                             cont_traits_varcov,
                                                                             n_time_bins_ex,
                                                                             n_cont_traits,
                                                                             n_cat_states_ex,
                                                                             self.cont_traits_effect_ex,
                                                                             self.cont_traits_effect_bellu_ex,
                                                                             self.cont_traits_effect_optimum_ex,
                                                                             verbose)

        # biogeography
        n_areas = np.random.choice(np.arange(min(self.n_areas), max(self.n_areas) + 1), 1)
        n_areas = int(n_areas)
        dispersal = np.zeros(1)
        extirpation = np.zeros(1)
        if n_areas > 1:
            dispersal = np.random.uniform(np.min(self.dispersal), np.max(self.dispersal), 1)
            extirpation = np.random.uniform(np.min(self.extirpation), np.max(self.extirpation), 1)

        # environmental effects
        sp_env_eff = np.random.uniform( 1.0 / np.min(self.sp_env_eff), 1.0 / np.max(self.sp_env_eff), 1)
        ex_env_eff = np.random.uniform(1.0 / np.min(self.ex_env_eff), 1.0 / np.max(self.ex_env_eff), 1)

        if self.sp_env_file is not None:
            time_vec = np.arange(int(np.abs(root) * self.scale) + 2)
            # What if temporal resolution of the environment is coarser than time_vec?
            self._env_sp_binned = get_binned_continuous_variable(sp_env_ts, time_vec, self.scale)
            self._env_sp_mean = np.mean(self._env_sp_binned)
            self._env_sp_std = np.std(self._env_sp_binned)
        else:
            sp_env_binned = None

        if self.ex_env_file is not None:
            time_vec = np.arange(int(np.abs(root) * self.scale) + 2)
            self._env_ex_binned = get_binned_continuous_variable(ex_env_ts, time_vec, self.scale)
            self._env_ex_mean = np.mean(self._env_ex_binned)
            self._env_ex_std = np.std(self._env_ex_binned)
        else:
            ex_env_binned = None

        return dT, L_shifts, M_shifts, L, M, timesL, timesM, linL, linM, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_varcov_clado, cont_traits_effect_sp, cont_traits_effect_ex, expected_sd_cont_traits, cont_traits_effect_shift_sp, cont_traits_effect_shift_ex, n_cat_traits, cat_states, cat_traits_Q, cat_trait_effect, n_areas, dispersal, extirpation, sp_env_eff, ex_env_eff


    def make_shifts_birth_death(self, root_scaled, poi_shifts, range_rate):
        timesR_temp = [root_scaled, 0.]
        # Number of rate shifts expected according to a Poisson distribution
        n_shifts = np.random.poisson(poi_shifts)
        R = np.random.uniform(np.min(range_rate), np.max(range_rate), n_shifts + 1)
        R = R / self.scale
        # random shift times
        shift_time_R = np.random.uniform(0, root_scaled, n_shifts)
        timesR = np.sort(np.concatenate((timesR_temp, shift_time_R), axis = 0))[::-1]
        # Rates through (scaled) time
        R_tt = np.zeros(root_scaled, dtype = 'float')
        idx_time_vec = np.arange(root_scaled)[::-1]
        for i in range(n_shifts + 1):
            Ridx = np.logical_and(idx_time_vec < timesR[i], idx_time_vec >= timesR[i + 1])
            R_tt[Ridx] = R[i]

        return R_tt, R, timesR


    def add_linear_time_effect(self, R_shifts, range_lin, fixed_rtt):
        t_vec = np.linspace(-0.5, 0.5, len(R_shifts))
        if range_lin and fixed_rtt is None:
            # Slope
            linR = np.random.uniform(np.min(range_lin), np.max(range_lin), 1)
        else:
            # No change through time
            linR = np.zeros(1)
        R_tt = R_shifts + linR * t_vec
        R_tt[R_tt < 0.0] = 1e-10

        return R_tt, linR


    def make_fixed_bd_through_time(self, root_scaled, fixed_rtt):
        rtt = np.zeros(root_scaled, dtype = 'float')
        idx_time_vec = np.arange(root_scaled)[::-1]
        fixed_rtt2 = fixed_rtt + 0.0
        fixed_rtt2[:, 0] = fixed_rtt2[:, 0] * self.scale
        fixed_rtt2[:, 1] = fixed_rtt2[:, 1] / self.scale
        for i in range(len(fixed_rtt2) - 1):
            idx = np.logical_and(idx_time_vec <= fixed_rtt2[i, 0], idx_time_vec > fixed_rtt2[i + 1, 0])
            rate_idx = np.linspace(fixed_rtt2[i, 1], fixed_rtt2[i + 1, 1], sum(idx))
            rtt[idx] = rate_idx

        return rtt, fixed_rtt2[:,1], fixed_rtt2[1:,0]


    def get_divdep_lam(self, lam, N, t):
        if self.fixed_K_lam[0] is None:
            K = self.K_lam[0]
        else:
            tt = t / self.scale
            idx_K = np.digitize(tt, self.fixed_K_lam[0][:, 0])
            K = self.fixed_K_lam[0][idx_K, 1]
        divdep_lam = lam * (1 - (N / K))
        if divdep_lam < 0.0:
            divdep_lam = 0.0

        return divdep_lam


    def get_divdep_mu(self, mu, N, t):
        if self.fixed_K_mu[0] is None:
            K = self.K_mu[0]
        else:
            tt = t / self.scale
            idx_K = np.digitize(tt, self.fixed_K_mu[0][:, 0])
            K = self.fixed_K_mu[0][idx_K, 1]
        divdep_mu = mu / (1 - (N / K))
        if divdep_mu < 0.0:
            divdep_mu = mu / (1 - ((N - 1e-5) / N))

        return divdep_mu


    def get_harmonic_mean(self, v):
        hm = np.nan
        v = v[np.isnan(v) == False]
        if len(v) > 0:
            v = v * self.scale
            hm = len(v) / np.sum(1.0 / v)

        return hm


    def empty_traits(self, past, n_cont_traits):
        tr = np.empty((past, n_cont_traits))
        tr[:] = np.nan

        return tr


    def nearestPD(self, A):
        """Find the nearest positive-definite matrix to input
        https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """
        B = (A + A.T) / 2
        _, s, V = la.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if self.isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3


    def isPD(self, B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False


    def make_cor_cont_traits(self, n_cont_traits):
        n_cor = int(n_cont_traits * (n_cont_traits - 1) / 2)
        cor = np.random.uniform(np.min(self.cont_traits_cor), np.max(self.cont_traits_cor), n_cor)

        return cor


    def make_cont_traits_varcov(self, n_cont_traits, s2, cor, scale):
        if n_cont_traits == 1:
            varcov = np.random.uniform(np.min(s2), np.max(s2), 1)
            varcov = np.array([varcov])
            varcov = np.sqrt(varcov / scale)
        else:
            sigma2 = np.random.uniform(np.min(s2), np.max(s2), n_cont_traits)
            sigma2 = np.diag(sigma2)
            cormat = np.ones([n_cont_traits**2]).reshape((n_cont_traits, n_cont_traits))
            cormat[np.triu_indices(n_cont_traits, k = 1)] = cor
            cormat[np.tril_indices(n_cont_traits, k = -1)] = cor
            varcov = sigma2 @ cormat @ sigma2 # correlation to covariance for multivariate random
            varcov = self.nearestPD(varcov)
            varcov = varcov / scale

        return varcov


    def make_cont_traits_Theta1(self, n_cont_traits):
        cont_traits_Theta1 = np.random.uniform(np.min(self.cont_traits_Theta1), np.max(self.cont_traits_Theta1), n_cont_traits)

        return cont_traits_Theta1


    def make_cont_traits_alpha(self, n_cont_traits, root):
        if self.cont_traits_alpha[0] == 0.0 and self.cont_traits_alpha[1] == 0.0:
            cont_traits_alpha = np.zeros(n_cont_traits)
        else:
            alpha = np.random.uniform(np.log(np.min(self.cont_traits_alpha)), np.log(np.max(self.cont_traits_alpha)), n_cont_traits) # half life
            alpha = np.exp(alpha)
            cont_traits_alpha = alpha * np.log(2.0) * (1 / np.abs(root))

        return cont_traits_alpha


    def evolve_cont_traits(self, cont_traits, n_cont_traits, cont_traits_alpha, cont_traits_Theta1, cont_traits_varcov):
        if n_cont_traits == 1:
            # Not possible to vectorize; sd needs to have the same size as the mean
            cont_traits = cont_traits + cont_traits_alpha * (cont_traits_Theta1 - cont_traits) + np.random.normal(0.0, cont_traits_varcov[0,0], 1)
        elif n_cont_traits > 1:
            cont_traits = cont_traits + cont_traits_alpha * (cont_traits_Theta1 - cont_traits) + np.random.multivariate_normal(np.zeros(n_cont_traits), cont_traits_varcov, 1)
            cont_traits = cont_traits[0]

        return cont_traits


    def get_rate_by_cont_trait_transformation(self, r, cont_trait_value, par, expected_sd, n_cont_traits):
        # print('r: ', r)
        # print('cont_trait_value: ', cont_trait_value)
        # print('par: ', par)
        # print('expected_sd: ', expected_sd)
        # print('n_cont_traits: ', n_cont_traits)
        if np.all(par[:, 0] > 10000.0):
            return r
        else:
            if n_cont_traits == 1:
                # print('cont_trait_value:' , cont_trait_value)
                # print('expected_sd[0]: ', expected_sd[0])
                trait_pdf = norm.pdf(cont_trait_value, par[0, 4], par[:, 0] * expected_sd[0])[0]
            else:
                # How to scale the multivariate SD by the effect? Only the diagonals and then make it positive definite again?
                cov_tmp = par[:, 0] * expected_sd
                cov_tmp = self.nearestPD(cov_tmp)
                trait_pdf = multivariate_normal.pdf(cont_trait_value,
                                                    mean = par[:, 4],
                                                    cov = cov_tmp)
            # Scale according to trait effect: ((bellu * trait_pdf - MinPDF) / (MaxPDF - MinPDF) ) * (2 * Effect) - Effect
            # cont_trait_effect = ((par[:, 1] * trait_pdf - par[:, 2]) / (par[:, 3] - par[:, 2])) * (2 * par[:, 0]) - par[:, 0]
            # cont_trait_effect = np.sum(cont_trait_effect)
            # transf_r = r * np.exp(cont_trait_effect)
            scaled_trait_pdf = (trait_pdf - par[:, 2]) / (par[:, 3] - par[:, 2])
            # rate * +/- f(delta, v)
            r_ushape = 0
            # What to do when there is more than one continuous trait?
            if par[0, 1] == -1:
                r_ushape = r
            transf_r = r * par[0, 1] * scaled_trait_pdf + r_ushape

            return transf_r[0]


    def get_cont_trait_effect_parameters(self, root, sigma2, n_time_bins, n_cont_traits, n_cat_states, cte, bellu, opt, verbose):
        # 1st time; 2nd axis: n_cont_traits; 3rd axis: n_cat_traits; 4th axis: trait effect, min effect, max effect
        effect_par = np.zeros((n_time_bins, n_cont_traits, n_cat_states, 5))
        # Expected standard deviation of traits after time = root
        if n_cont_traits == 1:
            expected_sd = np.sqrt(root * sigma2**2 * self.scale) # expected SD of traits after time = root
            # effect_par[:, :, :, 3] = norm.pdf(0.0, 0.0, expected_sd)
        else:
            expected_sd = root * sigma2 * self.scale
            # effect_par[:, :, :, 3] = multivariate_normal.pdf(np.zeros(n_cont_traits), mean = np.zeros(n_cont_traits), cov = expected_sd)
        # Make sure that specified effects of the continuous traits and their u/bell-shape are having the correct shape
        # print('n_time_bins: ', n_time_bins, opt.shape[0])
        # print('n_cont_traits: ', n_cont_traits, opt.shape[1])
        # print('n_cat_states: ', n_cat_states, opt.shape[2])
        cte = 1.0 / cte
        if (cte.shape[0] < n_time_bins) or (cte.shape[1] < n_cont_traits) or (cte.shape[2] < n_cat_states):
            if verbose:
                print('Dimensions of continuous traits effects do not match number of traits or time strata.\n'
                      'Using instead the range of the specified values.')
            #cte_range = np.array([np.min(cte), np.max(cte)])
            cte_range = np.repeat(np.random.uniform(np.min(cte), np.max(cte), 1), 2)
            cte = np.tile(cte_range, n_time_bins * n_cont_traits * n_cat_states).reshape((n_time_bins, n_cont_traits, n_cat_states, 2))
        if (bellu.shape[0] < n_time_bins) or (bellu.shape[1] < n_cont_traits) or (bellu.shape[2] < n_cat_states):
            if verbose:
                print('Dimensions of continuous traits bell/u-shape do not match number of traits or time strata.\n'
                      'Using instead the range of the specified values.')
            #bellu_range = np.array([np.min(bellu), np.max(bellu)])
            bellu_range = np.repeat(np.random.choice( np.array([np.min(bellu), np.max(bellu)]) , 1), 2)
            bellu = np.tile(bellu_range, n_time_bins * n_cont_traits * n_cat_states).reshape((n_time_bins, n_cont_traits, n_cat_states, 2))
        if (opt.shape[0] < n_time_bins) or (opt.shape[1] < n_cont_traits) or (opt.shape[2] < n_cat_states):
            if verbose:
                print('Dimensions of continuous traits optimum do not match number of traits or time strata.\n'
                      'Using instead the range of the specified values.')
            #opt_range = np.array([np.min(opt), np.max(opt)])
            opt_range = np.repeat(np.random.uniform(np.min(opt), np.max(opt), 1), 2)
            opt = np.tile(opt_range, n_time_bins * n_cont_traits * n_cat_states).reshape((n_time_bins, n_cont_traits, n_cat_states, 2))
        # Fill array for parameterizing continuous trait effects
        for i in range(n_time_bins):
            for k in range(n_cat_states):
                opt_tmp = opt[i, :, k, :]
                opt_tmp = np.sort(opt_tmp, axis = 1)
                effect_par[i, :, k, 4] = np.random.uniform(opt_tmp[:, 0], opt_tmp[:, 1], n_cont_traits)
                for j in range(n_cont_traits):
                    # Magnitude of the effect
                    cte_tmp = cte[i, j, k, :]
                    effect_par[i, j, k, 0] = np.random.uniform(np.min(cte_tmp), np.max(cte_tmp), 1)
                    # Whether effect has a bell (1) or u-shape (-1)
                    effect_par[i, j, k, 1] = np.random.choice(bellu[i, j, k, :], 1)
                    # Sort in so that the min is smaller than the max
                    #effect_par[i, j, k, 3] = effect_par[i, j, k, 3] * effect_par[i, j, k, 1]
                    #effect_par[i, j, k, 2:4] = np.sort(effect_par[i, j, k, 2:4])
                if n_cont_traits == 1:
                    effect_par[i, :, k, 3] = norm.pdf(effect_par[i, :, k, 4], effect_par[i, :, k, 4], effect_par[i, :, k, 0] * expected_sd)[0]
                else:
                    # How to scale the multivariate SD by the effect? Only the diagonals and then make it positive definite again?
                    effect_par[i, :, k, 3] = multivariate_normal.pdf(effect_par[i, :, k, 4],
                                                                     mean = effect_par[i, :, k, 4],
                                                                     cov = expected_sd,
                                                                     allow_singular = True)

        return effect_par, expected_sd


    def make_cont_trait_effect_time_vec(self, root_scaled, effect_shifts):
        time_vec = np.zeros(int(root_scaled + 1), dtype = int)
        if effect_shifts is not None:
            # From the past (root_scaled) to the present (0)
            idx_time_vec = np.arange(root_scaled + 1)[::-1]
            effect_shifts = effect_shifts * self.scale
            shift_time = np.concatenate((np.zeros(1), effect_shifts, np.array([root_scaled])))
            shift_time = np.sort(shift_time)[::-1]
            for i in range(len(shift_time) - 1):
                idx = np.logical_and(idx_time_vec < shift_time[i], idx_time_vec >= shift_time[i + 1])
                time_vec[idx] = i

        return time_vec[::-1]


    def evolve_cat_traits_ana(self, Q, s, ran, cat_states):
        s = int(s)
        state = s
        # print('ran', ran)
        # print('Q[s, s]', Q[s, s])
        if Q[s, s] > ran:
            pos_states = cat_states != s
            p = Q[s, pos_states]
            p = p / np.sum(p)
            state = np.random.choice(cat_states[pos_states], 1, p = p)

        return state


    def evolve_cat_traits_clado(self, Q, s, cat_states):
        s = int(s)
        p = Q[s,:]
        p = p / np.sum(p)
        state = np.random.choice(cat_states, 1, p = p)

        return state


    def make_cat_traits_Q(self, n_states):
        cat_traits_ordinal = np.random.choice(self.cat_traits_ordinal, 1)
        n_states = int(n_states)
        Q = np.zeros([n_states**2]).reshape((n_states, n_states))
        if self.cat_traits_diag is not None:
            self.cat_traits_dir = 73.0
        for i in range(n_states):
            dir_alpha = np.ones(n_states)
            dir_alpha[i] = self.cat_traits_dir
            q_idx = np.arange(0, n_states)
            if cat_traits_ordinal and n_states > 2:
                if i == 0:
                    dir_alpha = np.array([self.cat_traits_dir, 1])
                    q_idx = [0,1]
                elif i == (n_states - 1):
                    dir_alpha = np.array([1, self.cat_traits_dir])
                    q_idx = [n_states - 2, n_states - 1]
                else:
                    dir_alpha = np.ones(3)
                    dir_alpha[1] = self.cat_traits_dir
                    q_idx = np.arange(i - 1, i + 2)
            if self.cat_traits_diag is not None:
                fix_trans = np.zeros(len(q_idx))
                fix_trans[dir_alpha == 73.0] = self.cat_traits_diag
                fix_trans[dir_alpha != 73.0] = (1.0 - self.cat_traits_diag) / (len(q_idx) - 1)
                Q[i, q_idx] = fix_trans
            else:
                Q[i, q_idx] = np.random.dirichlet(dir_alpha, 1).flatten()

        return Q


    def get_stationary_distribution(self, Q):
        # Why do we need some jitter to get positive values in the eigenvector?
        Qtmp = Q + 0.0#+ np.random.uniform(0.0, 0.001, np.size(Q)).reshape(Q.shape)
        Qtmp = Qtmp / np.sum(Qtmp, axis = 1)
        eigenvals, left_eigenvec = scipy.linalg.eig(Qtmp, right = False, left = True)
        left_eigenvec1 = left_eigenvec[:, np.isclose(eigenvals, 1)]
        pi = left_eigenvec1[:, 0].real
        pi_normalized = pi / np.sum(pi)

        return pi_normalized


    def make_cat_trait_effect(self, n_states):
        n_states = n_states - 1
        cat_trait_effect = np.ones((2, n_states))
        # allows user provided values for trait effect in case of more than two states
        if self.cat_traits_effect.shape[1] == n_states:
            cat_trait_effect[0,:] = self.cat_traits_effect[0,:]
            cat_trait_effect[1,:] = self.cat_traits_effect[1,:]
        else:
            cat_trait_effect[0,:] = np.random.uniform(self.cat_traits_effect[0, 0], self.cat_traits_effect[0, 1], n_states) # effect on speciation
            cat_trait_effect[1,:] = np.random.uniform(self.cat_traits_effect[1, 0], self.cat_traits_effect[1, 1], n_states) # effect on extinction
            id = np.random.choice(self.cat_traits_effect_decr_incr[0,:], n_states)
            cat_trait_effect[0, id] = 1.0 / cat_trait_effect[0, id]
            id = np.random.choice(self.cat_traits_effect_decr_incr[1,:], n_states)
            cat_trait_effect[1, id] = 1.0 / cat_trait_effect[1, id]

        cat_trait_effect = np.hstack((np.ones((2,1)), cat_trait_effect))

        return cat_trait_effect


    def get_geographic_states(self, areas):
        a = np.arange(areas)
        comb_areas = []
        for i in range(1, areas + 1):
            comb_areas.append(list(combinations(a, i)))
        # flatten nested list: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        comb_areas = reduce(iconcat, comb_areas, [])

        return comb_areas


    def make_anagenetic_DEC_matrix(self, areas, d, e):
        # No state 0 i.e. global extinction is disconnected from the geographic evolution!
        n_geo_states = 2**areas - 1
        de_mat = np.zeros((n_geo_states, n_geo_states))
        comb_areas = self.get_geographic_states(areas)
        areas_per_state = np.zeros(n_geo_states)
        for i in range(n_geo_states):
            areas_per_state[i] = len(comb_areas[i])

        # Multiplier for dispersal rate (number of areas in state - 1)
        d_multi = np.ones(n_geo_states)
        for i in range(n_geo_states):
            d_multi[i] = len(comb_areas[i]) - 1

        for i in range(n_geo_states):
            disp_from_outgoing_state = np.zeros(n_geo_states, dtype = bool)
            areas_outgoing_state = int(areas_per_state[i])
            outgoing = set(comb_areas[i])
            # Check only increase by one area unit. E.g. A -> A,B or A,B -> A,B,C but not A -> A,B,C or A,B -> A,B,C,D
            candidate_states_for_dispersal = np.where(areas_per_state == (areas_outgoing_state + 1))[0]
            for y in candidate_states_for_dispersal:
                ingoing = set(comb_areas[y])
                inter = outgoing.intersection(ingoing)
                disp_from_outgoing_state[y] = inter == outgoing
            de_mat[i, disp_from_outgoing_state] = d * d_multi[disp_from_outgoing_state]

            ext_from_outgoing_state = np.zeros(n_geo_states, dtype = bool)
            # Check only decrease by one area unit. E.g. A,B -> A/B, A,B,C -> A,B/A,C/B,C but not A,B -> A,C/A,B,C or A,B,C -> A/B
            candidate_states_for_extinction = np.where(areas_per_state == (areas_outgoing_state - 1))[0]
            for y in candidate_states_for_extinction:
                ingoing = set(comb_areas[y])
                inter = outgoing.intersection(ingoing)
                ext_from_outgoing_state[y] = inter == ingoing
            de_mat[i, ext_from_outgoing_state] = e

        de_mat_colsums = np.sum(de_mat, axis = 0)
        np.fill_diagonal(de_mat, de_mat_colsums)

        return de_mat, comb_areas


    def get_DEC_clado_weight(self, areas):
        # there is probably an equation for this!
        # narrow sympatry (1 area only)
        range_inheritances = areas
        # subset sympatry
        for i in range(2, areas + 1):
            for j in range(1, i):
                range_inheritances += comb(i, j)
        # narrow vicariance
        range_inheritances += areas
        for i in range(5, areas + 1):
            # maximum number of areas for the smaller subset of the range
            max_areas_subset = np.floor(i/2) - 1 + (i % 2)
            max_areas_subset = int(max_areas_subset)
            for j in range(1, max_areas_subset + 1):
                range_inheritances += comb(i, j)

        return 1.0 / range_inheritances


    def get_range_idx_from_ranges_list(self, a, comb_areas):
        a = set(a)
        for i in range(len(comb_areas)):
            if a == set(comb_areas[i]):
                range_idx = i

        return range_idx



    def evolve_biogeo_clado(self, areas, comb_areas, range_idx):
        lr = np.random.choice([0, 1], 2, replace=False)  # random flip ranges for the two lineages
        if range_idx <= (areas - 1):
            range_ancestor = range_idx
            range_descendant = range_idx
        elif range_idx <= (areas + comb(areas, 2) - 1): # Range size == 2 (e.g. AB, AC)
            if np.random.uniform(0.0, 1.0, 1) <= 0.5:
                # narrow sympatry
                range_ancestor = range_idx
                range_descendant = range_idx
            else:
                # narrow vicariance
                outgoing_range = comb_areas[int(range_idx)]
                range_ancestor = outgoing_range[lr[0]] #self.get_range_idx_from_ranges_list(outgoing_range[lr[0]], comb_areas)
                range_descendant = outgoing_range[lr[1]] #self.get_range_idx_from_ranges_list(outgoing_range[lr[1]], comb_areas)
        else:
            outgoing_range = comb_areas[int(range_idx)]
            n_areas_outgoing_range = len(outgoing_range)
            max_areas_subset = np.floor(n_areas_outgoing_range / 2) - 1 + (n_areas_outgoing_range % 2)
            max_areas_subset = int(max_areas_subset)
            n_areas_subset = int(np.random.choice(np.arange(max_areas_subset) + 1, 1))
            ran_subset = np.random.choice(np.array(outgoing_range), n_areas_subset)
            subset_range = tuple(np.sort(ran_subset))
            range_idx2 = np.arange(2)
            range_idx2[0] = self.get_range_idx_from_ranges_list(subset_range, comb_areas)
            if np.random.uniform(0.0, 1.0, 1) <= 0.5:
                # subset sympatry
                range_idx2[1] = range_idx
                range_ancestor = range_idx2[lr[0]]
                range_descendant = range_idx2[lr[0]]
            else:
                # narrow vicariance
                remaining_range = tuple(set(outgoing_range) - set(subset_range))
                range_idx2[1] = self.get_range_idx_from_ranges_list(remaining_range, comb_areas)
                range_ancestor = range_idx2[lr[0]]
                range_descendant = range_idx2[lr[0]]

        return range_ancestor, range_descendant


    # def make_cladoenetic_DEC_matrix(self, areas):
    #     # No state 0 i.e. global extinction is disconnected from the geographic evolution!
    #     n_geo_states = 2 ** areas - 1
    #     c_mat = np.zeros((n_geo_states, n_geo_states), dtype = int)
    #     comb = self.get_geographic_states(areas)
    #     # narrow sympatry (rangesize of 1 area)
    #     for i in range(areas):
    #         c_mat[i,i] = 1
    #     # subset sympatry
    #     # narrow vicariance


    def get_true_rate_through_time(self, root, L_tt, M_tt, L_weighted_tt, M_weighted_tt):
        time_rates = np.linspace(0.0, np.abs(root), len(L_tt))
        d = np.stack((time_rates, L_tt[::-1] * self.scale, M_tt[::-1] * self.scale, L_weighted_tt, M_weighted_tt), axis = -1)
        div_rates = pd.DataFrame(data = d, columns = ['time', 'speciation', 'extinction', 'trait_weighted_speciation', 'trait_weighted_extinction'])

        return div_rates


    def check_proportion_cat_traits(self, n_cat_traits, cat_traits):
        proportion_ok = False
        if n_cat_traits == 0:
            proportion_ok = True
        else:
            cat_traits_min_freq = self.cat_traits_min_freq
            if len(cat_traits_min_freq) < n_cat_traits:
                cat_traits_min_freq = np.repeat(cat_traits_min_freq[0], n_cat_traits)
            ct = np.nanmean(cat_traits, axis = 0)
            min_freq = np.zeros(n_cat_traits)
            for i in range(n_cat_traits):
                counts = np.unique(ct[i, :], return_counts = True)[1]
                min_freq[i] = np.min(counts / np.sum(counts))
            if np.all(min_freq > cat_traits_min_freq):
                proportion_ok = True

        return proportion_ok


    def get_rate_by_env_transformation(self, r, t, env_eff, rate_type = 'l', cate_state = 0):
        if rate_type == 'l':
            env = self._env_sp_binned
            env_mean = self._env_sp_mean
            env_sd = self._env_sp_std
        else:
            env = self._env_ex_binned
            env_mean = self._env_ex_mean
            env_sd = self._env_ex_std
        sd = env_sd * np.abs(env_eff)
        max_pdf = norm.pdf(env_mean, env_mean, sd)
        env_pdf = norm.pdf(env[t], env_mean, sd)
        env_pdf_scaled = env_pdf / max_pdf
        r_env_transf = r * env_pdf_scaled
        if env_eff < 0.0:
            r_env_transf = -1 * r_env_transf + r
        if cate_state == 1:
            r_env_transf = -1 * r_env_transf + r

        return float(r_env_transf)


    def format_age_for_newick(self, age):
        frac_age, int_age = np.modf(age)
        int_age = str(int(int_age))
        frac_age = np.round(frac_age, self.nwk_decimal_digits)
        frac_age = str(frac_age).replace('0.', '')
        age_nwk = int_age.zfill(self.nwk_leading_digits) + '.' + frac_age.ljust(self.nwk_decimal_digits, '0')

        return age_nwk


    def update_nwk(self, nwkj, anagenetic = True):
        sp_name = nwkj[:(len(nwkj)-self.nwk_digits) ]
        new_age = 0.0
        if anagenetic:
            age = float(nwkj[-self.nwk_digits: ])
            new_age = age + 1.0 / self.scale
        new_age = self.format_age_for_newick(new_age)
        new_nwk = sp_name + new_age

        return new_nwk


    def add_speciation_to_nwk_str(self, nwk_str, nwkj, new_sp_idx):
        sp_name = nwkj[:(len(nwkj) - self.nwk_digits)]
        age = nwkj[-self.nwk_digits:]
        new_age = self.format_age_for_newick(0.0)
        replace_str = '(T' + new_sp_idx + ':' + new_age + ',' + sp_name + new_age + '):' + age
        nwk_str_new = nwk_str.replace(nwkj, replace_str)
        del nwk_str

        return nwk_str_new

    # def get_duplicates(self, a):
    #     uniques = np.unique(a, return_index=True)[1]
    #     result = a == a
    #     result[uniques] = False
    #     return result


    def get_number_of_equal_nodeages(self, tree):
        root_dist_list = []
        for nd in tree.preorder_node_iter():
            if nd.is_leaf() is False:
                root_dist = nd.distance_from_root()
                if root_dist > 0.0:
                    root_dist_list.append(root_dist)
        root_dist_array = np.array(root_dist_list)
        counts = np.unique(root_dist_array, return_counts=True)[1]
        nodeages_equal = np.sum(counts > 1)

        return nodeages_equal


    def make_equal_nodeages_different(self, tree):
        number_of_equal_nodeages = self.get_number_of_equal_nodeages(tree)
        a = 0.01 / self.scale
        while number_of_equal_nodeages > 0:
            for nd in tree.preorder_node_iter():
                if nd.is_leaf() is False:
                    root_dist = nd.distance_from_root()
                    if root_dist > 0.0:
                        jitter = np.random.uniform(-a, a, 1)[0]
                        nd.edge_length -= jitter
                        for child_nd in nd.child_nodes():
                            child_nd.edge_length += jitter
            number_of_equal_nodeages = self.get_number_of_equal_nodeages(tree)

        return tree


    # def resolve_polytomies(self, tree):
    #     # Not in newick sring but the a descendent may arise directly after the node (T0:1.2,T1:0.5):0.0
    #     # This will not work if the two speciation evens happens at the present (T0:0.0,T1:0.0):0.0
    #     a = 0.1 / self.scale
    #     for nd in tree.preorder_node_iter():
    #         if nd.is_leaf() is False and nd.edge_length == 0.0:
    #             jitter = np.random.uniform(0.0, a, 1)[0]
    #             nd.edge_length -= jitter
    #             for child_nd in nd.child_nodes():
    #                 print(str(child_nd.taxon))
    #                 print('is leaf', child_nd.is_leaf())
    #                 print('edge length before', child_nd.edge_length)
    #                 child_nd.edge_length += jitter
    #                 print('edge length after', child_nd.edge_length)
    #
    #     return tree



    def nwk_str_to_tree(self, nwk_str, ts_te, root):
        #root_abs = np.abs(root)
        # What if all species go extinct before the present?
        search_string = '):' + self.format_age_for_newick(0.0)
        replace_str = '):' + str(0.1 / self.scale)
        nwk_str_dichotomous = nwk_str.replace(search_string, replace_str)
        nwk_str_root = '[&R] ' + nwk_str_dichotomous
        tree = dendropy.Tree.get_from_string(nwk_str_root, 'newick')
        #tree = self.resolve_polytomies(tree)
        tree = self.make_equal_nodeages_different(tree)
        latest_extinction = np.min(ts_te[:, 1])
        root_abs = get_root_age(tree)
        # At least one extant species
        if latest_extinction == 0.0:
            for leaf in tree.leaf_node_iter():
                species_name = str(leaf.taxon)
                species_idx = int(species_name.replace("T", "").replace("'", ""))
                root_dist = leaf.distance_from_root()
                if ts_te[species_idx, 1] == 0.0:
                    delta_tip_height = root_abs - root_dist
                    leaf.edge_length += delta_tip_height
        # All species are extinct - this is not possible to specify with a newick string
        else:
            pass

        return tree


    def get_LTT(self, FA, LA, root):
        change_diversity_LA = np.repeat(-1, len(LA))
        change_diversity_LA[LA == 0.0] = 0
        change_diversity_FA = np.repeat(1, len(FA))
        change_diversity = np.concatenate((change_diversity_FA, change_diversity_LA))
        times_change = np.concatenate((FA, LA))
        times_order = np.argsort(times_change)[::-1]
        times_change = times_change[times_order]
        change_diversity = change_diversity[times_order]
        diversity = np.cumsum(change_diversity)
        interp1d_func = interp1d(times_change, diversity, kind = 'nearest', bounds_error = False, fill_value = (0.0, 0.0))
        timevec = np.linspace(-root, 0.0, int(-root * self.scale))
        interpolated_diversity = interp1d_func(timevec)

        return np.array([timevec, interpolated_diversity]).T


    def check_diversity_in_timewindow(self, FA, LA):
        timewindow = self.timewindow_rangeSP / self.scale
        FA_before = FA >= timewindow[0]
        FA_in = np.logical_and(FA <= timewindow[0], FA >= timewindow[1])
        LA_in = np.logical_and(LA <= timewindow[0], LA >= timewindow[1])
        LA_after = LA <= timewindow[1]
        FAbefore_LAin = np.logical_and(FA_before, LA_in)
        FAin_LAafter = np.logical_and(FA_in, LA_after)
        FAin_LAin = np.logical_and(FA_in, LA_in)
        diversity_in_window = np.sum(np.any((FAbefore_LAin, FAin_LAafter, FAin_LAin), axis = 0))
        diversity_in_rangeSP = diversity_in_window <= self.maxSP_timewindow and diversity_in_window >= self.minSP_timewindow

        return diversity_in_rangeSP, diversity_in_window


    def run_simulation(self, verbose = False):
        FAtrue = []
        LOtrue = []
        n_extinct = 0
        n_extant = 0
        prop_cat_traits_ok = False
        sp_env_ts = None
        if self.sp_env_file is not None:
            sp_env_ts = np.loadtxt(self.sp_env_file, skiprows=1)
        ex_env_ts = None
        if self.ex_env_file is not None:
            ex_env_ts = np.loadtxt(self.ex_env_file, skiprows=1)
        rangeSP_OK_in_timewindow = True
        if self.timewindow_rangeSP is not None:
            rangeSP_OK_in_timewindow = False
            self.timewindow_rangeSP = np.sort(self.timewindow_rangeSP)[::-1] * self.scale
            self.minSP = 0.0
            self.maxSP = np.inf
        exceeded_diversity = True
        while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP or n_extant < self.minExtant_SP or n_extant > self.maxExtant_SP or prop_cat_traits_ok == False or rangeSP_OK_in_timewindow == False or exceeded_diversity:
            if verbose:
                print('New round')
            root = -np.random.uniform(np.min(self.root_r), np.max(self.root_r))  # ROOT AGES
            dT, L_tt, M_tt, L, M, timesL, timesM, linL, linM, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_varcov_clado, cont_traits_effect_sp, cont_traits_effect_ex, expected_sd_cont_traits, cont_traits_effect_shift_sp, cont_traits_effect_shift_ex, n_cat_traits, cat_states, cat_traits_Q, cat_traits_effect, n_areas, dispersal, extirpation, env_eff_sp, env_eff_ex = self.get_random_settings(root, sp_env_ts, ex_env_ts, verbose)
            self.nwk_leading_digits = len(str(int(np.abs(root))))
            self.nwk_decimal_digits = len(str(int(self.scale)))
            self.nwk_digits = self.nwk_leading_digits + self.nwk_decimal_digits + 1

            FAtrue, LOtrue, anc_desc, cont_traits, cat_traits, mass_ext_time, mass_ext_mag, mass_ext_victim, lineage_weighted_lambda_tt, lineage_weighted_mu_tt, lineage_rates, biogeo, areas_comb, lineage_rates_through_time, nwk_str, exceeded_diversity = self.simulate(L_tt, M_tt, root, dT, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_varcov_clado, cont_traits_effect_sp, cont_traits_effect_ex, expected_sd_cont_traits, cont_traits_effect_shift_sp, cont_traits_effect_shift_ex, n_cat_traits, cat_states, cat_traits_Q, cat_traits_effect, n_areas, dispersal, extirpation, env_eff_sp, env_eff_ex)
            #print('exceeded_diversity', exceeded_diversity)
            prop_cat_traits_ok = self.check_proportion_cat_traits(n_cat_traits, cat_traits)

            n_extinct = len(LOtrue[LOtrue > 0.0])
            n_extant = len(LOtrue[LOtrue == 0.0])

            if self.timewindow_rangeSP is not None:
                rangeSP_OK_in_timewindow, diversity_in_window = self.check_diversity_in_timewindow(FAtrue, LOtrue)

            if verbose:
                print('N. species', len(LOtrue))
                if self.timewindow_rangeSP is not None:
                    print('N. species in timewindow', diversity_in_window)
                print('N. extant species', n_extant)
                print('Range speciation rate', np.round(np.nanmin(lineage_weighted_lambda_tt[1:-1]), 3), np.round(np.nanmax(lineage_weighted_lambda_tt[1:-1]), 3))
                print('Range extinction rate', np.round(np.nanmin(lineage_weighted_mu_tt[1:-1]), 3), np.round(np.nanmax(lineage_weighted_mu_tt[1:-1]), 3))

        ts_te = np.array([FAtrue, LOtrue]).T
        LTTtrue = self.get_LTT(FAtrue, LOtrue, root)
        true_rates_through_time = self.get_true_rate_through_time(root, L_tt, M_tt, lineage_weighted_lambda_tt, lineage_weighted_mu_tt)
        mass_ext_time = np.abs(np.array(mass_ext_time)) / self.scale
        mass_ext_mag = np.array(mass_ext_mag)

        tree = 'No tree when there is more than one starting species'
        tree_offset = 0.0
        if self.s_species == 1:
            tree = self.nwk_str_to_tree(nwk_str, ts_te, root)
            tree_offset = np.min(ts_te[:, 1])

        if isinstance(cat_traits, np.ndarray):
            cat_traits[0, :] = cat_traits[1, :]
        if isinstance(cont_traits, np.ndarray):
            cont_traits[0, :] = cont_traits[1, :]

        res_bd = {'lambda': L * self.scale,
                  'tshift_lambda': timesL / self.scale,
                  'mu': M * self.scale,
                  'tshift_mu': timesM / self.scale,
                  'true_rates_through_time': true_rates_through_time,
                  'mass_ext_time': mass_ext_time,
                  'mass_ext_magnitude': mass_ext_mag,
                  'mass_ext_victim': mass_ext_victim,
                  'linear_time_lambda': linL,
                  'linear_time_mu': linM,
                  'N_species': len(LOtrue),
                  'ts_te': ts_te,
                  'anc_desc': anc_desc,
                  'lineage_rates': lineage_rates,
                  'cont_traits': cont_traits,
                  'cat_traits': cat_traits,
                  'cont_traits_varcov': cont_traits_varcov,
                  'cont_traits_Theta1': cont_traits_Theta1,
                  'cont_traits_alpha': cont_traits_alpha,
                  'cont_traits_effect_sp': cont_traits_effect_sp,
                  'cont_traits_effect_ex': cont_traits_effect_ex,
                  'cont_traits_effect_shift_sp': cont_traits_effect_shift_sp,
                  'cont_traits_effect_shift_ex': cont_traits_effect_shift_ex,
                  'expected_sd_cont_traits': expected_sd_cont_traits,
                  'cat_traits_Q': cat_traits_Q,
                  'cat_traits_effect': cat_traits_effect,
                  'geographic_range': biogeo,
                  'range_states': areas_comb,
                  'env_eff_sp': 1.0 / env_eff_sp,
                  'env_eff_ex': 1.0 / env_eff_ex,
                  'lineage_rates_through_time': lineage_rates_through_time * self.scale,
                  'tree': tree,
                  'tree_offset': tree_offset,
                  'LTTtrue': LTTtrue,
                  'sim_scale': self.scale}
        if self.sp_env_file is not None:
            res_bd['env_sp'] = sp_env_ts
        if self.ex_env_file is not None:
            res_bd['env_ex'] = ex_env_ts
        if verbose:
            ltt = ""
            for i in range(int(max(FAtrue))):
                n = len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i])
                ltt += "\n%s\t%s\t%s" % (i, n, "*" * n)
            print(ltt)
        return res_bd



class fossil_simulator():
    def __init__(self,
                 range_q=[0.5, 5.],
                 range_alpha=None, # list of range e.g. [1.0, 5.0],
                 poi_shifts=0,
                 fixed_shift_times=[],
                 fixed_q=[],
                 q_loguniform=False,
                 alpha_loguniform=False,
                 age_effect=None, # list with alpha and beta of the beta distribution
                 age_mass_extinction=None, # list with alpha and beta of the beta distribution
                 cat_trait_effect=None, # dict of lists for each categorical trait e.g. for first trait with three states {'1': [0.5, 1.0, 2.0]}. For two traits {'1': [0.5, 1.0, 2.0], '2': [1.0, 1.0, 3.0]}
                 cont_trait_effect=None, # dict of lists for each continuous trait, growth rate, min, and max e.g. for first trait {'1': [2.0, 0.2, 2.0]}. For two traits {'1': [2.0, 0.2, 2.0], '2': [4.5, 0.5, 1.5]}
                 qtt_res=1, # time resolution for the sampling rate through time
                 seed = 0):
        self.range_q = range_q
        self.range_alpha = range_alpha
        self.poi_shifts = poi_shifts
        self.fixed_shift_times = fixed_shift_times
        self.fixed_q = fixed_q
        self.q_loguniform = q_loguniform
        self.alpha_loguniform = alpha_loguniform
        self.age_effect = age_effect
        self.age_mass_extinction = age_mass_extinction
        self.cat_trait_effect = cat_trait_effect
        self.cont_trait_effect = cont_trait_effect
        self.qtt_res = qtt_res
        if seed:
            np.random.seed(seed)


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
        shift_time_q_highres = np.arange(root_age, death_age, -self.qtt_res)
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
        len_q = len(q)
        self.make_cont_trait_multiplier(res_bd)
        self.make_taxon_age_multipliers(res_bd, len_q, shift_time_q)

        for i in range(len_q):
            dur, ts, te = self.get_duration(sp_x, upper=shift_time_q[i], lower=shift_time_q[i + 1])
            dur = dur.flatten()
            cat_trait_multiplier = self.make_cat_trait_multiplier(res_bd,
                                                                  upper=shift_time_q[i],
                                                                  lower=shift_time_q[i + 1])
            cont_trait_multipliers = self.get_cont_traits_multipliers_time_slice(res_bd,
                                                                                 upper=shift_time_q[i],
                                                                                 lower=shift_time_q[i + 1])
            age_multipliers = self.age_multipliers[:, i].reshape(-1)
            poi_rate_occ = q[i] * cat_trait_multiplier * cont_trait_multipliers * age_multipliers * sampl_hetero * dur
            exp_occ = np.round(np.random.poisson(poi_rate_occ))
            non_zero_branch_length = dur > 0.0
            qtt_taxa[non_zero_branch_length, i] = poi_rate_occ[non_zero_branch_length] / dur[non_zero_branch_length]

            for y in range(n_taxa):
                occ_y = np.random.uniform(ts[y], te[y], exp_occ[y])
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

        return occ2, lineages_sampled, alpha, qtt_taxa


    def harmonic_mean_q_through_time(self, q_rates, shift_time_q):
        qtt = np.full(q_rates.shape[1], np.nan)
        not_all_nan = np.sum(np.isnan(q_rates), axis=0) < q_rates.shape[0]
        qtt[not_all_nan] = 1 / np.nanmean(1 / q_rates[:, not_all_nan], axis=0)

        qtt = np.concatenate((qtt, qtt[-1]), axis=None)
        qtt = np.c_[shift_time_q, qtt]
        qtt = pd.DataFrame(qtt, columns=['time', 'q'])

        return qtt


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
                num_states = len(np.unique(cat_traits[:, i]))
                if num_states > 1 and str(i + 1) in self.cat_trait_effect.keys():
                    multipliers *= np.array(self.cat_trait_effect[str(i + 1)])[cat_traits[:, i]]
        return multipliers


    def get_cont_traits_multipliers_time_slice(self, res_bd, upper=np.inf, lower=-np.inf):
        time = res_bd['true_rates_through_time']['time']
        time = np.concatenate((np.inf, time, -np.inf), axis=None)
        trait_idx = np.logical_and(time > lower, time <= upper)
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
            if not self.age_mass_extinction is None and np.any(me_vict == 1):
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
                        if not self.age_mass_extinction is None and me_vict[i] == 1:
                            b = beta_distr.pdf(x, self.age_mass_extinction[1], self.age_mass_extinction[0])
                        self.age_multipliers[i, m_idx[j]] = np.mean(b)


    def run_simulation(self, res_bd):
        sp_x = res_bd['ts_te']
        self.n_taxa = sp_x.shape[0]
        is_alive = self.get_is_alive(sp_x)

        q, shift_time_q, shift_time_q_write = self.make_sampling_rate(sp_x)
        fossil_occ, taxa_sampled, alpha, qtt_taxa = self.get_fossil_occurrences(res_bd, q, shift_time_q, is_alive)

        taxon_names = self.get_taxon_names(taxa_sampled)
        qtt = self.harmonic_mean_q_through_time(qtt_taxa, shift_time_q)
        shift_time_q_write = shift_time_q_write[1:-1]
        qtt_taxa = pd.DataFrame(qtt_taxa, columns=shift_time_q[1:].tolist(), index=taxon_names)

        d = {'fossil_occurrences': fossil_occ,
             'taxon_names': taxon_names,
             'taxa_sampled': taxa_sampled,
             'q': q,
             'shift_time': shift_time_q_write,
             'alpha': alpha,
             'qtt': qtt,
             'qtt_taxa': qtt_taxa,
             'write_me_trait': self.write_me_trait}

        return d


class write_PyRate_files():
    def __init__(self,
                 output_wd = '',
                 delta_time = 1.0,
                 name = None):
        self.output_wd = output_wd
        self.delta_time = delta_time
        self.name_file = name

    def write_occurrences(self, sim_fossil, name_file):
        fossil_occ = sim_fossil['fossil_occurrences']
        taxon_names = sim_fossil['taxon_names']
        py = "%s/%s/%s.py" % (self.output_wd, name_file, name_file)
        pyfile = open(py, "w")
        pyfile.write('#!/usr/bin/env python')
        pyfile.write('\n')
        pyfile.write('from numpy import *')
        pyfile.write('\n')
        pyfile.write('data_1 = [')  # Open block with fossil occurrences
        pyfile.write('\n')
        for i in range(len(fossil_occ)):
            pyfile.write('array(')
            pyfile.write(str(fossil_occ[i].tolist()))
            pyfile.write(')')
            if i != (len(fossil_occ) - 1):
                pyfile.write(',')
            pyfile.write('\n')
        pyfile.write(']')  # End block with fossil occurrences
        pyfile.write('\n')
        pyfile.write('d = [data_1]')
        pyfile.write('\n')
        pyfile.write('names = ["%s"]' % name_file)
        pyfile.write('\n')
        pyfile.write('def get_data(i): return d[i]')
        pyfile.write('\n')
        pyfile.write('def get_out_name(i): return names[i]')
        pyfile.write('\n')
        pyfile.write('taxa_names = ')
        pyfile.write(str(taxon_names))
        pyfile.write('\n')
        pyfile.write('def get_taxa_names(): return taxa_names')
        pyfile.flush()


    def write_q_epochs(self, sim_fossil, name_file):
        file_q_epochs = '%s/%s/%s_q_epochs.txt' % (self.output_wd, name_file, name_file)
        np.savetxt(file_q_epochs, np.sort(sim_fossil['shift_time']), delimiter='\t', fmt='%f')


    def write_qtt(self, sim_fossil, name_file):
        file_qtt = '%s/%s/%s_true_qtt.txt' % (self.output_wd, name_file, name_file)
        sim_fossil['qtt'].to_csv(file_qtt, na_rep='NA', index=False, sep='\t', float_format="%.3f")


    def write_qtt_per_taxon(self, sim_fossil, name_file):
        file_qtt = '%s/%s/%s_true_qtt_per_taxon.txt' % (self.output_wd, name_file, name_file)
        sim_fossil['qtt_taxa'].to_csv(file_qtt, na_rep='NA', index=True, sep='\t', float_format="%.3f")


    def write_true_tste(self, res_bd, sim_fossil, name_file):
        tste = res_bd['ts_te'][sim_fossil['taxa_sampled'], :]
        tste_df = pd.DataFrame(data = tste, columns = ['ts', 'te'], index = sim_fossil['taxon_names'])
        tste_file = "%s/%s/%s_true_tste.csv" % (self.output_wd, name_file, name_file)
        tste_df.to_csv(tste_file, header = True, sep = '\t', index = True, na_rep = 'NA')


    # def get_mean_cont_traits_per_taxon(self, sim_fossil, res_bd):
    #     cont_traits = res_bd['cont_traits']
    #     cont_traits = cont_traits[:, :, sim_fossil['taxa_sampled']]
    #     means_cont_traits = np.nanmean(cont_traits, axis = 0)
    #     means_cont_traits = means_cont_traits.transpose()
    #
    #     return means_cont_traits


    def get_mean_cont_traits_per_taxon_from_sampling_events(self, sim_fossil, res_bd):
        fossil_occ = sim_fossil['fossil_occurrences']
        taxa_sampled = sim_fossil['taxa_sampled']
        cont_traits = res_bd['cont_traits']
        cont_traits = cont_traits[:, :, taxa_sampled]
        time = res_bd['true_rates_through_time']['time']
        n_lineages = len(taxa_sampled)
        means_cont_traits = np.zeros((n_lineages, cont_traits.shape[1]))
        for i in range(n_lineages):
            occ_i = fossil_occ[i]
            trait_idx = np.searchsorted(time, occ_i)
            cont_trait_i = cont_traits[trait_idx, :, i]
            means_cont_traits[i,:] = np.nanmean(cont_trait_i, axis = 0)

        return means_cont_traits


    def center_and_scale_unitvar(self, cont_traits):
        n = cont_traits.shape[1]
        cont_traits_mean_sd = np.zeros((2, n))
        cont_traits_mean_sd[0,:] = np.mean(cont_traits, axis = 0)
        cont_traits_mean_sd[1,:] = np.std(cont_traits, axis = 0)
        cont_traits -= cont_traits_mean_sd[0,:]
        cont_traits /= cont_traits_mean_sd[1,:]
        colnames = []
        for i in range(n):
            colnames.append('cont_trait_%s' % i)
        mean_sd = pd.DataFrame(data = cont_traits_mean_sd, columns = colnames)
        mean_sd.index = ['mean', 'sd']

        return cont_traits, mean_sd


    def get_majority_cat_trait_per_taxon(self, sim_fossil, res_bd):
        cat_traits = res_bd['cat_traits']
        n_cat_traits = cat_traits.shape[1]
        taxa_sampled = sim_fossil['taxa_sampled']
        n_taxa_sampled = len(taxa_sampled)
        maj_cat_traits = np.zeros(n_taxa_sampled * n_cat_traits, dtype = int).reshape((n_taxa_sampled, n_cat_traits))
        for i in range(n_cat_traits):
            cat_traits_i = cat_traits[:, i, taxa_sampled]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=Warning)
                maj_cat_traits_i = mode(cat_traits_i, nan_policy='omit')[0]#[0] # Did this change with newer scipy?
            maj_cat_traits[:, i] = maj_cat_traits_i.astype(int)

        return maj_cat_traits


    def is_ordinal_trait(self, Q):
        is_ordinal = False
        if np.all(Q[0,2:] == 0.0) and np.all(Q[-1,:-2] == 0.0):
            is_ordinal = True

        return is_ordinal


    def make_one_hot_encoding(self, a):
        b = np.unique(a)
        n_states = len(b)
        c = a - np.min(a)
        one_hot = np.eye(n_states)[c]
        one_hot = one_hot.astype(int)

        return one_hot, b


    def make_time_vector(self, res_bd):
        root_age = np.max(res_bd['ts_te'])
        root_age = root_age + 0.2 * root_age # Give a little extra time before the root?!
        time_vector = np.arange(0.0, root_age, self.delta_time)

        return time_vector


    def write_time_vector(self, res_bd, name_file):
        time_vector = self.make_time_vector(res_bd)
        file_time = '%s/%s/%s_time.txt' % (self.output_wd, name_file, name_file)
        np.savetxt(file_time, time_vector, delimiter='\t', fmt='%f')


    def write_true_rates_through_time(self, rates, name_file):
        rate_file = "%s/%s/%s_true_rates_through_time.csv" % (self.output_wd, name_file, name_file)
        rates.to_csv(rate_file, header = True, sep = '\t', index = False, na_rep = 'NA')


    def get_sampling_rates_through_time(self, sim_fossil, res_bd):
        q = sim_fossil['q']
        time_sampling = res_bd['true_rates_through_time']['time'].to_numpy()
        time_sampling = time_sampling[::-1]
        shift_time = np.concatenate(( np.array([time_sampling[0] + 0.01]), sim_fossil['shift_time'], np.zeros(1)))
        n_shifts = len(sim_fossil['shift_time'])

        # Until knowing what to do here, we simply bypass calculating age-dependent sampling rates
        q_tt = np.zeros(len(time_sampling), dtype='float')
        for i in range(n_shifts + 1):
            qidx = np.logical_and(time_sampling < shift_time[i], time_sampling >= shift_time[i + 1])
            q_tt[qidx] = q[i]

        return q_tt[::-1]


    def write_lineage_rates(self, sim_fossil, res_bd, name_file):
        taxa_sampled = sim_fossil['taxa_sampled']
        taxon_names = sim_fossil['taxon_names']
        lineage_rate = res_bd['lineage_rates']
        lineage_rate = lineage_rate[taxa_sampled,:]
        names_df = pd.DataFrame(data = taxon_names, columns = ['scientificName'])
        colnames = ['ts', 'te', 'speciation', 'extinction', 'ancestral_speciation']
        if res_bd['cont_traits'] is not None:
            n_cont_traits = res_bd['cont_traits'].shape[1]
            for y in range(n_cont_traits):
                colnames.append('cont_trait_anc_%s' % y)
            for y in range(n_cont_traits):
                colnames.append('cont_trait_ts_%s' % y)
            for y in range(n_cont_traits):
                colnames.append('cont_trait_te_%s' % y)
        if res_bd['cat_traits'] is not None:
            n_cat_traits = res_bd['cat_traits'].shape[1]
            for y in range(n_cat_traits):
                colnames.append('cat_trait_%s' % y)
            for y in range(n_cat_traits):
                colnames.append('cat_trait_anc_%s' % y)

        tste_rates = pd.DataFrame(data = lineage_rate, columns = colnames)
        df = pd.concat([names_df, tste_rates], axis=1)
        file = "%s/%s/%s_lineage_rates.csv" % (self.output_wd, name_file, name_file)
        df.to_csv(file, header = True, sep = '\t', index = False, na_rep = 'NA')


    def expand_grid(self, x, y, z):
        xG, yG, zG = np.meshgrid(x, y, z)  # create the actual grid
        xG = xG.flatten()  # make the grid 1d
        yG = yG.flatten()
        zG = zG.flatten()
        gr = np.stack((xG, yG, zG), axis = 1)
        return gr


    def write_cont_trait_effects(self, res_bd, name_file):
        cte_sp = res_bd['cont_traits_effect_sp']
        cte_ex = res_bd['cont_traits_effect_ex']
        # Probably there is something easier like cte_sp.flatten().reshape((, 5))
        if len(cte_sp) > 0:
            n_time_bins, n_cont_traits, n_cat_states, n_par = cte_sp.shape
            time_bins = np.arange(n_time_bins)
            cont_traits = np.arange(n_cont_traits)
            cat_states = np.arange(n_cat_states)
            gr = self.expand_grid(time_bins, cont_traits, cat_states)
            n_comb = gr.shape[0]
            # trait effect for all combinations of time bins, cont traits, and states
            cte = np.zeros(2 * n_comb * (4 + n_par)).reshape((2 * n_comb, 4 + n_par))
            cte[:n_comb, 1:4] = gr
            cte[n_comb:, 1:4] = gr
            cte[n_comb:, 0] = 1.0 # denotes extinction
            for h in range(gr.shape[0]):
                i, j, k = gr[h, :]
                cte[h, 4:] = cte_sp[i, j, k,:]
                cte[n_comb + h, 4:] = cte_ex[i, j, k, :]
            cte_df = pd.DataFrame(data = cte,
                                  columns = ['extinction', 'time_bin', 'trait', 'state',
                                             'magnitude', 'bell_or_u', 'min_pdf', 'max_pdf', 'optimum'])
            cont_trait_effect_name = "%s/%s/%s_cont_trait_effect.csv" % (self.output_wd, name_file, name_file)
            cte_df.to_csv(cont_trait_effect_name, header = True, sep = '\t', index = False, na_rep = 'NA')
            sd_traits_name = "%s/%s/%s_expected_sd_cont_traits.csv" % (self.output_wd, name_file, name_file)
            np.savetxt(sd_traits_name, res_bd['expected_sd_cont_traits'], delimiter='\t', fmt='%f')


    def bin_and_write_env(self, env_var, env_file_name):
        max_age_env = np.max(env_var[:, 0])
        time_vec = np.arange(0, max_age_env + self.delta_time, self.delta_time)
        binned_env = get_binned_continuous_variable(env_var, time_vec, 1.0)
        binned_env = np.stack((time_vec[:-1], binned_env), axis = 1)
        np.savetxt(env_file_name, binned_env, delimiter = '\t', fmt='%f')


    def get_cophenetic_distance_matrix(self, tree):
        pdm = tree.phylogenetic_distance_matrix().as_data_table()._data

        species = [tip.label for tip in tree.taxon_namespace]
        ntips = len(species)
        pD = np.zeros((ntips, ntips))
        for i in range(ntips):
            for j in range(i, ntips):
                d = pdm[species[i]][species[j]]
                pD[i][j] = d
                pD[j][i] = d

        return ({'pD': pD, 'species': species})


    def pvr_decomp(self, tree):
        """
        Phylogenetic distances matrix (eigen)decomposition.
        Arguments
        ----------
        tree : dendropy phylogeny
        Returns
        -------
        dictionary : {"eigenval": E, "eigenvect": V, "species":vcv["species"]}
        """
        pD = self.get_cophenetic_distance_matrix(tree)
        P = pD['pD']
        A = -0.5 * P
        L = np.ones(P.size).reshape((P.shape[0], P.shape[1]))
        D = np.eye(P.shape[0]) - ((1 / P.shape[1]) * L)
        P = la.multi_dot([D, A, D])
        E, V = la.eigh(P, UPLO='L')
        key = np.argsort(E)[::-1][:None]
        E, V = E[key], V[:, key]

        # Sort alphanumerically (T0, T1, T2, ..., T10, T11, ..., T20)
        species = pD['species']
        sort_idx = np.array(index_natsorted(species))
        species = np.array(species)[sort_idx].tolist()  # Why there is no fancy indexing of lists?
        E = E[sort_idx]
        V = V[sort_idx, :]

        # Center and scale to unitvariance
        V -= np.mean(V, axis = 0)
        V /= np.std(V, axis = 0)

        return ({'eigenval': E, 'eigenvect': V, 'species': species})


    def write_sampling_heterogeneity(self, sim_fossil, name_file):
        alpha_name = "%s/%s/%s_true_sampling_alpha.csv" % (self.output_wd, name_file, name_file)
        np.savetxt(alpha_name, sim_fossil['alpha'], delimiter='\t', fmt='%f')


    def run_writter(self, sim_fossil, res_bd, num_pvr=0, write_tree=False):
        # Create a directory for the output
        try:
            os.mkdir(self.output_wd)
        except OSError as error:
            print(error)
        # Create a subdirectory for PyRate with either a random name or a given name
        if self.name_file is None:
            name_file = ''.join(random.choices(string.ascii_lowercase, k = 10))
        else:
            name_file = self.name_file

        path_make_dir = os.path.join(self.output_wd, name_file)
        try:
            os.mkdir(path_make_dir)
        except OSError as error:
            print(error)

        self.write_occurrences(sim_fossil, name_file)
        self.write_q_epochs(sim_fossil, name_file)
        self.write_sampling_heterogeneity(sim_fossil, name_file)
        self.write_qtt(sim_fossil, name_file)
        self.write_qtt_per_taxon(sim_fossil, name_file)

        traits = pd.DataFrame(data = sim_fossil['taxon_names'], columns = ['scientificName'])

        if res_bd['cont_traits'].shape[1] > 0:
            #mean_cont_traits_taxon = self.get_mean_cont_traits_per_taxon(sim_fossil, res_bd)
            mean_cont_traits_taxon = self.get_mean_cont_traits_per_taxon_from_sampling_events(sim_fossil, res_bd)
            mean_cont_traits_taxon, backscale_cont_traits = self.center_and_scale_unitvar(mean_cont_traits_taxon)
            for i in range(mean_cont_traits_taxon.shape[1]):
                traits['cont_trait_%s' % i] = mean_cont_traits_taxon[:,i]
            backscale_file = "%s/%s/%s_backscale_cont_traits.csv" % (self.output_wd, name_file, name_file)
            backscale_cont_traits.to_csv(backscale_file, header = True, sep = '\t', index = True)

        if res_bd['cat_traits'].shape[1] > 0:
            maj_cat_traits_taxon = get_majority_cat_trait_per_taxon(res_bd, sim_fossil)
            for y in range(maj_cat_traits_taxon.shape[1]):
                is_ordinal = self.is_ordinal_trait(res_bd['cat_traits_Q'][y])
                if is_ordinal:
                    traits['cat_trait_%s' % y] = maj_cat_traits_taxon[:,y]
                else:
                    cat_traits_taxon_one_hot, names_one_hot = self.make_one_hot_encoding(maj_cat_traits_taxon[:,y])
                    for i in range(cat_traits_taxon_one_hot.shape[1]):
                        traits['cat_trait_%s_%s' % (y, names_one_hot[i])] = cat_traits_taxon_one_hot[:, i]

        if sim_fossil['write_me_trait']:
            traits['me_victim'] = res_bd['mass_ext_victim'][sim_fossil['taxa_sampled']]

        if write_tree or num_pvr > 0:
            tree_trimmed_by_lad, _ = trim_tree_by_lad(res_bd, sim_fossil)
            if write_tree:
                tree_file = "%s/%s/%s_phylo.tre" % (self.output_wd, name_file, name_file)
                tree_trimmed_by_lad.ladderize(ascending=False)
                tree_trimmed_by_lad.write(path=tree_file, schema='newick')
        if num_pvr > 0:
            pvr = self.pvr_decomp(tree_trimmed_by_lad)['eigenvect']
            traits_pvr = traits.copy()
            for i in range(num_pvr):
                traits_pvr['pvr_%s' % i] = pvr[:, i]

        if traits.shape[1] > 1:
            trait_file = "%s/%s/%s_traits.csv" % (self.output_wd, name_file, name_file)
            traits.to_csv(trait_file, header=True, sep='\t', index=False)
            if num_pvr:
                trait_pvr_file = "%s/%s/%s_traits_pvr.csv" % (self.output_wd, name_file, name_file)
                traits_pvr.to_csv(trait_pvr_file, header=True, sep='\t', index=False)


        self.write_time_vector(res_bd, name_file)

        qtt = self.get_sampling_rates_through_time(sim_fossil, res_bd)
        rates = res_bd['true_rates_through_time']
        rates['sampling'] = qtt
        self.write_true_rates_through_time(rates, name_file)

        self.write_lineage_rates(sim_fossil, res_bd, name_file)

        self.write_true_tste(res_bd, sim_fossil, name_file)

        key_names = list(res_bd.keys())
        if 'env_sp' in key_names:
            env_sp_name = "%s/%s/%s_env_sp.csv" % (self.output_wd, name_file, name_file)
            self.bin_and_write_env(res_bd['env_sp'], env_sp_name)
        if 'env_ex' in key_names:
            env_ex_name = "%s/%s/%s_env_ex.csv" % (self.output_wd, name_file, name_file)
            self.bin_and_write_env(res_bd['env_ex'], env_ex_name)

        self.write_cont_trait_effects(res_bd, name_file)



        return name_file


# Bin environmental variable into time time-bins.
# Needed to simulate diversification and to write files for analyses.
# def get_binned_continuous_variable(env_var, time_vec, scaletime):
#     times = env_var[:, 0]
#     times = times * scaletime
#     values = env_var[:, 1]
#     mean_var = np.zeros(len(time_vec) - 1)
#     mean_var[:] = np.nan
#     for i in range(1, len(time_vec)):
#         t_min = time_vec[i - 1]
#         t_max = time_vec[i]
#         in_range_M = (times <= t_max).nonzero()[0]
#         in_range_m = (times >= t_min).nonzero()[0]
#         mean_var[i - 1] = np.mean(values[np.intersect1d(in_range_m, in_range_M)])
#
#     return mean_var


def get_binned_continuous_variable(env_var, time_vec, scaletime):
    times = env_var[:, 0] * scaletime
    values = env_var[:, 1]
    bins = np.digitize(times, time_vec)
    mean_var = np.zeros(len(time_vec) - 1)
    mean_var[:] = np.nan
    for i in range(1, len(time_vec)):
        mean_var[i - 1] = np.mean(values[bins == i])

    return mean_var


def keep_fossils_in_interval(fossils, keep_in_interval, keep_extant = True):
    fossil_occ = fossils['fossil_occurrences']
    n_lineages = len(fossil_occ)
    occ = fossil_occ
    taxon_names = fossils['taxon_names']
    taxa_sampled = fossils['taxa_sampled']
    if keep_in_interval is not None:
        occ = []
        keep = []
        for i in range(n_lineages):
            occ_i = fossil_occ[i]
            is_alive = np.any(occ_i == 0)
            occ_keep = np.array([])
            for y in range(keep_in_interval.shape[0]):
                occ_keep_y = occ_i[np.logical_and(occ_i <= keep_in_interval[y,0], occ_i > keep_in_interval[y,1])]
                occ_keep = np.concatenate((occ_keep, occ_keep_y))
            occ_i = occ_keep
            if is_alive and keep_extant:
                occ_i = np.concatenate((occ_i, np.zeros(1)))
            if len(occ_i) > 0:
                occ.append(occ_i)
                keep.append(i)

        taxon_names = np.array(taxon_names)
        taxon_names = taxon_names[keep]
        taxon_names = taxon_names.tolist()
        taxa_sampled = taxa_sampled[keep]

    fossils['fossil_occurrences'] = occ
    fossils['taxon_names'] = taxon_names
    fossils['taxa_sampled'] = taxa_sampled

    return fossils


def get_interval_exceedings(fossils, ts_te, keep_in_interval):
    taxa_sampled = fossils['taxa_sampled']
    ts_te = ts_te[taxa_sampled, :]
    intervall_exceeds_df = pd.DataFrame(data = np.zeros((1,2)), columns = ['ts_before_upper_bound', 'te_after_lower_bound'])
    if keep_in_interval is not None:
        n_intervals = keep_in_interval.shape[0]
        intervall_exceeds = np.zeros((1, 2 * n_intervals))
        colnames = []
        ts = ts_te[:, 0] + 0.0
        te = ts_te[:, 1] + 0.0
        for y in range(n_intervals):
            # Speciation before upper interval boundary, discard lineages that speciate before but go also extinct
            intervall_exceeds[0, 2 * y] = np.sum(np.logical_and(ts > keep_in_interval[y,0], (te > keep_in_interval[y,0]) == False))
            # Extinction after lower interval boundary, discard lineages that speciate after the lower interval boundary
            intervall_exceeds[0, 1 + 2 * y] = np.sum(np.logical_and(te < keep_in_interval[y,1], (ts < keep_in_interval[y,1]) == False))
            colnames.append('ts_before_%s' % str(keep_in_interval[y,0]))
            colnames.append('te_after_%s' % str(keep_in_interval[y, 1]))

        intervall_exceeds_df = pd.DataFrame(data = intervall_exceeds, columns = colnames)

    return intervall_exceeds_df


def get_root_age(tree, include_root = True):
    mrca_age = tree.max_distance_from_root()
    root_age = mrca_age
    if include_root:
        root_length = tree.internal_edges()[0].length
        root_age += root_length

    return root_age


# Also used within write_FBD_tree. Try to replace this later!
def trim_tree_by_lad(res_bd, fossils, trim_edges=False, extant_only=False):
    taxon_names = fossils['taxon_names']
    fossil_occurrences = fossils['fossil_occurrences']
    tree_trimmed = res_bd['tree'].clone()
    tree_trimmed = tree_trimmed.extract_tree_with_taxa_labels(labels=set(taxon_names))
    # tree_trimmed.update_taxon_namespace()
    # dirty hack to update tree_trimmed.taxon_namespace
    tree_trimmed = dendropy.Tree.get(data=tree_trimmed.as_string(schema="newick"), schema="newick")
    ts_te = res_bd['ts_te']
    keep_taxa = []
    a = 1.0 / res_bd['sim_scale']
    for leaf in tree_trimmed.leaf_node_iter():
        if leaf.is_leaf():
            leaf_name = str(leaf.taxon)
            leaf_name = leaf_name.replace("'", "")
            ts_te_idx = int(leaf_name.replace("T", "").replace("'", ""))
            te_leaf = ts_te[ts_te_idx, 1]
            if (te_leaf != 0.0 or trim_edges) and extant_only is False:
                occ_idx = taxon_names.index(leaf_name)
                lad_leaf = np.min(fossil_occurrences[occ_idx])
                shorten_branch = lad_leaf - te_leaf
                if (leaf.edge_length - shorten_branch) >= 0:
                    # Avoid branches descending from a node to the past (b/c of max fossil age older than the node)
                    leaf.edge_length = leaf.edge_length - shorten_branch
                    keep_taxa.append(leaf_name)
                else:
                    # Nevermind, just shorten the branches until the node
                    jitter = np.random.uniform(-a, a, 1)[0]
                    leaf.edge_length = jitter + 1.0 / res_bd['sim_scale']
                    keep_taxa.append(leaf_name)
            elif te_leaf == 0.0:
                keep_taxa.append(leaf_name)
    # Remove taxa with maximum fossil age older than the node from which the taxa descends
    tree_trimmed = tree_trimmed.extract_tree_with_taxa_labels(labels=set(keep_taxa))
    # tree_trimmed.update_taxon_namespace()
    tree_trimmed = dendropy.Tree.get(data=tree_trimmed.as_string(schema="newick"), schema="newick")

    return tree_trimmed, keep_taxa


def get_majority_cat_trait_per_taxon(res_bd, sim_fossil=None, upper=np.inf, lower=-np.inf):
    cat_traits = res_bd['cat_traits']
    n_cat_traits = cat_traits.shape[1]
    n_taxa_sampled = cat_traits.shape[2]
    taxa_sampled = np.arange(n_taxa_sampled, dtype=int)
    if not sim_fossil is None:
        taxa_sampled = sim_fossil['taxa_sampled']
        n_taxa_sampled = len(taxa_sampled)
    maj_cat_traits = np.zeros(n_taxa_sampled * n_cat_traits, dtype = int).reshape((n_taxa_sampled, n_cat_traits))
    time = res_bd['true_rates_through_time']['time']
    time = np.concatenate((np.inf, time, -np.inf), axis=None)
    trait_idx = np.logical_and(time > lower, time <= upper)
    for i in range(n_cat_traits):
        cat_traits_i = cat_traits[:, i, taxa_sampled]
        cat_traits_i = cat_traits_i[trait_idx, :]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            maj_cat_traits_i = mode(cat_traits_i, nan_policy='omit')[0]
        maj_cat_traits_i[np.isnan(maj_cat_traits_i)] = 0
        maj_cat_traits[:, i] = maj_cat_traits_i.astype(int)

    return maj_cat_traits



class write_FBD_files():
    def __init__(self,
                 output_wd = '',
                 name_file = '',
                 interval_size = 1.0,
                 FBD_rate_prior = 'HSMRF',
                 translate = None, # move occurrence by time X (only useful if lineages are extinct)
                 interval_ages = None,
                 padding = [np.inf, 0.0],
                 fix_fake_bin = False):
        self.output_wd = output_wd
        self.name_file = name_file
        self.interval_size = interval_size
        self.FBD_rate_prior = FBD_rate_prior
        self.translate = translate
        self.interval_ages = interval_ages
        self.padding = padding
        self.fix_fake_bin = fix_fake_bin

    def get_ranges(self, fossils_copy):
        taxon_names = fossils_copy['taxon_names']
        n_lineages = len(taxon_names)
        ranges = pd.DataFrame(data = taxon_names, columns = ['taxon'])
        ranges['min'] = np.zeros(n_lineages)
        ranges['max'] = np.zeros(n_lineages)
        occ = fossils_copy['fossil_occurrences']
        for i in range(n_lineages):
            occ_i = occ[i]
            ranges.iloc[i, 1] = np.min(occ_i)
            ranges.iloc[i, 2] = np.max(occ_i)

        return ranges


    def get_occurrences_per_interval(self, fossils_copy, interval_ages):
        taxon_names = fossils_copy['taxon_names']
        n_lineages = len(taxon_names)
        names_df = pd.DataFrame(data = taxon_names, columns = ['taxon'])
        n_intervals = interval_ages.shape[0]
        counts = np.zeros((n_lineages, n_intervals), dtype = int)
        occ = fossils_copy['fossil_occurrences']
        for i in range(n_lineages):
            for y in range(n_intervals):
                occ_i = occ[i]
                # What to do with a record at the present? Is this rho for FBD?
                occ_i_y = occ_i[np.logical_and(occ_i <= interval_ages[y, 0], occ_i > interval_ages[y, 1])]
                counts[i, y] = len(occ_i_y)
        # Omit empty leading or trailing bins
        cs = np.sum(counts, axis = 0)
        cs1 = np.where(cs > 0)[0]
        incl_bins = np.arange(cs1[0], cs1[-1] + 1)
        counts = counts[:, incl_bins]
        interval_ages = interval_ages[incl_bins,:]

        # change to order present -> past
        counts = counts[:, ::-1]

        if self.padding[1] != 0.0:
            pad_interval = np.array([[interval_ages[-1,1],0.0]])
            interval_ages = np.concatenate((interval_ages, pad_interval), axis = 0)
            pad_counts = np.zeros(n_lineages, dtype = int).reshape((n_lineages, 1))
            counts = np.concatenate((pad_counts, counts), axis = 1)
        if self.padding[0] != np.inf:
            pad_interval = np.array([[np.inf, interval_ages[0, 0]]])
            interval_ages = np.concatenate((pad_interval, interval_ages), axis=0)
            pad_counts = np.zeros(n_lineages, dtype = int).reshape((n_lineages, 1))
            counts = np.concatenate((counts, pad_counts), axis=1)

        counts_df = pd.DataFrame(data = counts, columns = interval_ages[:, 1][::-1])
        counts_df = pd.concat([names_df, counts_df], axis = 1)

        return counts_df, interval_ages


    def get_age_range_fossils(self, fossils_copy):
        occ = fossils_copy['fossil_occurrences']
        len_occ = len(occ)
        max_ages_taxa = np.zeros(len_occ)
        min_ages_taxa = np.zeros(len_occ)
        for i in range(len(occ)):
            max_ages_taxa[i] = np.max(occ[i])
            min_ages_taxa[i] = np.min(occ[i])
        max_age = np.max(max_ages_taxa)
        max_age = np.ceil(max_age)
        min_age = np.min(min_ages_taxa)

        return max_age, min_age


    def translate_fossil_occurrences(self, fossils_copy):
        for i in range(len(fossils_copy['fossil_occurrences'])):
            fossils_copy['fossil_occurrences'][i] = fossils_copy['fossil_occurrences'][i] - self.translate


    def write_script(self, interval_ages, min_age):
        rho = 1.0
        bounded = "FALSE"
        if self.translate is not None or min_age > 0.0:
            rho = 0.0
        if min_age > 0.0:
            bounded = "TRUE"
        if self.FBD_rate_prior == 'HSMRF' and self.fix_fake_bin is False and self.padding[0] == np.inf and self.padding[1] == 0.0:
            # FBD for complete data or truncated but without padding
            scr = "%s/%s/%s/%s/%s_FBDR_HSMRF.Rev" % (self.output_wd, self.name_file, 'FBD', 'scripts', self.name_file)
            scrfile = open(scr, "w")
            scrfile.write('######################################')
            scrfile.write('\n')
            scrfile.write('# FBD using stratigraphic range data #')
            scrfile.write('\n')
            scrfile.write('######################################')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# read stratigraphic ranges')
            scrfile.write('\n')
            scrfile.write('taxa = readTaxonData(file = "data/%s_ranges.csv")' % self.name_file)
            scrfile.write('\n')
            scrfile.write('# read fossil counts')
            scrfile.write('\n')
            scrfile.write('k <- readDataDelimitedFile(file = "data/%s_counts.csv", header = true, rownames = true)' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            timeline = interval_ages[1:, 0]
            timeline = timeline[::-1]
            timeline = timeline.tolist()
            scrfile.write('# interval boundaries')
            scrfile.write('\n')
            scrfile.write('timeline <- v(')
            for i in range(len(timeline)):
                scrfile.write(str(timeline[i]))
                if i < (len(timeline) - 1):
                    scrfile.write(',')
            scrfile.write(')')
            scrfile.write('\n')
            scrfile.write('timeline_size <- timeline.size()')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# create some vector for the moves and monitors of this analysis')
            scrfile.write('\n')
            scrfile.write('moves = VectorMoves()')
            scrfile.write('\n')
            scrfile.write('monitors = VectorMonitors()')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# HSMRF')
            scrfile.write('\n')
            scrfile.write('# prior and hyperprior for overall amount of rate variation')
            scrfile.write('\n')
            scrfile.write('speciation_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('extinction_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('psi_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('speciation_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('extinction_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('psi_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# create a random variable at the present time')
            scrfile.write('\n')
            scrfile.write('log_speciation_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_speciation_at_present.setValue(0.0)')
            scrfile.write('\n')
            scrfile.write('log_extinction_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_extinction_at_present.setValue(-1.0)')
            scrfile.write('\n')
            scrfile.write('log_psi_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_psi_at_present.setValue(0.0)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_speciation_at_present, delta = 1.0, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_extinction_at_present, delta = 1.0, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_psi_at_present, delta = 1.0, weight = 5))')
            # scrfile.write('moves.append(mvSlideBactrian(log_speciation_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(log_extinction_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(log_psi_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_speciation_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_extinction_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_psi_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_speciation_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_extinction_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_psi_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('\n')
            # scrfile.write('# account for the correlation between the parameters by joint moves')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present = mvAVMVN(weight = 50)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_speciation_at_present)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_extinction_at_present)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_psi_at_present)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(avmvn_rates_at_present)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move = mvUpDownScale(weight = 5)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_speciation_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_extinction_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_psi_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(up_down_move)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            scrfile.write('for (i in 1:timeline_size) {')
            scrfile.write('\n')
            scrfile.write('    sigma_speciation[i] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('    sigma_extinction[i] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('    sigma_psi[i] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # Make sure values initialize to something reasonable')
            scrfile.write('\n')
            scrfile.write('    sigma_speciation[i].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    sigma_extinction[i].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    sigma_psi[i].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # moves on the single sigma values')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScale(sigma_speciation[i], lambda = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScale(sigma_extinction[i], lambda = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScale(sigma_psi[i], lambda = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_speciation[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_extinction[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_psi[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('\n')
            scrfile.write('    # non-centralized parameterization of horseshoe')
            scrfile.write('\n')
            scrfile.write('    delta_log_speciation[i] ~ dnNormal(mean = 0, sd = sigma_speciation[i] * speciation_global_scale * speciation_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('    delta_log_extinction[i] ~ dnNormal(mean = 0, sd = sigma_extinction[i] * extinction_global_scale * extinction_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('    delta_log_psi[i] ~ dnNormal(mean = 0, sd = sigma_psi[i] * psi_global_scale * psi_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # Make sure values initialize to something reasonable')
            scrfile.write('\n')
            scrfile.write('    delta_log_speciation[i].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    delta_log_extinction[i].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    delta_log_psi[i].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(delta_log_speciation[i], delta = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(delta_log_extinction[i], delta = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(delta_log_psi[i], delta = 1.0, weight = 5) )')
            scrfile.write('\n')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_speciation[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_extinction[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_psi[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i] = mvUpDownSlide(weight = 5)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_speciation[i], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_extinction[i], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_psi[i], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( delta_up_down_move[i] )')
            # scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Assemble first-order differences and speciation_rate at present into the random field')
            scrfile.write('\n')
            scrfile.write('speciation := fnassembleContinuousMRF(log_speciation_at_present, delta_log_speciation, initialValueIsLogScale = TRUE, order = 1) + 0.000001')
            scrfile.write('\n')
            scrfile.write('extinction := fnassembleContinuousMRF(log_extinction_at_present, delta_log_extinction, initialValueIsLogScale = TRUE, order = 1) + 0.000001')
            scrfile.write('\n')
            scrfile.write('psi := fnassembleContinuousMRF(log_psi_at_present, delta_log_psi, initialValueIsLogScale = TRUE, order = 1) + 0.000001')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Move all field parameters in one go')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_speciation, weight = 5, tune = FALSE, forceAccept = TRUE))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_extinction, weight = 5, tune = FALSE, forceAccept = TRUE))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_psi, weight = 5, tune = FALSE, forceAccept = TRUE))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Move all field hyperparameters in one go')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(speciation_global_scale, sigma_speciation, delta_log_speciation, speciation_global_scale_hyperprior, order = 1, weight = 10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(extinction_global_scale, sigma_extinction, delta_log_extinction, extinction_global_scale_hyperprior, order = 1, weight = 10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(psi_global_scale, sigma_psi, delta_log_psi, psi_global_scale_hyperprior, order = 1, weight = 10))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Swap moves to exchange adjacent delta,sigma pairs')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_speciation, sigma_speciation, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_extinction, sigma_extinction, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_psi, sigma_psi, weight = 5))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('rho <- %s' % str(rho))
            scrfile.write('\n')
            scrfile.write('# bounded = FALSE mitigates high extinction rate at the present under constant speciation and extinction')
            scrfile.write('\n')
            scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation, mu = extinction, psi = psi, rho = rho, timeline = timeline, k = k, bounded = %s)' % str(bounded))
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# times of speciation and extinction')
            scrfile.write('\n')
            scrfile.write('for (i in 1:taxa.size()) {')
            scrfile.write('\n')
            scrfile.write('    ts[i] := bd[i][1]')
            scrfile.write('\n')
            scrfile.write('    te[i] := bd[i][2]')
            scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.01, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.1, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 1.0, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.01, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.1, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 1.0, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# workspace model wrapper')
            scrfile.write('\n')
            scrfile.write('mymodel = model(bd)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# add monitors')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnScreen(speciation, extinction, psi, printgen = 5000))')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnModel(filename = "output/%s_model1_HSMRF.log", printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# monitors to print RevGagets input')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_speciation_rates.log", speciation, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_speciation_times.log", timeline, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_extinction_rates.log", extinction, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_sampling_rates.log", psi, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_ts.log", ts, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_te.log", te, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# run the analysis')
            scrfile.write('\n')
            scrfile.write('mymcmc = mcmc(mymodel, moves, monitors, moveschedule = "random")')
            scrfile.write('\n')
            scrfile.write('mymcmc.burnin(generations = 5000, tuningInterval = 500)')
            scrfile.write('\n')
            scrfile.write('mymcmc.run(50000)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('q()')
            scrfile.flush()
        elif self.FBD_rate_prior == 'HSMRF' and self.fix_fake_bin is False and self.padding[0] != np.inf and self.padding[1] != 0.0:
            # FBD for truncated data with padding using fake bins
            # speciation and extinction rates of fake bins are part of the horseshoe
            scr = "%s/%s/%s/%s/%s_FBDR_HSMRF.Rev" % (self.output_wd, self.name_file, 'FBD', 'scripts', self.name_file)
            scrfile = open(scr, "w")
            scrfile.write('######################################')
            scrfile.write('\n')
            scrfile.write('# FBD using stratigraphic range data #')
            scrfile.write('\n')
            scrfile.write('######################################')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# FBD for truncated data with padding using fake bins. Speciation and extinction rates of fake bins are part of the horseshoe.')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# read stratigraphic ranges')
            scrfile.write('\n')
            scrfile.write('taxa = readTaxonData(file = "data/%s_ranges.csv")' % self.name_file)
            scrfile.write('\n')
            scrfile.write('# read fossil counts')
            scrfile.write('\n')
            scrfile.write('k <- readDataDelimitedFile(file = "data/%s_counts.csv", header = true, rownames = true)' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            timeline = interval_ages[1:, 0]
            timeline = timeline[::-1]
            timeline = timeline.tolist()
            scrfile.write('# interval boundaries')
            scrfile.write('\n')
            scrfile.write('timeline <- v(')
            for i in range(len(timeline)):
                scrfile.write(str(timeline[i]))
                if i < (len(timeline) - 1):
                    scrfile.write(',')
            scrfile.write(')')
            scrfile.write('\n')
            scrfile.write('timeline_size <- timeline.size()')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# create some vector for the moves and monitors of this analysis')
            scrfile.write('\n')
            scrfile.write('moves = VectorMoves()')
            scrfile.write('\n')
            scrfile.write('monitors = VectorMonitors()')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# HSMRF')
            scrfile.write('\n')
            scrfile.write('# prior and hyperprior for overall amount of rate variation')
            scrfile.write('\n')
            scrfile.write('speciation_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('extinction_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('psi_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('speciation_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('extinction_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('psi_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# create a random variable at the present time')
            scrfile.write('\n')
            scrfile.write('log_speciation_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_speciation_at_present.setValue(0.0)')
            scrfile.write('\n')
            scrfile.write('log_extinction_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_extinction_at_present.setValue(-1.0)')
            scrfile.write('\n')
            scrfile.write('log_psi_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_psi_at_present.setValue(0.0)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlideBactrian(log_speciation_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlideBactrian(log_extinction_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlideBactrian(log_psi_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMirrorMultiplier(log_speciation_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMirrorMultiplier(log_extinction_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMirrorMultiplier(log_psi_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvRandomDive(log_speciation_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvRandomDive(log_extinction_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvRandomDive(log_psi_at_present, weight = 5))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# account for the correlation between the parameters by joint moves')
            scrfile.write('\n')
            scrfile.write('avmvn_rates_at_present = mvAVMVN(weight = 50)')
            scrfile.write('\n')
            scrfile.write('avmvn_rates_at_present.addVariable(log_speciation_at_present)')
            scrfile.write('\n')
            scrfile.write('avmvn_rates_at_present.addVariable(log_extinction_at_present)')
            scrfile.write('\n')
            scrfile.write('# avmvn_rates_at_present.addVariable(log_psi_at_present)')
            scrfile.write('\n')
            scrfile.write('moves.append(avmvn_rates_at_present)')
            scrfile.write('\n')
            scrfile.write('up_down_move = mvUpDownScale(weight = 5)')
            scrfile.write('\n')
            scrfile.write('up_down_move.addVariable(log_speciation_at_present, TRUE)')
            scrfile.write('\n')
            scrfile.write('up_down_move.addVariable(log_extinction_at_present, TRUE)')
            scrfile.write('\n')
            scrfile.write('# up_down_move.addVariable(log_psi_at_present, TRUE)')
            scrfile.write('\n')
            scrfile.write('moves.append(up_down_move)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('sigma_speciation[1] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('sigma_extinction[1] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Make sure values initialize to something reasonable')
            scrfile.write('\n')
            scrfile.write('sigma_speciation[1].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('sigma_extinction[1].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# moves on the single sigma values')
            scrfile.write('\n')
            scrfile.write('moves.append(mvScaleBactrian(sigma_speciation[1], weight = 5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvScaleBactrian(sigma_extinction[1], weight = 5))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# non-centralized parameterization of horseshoe')
            scrfile.write('\n')
            scrfile.write('delta_log_speciation[1] ~ dnNormal(mean=0, sd=sigma_speciation[1] * speciation_global_scale * speciation_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('delta_log_extinction[1] ~ dnNormal(mean=0, sd=sigma_extinction[1] * extinction_global_scale * extinction_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Make sure values initialize to something reasonable')
            scrfile.write('\n')
            scrfile.write('delta_log_speciation[1].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('delta_log_extinction[1].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlideBactrian(delta_log_speciation[1], weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlideBactrian(delta_log_extinction[1], weight=5))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('delta_up_down_move[1] = mvUpDownSlide(weight=5)')
            scrfile.write('\n')
            scrfile.write('delta_up_down_move[1].addVariable(delta_log_speciation[1], TRUE)')
            scrfile.write('\n')
            scrfile.write('delta_up_down_move[1].addVariable(delta_log_extinction[1], TRUE)')
            scrfile.write('\n')
            scrfile.write('moves.append(delta_up_down_move[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('for (i in 2:(timeline_size-1)) {')
            scrfile.write('\n')
            scrfile.write('    sigma_speciation[i] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('    sigma_extinction[i] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('    sigma_psi[i-1] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # Make sure values initialize to something reasonable')
            scrfile.write('\n')
            scrfile.write('    sigma_speciation[i].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    sigma_extinction[i].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    sigma_psi[i-1].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # moves on the single sigma values')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScaleBactrian(sigma_speciation[i], weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScaleBactrian(sigma_extinction[i], weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScaleBactrian(sigma_psi[i-1], weight = 5) )')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # non-centralized parameterization of horseshoe')
            scrfile.write('\n')
            scrfile.write('    delta_log_speciation[i] ~ dnNormal(mean = 0, sd = sigma_speciation[i] * speciation_global_scale * speciation_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('    delta_log_extinction[i] ~ dnNormal(mean = 0, sd = sigma_extinction[i] * extinction_global_scale * extinction_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('    delta_log_psi[i-1] ~ dnNormal(mean = 0, sd = sigma_psi[i-1] * psi_global_scale * psi_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # Make sure values initialize to something reasonable')
            scrfile.write('\n')
            scrfile.write('    delta_log_speciation[i].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    delta_log_extinction[i].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    delta_log_psi[i-1].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlideBactrian(delta_log_speciation[i], weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlideBactrian(delta_log_extinction[i], weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlideBactrian(delta_log_psi[i-1], weight = 5) )')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    delta_up_down_move[i] = mvUpDownSlide(weight = 5)')
            scrfile.write('\n')
            scrfile.write('    delta_up_down_move[i].addVariable(delta_log_speciation[i], TRUE)')
            scrfile.write('\n')
            scrfile.write('    delta_up_down_move[i].addVariable(delta_log_extinction[i], TRUE)')
            scrfile.write('\n')
            scrfile.write('    delta_up_down_move[i].addVariable(delta_log_psi[i-1], TRUE)')
            scrfile.write('\n')
            scrfile.write('    moves.append( delta_up_down_move[i] )')
            scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('sigma_speciation[timeline_size] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('sigma_extinction[timeline_size] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('sigma_speciation[timeline_size].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('sigma_extinction[timeline_size].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvScaleBactrian(sigma_speciation[timeline_size], weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvScaleBactrian(sigma_extinction[timeline_size], weight=5))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('delta_log_speciation[timeline_size] ~ dnNormal(mean=0, sd=sigma_speciation[timeline_size] * speciation_global_scale * speciation_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('delta_log_extinction[timeline_size] ~ dnNormal(mean=0, sd=sigma_extinction[timeline_size] * extinction_global_scale * extinction_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('delta_log_speciation[timeline_size].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('delta_log_extinction[timeline_size].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlideBactrian(delta_log_speciation[timeline_size], weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlideBactrian(delta_log_extinction[timeline_size], weight=5))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('delta_up_down_move[timeline_size] = mvUpDownSlide(weight=5)')
            scrfile.write('\n')
            scrfile.write('delta_up_down_move[timeline_size].addVariable(delta_log_speciation[timeline_size], TRUE)')
            scrfile.write('\n')
            scrfile.write('delta_up_down_move[timeline_size].addVariable(delta_log_extinction[timeline_size], TRUE)')
            scrfile.write('\n')
            scrfile.write('moves.append(delta_up_down_move[timeline_size])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Assemble first-order differences and rates at present into the random field')
            scrfile.write('\n')
            scrfile.write('#############################################################################')
            scrfile.write('\n')
            scrfile.write('speciation := fnassembleContinuousMRF(log_speciation_at_present, delta_log_speciation, initialValueIsLogScale=TRUE, order=1) + 0.000001')
            scrfile.write('\n')
            scrfile.write('extinction := fnassembleContinuousMRF(log_extinction_at_present, delta_log_extinction, initialValueIsLogScale=TRUE, order=1) + 0.000001')
            scrfile.write('\n')
            scrfile.write('psi := fnassembleContinuousMRF(log_psi_at_present, delta_log_psi, initialValueIsLogScale=TRUE, order=1) + 0.000001')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Move all field parameters in one go')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_speciation, weight=5, tune=FALSE, forceAccept=FALSE))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_extinction, weight=5, tune=FALSE, forceAccept=FALSE))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_psi, weight=5, tune=FALSE, forceAccept=FALSE))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Move all field hyperparameters in one go')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(speciation_global_scale, sigma_speciation, delta_log_speciation, speciation_global_scale_hyperprior, propGlobalOnly=0.75, weight=10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(extinction_global_scale, sigma_extinction, delta_log_extinction, extinction_global_scale_hyperprior, propGlobalOnly=0.75, weight=10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(psi_global_scale, sigma_psi, delta_log_psi, psi_global_scale_hyperprior, propGlobalOnly=0.75, weight=10))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Swap moves to exchange adjacent delta,sigma pairs')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_speciation, sigma_speciation, weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_extinction, sigma_extinction, weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_psi, sigma_psi, weight=5))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Add poor fossilization for the fake bins')
            scrfile.write('\n')
            scrfile.write('psi2[1] = 0.000001')
            scrfile.write('\n')
            scrfile.write('for (i in 1:psi.size()) {')
            scrfile.write('\n')
            scrfile.write('    psi2[i+1] := psi[i]')
            scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('psi2[psi2.size() + 1] = 0.000001')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('rho <- %s' % str(rho))
            scrfile.write('\n')
            scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation, mu = extinction, psi = psi2, rho = rho, timeline = timeline, k = k, bounded = TRUE)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# times of speciation and extinction')
            scrfile.write('\n')
            scrfile.write('for (i in 1:taxa.size()) {')
            scrfile.write('\n')
            scrfile.write('    ts[i] := bd[i][1]')
            scrfile.write('\n')
            scrfile.write('    te[i] := bd[i][2]')
            scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.01, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.1, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 1.0, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.01, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.1, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 1.0, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# workspace model wrapper')
            scrfile.write('\n')
            scrfile.write('mymodel = model(bd)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# add monitors')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnScreen(speciation, extinction, psi, printgen = 5000))')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnModel(filename = "output/%s_model1_HSMRF.log", printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# monitors to print RevGagets input')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_speciation_rates.log", speciation, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_speciation_times.log", timeline, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_extinction_rates.log", extinction, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_sampling_rates.log", psi2, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_ts.log", ts, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_te.log", te, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# run the analysis')
            scrfile.write('\n')
            scrfile.write('mymcmc = mcmc(mymodel, moves, monitors, moveschedule = "random")')
            scrfile.write('\n')
            scrfile.write('mymcmc.burnin(generations = 5000, tuningInterval = 500)')
            scrfile.write('\n')
            scrfile.write('mymcmc.run(50000)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('q()')
            scrfile.flush()
        elif self.FBD_rate_prior == 'HSMRF' and self.fix_fake_bin is True and self.padding[0] != np.inf and self.padding[1] != 0.0:
            # FBD for truncated data with padding using fake bins.
            # Speciation and extinction rates of fake bins are inferred separately of the horseshoe.
            scr = "%s/%s/%s/%s/%s_FBDR_HSMRF.Rev" % (self.output_wd, self.name_file, 'FBD', 'scripts', self.name_file)
            scrfile = open(scr, "w")
            scrfile.write('######################################')
            scrfile.write('\n')
            scrfile.write('# FBD using stratigraphic range data #')
            scrfile.write('\n')
            scrfile.write('######################################')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# FBD for truncated data with padding using fake bins. Speciation and extinction rates of fake bins are inferred separately of the horseshoe.')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# read stratigraphic ranges')
            scrfile.write('\n')
            scrfile.write('taxa = readTaxonData(file = "data/%s_ranges.csv")' % self.name_file)
            scrfile.write('\n')
            scrfile.write('# read fossil counts')
            scrfile.write('\n')
            scrfile.write('k <- readDataDelimitedFile(file = "data/%s_counts.csv", header = true, rownames = true)' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            timeline = interval_ages[1:, 0]
            timeline = timeline[::-1]
            timeline = timeline.tolist()
            scrfile.write('# interval boundaries')
            scrfile.write('\n')
            scrfile.write('timeline <- v(')
            for i in range(len(timeline)):
                scrfile.write(str(timeline[i]))
                if i < (len(timeline) - 1):
                    scrfile.write(',')
            scrfile.write(')')
            scrfile.write('\n')
            scrfile.write('timeline_size <- timeline.size()')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# create some vector for the moves and monitors of this analysis')
            scrfile.write('\n')
            scrfile.write('moves = VectorMoves()')
            scrfile.write('\n')
            scrfile.write('monitors = VectorMonitors()')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# HSMRF')
            scrfile.write('\n')
            scrfile.write('# prior and hyperprior for overall amount of rate variation')
            scrfile.write('\n')
            scrfile.write('speciation_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('extinction_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('psi_global_scale_hyperprior <- 0.021')
            scrfile.write('\n')
            scrfile.write('speciation_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('extinction_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('psi_global_scale ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# create a random variable at the present time')
            scrfile.write('\n')
            scrfile.write('log_speciation_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_speciation_at_present.setValue(0.0)')
            scrfile.write('\n')
            scrfile.write('log_extinction_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_extinction_at_present.setValue(-1.0)')
            scrfile.write('\n')
            scrfile.write('log_psi_at_present ~ dnUniform(-5.0, 5.0)')
            scrfile.write('\n')
            scrfile.write('log_psi_at_present.setValue(0.0)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_speciation_at_present, delta = 1.0, weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_extinction_at_present, delta = 1.0, weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_psi_at_present, delta = 1.0, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(log_speciation_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(log_extinction_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(log_psi_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_speciation_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_extinction_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_psi_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_speciation_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_extinction_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_psi_at_present, weight=5))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# account for the correlation between the parameters by joint moves')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present = mvAVMVN(weight=50)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_speciation_at_present)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_extinction_at_present)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_psi_at_present)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(avmvn_rates_at_present)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move = mvUpDownScale(weight=5)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_speciation_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_extinction_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_psi_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(up_down_move)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('for (i in 1:(timeline_size - 2)) {')
            scrfile.write('\n')
            scrfile.write('    sigma_speciation[i] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('    sigma_extinction[i] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('    sigma_psi[i] ~ dnHalfCauchy(0, 1)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # Make sure values initialize to something reasonable')
            scrfile.write('\n')
            scrfile.write('    sigma_speciation[i].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    sigma_extinction[i].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    sigma_psi[i].setValue(runif(1, 0.005, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # moves on the single sigma values')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScale(sigma_speciation[i], lambda = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScale(sigma_extinction[i], lambda = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvScale(sigma_psi[i], lambda = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_speciation[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_extinction[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_psi[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('\n')
            scrfile.write('    # non-centralized parameterization of horseshoe')
            scrfile.write('\n')
            scrfile.write('    delta_log_speciation[i] ~ dnNormal(mean = 0, sd = sigma_speciation[i] * speciation_global_scale * speciation_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('    delta_log_extinction[i] ~ dnNormal(mean = 0, sd = sigma_extinction[i] * extinction_global_scale * extinction_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('    delta_log_psi[i] ~ dnNormal(mean = 0, sd = sigma_psi[i] * psi_global_scale * psi_global_scale_hyperprior)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # Make sure values initialize to something reasonable')
            scrfile.write('\n')
            scrfile.write('    delta_log_speciation[i].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    delta_log_extinction[i].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('    delta_log_psi[i].setValue(runif(1, -0.1, 0.1)[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(delta_log_speciation[i], delta = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(delta_log_extinction[i], delta = 1.0, weight = 5) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(delta_log_psi[i], delta = 1.0, weight = 5) )')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_speciation[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_extinction[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_psi[i], weight = 5) )')
            scrfile.write('\n')
            scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i] = mvUpDownSlide(weight = 5)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_speciation[i], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_extinction[i], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_psi[i], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( delta_up_down_move[i] )')
            # scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Assemble first-order differences and speciation_rate at present into the random field')
            scrfile.write('\n')
            scrfile.write('speciation := fnassembleContinuousMRF(log_speciation_at_present, delta_log_speciation, initialValueIsLogScale = TRUE, order = 1) + 0.00001')
            scrfile.write('\n')
            scrfile.write('extinction := fnassembleContinuousMRF(log_extinction_at_present, delta_log_extinction, initialValueIsLogScale = TRUE, order = 1) + 0.00001')
            scrfile.write('\n')
            scrfile.write('psi := fnassembleContinuousMRF(log_psi_at_present, delta_log_psi, initialValueIsLogScale = TRUE, order = 1) + 0.00001')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('extinction_firstbin ~ dnUniform(0.00001, 5.0)')
            scrfile.write('\n')
            scrfile.write('speciation_lastbin ~ dnUniform(0.00001, 5.0)')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(extinction_firstbin, delta = 1.0, tune = TRUE, weight = 10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(speciation_lastbin, delta = 1.0, tune = TRUE, weight = 10))')
            scrfile.write('\n')
            scrfile.write('speciation2[1] = 0.00001')
            scrfile.write('\n')
            scrfile.write('extinction2[1] := extinction_firstbin')
            scrfile.write('\n')
            scrfile.write('for (i in 1:speciation.size()) {')
            scrfile.write('\n')
            scrfile.write('    speciation2[i+1] := speciation[i]')
            scrfile.write('\n')
            scrfile.write('    extinction2[i+1] := extinction[i]')
            scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('speciation2[speciation2.size() + 1] := speciation_lastbin')
            scrfile.write('\n')
            scrfile.write('extinction2[extinction2.size() + 1] = 0.00001')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Move all field parameters in one go')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_speciation, weight = 5, tune = FALSE, forceAccept = TRUE))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_extinction, weight = 5, tune = FALSE, forceAccept = TRUE))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_psi, weight = 5, tune = FALSE, forceAccept = TRUE))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Move all field hyperparameters in one go')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(speciation_global_scale, sigma_speciation, delta_log_speciation, speciation_global_scale_hyperprior, propGlobalOnly=0.75, weight=10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(extinction_global_scale, sigma_extinction, delta_log_extinction, extinction_global_scale_hyperprior, propGlobalOnly=0.75, weight=10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(psi_global_scale, sigma_psi, delta_log_psi, psi_global_scale_hyperprior, propGlobalOnly=0.75, weight=10))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Swap moves to exchange adjacent delta,sigma pairs')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_speciation, sigma_speciation, weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_extinction, sigma_extinction, weight=5))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_psi, sigma_psi, weight=5))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Add poor fossilization for the fake bins')
            scrfile.write('\n')
            scrfile.write('psi2[1] = 0.000001')
            scrfile.write('\n')
            scrfile.write('for (i in 1:psi.size()) {')
            scrfile.write('\n')
            scrfile.write('    psi2[i+1] := psi[i]')
            scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('psi2[psi2.size() + 1] = 0.000001')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('rho <- %s' % str(rho))
            scrfile.write('\n')
            scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation2, mu = extinction2, psi = psi2, rho = rho, timeline = timeline, k = k, bounded = TRUE)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# times of speciation and extinction')
            scrfile.write('\n')
            scrfile.write('for (i in 1:taxa.size()) {')
            scrfile.write('\n')
            scrfile.write('    ts[i] := bd[i][1]')
            scrfile.write('\n')
            scrfile.write('    te[i] := bd[i][2]')
            scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.01, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.1, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 1.0, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.01, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.1, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 1.0, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# workspace model wrapper')
            scrfile.write('\n')
            scrfile.write('mymodel = model(bd)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# add monitors')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnScreen(speciation, extinction, psi, printgen = 5000))')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnModel(filename = "output/%s_model1_HSMRF.log", printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# monitors to print RevGagets input')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_speciation_rates.log", speciation2, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_speciation_times.log", timeline, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_extinction_rates.log", extinction2, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_sampling_rates.log", psi2, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_ts.log", ts, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_te.log", te, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# run the analysis')
            scrfile.write('\n')
            scrfile.write('mymcmc = mcmc(mymodel, moves, monitors, moveschedule = "random")')
            scrfile.write('\n')
            scrfile.write('mymcmc.burnin(generations = 5000, tuningInterval = 500)')
            scrfile.write('\n')
            scrfile.write('mymcmc.run(50000)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('q()')
            scrfile.flush()
            # Center HSMRF at mid-time
            # scr = "%s/%s/%s/%s/%s_FBDR_HSMRF.Rev" % (self.output_wd, self.name_file, 'FBD', 'scripts', self.name_file)
            # scrfile = open(scr, "w")
            # scrfile.write('######################################')
            # scrfile.write('\n')
            # scrfile.write('# FBD using stratigraphic range data #')
            # scrfile.write('\n')
            # scrfile.write('######################################')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# read stratigraphic ranges')
            # scrfile.write('\n')
            # scrfile.write('taxa = readTaxonData(file = "data/%s_ranges.csv")' % self.name_file)
            # scrfile.write('\n')
            # scrfile.write('# read fossil counts')
            # scrfile.write('\n')
            # scrfile.write('k <- readDataDelimitedFile(file = "data/%s_counts.csv", header = true, rownames = true)' % self.name_file)
            # scrfile.write('\n')
            # scrfile.write('\n')
            # timeline = interval_ages[1:, 0]
            # timeline = timeline[::-1]
            # timeline = timeline.tolist()
            # scrfile.write('# interval boundaries')
            # scrfile.write('\n')
            # scrfile.write('timeline <- v(')
            # for i in range(len(timeline)):
            #     scrfile.write(str(timeline[i]))
            #     if i < (len(timeline) - 1):
            #         scrfile.write(',')
            # scrfile.write(')')
            # scrfile.write('\n')
            # scrfile.write('timeline_size <- timeline.size()')
            # scrfile.write('\n')
            # scrfile.write('timeline_size_before = ceil(timeline_size / 2)')
            # scrfile.write('\n')
            # scrfile.write('timeline_size_after = timeline_size - timeline_size_before')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# create some vector for the moves and monitors of this analysis')
            # scrfile.write('\n')
            # scrfile.write('moves = VectorMoves()')
            # scrfile.write('\n')
            # scrfile.write('monitors = VectorMonitors()')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# HSMRF')
            # scrfile.write('\n')
            # scrfile.write('# prior and hyperprior for overall amount of rate variation')
            # scrfile.write('\n')
            # scrfile.write('speciation_global_scale_hyperprior <- 0.021')
            # scrfile.write('\n')
            # scrfile.write('extinction_global_scale_hyperprior <- 0.021')
            # scrfile.write('\n')
            # scrfile.write('psi_global_scale_hyperprior <- 0.021')
            # scrfile.write('\n')
            # scrfile.write('speciation_global_scale ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('extinction_global_scale ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('psi_global_scale ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('# create a random variable at the present time')
            # scrfile.write('\n')
            # scrfile.write('log_speciation_at_present ~ dnUniform(-5.0, 5.0)')
            # scrfile.write('\n')
            # scrfile.write('log_speciation_at_present.setValue(0.0)')
            # scrfile.write('\n')
            # scrfile.write('log_extinction_at_present ~ dnUniform(-5.0, 5.0)')
            # scrfile.write('\n')
            # scrfile.write('log_extinction_at_present.setValue(-1.0)')
            # scrfile.write('\n')
            # scrfile.write('log_psi_at_present ~ dnUniform(-5.0, 5.0)')
            # scrfile.write('\n')
            # scrfile.write('log_psi_at_present.setValue(0.0)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(log_speciation_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(log_extinction_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(log_psi_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_speciation_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_extinction_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMirrorMultiplier(log_psi_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_speciation_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_extinction_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvRandomDive(log_psi_at_present, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# account for the correlation between the parameters by joint moves')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present = mvAVMVN(weight = 50)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_speciation_at_present)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_extinction_at_present)')
            # scrfile.write('\n')
            # scrfile.write('avmvn_rates_at_present.addVariable(log_psi_at_present)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(avmvn_rates_at_present)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move = mvUpDownScale(weight = 5)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_speciation_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_extinction_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('up_down_move.addVariable(log_psi_at_present, TRUE)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(up_down_move)')
            # scrfile.write('\n')
            # scrfile.write('sigma_speciation[1] ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('sigma_extinction[1] ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('# Make sure values initialize to something reasonable')
            # scrfile.write('\n')
            # scrfile.write('sigma_speciation[1].setValue(runif(1, 0.005, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('sigma_extinction[1].setValue(runif(1, 0.005, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# moves on the single sigma values')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvScaleBactrian(sigma_speciation[1], weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvScaleBactrian(sigma_extinction[1], weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# non-centralized parameterization of horseshoe')
            # scrfile.write('\n')
            # scrfile.write('delta_log_speciation[1] ~ dnNormal(mean=0, sd=sigma_speciation[1] * speciation_global_scale * speciation_global_scale_hyperprior)')
            # scrfile.write('\n')
            # scrfile.write('delta_log_extinction[1] ~ dnNormal(mean=0, sd=sigma_extinction[1] * extinction_global_scale * extinction_global_scale_hyperprior)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# Make sure values initialize to something reasonable')
            # scrfile.write('\n')
            # scrfile.write('delta_log_speciation[1].setValue(runif(1, -0.1, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('delta_log_extinction[1].setValue(runif(1, -0.1, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(delta_log_speciation[1], weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(delta_log_extinction[1], weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('delta_up_down_move[1] = mvUpDownSlide(weight = 5)')
            # scrfile.write('\n')
            # scrfile.write('delta_up_down_move[1].addVariable(delta_log_speciation[1], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('delta_up_down_move[1].addVariable(delta_log_extinction[1], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(delta_up_down_move[1])')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('for (i in 2:(timeline_size-1)) {')
            # scrfile.write('\n')
            # scrfile.write('    sigma_speciation[i] ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('    sigma_extinction[i] ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('    sigma_psi[i-1] ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('    # Make sure values initialize to something reasonable')
            # scrfile.write('\n')
            # scrfile.write('    sigma_speciation[i].setValue(runif(1, 0.005, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('    sigma_extinction[i].setValue(runif(1, 0.005, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('    sigma_psi[i-1].setValue(runif(1, 0.005, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('    # moves on the single sigma values')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_speciation[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_extinction[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvScaleBactrian(sigma_psi[i-1], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('    # non-centralized parameterization of horseshoe')
            # scrfile.write('\n')
            # scrfile.write('    delta_log_speciation[i] ~ dnNormal(mean = 0, sd = sigma_speciation[i] * speciation_global_scale * speciation_global_scale_hyperprior)')
            # scrfile.write('\n')
            # scrfile.write('    delta_log_extinction[i] ~ dnNormal(mean = 0, sd = sigma_extinction[i] * extinction_global_scale * extinction_global_scale_hyperprior)')
            # scrfile.write('\n')
            # scrfile.write('    delta_log_psi[i-1] ~ dnNormal(mean = 0, sd = sigma_psi[i-1] * psi_global_scale * psi_global_scale_hyperprior)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('    # Make sure values initialize to something reasonable')
            # scrfile.write('\n')
            # scrfile.write('    delta_log_speciation[i].setValue(runif(1, -0.1, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('    delta_log_extinction[i].setValue(runif(1, -0.1, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('    delta_log_psi[i-1].setValue(runif(1, -0.1, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_speciation[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_extinction[i], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( mvSlideBactrian(delta_log_psi[i-1], weight = 5) )')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i] = mvUpDownSlide(weight = 5)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_speciation[i], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_extinction[i], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    delta_up_down_move[i].addVariable(delta_log_psi[i-1], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('    moves.append( delta_up_down_move[i] )')
            # scrfile.write('\n')
            # scrfile.write('}')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# [timeline_size] works WITHOUT timeline_size_before but not with!')
            # scrfile.write('\n')
            # scrfile.write('sigma_speciation[sigma_speciation.size() + 1] ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('sigma_extinction[sigma_extinction.size() + 1] ~ dnHalfCauchy(0, 1)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('sigma_speciation[sigma_speciation.size()].setValue(runif(1, 0.005, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('sigma_extinction[sigma_extinction.size()].setValue(runif(1, 0.005, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvScaleBactrian(sigma_speciation[sigma_speciation.size()], weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvScaleBactrian(sigma_extinction[sigma_extinction.size()], weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('delta_log_speciation[delta_log_speciation.size() + 1] ~ dnNormal(mean = 0, sd = sigma_speciation[sigma_speciation.size()] * speciation_global_scale * speciation_global_scale_hyperprior)')
            # scrfile.write('\n')
            # scrfile.write('delta_log_extinction[delta_log_extinction.size() + 1] ~ dnNormal(mean = 0, sd = sigma_extinction[sigma_extinction.size()] * extinction_global_scale * extinction_global_scale_hyperprior)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('delta_log_speciation[delta_log_speciation.size()].setValue(runif(1, -0.1, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('delta_log_extinction[delta_log_extinction.size()].setValue(runif(1, -0.1, 0.1)[1])')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(delta_log_speciation[delta_log_speciation.size()], weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvSlideBactrian(delta_log_extinction[delta_log_extinction.size()], weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('delta_up_down_move[delta_log_speciation.size()] = mvUpDownSlide(weight = 5)')
            # scrfile.write('\n')
            # scrfile.write('delta_up_down_move[delta_log_speciation.size()].addVariable(delta_log_speciation[delta_log_speciation.size()], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('delta_up_down_move[delta_log_extinction.size()].addVariable(delta_log_extinction[delta_log_extinction.size()], TRUE)')
            # scrfile.write('\n')
            # scrfile.write('moves.append(delta_up_down_move[delta_log_speciation.size()])')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# Assemble first-order differences and rates at present into the random field')
            # scrfile.write('\n')
            # scrfile.write('#############################################################################')
            # scrfile.write('\n')
            # scrfile.write('for (i in 1:timeline_size_before) {')
            # scrfile.write('\n')
            # scrfile.write('    cumdiff_sp = 0')
            # scrfile.write('\n')
            # scrfile.write('    for (y in 1:i) {')
            # scrfile.write('\n')
            # scrfile.write('        cumdiff_sp := cumdiff_sp - delta_log_speciation[timeline_size_before + 1 - y]')
            # scrfile.write('\n')
            # scrfile.write('    }')
            # scrfile.write('\n')
            # scrfile.write('    speciation[i] := exp(log_speciation_at_present - cumdiff_sp) + 0.000001')
            # scrfile.write('\n')
            # scrfile.write('}')
            # scrfile.write('\n')
            # scrfile.write('speciation[speciation.size() + 1] := exp(log_speciation_at_present)')
            # scrfile.write('\n')
            # scrfile.write('for (i in 1:timeline_size_after) {')
            # scrfile.write('\n')
            # scrfile.write('    cumsum_sp = 0')
            # scrfile.write('\n')
            # scrfile.write('    for (y in 1:i) {')
            # scrfile.write('\n')
            # scrfile.write('        cumsum_sp := cumsum_sp + delta_log_speciation[timeline_size_before + y]')
            # scrfile.write('\n')
            # scrfile.write('    }')
            # scrfile.write('\n')
            # scrfile.write('    speciation[timeline_size_before + 1 + i] := exp(log_speciation_at_present + cumsum_sp) + 0.000001')
            # scrfile.write('\n')
            # scrfile.write('}')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('for (i in 1:timeline_size_before) {')
            # scrfile.write('\n')
            # scrfile.write('    cumdiff_ex = 0')
            # scrfile.write('\n')
            # scrfile.write('    for (y in 1:i) {')
            # scrfile.write('\n')
            # scrfile.write('        cumdiff_ex := cumdiff_ex - delta_log_extinction[timeline_size_before + 1 - y]')
            # scrfile.write('\n')
            # scrfile.write('    }')
            # scrfile.write('\n')
            # scrfile.write('    extinction[i] := exp(log_extinction_at_present - cumdiff_ex) + 0.000001')
            # scrfile.write('\n')
            # scrfile.write('}')
            # scrfile.write('\n')
            # scrfile.write('extinction[extinction.size() + 1] := exp(log_extinction_at_present)')
            # scrfile.write('\n')
            # scrfile.write('for (i in 1:timeline_size_after) {')
            # scrfile.write('\n')
            # scrfile.write('    cumsum_ex = 0')
            # scrfile.write('\n')
            # scrfile.write('    for (y in 1:i) {')
            # scrfile.write('\n')
            # scrfile.write('        cumsum_ex := cumsum_ex + delta_log_extinction[timeline_size_before + y]')
            # scrfile.write('\n')
            # scrfile.write('    }')
            # scrfile.write('\n')
            # scrfile.write('    extinction[timeline_size_before + 1 + i] := exp(log_extinction_at_present + cumsum_ex) + 0.000001')
            # scrfile.write('\n')
            # scrfile.write('}')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# Add poor fossilization for the fake bins')
            # scrfile.write('\n')
            # scrfile.write('psi[1] = 0.000001')
            # scrfile.write('\n')
            # scrfile.write('for (i in 1:(timeline_size_before - 1)) {')
            # scrfile.write('\n')
            # scrfile.write('    cumdiff_psi = 0')
            # scrfile.write('\n')
            # scrfile.write('    for (y in 1:i) {')
            # scrfile.write('\n')
            # scrfile.write('        cumdiff_psi := cumdiff_psi - delta_log_psi[timeline_size_before + 1 - y]')
            # scrfile.write('\n')
            # scrfile.write('    }')
            # scrfile.write('\n')
            # scrfile.write('    psi[i + 1] := exp(log_psi_at_present - cumdiff_psi) + 0.000001')
            # scrfile.write('\n')
            # scrfile.write('}')
            # scrfile.write('\n')
            # scrfile.write('psi[psi.size() + 1] := exp(log_psi_at_present)')
            # scrfile.write('\n')
            # scrfile.write('for (i in 1:(timeline_size_after - 1)) {')
            # scrfile.write('\n')
            # scrfile.write('    cumsum_psi = 0')
            # scrfile.write('\n')
            # scrfile.write('    for (y in 1:i) {')
            # scrfile.write('\n')
            # scrfile.write('        cumsum_psi := cumsum_psi + delta_log_psi[timeline_size_before - 1 + y]')
            # scrfile.write('\n')
            # scrfile.write('    }')
            # scrfile.write('\n')
            # scrfile.write('    psi[timeline_size_before + 1 + i] := exp(log_psi_at_present + cumsum_psi) + 0.000001')
            # scrfile.write('\n')
            # scrfile.write('}')
            # scrfile.write('\n')
            # scrfile.write('psi[psi.size() + 1] = 0.000001')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# Move all field parameters in one go')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_speciation, weight=5, tune=FALSE, forceAccept = TRUE))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_extinction, weight=5, tune=FALSE, forceAccept = TRUE))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvEllipticalSliceSamplingSimple(delta_log_psi, weight=5, tune=FALSE, forceAccept = TRUE))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# Move all field hyperparameters in one go')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(speciation_global_scale, sigma_speciation, delta_log_speciation, speciation_global_scale_hyperprior, propGlobalOnly = 0.75, weight = 10))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(extinction_global_scale, sigma_extinction, delta_log_extinction, extinction_global_scale_hyperprior, propGlobalOnly = 0.75, weight = 10))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvHSRFHyperpriorsGibbs(psi_global_scale, sigma_psi, delta_log_psi, psi_global_scale_hyperprior, propGlobalOnly = 0.75, weight = 10))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# Swap moves to exchange adjacent delta,sigma pairs')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_speciation, sigma_speciation, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_extinction, sigma_extinction, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvHSRFIntervalSwap(delta_log_psi, sigma_psi, weight = 5))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('rho <- %s' % str(rho))
            # scrfile.write('\n')
            # scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation, mu = extinction, psi = psi, rho = rho, timeline = timeline, k = k, bounded = FALSE)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.01, weight = taxa.size()))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.1, weight = taxa.size()))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 1.0, weight = taxa.size()))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.01, weight = taxa.size()))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.1, weight = taxa.size()))')
            # scrfile.write('\n')
            # scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 1.0, weight = taxa.size()))')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# workspace model wrapper')
            # scrfile.write('\n')
            # scrfile.write('mymodel = model(bd)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# add monitors')
            # scrfile.write('\n')
            # scrfile.write('monitors.append(mnScreen(speciation, extinction, psi, printgen = 5000))')
            # scrfile.write('\n')
            # scrfile.write('monitors.append(mnModel(filename = "output/%s_model1_HSMRF.log", printgen = 50))' % self.name_file)
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# monitors to print RevGagets input')
            # scrfile.write('\n')
            # scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_speciation_rates.log", speciation, printgen = 50))' % self.name_file)
            # scrfile.write('\n')
            # scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_speciation_times.log", timeline, printgen = 50))' % self.name_file)
            # scrfile.write('\n')
            # scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_extinction_rates.log", extinction, printgen = 50))' % self.name_file)
            # scrfile.write('\n')
            # scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_sampling_rates.log", psi, printgen = 50))' % self.name_file)
            # scrfile.write('\n')
            # scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_ts_te.log", bd, printgen = 50))' % self.name_file)
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('# run the analysis')
            # scrfile.write('\n')
            # scrfile.write('mymcmc = mcmc(mymodel, moves, monitors, moveschedule = "random")')
            # scrfile.write('\n')
            # scrfile.write('mymcmc.burnin(generations = 5000, tuningInterval = 500)')
            # scrfile.write('\n')
            # scrfile.write('mymcmc.run(50000)')
            # scrfile.write('\n')
            # scrfile.write('\n')
            # scrfile.write('q()')
            # scrfile.flush()
        else:
            scr = "%s/%s/%s/%s/%s_FBDR.Rev" % (self.output_wd, self.name_file, 'FBD', 'scripts', self.name_file)
            print('Fourth FBD')
            scrfile = open(scr, "w")
            scrfile.write('######################################')
            scrfile.write('\n')
            scrfile.write('# FBD using stratigraphic range data #')
            scrfile.write('\n')
            scrfile.write('######################################')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# read stratigraphic ranges')
            scrfile.write('\n')
            scrfile.write('taxa = readTaxonData(file = "data/%s_ranges.csv")' % self.name_file)
            scrfile.write('\n')
            scrfile.write('# read fossil counts')
            scrfile.write('\n')
            scrfile.write('k <- readDataDelimitedFile(file = "data/%s_counts.csv", header = true, rownames = true)' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            timeline = interval_ages[1:,0]
            timeline = timeline[::-1]
            timeline = timeline.tolist()
            scrfile.write('# interval boundaries')
            scrfile.write('\n')
            scrfile.write('timeline <- v(')
            for i in range(len(timeline)):
                scrfile.write(str(timeline[i]))
                if i < (len(timeline) - 1):
                    scrfile.write(',')
            scrfile.write(')')
            scrfile.write('\n')
            scrfile.write('timeline_size <- timeline.size()')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# create some vector for the moves and monitors of this analysis')
            scrfile.write('\n')
            scrfile.write('moves = VectorMoves()')
            scrfile.write('\n')
            scrfile.write('monitors = VectorMonitors()')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# first we create the standard deviation of the rates between intervals')
            scrfile.write('\n')
            scrfile.write('speciation_sd ~ dnExponential(1.0)')
            scrfile.write('\n')
            scrfile.write('moves.append(mvScale(speciation_sd, weight = 10))')
            scrfile.write('\n')
            scrfile.write('extinction_sd ~ dnExponential(1.0)')
            scrfile.write('\n')
            scrfile.write('moves.append(mvScale(extinction_sd, weight = 10))')
            scrfile.write('\n')
            scrfile.write('psi_sd ~ dnExponential(1.0)')
            scrfile.write('\n')
            scrfile.write('moves.append(mvScale(psi_sd, weight = 10))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# create a random variable at the present time')
            scrfile.write('\n')
            scrfile.write('log_speciation[1] ~ dnUniform(-10.0, 10.0)')
            scrfile.write('\n')
            scrfile.write('log_extinction[1] ~ dnUniform(-10.0, 10.0)')
            scrfile.write('\n')
            scrfile.write('log_psi[1] ~ dnUniform(-10.0, 10.0)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# apply moves on the rates')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_speciation[1], weight = 2))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_extinction[1], weight = 2))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(log_psi[1], weight = 2))')
            scrfile.write('\n')
            scrfile.write('speciation[1] := exp(log_speciation[1])')
            scrfile.write('\n')
            scrfile.write('extinction[1] := exp(log_extinction[1])')
            scrfile.write('\n')
            scrfile.write('psi[1] := exp(log_psi[1])')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('for (i in 1:timeline_size) {')
            scrfile.write('\n')
            scrfile.write('    index = i + 1')
            scrfile.write('\n')
            scrfile.write('    # specify normal priors (= Brownian motion) on the log of the rates')
            scrfile.write('\n')
            scrfile.write('    log_speciation[index] ~ dnNormal( mean = log_speciation[i], sd = speciation_sd )')
            scrfile.write('\n')
            scrfile.write('    log_extinction[index] ~ dnNormal( mean = log_extinction[i], sd = extinction_sd )')
            scrfile.write('\n')
            scrfile.write('    log_psi[index] ~ dnNormal( mean=log_psi[i], sd = psi_sd )')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # apply moves on the rates')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(log_speciation[index], weight = 2) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(log_extinction[index], weight = 2) )')
            scrfile.write('\n')
            scrfile.write('    moves.append( mvSlide(log_psi[index], weight = 2) )')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('    # transform the log-rate into actual rates')
            scrfile.write('\n')
            scrfile.write('    speciation[index] := exp( log_speciation[index] )')
            scrfile.write('\n')
            scrfile.write('    extinction[index] := exp( log_extinction[index] )')
            scrfile.write('\n')
            scrfile.write('    psi[index] := exp( log_psi[index] )')
            scrfile.write('\n')
            scrfile.write('}')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvVectorSlide(log_speciation, weight = 10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvVectorSlide(log_extinction, weight = 10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvVectorSlide(log_psi, weight = 10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvShrinkExpand(log_speciation, sd = speciation_sd, weight = 10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvShrinkExpand(log_extinction, sd = extinction_sd, weight = 10))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvShrinkExpand(log_psi, sd = psi_sd, weight = 10))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('rho <- %s' % str(rho))
            scrfile.write('\n')
            scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation, mu = extinction, psi = psi, rho = rho, timeline = timeline, k = k, bounded = FALSE)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.01, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.1, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 1.0, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.01, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 0.1, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('moves.append(mvMatrixElementSlide(bd, delta = 1.0, weight = taxa.size()))')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# workspace model wrapper')
            scrfile.write('\n')
            scrfile.write('mymodel = model(bd)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# add monitors')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnScreen(speciation, extinction, psi, printgen = 5000))')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnModel(filename = "output/%s_model1_HSMRF.log", printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# monitors to print RevGagets input')
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_speciation_rates.log", speciation, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_speciation_times.log", timeline, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_extinction_rates.log", extinction, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_extinction_times.log", timeline, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_sampling_rates.log", psi, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_sampling_times.log", timeline, printgen = 50))' % self.name_file)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# run the analysis')
            scrfile.write('\n')
            scrfile.write('mymcmc = mcmc(mymodel, moves, monitors, moveschedule = "random")')
            scrfile.write('\n')
            scrfile.write('mymcmc.burnin(generations = 5000, tuningInterval = 500)')
            scrfile.write('\n')
            scrfile.write('mymcmc.run(50000)')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('q()')
            scrfile.flush()



    def run_FBD_writter(self, fossils):
        path_dir = os.path.join(self.output_wd, self.name_file)

        path_make_dir_FBD = os.path.join(path_dir, 'FBD')
        try:
            os.mkdir(path_make_dir_FBD)
        except OSError as error:
            print(error)

        path_make_dir_scripts = os.path.join(path_make_dir_FBD, 'scripts')
        try:
            os.mkdir(path_make_dir_scripts)
        except OSError as error:
            print(error)

        path_make_dir_data = os.path.join(path_make_dir_FBD, 'data')
        try:
            os.mkdir(path_make_dir_data)
        except OSError as error:
            print(error)

        fossils_deepcopy = copy.deepcopy(fossils)

        if self.translate is not None and self.padding is None:
            self.translate_fossil_occurrences(fossils_deepcopy)

        if self.interval_ages is None:
            root_height, min_age = self.get_age_range_fossils(fossils_deepcopy)
            interval_ages = np.stack((np.arange(self.interval_size, root_height + self.interval_size, self.interval_size, dtype = float)[::-1],
                                      np.arange(0.0, root_height, self.interval_size, dtype = float)[::-1]),
                                     axis = 1)
        interval_ages = interval_ages[interval_ages[:, 0] <= self.padding[0], :]
        interval_ages = interval_ages[interval_ages[:, 1] >=self.padding[1], :]

        ranges = self.get_ranges(fossils_deepcopy)
        ranges_file = "%s/%s/%s/%s/%s_ranges.csv" % (self.output_wd, self.name_file, 'FBD', 'data', self.name_file)
        ranges.to_csv(ranges_file, header = True, sep = '\t', index = False)

        counts, interval_ages = self.get_occurrences_per_interval(fossils_deepcopy, interval_ages)
        counts_file = "%s/%s/%s/%s/%s_counts.csv" % (self.output_wd, self.name_file, 'FBD', 'data', self.name_file)
        counts.to_csv(counts_file, header = True, sep = '\t', index = False)

        self.write_script(interval_ages, min_age)


def write_occurrence_table(fossils, output_wd, name_file):
    occ_list = fossils['fossil_occurrences']
    occ = np.concatenate(occ_list).ravel()
    occ = np.stack((occ, occ), axis = 1)
    n_occ = np.zeros(len(fossils['taxon_names']), dtype = int)
    for i in range(len(n_occ)):
        n_occ[i] = len(occ_list[i])
    taxon_names = np.repeat(fossils['taxon_names'], n_occ)
    names_df = pd.DataFrame(data = taxon_names, columns = ['sp'])
    occ_df = pd.DataFrame(data = occ, columns = ['hmin', 'hmax'])
    occ_df = pd.concat([names_df, occ_df], axis = 1)
    try:
        os.mkdir(output_wd)
    except OSError as error:
        print(error)
    occ_file = "%s/%s/%s_fossil_occurrences.csv" % (output_wd, name_file, name_file)
    occ_df.to_csv(occ_file, header = True, sep = '\t', index = False, na_rep = 'NA')


def write_ltt(res_bd, output_wd, name_file):
    ltt = res_bd['LTTtrue']
    ltt_df = pd.DataFrame(data=ltt, columns=['time', 'taxa'])
    ltt_file = os.path.join(output_wd, name_file, '%s_simulated_ltt.csv' % name_file)
    ltt_df.to_csv(ltt_file, header=True, sep='\t', index=False, na_rep='NA')


def prune_extinct(tree, tol = 1e-7):
    # mrca_age = tree.max_distance_from_root()
    # root_length = tree.internal_edges()[0].length
    # root_age = mrca_age + root_length
    root_age = get_root_age(tree)
    extant = []
    for leaf in tree.leaf_node_iter():
        species_name = str(leaf.taxon)
        root_dist = leaf.distance_from_root()
        delta_time_present = root_age - root_dist
        if delta_time_present < tol:
            extant.append(species_name.replace("'", ""))
    labels = set(extant)
    tree_ex = tree.extract_tree_with_taxa_labels(labels = labels)

    return tree_ex


class write_FBD_tree():
    def __init__(self,
                 fossils = None,
                 res_bd = None,
                 output_wd = '',
                 edges = None,
                 delta_time = 1.0):
        self.fossils = copy.deepcopy(fossils)
        self.res_bd = copy.deepcopy(res_bd)
        self.output_wd = output_wd
        self.edges = copy.deepcopy(edges)
        self.delta_time = delta_time
        #self.modify_fossils_by_treeoffset()


    def make_FBD_dir(self, name):
        self._path_FBD_data = os.path.join(self.output_wd, name, 'FBDtree', 'data')
        os.makedirs(self._path_FBD_data, exist_ok = True)
        self._path_FBD_scripts = os.path.join(self.output_wd, name, 'FBDtree', 'scripts')
        os.makedirs(self._path_FBD_scripts, exist_ok = True)
        self._path_FBD_output = os.path.join(self.output_wd, name, 'FBDtree', 'output')
        os.makedirs(self._path_FBD_output, exist_ok = True)


    def modify_fossils_by_treeoffset(self):
        # If tree_offset > 0, we need to subtract the offset from all fossil occurrences.
        # If not, will lad_leaf - te_leaf of trim_tree_by_lad() be wrong?
        tree_offset = self.res_bd['tree_offset']
        if tree_offset > 0.0:
            for i in range(len(self.fossils['fossil_occurrences'])):
                self.fossils['fossil_occurrences'][i] = self.fossils['fossil_occurrences'][i] - tree_offset


    # def prune_tree_by_fossil(self):
    #     # Keep only taxa in the tree that have fossil samples
    #     tree = self.res_bd['tree']
    #     labels = set(self.fossils['taxon_names'])
    #     self.tree_pruned = tree.extract_tree_with_taxa_labels(labels = labels)


    # def write_tree_pruned(self):
    #     self.prune_tree_by_fossil()
    #     self.write_ranges('pruned')
    #     path_pruned_tree = os.path.join(self._path_FBD_data, 'PhyloPruned.tre')
    #     self.tree_pruned.write(path = path_pruned_tree, schema = 'newick')


    def get_node_ages(self, tree, tree_offset):
        # Get node age (does not include a potential offset!)
        node_distance_from_root = []
        for nd in tree.postorder_node_iter():
            if nd.is_leaf() is False:
                node_distance_from_root.append(nd.distance_from_root())
        node_distance_from_root = np.sort(np.array(node_distance_from_root))
        root_length = tree.seed_node.edge_length
        node_ages = root_length + tree.max_distance_from_root() - node_distance_from_root + tree_offset

        return node_ages

    def get_range_branches(self, tree, tree_offset = 0.0):
        mrca_age = tree.max_distance_from_root()
        root_length = tree.seed_node.edge_length
        root_age = mrca_age + root_length + tree_offset
        range_branches = []
        for leaf in tree.leaf_node_iter():
            if leaf.is_leaf():
                leaf_name = str(leaf.taxon)
                leaf_name = leaf_name.replace("'", "")
                min_age = root_age - leaf.distance_from_root()
                if min_age < 1e-10:
                    min_age = 0.0
                max_age = min_age + leaf.edge_length
                range_branches.append([leaf_name, min_age, max_age])
        range_branches_df = pd.DataFrame(range_branches, columns = ['taxon', 'min_age', 'max_age'])
        return range_branches_df


    # def trim_tree_by_lad(self, res_bd, fossils, trim_edges = False):
    #     taxon_names = fossils['taxon_names']
    #     fossil_occurrences = fossils['fossil_occurrences']
    #     tree = res_bd['tree']
    #     tree_trimmed = tree.clone()
    #     tree_trimmed = tree_trimmed.extract_tree_with_taxa_labels(labels=set(taxon_names))
    #     tree_trimmed = dendropy.Tree.get(data=tree_trimmed.as_string(schema="newick"), schema="newick")
    #     ts_te = res_bd['ts_te']
    #     keep_taxa = []
    #     for leaf in tree_trimmed.leaf_node_iter():
    #         leaf_name = str(leaf.taxon)
    #         leaf_name = leaf_name.replace("'", "")
    #         ts_te_idx = int(leaf_name.replace("T", "").replace("'", ""))
    #         te_leaf = ts_te[ts_te_idx, 1]
    #         if te_leaf != 0.0 or trim_edges:
    #             occ_idx = taxon_names.index(leaf_name)
    #             lad_leaf = np.min(fossil_occurrences[occ_idx])
    #             shorten_branch = lad_leaf - te_leaf
    #             # Avoid branches descending from a node to the past (b/c of max fossil age older than the node)
    #             if (leaf.edge_length - shorten_branch) >= 0:
    #                 leaf.edge_length = leaf.edge_length - shorten_branch
    #                 keep_taxa.append(leaf_name)
    #         else:
    #             keep_taxa.append(leaf_name)
    #     # Remove taxa with maximum fossil age older than the node from which the taxa descends
    #     tree_trimmed = tree_trimmed.extract_tree_with_taxa_labels(labels=set(keep_taxa))
    #     tree_trimmed = dendropy.Tree.get(data=tree_trimmed.as_string(schema="newick"), schema="newick")
    #
    #     return tree_trimmed, keep_taxa


    def write_tree(self, tree, name, extant_only=False):
        ext = ''
        if extant_only:
            ext = '_extant'
        path_tree = os.path.join(self._path_FBD_data, '%s_tree%s.tre' % (name, ext))
        tree.ladderize(ascending=False)
        tree.write(path=path_tree, schema='newick')


    def get_taxon_data(self, fossils, edges, keep_taxa = None, tree = None, tree_offset = 0.0, translate = 0.0, tip_ages = True):
        taxon_names = fossils['taxon_names']
        n_lineages = len(taxon_names)
        taxon_data = pd.DataFrame(data = taxon_names, columns = ['taxon'])
        taxon_data['min_age'] = np.zeros(n_lineages) # min for RB1.1, RB1.2 needs min_age
        taxon_data['max_age'] = np.zeros(n_lineages)
        occ = fossils['fossil_occurrences']
        if tree is None:
            for i in range(n_lineages):
                occ_i = occ[i]
                if edges is not None:
                    younger = occ_i <= edges[0, 1]
                    if np.min(occ_i) > 0.0 or tree_offset > 0.0:
                        occ_i[younger] = np.nan
                    older = occ_i >= edges[0, 0]
                    occ_i[older] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category = RuntimeWarning)
                    min_age = np.nanmin(occ_i) - translate
                    taxon_data.iloc[i, 1] = min_age
                    if tip_ages:
                        taxon_data.iloc[i, 2] = min_age
                    else:
                        # Stratigraphic ages for fossilized birth-death range process
                        taxon_data.iloc[i, 2] = np.nanmax(occ_i) - translate
        else:
            taxon_data = self.get_range_branches(tree, tree_offset)
        if keep_taxa is not None:
            taxon_data = taxon_data[taxon_data['taxon'].isin(keep_taxa)]

        return taxon_data


    def write_taxon_data(self, name, fossils, edges = None, keep_taxa = None, tree = None, tree_offset = 0.0, translate = 0.0):
        self.tip_ages = self.get_taxon_data(fossils, edges, keep_taxa, tree, tree_offset, translate, tip_ages = True)
        tip_ages_file = os.path.join(self._path_FBD_data, '%s_tip_ages.csv') % name
        self.tip_ages.to_csv(tip_ages_file, header = True, sep = '\t', index = False)
        self.stratigraphic_ranges = self.get_taxon_data(fossils, edges, keep_taxa, tree, tree_offset, translate, tip_ages = False)
        stratigraphic_ranges_file = os.path.join(self._path_FBD_data, '%s_stratigraphic_ranges.csv') % name
        self.stratigraphic_ranges.to_csv(stratigraphic_ranges_file, header = True, sep = '\t', index = False)


    def write_occurrence(self, name, edges):
        occ = self.fossils['fossil_occurrences']
        n_lineages = len(occ)
        occ_concat = np.array([])
        for i in range(n_lineages):
            occ_i = occ[i]
            occ_i = occ_i[occ_i > 0.0]
            if edges is not None:
                keep = np.logical_and(occ_i <= edges[0, 0], occ_i >= edges[0, 1])
                occ_i = occ_i[keep]
            occ_concat = np.concatenate((occ_concat, occ_i), axis = None)
        occ_concat = occ_concat.reshape((len(occ_concat), 1))
        # occ_concat = np.column_stack((occ_concat, occ_concat)) # Can RevBay not read a single column text file?
        path_occ = os.path.join(self._path_FBD_data, '%s_occurrences.txt' % name)
        np.savetxt(path_occ, X = occ_concat, delimiter='\t', fmt='%f')


    def get_new_offset(self, edges, tree_offset):
        num_taxa_in_edges = len(self.trunc_fossil['fossil_occurrences'])
        latest_occ = np.zeros(num_taxa_in_edges)
        for i in range(num_taxa_in_edges):
            latest_occ[i] = np.min(self.trunc_fossil['fossil_occurrences'][i])
        distance_to_edge = np.min(latest_occ) - edges[0, 1]
        new_offset = tree_offset + distance_to_edge

        return new_offset


    def write_tree_offset(self, offset, name):
        path_offset = os.path.join(self._path_FBD_data, '%s_tree_offset.txt' % name)
        np.savetxt(path_offset, X = np.array([offset]), delimiter = '\t', fmt = '%f')


    def prune_and_trim_tree_at_edges(self, res_bd, tree_offset, fossils, edges, keep_extant = False):
        # Remove taxa not within edges and trim by fossil occurrences
        self.trunc_fossil = keep_fossils_in_interval(copy.deepcopy(fossils),
                                                     keep_in_interval = edges,
                                                     keep_extant = keep_extant)
        # mrca_age = tree.max_distance_from_root()  # + tree_offset # age of the first speciation
        # tree_pruned_edges, keep_taxa = self.trim_tree_by_lad(res_bd, self.trunc_fossil, trim_edges = True)
        tree_pruned_edges, keep_taxa = trim_tree_by_lad(res_bd, self.trunc_fossil, trim_edges = True)
        self.new_tree_offset = tree_offset + self.get_new_offset(edges, tree_offset) + edges[0, 1]

        return keep_taxa, tree_pruned_edges


    def get_cat_trait_per_taxon(self, res_bd, fossils, edges = None, majority = True):
        cat_traits = res_bd['cat_traits']
        n_cat_traits = cat_traits.shape[1]
        taxa_sampled = fossils['taxa_sampled']
        n_taxa_sampled = len(taxa_sampled)
        occ = fossils['fossil_occurrences']
        time = res_bd['true_rates_through_time']['time']
        maj_cat_traits = np.zeros(n_taxa_sampled * n_cat_traits).reshape((n_taxa_sampled, n_cat_traits))
        if majority:
            for i in range(n_cat_traits):
                cat_traits_i = cat_traits[:, i, taxa_sampled]
                maj_cat_traits[:, i] = mode(cat_traits_i, axis = 0, nan_policy = 'omit')[0]
        else:
            min_age = np.zeros(n_taxa_sampled)
            for i in range(n_taxa_sampled):
                occ_i = occ[i]
                if edges is not None:
                    keep = np.logical_and(occ_i <= edges[0, 0], occ_i >= edges[0, 1])
                    occ_i = occ_i[keep]
                if len(occ_i) > 0:
                    min_age[i] = np.nanmin(occ_i)
            trait_idx = np.searchsorted(time, min_age)
            for i in range(n_cat_traits):
                cat_traits_i = cat_traits[:, i, taxa_sampled]
                for j in range(n_taxa_sampled):
                    maj_cat_traits[j, i] = cat_traits_i[trait_idx[j], j]

        return maj_cat_traits


    def write_trait_nexus(self, name, res_bd, fossils, keep_taxa, edges, majority=False, extant_only=False):
        keep_taxa_array = np.array(keep_taxa)
        maj_cat_traits = self.get_cat_trait_per_taxon(res_bd, fossils, edges=edges, majority=majority)
        taxon_names = fossils['taxon_names']
        nex = nexus.NexusWriter()
        for i in range(len(taxon_names)):
            taxon = taxon_names[i]
            if np.isin(taxon, keep_taxa_array):
                for j in range(maj_cat_traits.shape[1]):
                    nex.add(taxon, 'Trait' + str(j), int(maj_cat_traits[i, j]))
        ext = ''
        if extant_only:
            ext = '_extant'
        path_nexus_traits = os.path.join(self._path_FBD_data, '%s_morphology%s.nex' % (name, ext))
        nex.write_to_file(path_nexus_traits, interleave = False, charblock = True, preserve_order = True)


    def write_rb_script(self, name, tree = None, tree_offset = 0.0, edges = np.array([[np.inf, 0.0]]), episodic_sampling = False, infer_mass_extinctions = False):
        # Pre-computed hyperpriors for 2-99 shifts (using RevGadgets:::setMRFGlobalScaleHyperpriorNShifts(N, "HSMRF"))
        GlobalScaleHyperprior_NShifts = [0.999952567, 0.507331511, 0.234871326, 0.145767331, 0.103241578,
                                         0.078804639, 0.063162029, 0.052340099, 0.044464321, 0.038521775,
                                         0.033868844, 0.030150525, 0.027111679, 0.024594979, 0.022462786,
                                         0.020649951, 0.019089467, 0.017725212, 0.016534940, 0.015479044,
                                         0.014531150, 0.013706527, 0.012959935, 0.012257861, 0.011636135,
                                         0.011085222, 0.010571979, 0.010110646, 0.009646429, 0.009252232,
                                         0.008893276, 0.008528772, 0.008217924, 0.007928209, 0.007628666,
                                         0.007354106, 0.007116858, 0.006874918, 0.006665834, 0.006459795,
                                         0.006259011, 0.006079703, 0.005904844, 0.005757506, 0.005568076,
                                         0.005447242, 0.005303322, 0.005130847, 0.005024999, 0.004880406,
                                         0.004789234, 0.004666251, 0.004538062, 0.004432210, 0.004369065,
                                         0.004250532, 0.004158740, 0.004070418, 0.003987883, 0.003920844,
                                         0.003831751, 0.003746992, 0.003666860, 0.003591752, 0.003531298,
                                         0.003457690, 0.003402831, 0.003355132, 0.003293772, 0.003234674,
                                         0.003180808, 0.003105620, 0.003064930, 0.002999167, 0.002943756,
                                         0.002891630, 0.002881929, 0.002833793, 0.002788829, 0.002744748,
                                         0.002701360, 0.002659138, 0.002617455, 0.002579869, 0.002544178,
                                         0.002510394, 0.002441295, 0.002434237, 0.002399183, 0.002362197,
                                         0.002326165, 0.002291734, 0.002276365, 0.002226954, 0.002196094,
                                         0.002156608, 0.002166308, 0.002116517, 0.002093737]
        epi_sam = ''
        if episodic_sampling:
            epi_sam = 'Full_'
        fixed_tree = True
        tree_type = 'Fixedtree'
        if tree is None:
            fixed_tree = False
            tree_type = 'Infertree'
        upper_edge = edges[0, 1]
        lower_edge = edges[0, 0]
        scr = os.path.join(self._path_FBD_scripts, '%s_%s_%sEpisodic.Rev' % (name, tree_type, epi_sam))
        scrfile = open(scr, "w")
        scrfile.write('####################################\n')
        scrfile.write('# EPISODIC FOSSIILIZED BIRTH-DEATH #\n')
        scrfile.write('####################################\n')
        scrfile.write('\n')
        scrfile.write('# Data and helpers\n')
        scrfile.write('#-----------------\n')
        scrfile.write('\n')
        rho = 1.0
        if tree_offset > 0.0:
            rho = 0.0
        if fixed_tree:
            scrfile.write('# Read the phylogeny\n')
            scrfile.write('observed_phylogeny = readTrees(file = "data/%s_tree.tre")[1]' % name)
            scrfile.write('\n')
            root_argument = "root"
            if tree_offset > 0.0:
                root_argument = "root + offset"
                scrfile.write('offset <- %s' % tree_offset)
                scrfile.write('\n')
                scrfile.write('# Add offset\n')
                scrfile.write('if (offset > 0.0) {\n')
                scrfile.write('    observed_phylogeny.offset(offset)\n')
                scrfile.write('}\n')
            scrfile.write('\n')
            scrfile.write('# Get the names of the taxa in the tree and the age of the tree. We need these later on.\n')
            scrfile.write('taxa <- observed_phylogeny.taxa()\n')
            scrfile.write('root <- observed_phylogeny.rootAge()\n')
            mrca_age = np.round(tree.max_distance_from_root() + tree_offset)
        else:
            scrfile.write('# Read morphology and ranges\n')
            scrfile.write('taxa <- readTaxonData("data/%s_tip_ages.csv")' % name)
            scrfile.write('\n')
            scrfile.write('morpho <- readDiscreteCharacterData("data/%s_morphology.nex")' % name)
            scrfile.write('\n')
            scrfile.write('n_taxa <- taxa.size()\n')
            mrca_age = np.sort(self.tip_ages['max_age'].to_numpy())[-1] # Why initially -2?
            if lower_edge < mrca_age:
                # Truncated case
                mrca_age = lower_edge
        scrfile.write('\n')
        scrfile.write('# Init moves and monitors of this analysis\n')
        scrfile.write('moves = VectorMoves()\n')
        scrfile.write('monitors = VectorMonitors()\n')
        scrfile.write('\n')
        start_timeline = self.delta_time
        # When the tree (truncated or not by edges) only includes extinct species, we should use a single, large time-bin
        # from the present until the most recent tip
        if upper_edge > self.delta_time or tree_offset > self.delta_time:
            new_start_timeline = np.floor(tree_offset)
            if new_start_timeline > start_timeline:
                start_timeline = new_start_timeline
        timeline = np.arange(start_timeline, mrca_age - self.delta_time, step = self.delta_time)
        # Check if hyperpriors should be for an uneven grid
        diff_timeline = np.abs(np.diff(np.concatenate((np.zeros(1), timeline))))
        even_grid_hyperprior = np.all(diff_timeline == diff_timeline[0])
        even_grid_hyperprior = True # Uneven grid hyperprior not working, so overwrite it for the moment.
        if fixed_tree:
            # Check if any boundaries (as defined by the timeline) is equal to a node age.
            # RevBayes will not find an initial likelihood in that case.
            node_ages = self.get_node_ages(tree, tree_offset)
            sim_fractional_digits = len(str(int(np.round(self.res_bd['sim_scale'])))) - 1
            node_ages = np.round(node_ages, sim_fractional_digits)
            problematic_boundaries = np.isin(timeline, node_ages)
            timeline[problematic_boundaries] = timeline[problematic_boundaries] + 10**-(sim_fractional_digits + 1)
        timeline = timeline.tolist()
        scrfile.write('# Skyline time boundaries\n')
        scrfile.write('timeline <- v(')
        for i in range(len(timeline)):
            scrfile.write(str(timeline[i]))
            if i < (len(timeline) - 1):
                scrfile.write(',')
        scrfile.write(')\n')
        scrfile.write('timeline_size <- timeline.size()\n')
        scrfile.write('\n')
        scrfile.write('\n')
        scrfile.write('# Priors and moves\n')
        scrfile.write('#-----------------\n')
        scrfile.write('\n')
        scrfile.write('# prior and hyperprior for overall amount of rate variation\n')
        GlobalScaleHyperprior = GlobalScaleHyperprior_NShifts[len(timeline) - 1]
        scrfile.write('speciation_global_scale_hyperprior <- %s\n' % GlobalScaleHyperprior)
        scrfile.write('speciation_global_scale ~ dnHalfCauchy(0, 1)\n')
        scrfile.write('extinction_global_scale_hyperprior <- %s\n' % GlobalScaleHyperprior)
        scrfile.write('extinction_global_scale ~ dnHalfCauchy(0, 1)\n')
        scrfile.write('\n')
        scrfile.write('# create a random variable at the present time\n')
        scrfile.write('log_speciation_at_present ~ dnUniform(-5.0, 1.0)\n')
        scrfile.write('log_speciation_at_present.setValue(-1.0)\n')
        scrfile.write('log_extinction_at_present ~ dnUniform(-5.0, 1.0)\n')
        scrfile.write('log_extinction_at_present.setValue(-3.0)\n')
        scrfile.write('\n')
        scrfile.write('moves.append( mvSlide(log_speciation_at_present, delta = 1.0, weight = 5) )\n')
        scrfile.write('moves.append( mvSlide(log_extinction_at_present, delta = 1.0, weight = 5) )\n')
        scrfile.write('\n')
        scrfile.write('for (i in 1:timeline_size) {\n')
        scrfile.write('    sigma_speciation[i] ~ dnHalfCauchy(0, 1)\n')
        scrfile.write('    sigma_extinction[i] ~ dnHalfCauchy(0, 1)\n')
        scrfile.write('\n')
        scrfile.write('    # Initialize to something reasonable\n')
        scrfile.write('    sigma_speciation[i].setValue(runif(1, 0.005, 0.1)[1])\n')
        scrfile.write('    sigma_extinction[i].setValue(runif(1, 0.005, 0.1)[1])\n')
        scrfile.write('\n')
        scrfile.write('    # Moves for the single sigma values\n')
        scrfile.write('    moves.append( mvScaleBactrian(sigma_speciation[i], weight = 5) )\n')
        scrfile.write('    moves.append( mvScaleBactrian(sigma_extinction[i], weight = 5) )\n')
        scrfile.write('\n')
        scrfile.write('    # Non-centralized parameterization of horseshoe\n')
        scrfile.write('    delta_log_speciation[i] ~ dnNormal(mean = 0, sd = sigma_speciation[i] * speciation_global_scale * speciation_global_scale_hyperprior)\n')
        scrfile.write('    delta_log_extinction[i] ~ dnNormal(mean = 0, sd = sigma_extinction[i] * extinction_global_scale * extinction_global_scale_hyperprior)\n')
        scrfile.write('\n')
        scrfile.write('    # Initialize to something reasonable\n')
        scrfile.write('    delta_log_speciation[i].setValue(runif(1, -0.1, 0.1)[1])\n')
        scrfile.write('    delta_log_extinction[i].setValue(runif(1, -0.1, 0.1)[1])\n')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvSlideBactrian(delta_log_speciation[i], weight = 5) )\n')
        scrfile.write('    moves.append( mvSlideBactrian(delta_log_extinction[i], weight = 5) )\n')
        scrfile.write('\n')
        scrfile.write('    delta_up_down_move[i] = mvUpDownSlide(weight = 5)\n')
        scrfile.write('    delta_up_down_move[i].addVariable(delta_log_speciation[i], TRUE)\n')
        scrfile.write('    delta_up_down_move[i].addVariable(delta_log_extinction[i], TRUE)\n')
        scrfile.write('    moves.append( delta_up_down_move[i] )\n')
        scrfile.write('}\n')
        scrfile.write('\n')
        scrfile.write('# Assemble first-order differences and speciation_rate at present into the random field\n')
        scrfile.write('speciation_rate := fnassembleContinuousMRF(log_speciation_at_present, delta_log_speciation, initialValueIsLogScale = TRUE, order = 1) + 0.000001\n')
        scrfile.write('extinction_rate := fnassembleContinuousMRF(log_extinction_at_present, delta_log_extinction, initialValueIsLogScale = TRUE, order = 1) + 0.000001\n')
        scrfile.write('\n')
        scrfile.write('# Move all field parameters in one go\n')
        scrfile.write('moves.append( mvEllipticalSliceSamplingSimple(delta_log_speciation, weight = 5, tune = FALSE, forceAccept = TRUE) )\n')
        scrfile.write('moves.append( mvEllipticalSliceSamplingSimple(delta_log_extinction, weight = 5, tune = FALSE, forceAccept = TRUE) )\n')
        scrfile.write('\n')
        scrfile.write('# Move all field hyperparameters in one go\n')
        if even_grid_hyperprior:
            scrfile.write('moves.append( mvHSRFHyperpriorsGibbs(speciation_global_scale, sigma_speciation, delta_log_speciation, speciation_global_scale_hyperprior, order = 1, weight = 10) )\n')
            scrfile.write('moves.append( mvHSRFHyperpriorsGibbs(extinction_global_scale, sigma_extinction, delta_log_extinction, extinction_global_scale_hyperprior, order = 1, weight = 10) )\n')
        else:
            # Not working because grid should be of type deterministic but this code makes it constant. No idea how to change it.
            scrfile.write('grid <- v(')
            for i in range(len(diff_timeline)):
                scrfile.write(str(diff_timeline[i]))
                if i < (len(diff_timeline) - 1):
                    scrfile.write(',')
            scrfile.write(')\n')
            scrfile.write('moves.append( mvHSRFUnevenGridHyperpriorsGibbs(speciation_global_scale, sigma_speciation, delta_log_speciation, grid, speciation_global_scale_hyperprior, order = 1, weight = 10) )\n')
            scrfile.write('moves.append( mvHSRFUnevenGridHyperpriorsGibbs(extinction_global_scale, sigma_extinction, delta_log_extinction, grid, extinction_global_scale_hyperprior, order = 1, weight = 10) )\n')
        scrfile.write('\n')
        scrfile.write('# Swap moves to exchange adjacent delta,sigma pairs\n')
        scrfile.write('moves.append( mvHSRFIntervalSwap(delta_log_speciation, sigma_speciation, weight = 5) )\n')
        scrfile.write('moves.append( mvHSRFIntervalSwap(delta_log_extinction, sigma_extinction, weight = 5) )\n')
        scrfile.write('\n')
        if episodic_sampling is False:
            scrfile.write('# Assume an exponential prior on the rate of sampling fossils (psi)\n')
            scrfile.write('psi2 ~ dnExponential(10.0)\n')
            scrfile.write('\n')
            scrfile.write('# Specify a scale move on the psi parameter\n')
            scrfile.write('moves.append( mvScale(psi2, lambda = 0.01, weight = 1) )\n')
            scrfile.write('moves.append( mvScale(psi2, lambda = 0.1,  weight = 1) )\n')
            scrfile.write('moves.append( mvScale(psi2, lambda = 1.0,  weight = 1) )\n')
            scrfile.write('for (i in 1:(timeline_size + 1)) {\n')
            scrfile.write('    psi[i] := psi2\n')
            scrfile.write('}\n')
        else:
            scrfile.write('# Horseshoe prior for sampling (psi)\n')
            scrfile.write('psi_global_scale_hyperprior <- %s\n' % GlobalScaleHyperprior)
            scrfile.write('psi_global_scale ~ dnHalfCauchy(0, 1)\n')
            scrfile.write('log_psi_at_present ~ dnUniform(-5.0, 1.0)\n')
            scrfile.write('log_psi_at_present.setValue(-4.0)\n')
            scrfile.write('moves.append( mvSlide(log_psi_at_present, delta = 1.0, weight = 5) )\n')
            scrfile.write('for (i in 1:timeline_size) {\n')
            scrfile.write('    sigma_psi[i] ~ dnHalfCauchy(0, 1)\n')
            scrfile.write('    sigma_psi[i].setValue(runif(1, 0.005, 0.1)[1])\n')
            scrfile.write('    delta_log_psi[i] ~ dnNormal(mean = 0, sd = sigma_psi[i] * psi_global_scale * psi_global_scale_hyperprior)\n')
            scrfile.write('    delta_log_psi[i].setValue(runif(1, -0.1, 0.1)[1])\n')
            scrfile.write('    moves.append( mvSlideBactrian(delta_log_psi[i], weight = 5) )\n')
            scrfile.write('}\n')
            scrfile.write('psi := fnassembleContinuousMRF(log_psi_at_present, delta_log_psi, initialValueIsLogScale = TRUE, order = 1) + 0.000001\n')
            scrfile.write('moves.append( mvEllipticalSliceSamplingSimple(delta_log_psi, weight = 5, tune = FALSE, forceAccept = TRUE) )\n')
            scrfile.write('moves.append( mvHSRFHyperpriorsGibbs(psi_global_scale, sigma_psi, delta_log_psi, psi_global_scale_hyperprior, order = 1, weight = 10))\n')
            scrfile.write('moves.append( mvHSRFIntervalSwap(delta_log_psi, sigma_psi, weight = 5) )\n')
        scrfile.write('\n')
        scrfile.write('# Probability of sampling species at the present\n')
        scrfile.write('rho <- %s' % str(rho))
        scrfile.write('\n')
        scrfile.write('\n')
        Mu_argument = ""
        if infer_mass_extinctions:
            scrfile.write('# Mass extinction\n')
            scrfile.write('#----------------\n')
            scrfile.write('expected_number_of_mass_extinctions <- 0.005\n')
            scrfile.write('mix_p <- Probability(1.0 - expected_number_of_mass_extinctions / timeline_size)\n')
            scrfile.write('for (i in 1:timeline_size) {\n')
            scrfile.write('    mass_extinction_probabilities[i] ~ dnReversibleJumpMixture(0.0, dnBeta(18.0, 2.0), mix_p)\n')
            scrfile.write('    moves.append( mvRJSwitch(mass_extinction_probabilities[i]) )\n')
            scrfile.write('    moves.append( mvSlideBactrian(mass_extinction_probabilities[i]) )\n')
            scrfile.write('}\n')
            scrfile.write('\n')
            Mu_argument = " Mu = mass_extinction_probabilities,"
        if fixed_tree:
            scrfile.write('# Define the tree-prior distribution as the fossilized birth-death process\n')
            scrfile.write('fbd_tree ~ dnBDSTP(originAge = %s, lambda = speciation_rate, mu = extinction_rate, psi = psi, Phi = rho, timeline = timeline,%s taxa = taxa, condition = "time", initialTree = observed_phylogeny)' % (root_argument, Mu_argument))
            scrfile.write('\n')
            scrfile.write('fbd_tree.clamp(observed_phylogeny)\n')
        else:
            scrfile.write('# MRCA age\n')
            scrfile.write('origin_time ~ dnUnif(%s, %s)' % (str(mrca_age), str(100.0 - upper_edge)))
            scrfile.write('\n')
            scrfile.write('moves.append(mvSlide(origin_time, delta = 0.01, weight = 5))\n')
            scrfile.write('moves.append(mvSlide(origin_time, delta = 0.1, weight = 5))\n')
            scrfile.write('moves.append(mvSlide(origin_time, delta = 1.0, weight = 5))\n')
            scrfile.write('\n')
            scrfile.write('# Define the tree-prior distribution as the fossilized birth-death process\n')
            scrfile.write('fbd_tree ~ dnBDSTP(originAge = origin_time, lambda = speciation_rate, mu = extinction_rate, psi = psi, Phi = rho, timeline = timeline,%s taxa = taxa, condition = "time")' % Mu_argument)
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Moves on tree topology\n')
            scrfile.write('moves.append( mvFNPR(fbd_tree, weight = n_taxa/10) )\n')
            scrfile.write('moves.append( mvNNI(fbd_tree, weight = n_taxa/20) )\n')
            scrfile.write('moves.append( mvNarrow(fbd_tree, weight = n_taxa/20) )\n')
            scrfile.write('\n')
            scrfile.write('# Moves on branch lengths and node ages\n')
            scrfile.write('moves.append( mvNodeTimeSlideUniform(fbd_tree, weight = n_taxa/10) )\n')
            scrfile.write('moves.append( mvRootTimeSlideUniform(fbd_tree, origin_time, weight = n_taxa/20) )\n')
            scrfile.write('moves.append( mvSubtreeScale(fbd_tree, weight = n_taxa/20) )\n')
            scrfile.write('moves.append( mvTreeScale(fbd_tree, origin_time, weight = n_taxa/20) )\n')
            scrfile.write('\n')
            scrfile.write('# Sampled ancestors\n')
            scrfile.write('moves.append(mvCollapseExpandFossilBranch(fbd_tree, origin_time, weight = 5))\n')
            scrfile.write('num_samp_anc := fbd_tree.numSampledAncestors()\n')
            scrfile.write('\n')
            scrfile.write('\n')
            scrfile.write('# Binary morphological substitution model\n')
            scrfile.write('#----------------------------------------\n')
            scrfile.write('Q_morpho := fnJC(2)\n')
            scrfile.write('\n')
            scrfile.write('# Set up Gamma-distributed rate variation\n')
            scrfile.write('alpha_morpho_inv ~ dnExponential( 1.0 )\n')
            scrfile.write('alpha_morpho := 1.0 / alpha_morpho_inv\n')
            scrfile.write('rates_morpho := fnDiscretizeGamma( alpha_morpho, alpha_morpho, 4 )\n')
            scrfile.write('\n')
            scrfile.write('# Moves on the parameters to the Gamma distribution\n')
            # scrfile.write('moves.append( mvScale(alpha_morpho, lambda = 0.01, weight = 5) )\n')
            # scrfile.write('moves.append( mvScale(alpha_morpho, lambda = 0.1,  weight = 3) )\n')
            # scrfile.write('moves.append( mvScale(alpha_morpho, lambda = 1.0,  weight = 1) )\n')
            scrfile.write('\n')
            scrfile.write('# We assume a strict morphological clock rate, drawn from an exponential prior\n')
            scrfile.write('clock_morpho ~ dnExponential(10.0)\n')
            scrfile.write('\n')
            # scrfile.write('moves.append( mvScale(clock_morpho, lambda = 0.01, weight = 10) )\n')
            # scrfile.write('moves.append( mvScale(clock_morpho, lambda = 0.1,  weight = 10) )\n')
            # scrfile.write('moves.append( mvScale(clock_morpho, lambda = 1.0,  weight = 10) )\n')
            scrfile.write('avmvn_morpho = mvAVMVN(weight = 10)\n')
            scrfile.write('avmvn_morpho.addVariable(alpha_morpho_inv)\n')
            scrfile.write('avmvn_morpho.addVariable(clock_morpho)\n')
            scrfile.write('moves.append( avmvn_morpho )\n')
            scrfile.write('up_down_move_morpho = mvUpDownScale(weight = 10)\n')
            scrfile.write('up_down_move_morpho.addVariable(alpha_morpho_inv, TRUE)\n')
            scrfile.write('up_down_move_morpho.addVariable(clock_morpho, TRUE)\n')
            scrfile.write('moves.append( up_down_move_morpho )\n')
            scrfile.write('moves.append( mvScaleBactrian(alpha_morpho_inv, weight = 10) )\n')
            scrfile.write('moves.append( mvScaleBactrian(clock_morpho, weight = 10) )\n')
            scrfile.write('moves.append( mvMirrorMultiplier(alpha_morpho_inv, weight = 10) )\n')
            scrfile.write('moves.append( mvMirrorMultiplier(clock_morpho, weight = 10) )\n')
            scrfile.write('moves.append( mvRandomDive(alpha_morpho_inv, weight = 10) )\n')
            scrfile.write('moves.append( mvRandomDive(clock_morpho, weight = 10) )\n')
            scrfile.write('\n')
            scrfile.write('# Create the substitution model and clamp with our observed Standard data\n')
            scrfile.write('# Here we use the option siteMatrices=true specify that the vector Q represents a site-specific mixture of rate matrices.\n')
            scrfile.write('# We also condition on observing only variable characters using coding="variable".\n')
            scrfile.write('phyMorpho ~ dnPhyloCTMC(tree = fbd_tree, siteRates = rates_morpho, branchRates = clock_morpho, Q = Q_morpho, type = "Standard", coding = "variable")\n')
            scrfile.write('phyMorpho.clamp(morpho)\n')
        scrfile.write('\n')
        scrfile.write('\n')
        scrfile.write('# The Model\n')
        scrfile.write('#----------\n')
        scrfile.write('\n')
        scrfile.write('# Workspace model wrapper\n')
        scrfile.write('mymodel = model(speciation_rate)\n')
        scrfile.write('\n')
        scrfile.write('# Set up the monitors that will output parameter values to file and screen\n')
        scrfile.write('monitors.append( mnScreen(printgen = 5000) )\n')
        scrfile.write('monitors.append( mnModel(filename = "output/%s_%s_%sEpisodic.log", printgen = 50, separator = TAB) )' % (name, tree_type, epi_sam))
        scrfile.write('\n')
        scrfile.write('monitors.append( mnFile(filename = "output/%s_%s_%sTimeline.log", timeline, printgen = 50) )' % (name, tree_type, epi_sam))
        scrfile.write('\n')
        if fixed_tree is False:
            scrfile.write('monitors.append( mnFile(filename = "output/%s_%s_%sEpisodic.trees", fbd_tree, printgen = 50) )' % (name, tree_type, epi_sam))
        scrfile.write('\n')
        scrfile.write('\n')
        scrfile.write('# The Analysis\n')
        scrfile.write('#-------------\n')
        scrfile.write('\n')
        scrfile.write('# Workspace mcmc\n')
        scrfile.write('mymcmc = mcmc(mymodel, monitors, moves, nruns = 1, ntries = 10000)\n')
        scrfile.write('\n')
        scrfile.write('# Run the MCMC\n')
        scrfile.write('mymcmc.burnin(generations = 5000, tuningInterval = 500)\n')
        scrfile.write('mymcmc.run(50000)\n')
        scrfile.write('\n')
        if fixed_tree is False:
            scrfile.write('# Summarize phylogeny\n')
            scrfile.write('posterior_trees = readTreeTrace("output/%s_%s_%sEpisodic.trees", burnin = 0.0)' % (name, tree_type, epi_sam))
            scrfile.write('\n')
            scrfile.write('mccTree(posterior_trees, file="output/%s_%s_%sEpisodic_MCC.tre", positiveBranchLengths = TRUE)' % (name, tree_type, epi_sam))
            scrfile.write('\n')
            scrfile.write('\n')
        scrfile.write('# quit\n')
        scrfile.write('q()')
        scrfile.flush()


    def run_writter(self,
                    name,
                    edges=None,
                    keep_extant=True,
                    infer_mass_extinctions=False,
                    translate_to_present=False,
                    write_RevBayes_script=True,
                    extant_only=False):
        self.make_FBD_dir(name)
        fossils_copy = copy.deepcopy(self.fossils)
        res_bd_copy = copy.deepcopy(self.res_bd)
        if edges is None:
            # tree_trimmed, keep_taxa = self.trim_tree_by_lad(res_bd_copy, fossils_copy)
            tree_trimmed, keep_taxa = trim_tree_by_lad(res_bd_copy, fossils_copy, extant_only=extant_only)
            # Delete this one below
            # taxon_data = self.get_taxon_data(fossils = fossils_copy, edges = edges, keep_taxa = keep_taxa, tree=None, tree_offset=0.0, translate = translate_to_present, tip_ages=True)
            self.write_tree(tree_trimmed, name=name, extant_only=extant_only)
            self.write_tree_offset(offset = res_bd_copy['tree_offset'], name = name)
            self.write_taxon_data(name = name, fossils = fossils_copy, edges=None, keep_taxa = keep_taxa, tree=None, tree_offset=0.0, translate=0.0)
            if write_RevBayes_script:
                self.write_rb_script(name = name,
                                     tree = tree_trimmed,
                                     tree_offset = res_bd_copy['tree_offset'],
                                     infer_mass_extinctions = infer_mass_extinctions)
                self.write_rb_script(name = name,
                                     tree = tree_trimmed,
                                     tree_offset = res_bd_copy['tree_offset'],
                                     episodic_sampling = True,
                                     infer_mass_extinctions = infer_mass_extinctions)
            if self.res_bd['cat_traits'].size > 0:
                self.write_trait_nexus(name = name,
                                       res_bd = res_bd_copy,
                                       fossils = fossils_copy,
                                       keep_taxa = keep_taxa,
                                       edges = edges,
                                       extant_only=extant_only)
                if write_RevBayes_script:
                    self.write_rb_script(name = name,
                                         tree_offset = res_bd_copy['tree_offset'],
                                         infer_mass_extinctions = infer_mass_extinctions)
                    self.write_rb_script(name = name,
                                         tree_offset = res_bd_copy['tree_offset'],
                                         episodic_sampling = True,
                                         infer_mass_extinctions = infer_mass_extinctions)
            # Use the simulated tree
            #self.write_tree(self.res_bd['tree'], 'Simulated')
            #self.write_rb_script('Simulated', self.res_bd['tree'], self.res_bd['tree_offset'])
            # Species ranges trimmed to their branch length
            # self.write_ranges(self.name + '_branches', keep_taxa, tree_trimmed, self.res_bd['tree_offset'])
        else:
            translate = 0.0
            if translate_to_present:
                translate = edges[0, 1]
            keep_taxa, tree_pruned_edges = self.prune_and_trim_tree_at_edges(res_bd = res_bd_copy,
                                                                             tree_offset = res_bd_copy['tree_offset'],
                                                                             fossils = fossils_copy,
                                                                             edges = edges,
                                                                             keep_extant = keep_extant)
            self.write_tree(tree_pruned_edges, name = name)
            self.write_tree_offset(offset = self.new_tree_offset, name = name)
            self.write_taxon_data(name=name,
                                  fossils=fossils_copy,
                                  edges=edges,
                                  keep_taxa=keep_taxa,
                                  tree_offset=self.new_tree_offset,
                                  translate=translate)
            if write_RevBayes_script:
                self.write_rb_script(name = name,
                                     tree = tree_pruned_edges,
                                     tree_offset = self.new_tree_offset - translate,
                                     edges = edges,
                                     infer_mass_extinctions = infer_mass_extinctions)
                self.write_rb_script(name = name,
                                     tree = tree_pruned_edges,
                                     tree_offset = self.new_tree_offset - translate,
                                     edges = edges,
                                     episodic_sampling = True,
                                     infer_mass_extinctions = infer_mass_extinctions)
            if self.res_bd['cat_traits'].size > 0:
                self.write_trait_nexus(name=name,
                                       res_bd=res_bd_copy,
                                       fossils=fossils_copy,
                                       keep_taxa=keep_taxa,
                                       edges=edges,
                                       extant_only=extant_only)
                if write_RevBayes_script:
                    self.write_rb_script(name = name,
                                         tree_offset = self.new_tree_offset - translate,
                                         edges = edges,
                                         infer_mass_extinctions = infer_mass_extinctions)
                    self.write_rb_script(name = name,
                                         tree_offset = self.new_tree_offset - translate,
                                         edges = edges,
                                         episodic_sampling = True,
                                         infer_mass_extinctions = infer_mass_extinctions)
        self.write_occurrence(name = name, edges = edges)

