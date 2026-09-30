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


class BdnnSimulator():
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
                 seed = -1):
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
        if self.seed == -1:
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
                    cat_trait_yi = int(np.random.choice(cat_states[y], 1, p = pi).item())
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
                                cat_trait_new = int(cat_trait_new.item())
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
        n_cat_traits = int(n_cat_traits.item())
        cat_traits_Q = []
        n_cat_traits_states = np.zeros(n_cat_traits, dtype = int)
        cat_states = []
        cat_trait_effect = []
        if n_cat_traits > 0:
            for i in range(n_cat_traits):
                n_cat_traits_states[i] = np.random.choice(np.arange(min(self.n_cat_traits_states),
                                                                    max(self.n_cat_traits_states) + 1),
                                                          1)[0]
                Qi = self.make_cat_traits_Q(n_cat_traits_states[i])
                cat_traits_Q.append(Qi)
                cat_states_i = np.arange(n_cat_traits_states[i])
                cat_states.append(cat_states_i)
                cat_trait_effect_i = self.make_cat_trait_effect(n_cat_traits_states[i])
                cat_trait_effect.append(cat_trait_effect_i)

        # continuous traits
        n_cont_traits = np.random.choice(np.arange(min(self.n_cont_traits), max(self.n_cont_traits) + 1), 1)
        n_cont_traits = int(n_cont_traits.item())
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
        n_areas = int(n_areas.item())
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
                    effect_par[i, j, k, 0] = np.random.uniform(np.min(cte_tmp), np.max(cte_tmp), 1)[0]
                    # Whether effect has a bell (1) or u-shape (-1)
                    effect_par[i, j, k, 1] = np.random.choice(bellu[i, j, k, :], 1)[0]
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
