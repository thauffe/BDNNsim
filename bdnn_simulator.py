import copy
import sys
import os
from numpy import linalg as la
from scipy.stats import mode
from scipy.stats import norm
from scipy.stats import multivariate_normal
from itertools import combinations
from functools import reduce
from operator import iconcat
from math import comb
import numpy as np
import pandas as pd
import scipy.linalg
import random
import string
np.set_printoptions(suppress = True, precision = 3)
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
                 root_r = [30., 100],  # range root ages
                 rangeL = [0.2, 0.5], # range speciation rate
                 rangeM = [0.2, 0.5], # range extinction rate
                 scale = 100., # root * scale = steps for the simulation
                 p_mass_extinction = 0.00924,
                 magnitude_mass_ext = [0.8, 0.95],
                 poiL = 3, # Number of rate shifts expected according to a Poisson distribution
                 poiM = 3, # Number of rate shifts expected according to a Poisson distribution
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
                 fixed_Mtt = None, # fix extinction rate through time (see fixed Mtt)
                 n_cont_traits = [0, 0], # number of continuous traits
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
                 cat_traits_ordinal = [True, False], # is categorical trait ordinal or discrete?
                 cat_traits_dir = 1, # concentration parameter dirichlet distribution for transition probabilities between categorical states
                 cat_traits_diag = None, # fix diagonal of categorical transition matrix to this value (overwrites cat_traits_dir)
                 # range of effect of categorical traits on speciation (1st row) and extinction (2nd row) (1 is no effect)
                 # effects can be fixed with e.g. np.array([[2.3., 2.3.],[1.5.,1.5.]]) and cat_traits_effect_decr_incr = np.array([[True, True],[False, False]])
                 # effect of n_cat_traits_states > 2 can be fixed with n_cat_traits_states = [3, 3] AND np.array([[1.5., 2.3.],[0.2.,1.5.]]) (no need for cat_traits_effect_decr_incr)
                 # or in case of 4 states with n_cat_traits_states = [4, 4] AND np.array([[1.5., 2.3., 0.6],[0.2.,1.5., 1.9]])
                 cat_traits_effect = np.array([[1., 1.],[1.,1.]]),
                 cat_traits_effect_decr_incr = np.array([[True, False],[True, False]]), # should categorical effect cause a decrease (True) or increase (False) in speciation (1st row) and extinction (2nd row)?
                 n_areas = [0, 0], # number of biogeographic areas (minimum of 2)
                 dispersal = [0.1, 0.3], # range for the rate of area expansion
                 extirpation = [0.05, 0.1], # range for the rate of area loss
                 sp_env_file = None, # Path to environmental file influencing speciation
                 sp_env_eff = [0.0, 0.0],  # range environmental effect on speciation rate
                 ex_env_file = None,  # Path to environmental file influencing speciation
                 ex_env_eff = [0.0, 0.0],  # range environmental effect on speciation rate
                 seed = 0):
        self.s_species = s_species
        self.rangeSP = rangeSP
        self.minSP = np.min(rangeSP)
        self.maxSP = np.max(rangeSP)
        self.minEX_SP = minEX_SP
        self.minExtant_SP = minExtant_SP
        self.maxExtant_SP = maxExtant_SP
        self.root_r = root_r
        self.rangeL = rangeL
        self.rangeM = rangeM
        self.scale = scale
        self.p_mass_extinction = p_mass_extinction
        self.magnitude_mass_ext = np.sort(magnitude_mass_ext)
        self.poiL = poiL
        self.poiM = poiM
        self.range_linL = range_linL
        self.range_linM = range_linM
        self.fixed_Ltt = fixed_Ltt
        self.fixed_Mtt = fixed_Mtt
        self.n_cont_traits = n_cont_traits
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
        self.n_areas = n_areas
        self.dispersal = dispersal
        self.extirpation = extirpation
        self.sp_env_file = sp_env_file
        self.sp_env_eff = sp_env_eff
        self.ex_env_file = ex_env_file
        self.ex_env_eff = ex_env_eff
        if seed:
            np.random.seed(seed)


    def simulate(self, L, M, root, dT, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_trait_effect_sp, cont_trait_effect_ex, expected_sd_cont_traits, cont_traits_effect_shift_sp, cont_traits_effect_shift_ex, n_cat_traits, cat_states, cat_traits_Q, cat_trait_effect, n_areas, dispersal, extirpation):
        ts = list()
        te = list()

        root = int(root * self.scale)
        # Trace ancestor descendant relationship
        # First entry: ancestor (for the seeding specis, this is an index of themselfs)
        # Following entries: descendants
        anc_desc = []

        # Track time of origin/extinction (already in ts and te) and rates per lineage
        lineage_rates = []

        for i in range(self.s_species):
            ts.append(root)
            te.append(-0.0)
            anc_desc.append(np.array([i]))
            lineage_rates_tmp = np.zeros(5 + 2 * n_cont_traits + 2 * n_cat_traits)
            lineage_rates_tmp[:] = np.nan
            lineage_rates_tmp[:5] = np.array([root, -0.0, L[root], M[root], 0.0])
            lineage_rates.append(lineage_rates_tmp)

        # init continuous traits (if there are any to simulate)
        root_plus_1 = np.abs(root) + 2

        # init categorical traits
        cat_traits = np.empty((root_plus_1, n_cat_traits, self.s_species))
        cat_traits[:] = np.nan
        # init continuous traits
        cont_traits = np.empty((root_plus_1, n_cont_traits, self.s_species))
        cont_traits[:] = np.nan

        for i in range(self.s_species):
            #cat_traits_Q[y] = dT * cat_traits_Q[y]  # Only for anagenetic evolution of categorical traits
            cat_trait_yi = 0
            if n_cat_traits > 0:
                for y in range(n_cat_traits):
                    pi = self.get_stationary_distribution(cat_traits_Q[y])
                    cat_trait_yi = int(np.random.choice(cat_states[y], 1, p = pi))
                    cat_traits[-1,y,i] = cat_trait_yi
                    lineage_rates[i][2] = lineage_rates[i][2] * cat_trait_effect[y][0, cat_trait_yi]
                    lineage_rates[i][3] = lineage_rates[i][3] * cat_trait_effect[y][1, cat_trait_yi]
                    lineage_rates[i][(5 + 2 * n_cont_traits + y):(6 + 2 * n_cont_traits + y)] = cat_trait_yi
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

        # evolution (from the past to the present)
        for t in range(root, 0):
            # time i.e. integers self.root * self.scale
            # t = 0 not simulated!
            t_abs = abs(t)
            # print('t: ', t)
            l = L[t]
            m = M[t]

            TE = len(te)
            if TE > self.maxSP:
                #print(t_abs)
                break
            ran_vec = np.random.random(TE)
            te_extant = np.where(np.array(te) == 0)[0]
            ran_vec_cat_trait = np.random.random(TE)
            ran_vec_biogeo = np.random.random(TE)

            no = np.random.uniform(0, 1)  # draw a random number
            no_extant_lineages = len(te_extant)  # the number of currently extant species
            mass_extinction_prob = self.p_mass_extinction/self.scale
            if no < mass_extinction_prob and no_extant_lineages > 10:  # mass extinction condition
                # print("Mass extinction", t / self.scale)
                # increased loss of species: increased ext probability for this time bin
                m = np.random.uniform(self.magnitude_mass_ext[0], self.magnitude_mass_ext[1])
                mass_ext_time.append(t)
                mass_ext_mag.append(m)

            # Trait dependent rates for all extant lineages
            lineage_lambda = np.zeros(TE)
            lineage_lambda[:] = np.nan
            lineage_mu = np.zeros(TE)
            lineage_mu[:] = np.nan

            for j in te_extant:  # extant lineages
                l_j = l + 0.
                m_j = m + 0.

                # categorical trait evolution
                cat_trait_j = 0
                if n_cat_traits > 0:
                    for y in range(n_cat_traits):
                        cat_trait_j = cat_traits[t_abs + 1, y, j] # No change along branches
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
                                                                     expected_sd_cont_traits, n_cont_traits)
                    cont_traits_bin = cont_traits_effect_shift_ex[t_abs]
                    m_j = self.get_rate_by_cont_trait_transformation(m_j,
                                                                     cont_trait_j,
                                                                     cont_trait_effect_ex[cont_traits_bin, :, cat_trait_j, :],
                                                                     expected_sd_cont_traits,
                                                                     n_cont_traits)

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
                    te.append(-0.0)  # add species
                    ts.append(t)  # sp time

                    # Keep track of ancestor-descendent relationship - double-check this!
                    desc = np.array([len(ts)])
                    anc_desc[j] = np.concatenate((anc_desc[j], desc))
                    anc = np.random.choice(anc_desc[j], 1)  # If a lineage already has multiple descendents
                    anc_desc.append(anc)

                    lineage_rates_tmp = np.zeros(5 + 2 * n_cont_traits + 2 * n_cat_traits)
                    l_new = l + 0.0
                    m_new = m + 0.0

                    # Inherit traits
                    cat_trait_new = 0
                    if n_cat_traits > 0:
                        cat_traits_new_species = self.empty_traits(root_plus_1, n_cat_traits)
                        # cat_traits_new_species[t_abs,] = cat_traits[t_abs,:,j] # inherit state at speciation
                        for y in range(n_cat_traits):
                            # Change of categorical trait at speciation
                            ancestral_cat_trait = cat_traits[t_abs, y, j]
                            cat_trait_new = self.evolve_cat_traits_clado(cat_traits_Q[y], ancestral_cat_trait, cat_states[y])
                            cat_trait_new = int(cat_trait_new)
                            cat_traits_new_species[t_abs, y] = cat_trait_new
                            # trait state for the just originated lineage
                            lineage_rates_tmp[(5 + 2 * n_cont_traits + y):(6 + 2 * n_cont_traits + y)] = cat_trait_new
                            # trait state of the ancestral lineage
                            lineage_rates_tmp[(5 + 2 * n_cont_traits + y + n_cat_traits):(6 + 2 * n_cont_traits + y + n_cat_traits)] = ancestral_cat_trait
                            l_new = l_new * cat_trait_effect[y][0, cat_trait_new]
                            m_new = m_new * cat_trait_effect[y][1, cat_trait_new]
                        cat_traits = np.dstack((cat_traits, cat_traits_new_species))
                    if n_cont_traits > 0:
                        cont_traits_new_species = self.empty_traits(root_plus_1, n_cont_traits)
                        cont_traits_at_origin = cont_traits[t_abs, :, j]
                        cont_traits_new_species[t_abs,:] = cont_traits_at_origin
                        cont_traits = np.dstack((cont_traits, cont_traits_new_species))
                        lineage_rates_tmp[5:(5 + n_cont_traits)] = cont_traits_at_origin
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

                # extinction
                if (ran > l_j and ran < (l_j + m_j) ) or t == -1:
                    if t != -1:
                        te[j] = t
                        lineage_rates[j][1] = t
                    lineage_rates[j][3] = m_j # Extinction rate at extinction time (or present for extant species)
                    if n_cont_traits > 0:
                        lineage_rates[j][(5 + n_cont_traits):(5 + 2 * n_cont_traits)] = cont_trait_j

            if t != -1:
                lineage_weighted_lambda_tt[t_abs-1] = self.get_harmonic_mean(lineage_lambda)
                lineage_weighted_mu_tt[t_abs-1] = self.get_harmonic_mean(lineage_mu)

        lineage_rates = np.array(lineage_rates)
        lineage_rates[:, 0] = -lineage_rates[:, 0] / self.scale # Why is it not working? lineage_rates[:,:2] = -lineage_rates[:,:2] / self.scale
        lineage_rates[:, 1] = -lineage_rates[:, 1] / self.scale
        lineage_rates[:, 2] = lineage_rates[:, 2] * self.scale
        lineage_rates[:, 3] = lineage_rates[:, 3] * self.scale
        lineage_rates[:, 4] = lineage_rates[:, 4] * self.scale

        return -np.array(ts) / self.scale, -np.array(te) / self.scale, anc_desc, cont_traits, cat_traits, mass_ext_time, mass_ext_mag, lineage_weighted_lambda_tt, lineage_weighted_mu_tt, lineage_rates, biogeo, areas_comb


    def get_random_settings(self, root, verbose):
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
        cont_traits_effect_sp = []
        cont_traits_effect_ex = []
        expected_sd_cont_traits = []
        cont_traits_effect_shift_sp = []
        cont_traits_effect_shift_ex = []
        if n_cont_traits > 0:
            cont_traits_varcov = self.make_cont_traits_varcov(n_cont_traits)
            cont_traits_Theta1 = self.make_cont_traits_Theta1(n_cont_traits)
            cont_traits_alpha = self.make_cont_traits_alpha(n_cont_traits, root_scaled)
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
        env_eff_sp = np.random.uniform(np.min(self.sp_env_eff), np.max(self.sp_env_eff), 1)
        env_eff_ex = np.random.uniform(np.min(self.ex_env_eff), np.max(self.ex_env_eff), 1)

        return dT, L_shifts, M_shifts, L, M, timesL, timesM, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_effect_sp, cont_traits_effect_ex, expected_sd_cont_traits, cont_traits_effect_shift_sp, cont_traits_effect_shift_ex, n_cat_traits, cat_states, cat_traits_Q, cat_trait_effect, n_areas, dispersal, extirpation, env_eff_sp, env_eff_ex


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


    def make_cont_traits_varcov(self, n_cont_traits):
        if n_cont_traits == 1:
            varcov = np.random.uniform(np.min(self.cont_traits_sigma), np.max(self.cont_traits_sigma), 1)
            varcov = np.array([varcov])
            varcov = np.sqrt(varcov / self.scale)
        else:
            sigma2 = np.random.uniform(np.min(self.cont_traits_sigma), np.max(self.cont_traits_sigma), n_cont_traits)
            sigma2 = np.diag(sigma2)
            n_cor = int(n_cont_traits * (n_cont_traits - 1) / 2)
            cor = np.random.uniform(np.min(self.cont_traits_cor), np.max(self.cont_traits_cor), n_cor)
            cormat = np.ones([n_cont_traits**2]).reshape((n_cont_traits, n_cont_traits))
            cormat[np.triu_indices(n_cont_traits, k = 1)] = cor
            cormat[np.tril_indices(n_cont_traits, k = -1)] = cor
            varcov = sigma2 @ cormat @ sigma2 # correlation to covariance for multivariate random
            varcov = self.nearestPD(varcov)
            varcov = varcov / self.scale

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
        if n_cont_traits == 1:
            # print('cont_trait_value:' , cont_trait_value)
            # print('expected_sd[0]: ', expected_sd[0])
            trait_pdf = norm.pdf(cont_trait_value, par[0, 4], par[:, 0] * expected_sd[0])[0]
        else:
            # How to scale the multivariate SD by the effect? Only the diagonals and then make it positive definite again?
            trait_pdf = multivariate_normal.pdf(cont_trait_value,
                                                mean = par[:, 4],
                                                cov = expected_sd)
        # Scale according to trait effect: ((bellu * trait_pdf - MinPDF) / (MaxPDF - MinPDF) ) * (2 * Effect) - Effect
        # cont_trait_effect = ((par[:, 1] * trait_pdf - par[:, 2]) / (par[:, 3] - par[:, 2])) * (2 * par[:, 0]) - par[:, 0]
        # cont_trait_effect = np.sum(cont_trait_effect)
        # transf_r = r * np.exp(cont_trait_effect)
        scaled_trait_pdf = (trait_pdf - par[:, 2]) / (par[:, 3] - par[:, 2])
        # rate * +/- f(delta, v)
        r_ushape = 0
        if par[:, 1] == -1:
            r_ushape = r
        transf_r = r * par[:, 1] * scaled_trait_pdf + r_ushape

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
                                                                     cov = expected_sd)

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


    def run_simulation(self, verbose = False):
        LOtrue = []
        n_extinct = 0
        n_extant = 0
        while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP or n_extant < self.minExtant_SP or n_extant > self.maxExtant_SP:
            root = -np.random.uniform(np.min(self.root_r), np.max(self.root_r))  # ROOT AGES
            dT, L_shifts, M_shifts, L, M, timesL, timesM, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_effect_sp, cont_traits_effect_ex, expected_sd_cont_traits, cont_traits_effect_shift_sp, cont_traits_effect_shift_ex, n_cat_traits, cat_states, cat_traits_Q, cat_traits_effect, n_areas, dispersal, extirpation, env_eff_sp, env_eff_ex = self.get_random_settings(root, verbose)

            L_tt, linL = self.add_linear_time_effect(L_shifts, self.range_linL, self.fixed_Ltt)
            M_tt, linM = self.add_linear_time_effect(M_shifts, self.range_linM, self.fixed_Mtt)

            # environmental effect
            sp_env = np.stack((np.linspace(0, 1.2 * np.abs(root), 1000), np.zeros(1000)), axis = 1)
            if self.sp_env_file is not None:
                sp_env = np.loadtxt(self.sp_env_file, skiprows = 1)
                time_vec = np.arange( int(np.abs(root) * self.scale) + 1 )
                sp_env_binned = get_binned_continuous_variable(sp_env, time_vec, self.scale)
                L_tt = L_tt * np.exp(env_eff_sp * sp_env_binned)
            ex_env = np.stack((np.linspace(0, 1.2 * np.abs(root), 1000), np.zeros(1000)), axis=1)
            if self.ex_env_file is not None:
                ex_env = np.loadtxt(self.ex_env_file, skiprows = 1)
                time_vec = np.arange(int(np.abs(root) * self.scale) + 1)
                ex_env_binned = get_binned_continuous_variable(ex_env, time_vec, self.scale)
                M_tt = M_tt * np.exp(env_eff_ex * ex_env_binned)

            FAtrue, LOtrue, anc_desc, cont_traits, cat_traits, mass_ext_time, mass_ext_mag, lineage_weighted_lambda_tt, lineage_weighted_mu_tt, lineage_rates, biogeo, areas_comb = self.simulate(L_tt, M_tt, root, dT, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_effect_sp, cont_traits_effect_ex, expected_sd_cont_traits, cont_traits_effect_shift_sp, cont_traits_effect_shift_ex, n_cat_traits, cat_states, cat_traits_Q, cat_traits_effect, n_areas, dispersal, extirpation)

            n_extinct = len(LOtrue[LOtrue > 0.0])
            n_extant = len(LOtrue[LOtrue == 0.0])

            if verbose:
                print('N. species', len(LOtrue))
                print('N. extant species', n_extant)
                print('Range speciation rate', np.round(np.nanmin(lineage_weighted_lambda_tt[1:-1]), 3), np.round(np.nanmax(lineage_weighted_lambda_tt[1:-1]), 3))
                print('Range extinction rate', np.round(np.nanmin(lineage_weighted_mu_tt[1:-1]), 3), np.round(np.nanmax(lineage_weighted_mu_tt[1:-1]), 3))

        ts_te = np.array([FAtrue, LOtrue]).T
        true_rates_through_time = self.get_true_rate_through_time(root, L_tt, M_tt, lineage_weighted_lambda_tt, lineage_weighted_mu_tt)
        mass_ext_time = np.abs(np.array([mass_ext_time])) / self.scale
        mass_ext_mag = np.array([mass_ext_mag])

        res_bd = {'lambda': L * self.scale,
                  'tshift_lambda': timesL / self.scale,
                  'mu': M * self.scale,
                  'tshift_mu': timesM / self.scale,
                  'true_rates_through_time': true_rates_through_time,
                  'mass_ext_time': mass_ext_time,
                  'mass_ext_magnitude': mass_ext_mag,
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
                  'env_eff_sp': env_eff_sp,
                  'sp_env': sp_env,
                  'env_eff_ex': env_eff_ex,
                  'ex_env': ex_env}
        if verbose:
            print("N. species", len(LOtrue))
            ltt = ""
            for i in range(int(max(FAtrue))):
                n = len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i])
                ltt += "\n%s\t%s\t%s" % (i, n, "*" * n)
            print(ltt)
        return res_bd



class fossil_simulator():
    def __init__(self,
                 range_q = [0.5, 5.],
                 range_alpha = [1.0, 10.0],
                 poi_shifts = 2,
                 seed = 0):
        self.range_q = range_q
        self.range_alpha = range_alpha
        self.poi_shifts = poi_shifts
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


    def get_sampling_heterogeneity(self, sp_x):
        alpha = np.random.uniform(np.min(self.range_alpha), np.max(self.range_alpha), 1)
        h = np.random.gamma(alpha, 1.0 / alpha, len(sp_x))
        return alpha, h


    def get_fossil_occurrences(self, sp_x, q, shift_time_q, is_alive):
        alpha, sampl_hetero = self.get_sampling_heterogeneity(sp_x)
        n_taxa = len(sp_x)
        occ = [np.array([])] * n_taxa
        len_q = len(q)
        for i in range(len_q):
            dur, ts, te = self.get_duration(sp_x, shift_time_q[i], shift_time_q[i + 1])
            dur = dur.flatten()
            poi_rate_occ = q[i] * sampl_hetero * dur
            exp_occ = np.round(np.random.poisson(poi_rate_occ))

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
        #occ = occ[lineages_sampled] # Cannot slice list with an array

        return occ2, lineages_sampled, alpha


    def get_taxon_names(self, lineages_sampled):
        num_taxa = len(lineages_sampled)
        taxon_names = []
        for i in range(num_taxa):
            taxon_names.append('T%s' % lineages_sampled[i])

        return taxon_names


    def run_simulation(self, sp_x):
        is_alive = self.get_is_alive(sp_x)

        # Number of rate shifts expected according to a Poisson distribution
        nS = np.random.poisson(self.poi_shifts)
        q = np.random.uniform(np.min(self.range_q), np.max(self.range_q), nS + 1)
        root = np.max(sp_x)
        shift_time_q = np.random.uniform(0, root, nS)
        shift_time_q = np.sort(shift_time_q)[::-1]
        shift_time_q = np.concatenate((np.array([root]), shift_time_q, np.zeros(1)))

        fossil_occ, taxa_sampled, alpha = self.get_fossil_occurrences(sp_x, q, shift_time_q, is_alive)
        taxon_names = self.get_taxon_names(taxa_sampled)
        shift_time_q = shift_time_q[1:-1]

        d = {'fossil_occurrences': fossil_occ,
             'taxon_names': taxon_names,
             'taxa_sampled': taxa_sampled,
             'q': q,
             'shift_time': shift_time_q,
             'alpha': alpha}

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
        np.savetxt(file_q_epochs, sim_fossil['shift_time'], delimiter='\t')


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
        cont_traits -= np.mean(cont_traits, axis = 0)
        cont_traits /= np.std(cont_traits, axis = 0)

        return cont_traits


    def get_majority_cat_trait_per_taxon(self, sim_fossil, res_bd):
        cat_traits = res_bd['cat_traits']
        n_cat_traits = cat_traits.shape[1]
        taxa_sampled = sim_fossil['taxa_sampled']
        n_taxa_sampled = len(taxa_sampled)
        maj_cat_traits = np.zeros(n_taxa_sampled * n_cat_traits, dtype = int).reshape((n_taxa_sampled, n_cat_traits))
        for i in range(n_cat_traits):
            cat_traits_i = cat_traits[:, i, taxa_sampled]
            maj_cat_traits_i = mode(cat_traits_i, nan_policy='omit')[0][0]
            maj_cat_traits_i = maj_cat_traits_i.compressed()
            maj_cat_traits[:,i] = maj_cat_traits_i.astype(int)

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
        np.savetxt(file_time, time_vector, delimiter='\t')


    def write_true_rates_through_time(self, rates, name_file):
        rate_file = "%s/%s/%s_true_rates_through_time.csv" % (self.output_wd, name_file, name_file)
        rates.to_csv(rate_file, header = True, sep = '\t', index = False, na_rep = 'NA')


    def get_sampling_rates_through_time(self, sim_fossil, res_bd):
        q = sim_fossil['q']
        time_sampling = res_bd['true_rates_through_time']['time'].to_numpy()
        time_sampling = time_sampling[::-1]
        shift_time = np.concatenate(( np.array([time_sampling[0] + 0.01]), sim_fossil['shift_time'], np.zeros(1)))
        n_shifts = len(sim_fossil['shift_time'])

        q_tt = np.zeros(len(time_sampling), dtype = 'float')

        for i in range(n_shifts + 1):
            qidx = np.logical_and(time_sampling < shift_time[i], time_sampling >= shift_time[i + 1])
            q_tt[qidx] = q[i]

        return q_tt[::-1]


    def write_lineage_rates(self, sim_fossil, res_bd, name_file):
        taxa_sampled = sim_fossil['taxa_sampled']
        taxon_names = sim_fossil['taxon_names']
        lineage_rate = res_bd['lineage_rates']
        lineage_rate = lineage_rate[taxa_sampled,:]
        names_df = pd.DataFrame(data = taxon_names, columns=['scientificName'])
        colnames = ['ts', 'te', 'speciation', 'extinction', 'ancestral_speciation']
        if res_bd['cont_traits'] is not None:
            n_cont_traits = res_bd['cont_traits'].shape[1]
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
            np.savetxt(sd_traits_name, res_bd['expected_sd_cont_traits'], delimiter='\t')


    def bin_and_write_env(self, env_var, env_file_name):
        max_age_env = np.max(env_var[:, 0])
        time_vec = np.arange(0, max_age_env + self.delta_time, self.delta_time)
        binned_env = get_binned_continuous_variable(env_var, time_vec, 1.0)
        binned_env = np.stack((time_vec[:-1], binned_env), axis = 1)
        np.savetxt(env_file_name, binned_env, delimiter = '\t')


    def run_writter(self, sim_fossil, res_bd):
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

        traits = pd.DataFrame(data = sim_fossil['taxon_names'], columns = ['scientificName'])

        if res_bd['cont_traits'].shape[1] > 0:
            #mean_cont_traits_taxon = self.get_mean_cont_traits_per_taxon(sim_fossil, res_bd)
            mean_cont_traits_taxon = self.get_mean_cont_traits_per_taxon_from_sampling_events(sim_fossil, res_bd)
            mean_cont_traits_taxon = self.center_and_scale_unitvar(mean_cont_traits_taxon)
            for i in range(mean_cont_traits_taxon.shape[1]):
                traits['cont_trait_%s' % i] = mean_cont_traits_taxon[:,i]

        if res_bd['cat_traits'].shape[1] > 0:
            maj_cat_traits_taxon = self.get_majority_cat_trait_per_taxon(sim_fossil, res_bd)
            for y in range(maj_cat_traits_taxon.shape[1]):
                is_ordinal = self.is_ordinal_trait(res_bd['cat_traits_Q'][y])
                if is_ordinal:
                    traits['cat_trait_%s' % y] = maj_cat_traits_taxon[:,y]
                else:
                    cat_traits_taxon_one_hot, names_one_hot = self.make_one_hot_encoding(maj_cat_traits_taxon[:,y])
                    for i in range(cat_traits_taxon_one_hot.shape[1]):
                        traits['cat_trait_%s_%s' % (y, names_one_hot[i])] = cat_traits_taxon_one_hot[:, i]

        if traits.shape[1] > 1:
            trait_file = "%s/%s/%s_traits.csv" % (self.output_wd, name_file, name_file)
            traits.to_csv(trait_file, header = True, sep = '\t', index = False)

        self.write_time_vector(res_bd, name_file)

        qtt = self.get_sampling_rates_through_time(sim_fossil, res_bd)
        rates = res_bd['true_rates_through_time']
        rates['sampling'] = qtt
        self.write_true_rates_through_time(rates, name_file)

        self.write_lineage_rates(sim_fossil, res_bd, name_file)

        self.write_true_tste(res_bd, sim_fossil, name_file)

        env_sp_name = "%s/%s/%s_env_sp.csv" % (self.output_wd, name_file, name_file)
        self.bin_and_write_env(res_bd['sp_env'], env_sp_name)
        env_ex_name = "%s/%s/%s_env_ex.csv" % (self.output_wd, name_file, name_file)
        self.bin_and_write_env(res_bd['ex_env'], env_ex_name)

        self.write_cont_trait_effects(res_bd, name_file)

        return name_file


# Bin environmental variable into time time-bins.
# Needed to simulate diversification and to write files for analyses.
def get_binned_continuous_variable(env_var, time_vec, scaletime):
    times = env_var[:, 0]
    times = times * scaletime
    values = env_var[:, 1]
    mean_var = np.zeros(len(time_vec) - 1)
    mean_var[:] = np.nan
    for i in range(len(time_vec) - 1):
        t_min = time_vec[i]
        t_max = time_vec[i + 1]
        in_range_M = (times <= t_max).nonzero()[0]
        in_range_m = (times >= t_min).nonzero()[0]
        mean_var[i] = np.mean(values[np.intersect1d(in_range_m, in_range_M)])

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
            if len(occ_i) > 0:
                if is_alive and keep_extant:
                    occ_i = np.concatenate((occ_i, np.zeros(1)))
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
        if self.translate is not None or min_age > 0.0:
            rho = 0.0
        if self.FBD_rate_prior == 'HSMRF' and self.fix_fake_bin is False and self.padding[0] == np.inf and self.padding[1] == 0.0:
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
            scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation, mu = extinction, psi = psi, rho = rho, timeline = timeline, k = k)')
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
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_ts_te.log", bd, printgen = 50))' % self.name_file)
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
            scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation, mu = extinction, psi = psi2, rho = rho, timeline = timeline, k = k)')
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
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_ts_te.log", bd, printgen = 50))' % self.name_file)
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
        elif self.FBD_rate_prior == 'HSMRF'  and self.fix_fake_bin is True and self.padding[0] != np.inf and self.padding[1] != 0.0:
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
            scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation2, mu = extinction2, psi = psi2, rho = rho, timeline = timeline, k = k)')
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
            scrfile.write('monitors.append(mnFile(filename = "output/%s_model1_HSMRF_ts_te.log", bd, printgen = 50))' % self.name_file)
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
            # scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation, mu = extinction, psi = psi, rho = rho, timeline = timeline, k = k)')
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
            scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = speciation, mu = extinction, psi = psi, rho = rho, timeline = timeline, k = k)')
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

