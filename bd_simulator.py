import sys
import os
from numpy import linalg as la
from scipy.stats import mode
from scipy.stats import norm
from itertools import combinations
from functools import reduce
from operator import iconcat
from math import comb
import numpy as np
import pandas as pd
import scipy.linalg
import random
import string
np.set_printoptions(suppress=True, precision=3)
from collections.abc import Iterable
#from .extract_properties import *
SMALL_NUMBER = 1e-10


class bdnn_simulator():
    def __init__(self,
                 s_species = 1,  # number of starting species
                 rangeSP = [100, 1000],  # min/max size data set
                 minEX_SP = 0,  # minimum number of extinct lineages allowed
                 minExtant_SP = 0, # minimum number of extant lineages
                 root_r = [30., 100],  # range root ages
                 rangeL = [0.2, 0.5], # range speciation rate
                 rangeM = [0.2, 0.5], # range extinction rate
                 scale = 100., # root * scale = steps for the simulation
                 p_mass_extinction = 0.00924,
                 magnitude_mass_ext = [0.8, 0.95],
                 poiL = 3, # Number of rate shifts expected according to a Poisson distribution
                 poiM = 3, # Number of rate shifts expected according to a Poisson distribution
                 range_linL = None, # Or a range (e.g. [-0.2, 0.2])
                 range_linM = None, # Or a range (e.g. [-0.2, 0.2])
                 n_cont_traits = [2, 5], # number of continuous traits
                 cont_traits_sigma = [0.1, 0.5], # evolutionary rates for continuous traits
                 cont_traits_cor = [-1, 1], # evolutionary correlation between continuous traits
                 cont_traits_Theta1 = [0, 0],  # morphological optima; 0 is no directional change from the ancestral values
                 cont_traits_alpha = [0, 0],  # strength of attraction towards Theta1; 0 is pure Brownian motion; [0.5, 2.0] is sensible
                 cont_traits_effect = [0., 0.], # range of effect of categorical traits on speciation and extinction (1 is no effect)
                 n_cat_traits = [1, 2], # number of categorical traits
                 n_cat_traits_states = [2, 5], # range number of states for categorical trait, can be set to [0,0] to avid any trait
                 cat_traits_ordinal = [True, False], # is categorical trait ordinal or discrete?
                 cat_traits_dir = 1, # concentration parameter dirichlet distribution for transition probabilities between categorical states
                 cat_traits_effect = [1., 1.], # range of effect of categorical traits on speciation and extinction (1 is no effect)
                 n_areas = [0, 0], # number of biogeographic areas (minimum of 2)
                 dispersal = [0.1, 0.3], # range for the rate of area expansion
                 extirpation = [0.05, 0.1], # range for the rate of area loss
                 seed = 0):
        self.s_species = s_species
        self.rangeSP = rangeSP
        self.minSP = np.min(rangeSP)
        self.maxSP = np.max(rangeSP)
        self.minEX_SP = minEX_SP
        self.minExtant_SP = minExtant_SP
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
        self.n_cont_traits = n_cont_traits
        self.cont_traits_sigma = cont_traits_sigma
        self.cont_traits_cor = cont_traits_cor
        self.cont_traits_Theta1 = cont_traits_Theta1
        self.cont_traits_alpha = cont_traits_alpha
        self.cont_traits_effect = cont_traits_effect
        self.n_cat_traits = n_cat_traits
        self.n_cat_traits_states = n_cat_traits_states
        self.cat_traits_ordinal = cat_traits_ordinal
        self.cat_traits_dir = cat_traits_dir
        self.cat_traits_effect = cat_traits_effect
        self.n_areas = n_areas
        self.dispersal = dispersal
        self.extirpation = extirpation
        if seed:
            np.random.seed(seed)


    def simulate(self, L, M, root, dT, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_trait_effect, n_cat_traits, cat_states, cat_traits_Q, cat_trait_effect, n_areas, dispersal, extirpation):
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
            lineage_rates_tmp = np.zeros(5 + n_cont_traits +  2 * n_cat_traits)
            lineage_rates_tmp[:] = np.nan
            lineage_rates_tmp[:5] = np.array([root, -0.0, L[root], M[root], 0.0])
            lineage_rates.append(lineage_rates_tmp)

        # init continuous traits (if there are any to simulate)
        root_plus_1 = np.abs(root) + 2

        # init categorical traits
        cat_traits = np.empty((root_plus_1, n_cat_traits, self.s_species))
        cat_traits[:] = np.nan

        for y in range(n_cat_traits):
            #cat_traits_Q[y] = dT * cat_traits_Q[y]  # Only for anagenetic evolution of categorical traits
            pi = self.get_stationary_distribution(cat_traits_Q[y])
            for i in range(self.s_species):
                cat_trait_yi = int(np.random.choice(cat_states[y], 1, p = pi))
                cat_traits[-1,y,i] = cat_trait_yi
                lineage_rates[i][2] = lineage_rates[i][2] * cat_trait_effect[y][0, cat_trait_yi]
                lineage_rates[i][3] = lineage_rates[i][3] * cat_trait_effect[y][1, cat_trait_yi]
                lineage_rates[i][(5 + y + n_cont_traits):(6 + y + n_cont_traits)] = cat_trait_yi
                lineage_rates[i][2] = L[root] * cat_trait_effect[y][0, int(cat_trait_yi)]
                lineage_rates[i][3] = M[root] * cat_trait_effect[y][1, int(cat_trait_yi)]

        # init continuous traits
        if n_cont_traits > 0:
            if n_cont_traits == 0:
                cont_traits_varcov = np.sqrt(cont_traits_varcov + 0.0) # standard deviation to variance
            cont_traits_varcov = dT * (cont_traits_varcov + 0.0)

        cont_traits = np.empty((root_plus_1, n_cont_traits, self.s_species))
        cont_traits[:] = np.nan
        if n_cont_traits > 0:
            for i in range(self.s_species):
                Theta0 = np.zeros(n_cont_traits)
                cont_traits_i = self.evolve_cont_traits(Theta0, n_cont_traits, cont_traits_alpha, cont_traits_Theta1, cont_traits_varcov) # from past to present
                cont_traits[-1, :, i] = cont_traits_i
                lineage_rates[i][5:(5 + n_cont_traits)] = cont_traits_i
                for y in range(n_cont_traits):
                    lineage_rates[i][2] = lineage_rates[i][2] + self.get_cont_trait_effect(cont_traits_i[y], cont_trait_effect[y][0, :])
                    lineage_rates[i][3] = lineage_rates[i][3] + self.get_cont_trait_effect(cont_traits_i[y], cont_trait_effect[y][1, :])

        # init biogeography
        biogeo = np.empty((root_plus_1, 1, self.s_species))
        biogeo[:] = np.nan
        biogeo[-1,:,:] = np.random.choice(np.arange(n_areas + 1), self.s_species)
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
            #print('t_abs', t_abs)
            l = L[t]
            m = M[t]

            TE = len(te)
            if TE > self.maxSP:
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
                for y in range(n_cat_traits):
                    #cat_trait_j = self.evolve_cat_traits_ana(cat_traits_Q[y], cat_traits[t_abs + 1, y, j], ran_vec_cat_trait[j], cat_states[y])
                    cat_trait_j = cat_traits[t_abs + 1, y, j] # No change along branches
                    cat_trait_j = int(cat_trait_j)
                    cat_traits[t_abs, y, j] = cat_trait_j
                    l_j = l_j * cat_trait_effect[y][0, cat_trait_j]
                    m_j = m_j * cat_trait_effect[y][1, cat_trait_j]

                # continuous trait evolution
                if n_cont_traits > 0:
                    cont_trait_j = self.evolve_cont_traits(cont_traits[t_abs + 1, :, j], n_cont_traits, cont_traits_alpha, cont_traits_Theta1, cont_traits_varcov)
                    cont_traits[t_abs, :, j] = cont_trait_j
                    for y in range(n_cont_traits):
                        l_j = l_j + self.get_cont_trait_effect(cont_trait_j[y], cont_trait_effect[y][0,:])
                        m_j = m_j + self.get_cont_trait_effect(cont_trait_j[y], cont_trait_effect[y][1,:])

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

                    desc = np.array([len(ts)])
                    anc_desc[j] = np.concatenate((anc_desc[j], desc))
                    anc = np.random.choice(anc_desc[j], 1)  # If a lineage already has multiple descendents
                    anc_desc.append(anc)

                    lineage_rates_tmp = np.zeros(5 + n_cont_traits + 2 * n_cat_traits)
                    l_new = l + 0.0
                    m_new = m + 0.0

                    # Inherit traits
                    if n_cat_traits > 0:
                        cat_traits_new_species = self.empty_traits(root_plus_1, n_cat_traits)
                        # cat_traits_new_species[t_abs,] = cat_traits[t_abs,:,j] # inherit state at speciation
                        for y in range(n_cat_traits):
                            # Change of categorical trait at speciation
                            ancestral_cat_trait = cat_traits[t_abs, y, j]
                            cat_trait_new = self.evolve_cat_traits_clado(cat_traits_Q[y], ancestral_cat_trait, cat_states[y])
                            cat_traits_new_species[t_abs, y] = cat_trait_new
                            # trait state for the just originated lineage
                            lineage_rates_tmp[(5 + y + n_cont_traits):(6 + y + n_cont_traits)] = cat_trait_new
                            # trait state of the ancestral lineage
                            lineage_rates_tmp[(5 + y + n_cont_traits + n_cat_traits):(6 + y + n_cont_traits + n_cat_traits)] = ancestral_cat_trait
                            l_new = l_new * cat_trait_effect[y][0, int(cat_trait_new)]
                            m_new = m_new * cat_trait_effect[y][1, int(cat_trait_new)]
                        cat_traits = np.dstack((cat_traits, cat_traits_new_species))
                    if n_cont_traits > 0:
                        cont_traits_new_species = self.empty_traits(root_plus_1, n_cont_traits)
                        cont_traits_at_origin = cont_traits[t_abs, :, j]
                        cont_traits_new_species[t_abs,:] = cont_traits_at_origin
                        cont_traits = np.dstack((cont_traits, cont_traits_new_species))
                        lineage_rates_tmp[5:(5 + n_cont_traits)] = cont_traits_at_origin
                        for y in range(n_cont_traits):
                            l_new = l_new + self.get_cont_trait_effect(cont_traits_at_origin[y], cont_trait_effect[y][0,:])
                            m_new = m_new + self.get_cont_trait_effect(cont_traits_at_origin[y], cont_trait_effect[y][1,:])
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
                elif ran > l_j and ran < (l_j + m_j):
                    te[j] = t
                    lineage_rates[j][1] = t

            if t != -1:
                lineage_weighted_lambda_tt[t_abs-1] = self.get_harmonic_mean(lineage_lambda)
                lineage_weighted_mu_tt[t_abs-1] = self.get_harmonic_mean(lineage_mu)

        lineage_rates = np.array(lineage_rates)
        lineage_rates[:, 0] = -lineage_rates[:, 0] / self.scale # Why is it not working? lineage_rates[:,:2] = -lineage_rates[:,:2] / self.scale
        lineage_rates[:, 1] = -lineage_rates[:, 1] / self.scale
        lineage_rates[:, 2] = lineage_rates[:, 2] * self.scale
        lineage_rates[:, 3] = lineage_rates[:, 3] * self.scale
        lineage_rates[:, 4] = lineage_rates[:, 4] * self.scale

        return -np.array(ts) / self.scale, -np.array(te) / self.scale, anc_desc, cont_traits, cat_traits, mass_ext_time, mass_ext_mag, lineage_weighted_lambda_tt, lineage_weighted_mu_tt, lineage_rates, biogeo


    def get_random_settings(self, root):
        root = np.abs(root)
        root_scaled = int(root * self.scale)
        timesL_temp = [root_scaled, 0.]
        timesM_temp = [root_scaled, 0.]

        dT = root / root_scaled

        # Number of rate shifts expected according to a Poisson distribution
        nL = np.random.poisson(self.poiL)
        nM = np.random.poisson(self.poiM)

        L = np.random.uniform(np.min(self.rangeL), np.max(self.rangeL), nL + 1) / self.scale
        M = np.random.uniform(np.min(self.rangeM), np.max(self.rangeM), nM + 1) / self.scale

        shift_time_L = np.random.uniform(0, root_scaled, nL)
        shift_time_M = np.random.uniform(0, root_scaled, nM)

        timesL = np.sort(np.concatenate((timesL_temp, shift_time_L), axis=0))[::-1]
        timesM = np.sort(np.concatenate((timesM_temp, shift_time_M), axis=0))[::-1]

        # Rates through (scaled) time
        L_shifts = np.zeros(root_scaled, dtype = 'float')
        M_shifts = np.zeros(root_scaled, dtype='float')
        idx_time_vec = np.arange(root_scaled)[::-1]

        for i in range(nL + 1):
            Lidx = np.logical_and(idx_time_vec <= timesL[i], idx_time_vec > timesL[i + 1])
            L_shifts[Lidx] = L[i]
        for i in range(nM + 1):
            Midx = np.logical_and(idx_time_vec <= timesM[i], idx_time_vec > timesM[i + 1])
            M_shifts[Midx] = M[i]

        # continuous traits
        n_cont_traits = np.random.choice(np.arange(min(self.n_cont_traits), max(self.n_cont_traits) + 1), 1)
        n_cont_traits = int(n_cont_traits)
        cont_trait_effect = []
        if n_cont_traits > 0:
            cont_traits_varcov = self.make_cont_traits_varcov(n_cont_traits)
            cont_traits_Theta1 = self.make_cont_traits_Theta1(n_cont_traits)
            cont_traits_alpha = self.make_cont_traits_alpha(n_cont_traits, root_scaled)
            for i in range(n_cont_traits):
                cont_trait_effect_i = self.get_cont_trait_effect_parameters(root, dT, cont_traits_varcov[i, i])
                cont_trait_effect.append(cont_trait_effect_i)

        # categorical traits
        n_cat_traits = np.random.choice(np.arange(min(self.n_cat_traits), max(self.n_cat_traits) + 1), 1)
        n_cat_traits = int(n_cat_traits)
        cat_traits_Q = []
        n_cat_traits_states = np.zeros(n_cat_traits, dtype = int)
        cat_states = []
        cat_trait_effect = []
        for i in range(n_cat_traits):
            n_cat_traits_states[i] = np.random.choice(np.arange(min(self.n_cat_traits_states), max(self.n_cat_traits_states) + 1), 1)
            Qi = self.make_cat_traits_Q(n_cat_traits_states[i])
            cat_traits_Q.append(Qi)
            cat_states_i = np.arange(n_cat_traits_states[i])
            cat_states.append(cat_states_i)
            cat_trait_effect_i = self.make_cat_trait_effect(n_cat_traits_states[i])
            cat_trait_effect.append(cat_trait_effect_i)

        # biogeography
        n_areas = np.random.choice(np.arange(min(self.n_areas), max(self.n_areas) + 1), 1)
        n_areas = int(n_areas)
        dispersal = np.zeros(1)
        extirpation = np.zeros(1)
        if n_areas > 1:
            dispersal = np.random.uniform(np.min(self.dispersal), np.max(self.dispersal), nL + 1)
            extirpation = np.random.uniform(np.min(self.extirpation), np.max(self.extirpation), nL + 1)

        return dT, L_shifts, M_shifts, L, M, timesL, timesM, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_trait_effect, n_cat_traits, cat_states, cat_traits_Q, cat_trait_effect, n_areas, dispersal, extirpation


    def add_linear_time_effect(self, L_shifts, M_shifts):
        # Effect sizes
        if self.range_linL:
            linL = np.random.uniform(np.min(self.range_linL), np.max(self.range_linL), 1)
        else:
            linL = np.zeros(1)
        if self.range_linM:
            linM = np.random.uniform(np.min(self.range_linM), np.max(self.range_linM), 1)
        else:
            linM = np.zeros(1)

        t_vec = np.linspace(-0.5, 0.5, len(L_shifts))

        L_tt = L_shifts + linL * t_vec
        M_tt = M_shifts + linM * t_vec

        L_tt[L_tt < 0.0] = 1e-10
        M_tt[M_tt < 0.0] = 1e-10

        return L_tt, M_tt, linL, linM


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
            # Not possible to vectorize; sd needs to have the same size than mean
            cont_traits = cont_traits + cont_traits_alpha * (cont_traits_Theta1 - cont_traits) + np.random.normal(0.0, cont_traits_varcov[0,0], 1)
        elif n_cont_traits > 1:
            cont_traits = cont_traits + cont_traits_alpha * (cont_traits_Theta1 - cont_traits) + np.random.multivariate_normal(np.zeros(n_cont_traits), cont_traits_varcov, 1)
            cont_traits = cont_traits[0]

        return cont_traits


    # def get_cont_trait_effect(self, root, sigma2, cont_trait_value, effect):
    #     m = np.abs(root)
    #     max_pdf = norm.pdf(0.0, 0.0, m * sigma2)
    #     cont_trait_effect = ((norm.pdf(cont_trait_value, 0, m * sigma2) * -1) + max_pdf) * effect
    #
    #     return cont_trait_effect


    def get_cont_trait_effect(self, cont_trait_value, par):
        effect = ((norm.pdf(cont_trait_value, 0.0, par[1]) * par[2]) + par[3]) * par[0]

        return effect


    def get_cont_trait_effect_parameters(self, root, dT, sigma2):
        effect_par = np.zeros((2, 4)) # rows: sp/ext cols: trait effect, expected SD, u/-bell-shape, max_pdf
        eff = np.random.uniform(np.min(self.cont_traits_effect), np.max(self.cont_traits_effect), 2)
        bell_shape = np.random.choice([True, False], 2) # np.array([False, True]) works too
        effect_par[:,0] = eff
        effect_par[:,1] = np.sqrt(root * (dT * sigma2**2))
        ub = np.zeros(2) - 1
        ub[bell_shape] = 1
        effect_par[:, 2] = ub
        max_pdf = norm.pdf(np.zeros(2), 0.0, effect_par[0,1])
        max_pdf[bell_shape] = 0.0
        effect_par[:, 3] = max_pdf

        return effect_par


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
        cat_trait_effect[0, :] = np.random.uniform(self.cat_traits_effect[0], self.cat_traits_effect[1], n_states) # effect on speciation
        cat_trait_effect[1, :] = np.random.uniform(self.cat_traits_effect[0], self.cat_traits_effect[1], n_states) # effect on extinction
        id = np.random.choice([True, False], n_states)
        cat_trait_effect[0, id] = 1.0 / cat_trait_effect[0, id]
        id = np.random.choice([True, False], n_states)
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
            print('range copying')
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
        while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP or n_extant < self.minExtant_SP:
            root = -np.random.uniform(np.min(self.root_r), np.max(self.root_r))  # ROOT AGES
            dT, L_shifts, M_shifts, L, M, timesL, timesM, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_effect, n_cat_traits, cat_states, cat_traits_Q, cat_traits_effect, n_areas, dispersal, extirpation = self.get_random_settings(root)
            L_tt, M_tt, linL, linM = self.add_linear_time_effect(L_shifts, M_shifts)

            FAtrue, LOtrue, anc_desc, cont_traits, cat_traits, mass_ext_time, mass_ext_mag, lineage_weighted_lambda_tt, lineage_weighted_mu_tt, lineage_rates, biogeo = self.simulate(L_tt, M_tt, root, dT, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, cont_traits_effect, n_cat_traits, cat_states, cat_traits_Q, cat_traits_effect, n_areas, dispersal, extirpation)

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
                  'cont_traits_effect': cont_traits_effect,
                  'cat_traits_Q': cat_traits_Q,
                  'cat_traits_effect': cat_traits_effect,
                  'biogeography': biogeo}
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


    def write_lineage_rates(self, res_bd, name_file):
        taxa_sampled = sim_fossil['taxa_sampled']
        taxon_names = sim_fossil['taxon_names']
        lineage_rate = res_bd['lineage_rates']
        lineage_rate = lineage_rate[taxa_sampled,:]
        names_df = pd.DataFrame(data = taxon_names, columns=['scientificName'])
        colnames = ['ts', 'te', 'speciation', 'extinction', 'ancestral_speciation']
        if res_bd['cont_traits'] is not None:
            n_cont_traits = res_bd['cont_traits'].shape[1]
            for y in range(n_cont_traits):
                colnames.append('cont_trait_%s' % y)
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


    def run_writter(self, sim_fossil, res_bd):
        # Get random name and create a subdirectory
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

        if res_bd['cont_traits'] is not None:
            #mean_cont_traits_taxon = self.get_mean_cont_traits_per_taxon(sim_fossil, res_bd)
            mean_cont_traits_taxon = self.get_mean_cont_traits_per_taxon_from_sampling_events(sim_fossil, res_bd)
            mean_cont_traits_taxon = self.center_and_scale_unitvar(mean_cont_traits_taxon)
            for i in range(mean_cont_traits_taxon.shape[1]):
                traits['cont_trait_%s' % i] = mean_cont_traits_taxon[:,i]

        if res_bd['cat_traits'] is not None:
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

        self.write_lineage_rates(res_bd, name_file)

        return name_file



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
                 interval_ages = np.array([[np.inf, 0.0]])):
        self.output_wd = output_wd
        self.name_file = name_file
        self.interval_ages = interval_ages

    def get_ranges(self, fossils):
        taxon_names = fossils['taxon_names']
        n_lineages = len(taxon_names)
        ranges = pd.DataFrame(data = taxon_names, columns = ['taxon'])
        ranges['min'] = np.zeros(n_lineages)
        ranges['max'] = np.zeros(n_lineages)
        occ = fossils['fossil_occurrences']
        for i in range(n_lineages):
            occ_i = occ[i]
            ranges.iloc[i, 1] = np.min(occ_i)
            ranges.iloc[i, 2] = np.max(occ_i)

        return ranges


    def get_occurrences_per_interval(self, fossils):
        taxon_names = fossils['taxon_names']
        n_lineages = len(taxon_names)
        names_df = pd.DataFrame(data = taxon_names, columns = ['taxon'])
        n_intervals = self.interval_ages.shape[0]
        counts = np.zeros((n_lineages, n_intervals), dtype = int)
        occ = fossils['fossil_occurrences']
        for i in range(n_lineages):
            for y in range(n_intervals):
                occ_i = occ[i]
                # What to do with a record at the present? Is this rho for FBD?
                occ_i_y = occ_i[np.logical_and(occ_i <= self.interval_ages[y, 0], occ_i > self.interval_ages[y, 1])]
                counts[i, y] = len(occ_i_y)

        counts_df = pd.DataFrame(data = counts, columns = np.arange(n_intervals))
        counts_df = pd.concat([names_df, counts_df], axis=1)

        return counts_df


    def write_script(self, name_file):
        scr = "%s/%s/%s/%s/%s_mcmc_FBDRMatrix_model1.Rev" % (self.output_wd, name_file, 'FBD', 'scripts', name_file)
        scrfile = open(scr, "w")
        scrfile.write('taxa = readTaxonData(file = "data/%s_ranges.csv")' % name_file)
        scrfile.write('\n')
        scrfile.write('k <- readDataDelimitedFile(file = "data/%s_counts.csv", header = true, rownames = true)' % name_file)
        scrfile.write('\n')
        if self.interval_ages.shape[0] > 1: # Skyline model
            timeline = self.interval_ages[1:,0]
            timeline = timeline.tolist()
            scrfile.write('timeline_size <- timeline.size()')
            scrfile.write('timeline <- v(')
            for i in range(len(timeline)):
                scrfile.write(str(timeline[i]))
                if i != 0 or i != (len(timeline) - 1):
                    scrfile.write(',')
            scrfile.write(')')
        else:
            scrfile.write('timeline_size <- 0')
        scrfile.write('\n')
        scrfile.write('moves = VectorMoves()')
        scrfile.write('\n')
        scrfile.write('monitors = VectorMonitors()')
        scrfile.write('\n')
        scrfile.write('alpha <- 10')
        scrfile.write('\n')
        scrfile.write('for (i in 1:(timeline_size+1)) {')
        scrfile.write('\n')
        scrfile.write('    mu[i] ~ dnExp(alpha)')
        scrfile.write('\n')
        scrfile.write('    lambda [i] ~ dnExp(alpha)')
        scrfile.write('\n')
        scrfile.write('    psi[i] ~ dnExp(alpha)')
        scrfile.write('\n')
        scrfile.write('    div[i] := lambda[i] - mu[i]')
        scrfile.write('\n')
        scrfile.write('    turnover[i] := mu[i] / lambda[i]')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale(mu[i], lambda = 0.01) )')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale(mu[i], lambda = 0.1) )')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale(mu[i], lambda = 1) )')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale( lambda[i], lambda = 0.01) )')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale( lambda[i], lambda = 0.1) )')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale( lambda[i], lambda = 1) )')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale(psi[i], lambda = 0.01) )')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale(psi[i], lambda = 0.1) )')
        scrfile.write('\n')
        scrfile.write('    moves.append( mvScale(psi[i], lambda = 1) )')
        scrfile.write('\n')
        scrfile.write('}')
        scrfile.write('\n')
        scrfile.write('rho <- 1')
        scrfile.write('\n')
        scrfile.write('bd ~ dnFBDRMatrix(taxa=taxa, lambda = lambda, mu=mu, psi=psi, rho=rho, timeline=timeline, k=k)')
        scrfile.write('\n')
        scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.01, weight=taxa.size()))')
        scrfile.write('\n')
        scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 0.1, weight=taxa.size()))')
        scrfile.write('\n')
        scrfile.write('moves.append(mvMatrixElementScale(bd, lambda = 1, weight=taxa.size()))')
        scrfile.write('\n')
        scrfile.write('moves.append(mvMatrixElementSlide(bd, delta=0.01, weight=taxa.size()))')
        scrfile.write('\n')
        scrfile.write('moves.append(mvMatrixElementSlide(bd, delta=0.1, weight=taxa.size()))')
        scrfile.write('\n')
        scrfile.write('moves.append(mvMatrixElementSlide(bd, delta=1, weight=taxa.size()))')
        scrfile.write('\n')
        scrfile.write('mymodel = model(bd)')
        scrfile.write('\n')
        scrfile.write('monitors.append(mnScreen(lambda , mu, psi, printgen=100))')
        scrfile.write('\n')
        scrfile.write('monitors.append(mnModel(filename="output/%s_model1.log", printgen=100))' % name_file)
        scrfile.write('\n')
        scrfile.write('monitors.append(mnFile(filename="output/%s_model1_speciation_rates.log", lambda , printgen=10))' % name_file)
        scrfile.write('\n')
        scrfile.write('monitors.append(mnFile(filename="output/%s_model1_speciation_times.log", timeline, printgen=10))' % name_file)
        scrfile.write('\n')
        scrfile.write('monitors.append(mnFile(filename="output/%s_model1_extinction_rates.log", mu, printgen=10))' % name_file)
        scrfile.write('\n')
        scrfile.write('monitors.append(mnFile(filename="output/%s_model1_extinction_times.log", timeline, printgen=10))' % name_file)
        scrfile.write('\n')
        scrfile.write('monitors.append(mnFile(filename="output/%s_model1_sampling_rates.log", psi, printgen=10))' % name_file)
        scrfile.write('\n')
        scrfile.write('monitors.append(mnFile(filename="output/%s_model1_sampling_times.log", timeline, printgen=10))' % name_file)
        scrfile.write('\n')
        scrfile.write('mymcmc = mcmc(mymodel, moves, monitors, moveschedule="random")')
        scrfile.write('\n')
        scrfile.write('mymcmc.run(30000)')
        scrfile.write('\n')
        scrfile.write('q()')
        scrfile.flush()



    def run_FBD_writter(self, fossils):
        path_dir = os.path.join(self.output_wd, name_file)

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

        ranges = self.get_ranges(fossils)
        ranges_file = "%s/%s/%s/%s/%s_ranges.csv" % (self.output_wd, name_file, 'FBD', 'data', name_file)
        ranges.to_csv(ranges_file, header = True, sep = '\t', index = False)

        counts = self.get_occurrences_per_interval(fossils)
        counts_file = "%s/%s/%s/%s/%s_counts.csv" % (self.output_wd, name_file, 'FBD', 'data', name_file)
        counts.to_csv(counts_file, header = True, sep = '\t', index = False)

        self.write_script(name_file)
