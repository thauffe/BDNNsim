import sys
from numpy import linalg as la
from scipy.stats import mode
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
                 #cont_traits_varcov = None, # variance-covariance matrix for evolving continuous traits
                 n_cont_traits = [2, 5], # number of continuous traits
                 cont_traits_sigma = [0.1, 0.5], # evolutionary rates for continuous traits
                 cont_traits_cor = [-1, 1], # evolutionary correlation between continuous traits
                 cont_traits_Theta1 = [0, 0],  # morphological optima; 0 is no directional change from the ancestral values
                 cont_traits_alpha = [0, 0],  # strength of attraction towards Theta1; 0 is pure Brownian motion; [0.5, 2.0] is sensible
                 n_cat_traits = [1,2], # number of categorical traits
                 n_cat_traits_states = [2, 5], # range number of states for categorical trait, can be set to [0,0] to avid any trait
                 cat_traits_ordinal = [True, False], # is categorical trait ordinal or discrete?
                 cat_traits_dir = 1, # concentration parameter dirichlet distribution for transition probabilities between categorical states
                 seed = 0):
        self.s_species = s_species
        self.rangeSP = rangeSP
        self.minSP = np.min(rangeSP)
        self.maxSP = np.max(rangeSP)
        self.minEX_SP = minEX_SP
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
        self.n_cat_traits = n_cat_traits
        self.n_cat_traits_states = n_cat_traits_states
        self.cat_traits_ordinal = cat_traits_ordinal
        self.cat_traits_dir = cat_traits_dir
        self.s_species = s_species
        if seed:
            np.random.seed(seed)


    def simulate(self, L, M, root, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, n_cat_traits, cat_states, cat_traits_Q):
        ts = list()
        te = list()

        root = int(root * self.scale)
        dT = 1.0 / self.scale

        # Trace ancestor descendant relationship
        # First entry: ancestor (for the seeding specis, this is an index of themselfs)
        # Following entries: descendants
        anc_desc = []

        for i in range(self.s_species):
            ts.append(root)
            te.append(0)
            anc_desc.append(np.array([i]))

        # init continuous traits (if there are any to simulate)
        root_plus_1 = np.abs(root) + 2

        if n_cont_traits > 0:
            if n_cont_traits == 0:
                cont_traits_varcov = np.sqrt(cont_traits_varcov + 0.0) # standard deviation to variance
            cont_traits_varcov = dT * (cont_traits_varcov + 0.0)

        cont_traits = np.empty((root_plus_1, n_cont_traits, self.s_species))
        cont_traits[:] = np.nan
        if n_cont_traits > 0:
            for i in range(self.s_species):
                Theta0 = np.zeros(n_cont_traits)
                cont_traits[-1,:,i] = self.evolve_cont_traits(Theta0, n_cont_traits, cont_traits_alpha, cont_traits_Theta1, cont_traits_varcov) # from past to present

        # init single categorical trait
        cat_traits = np.empty((root_plus_1, n_cat_traits, self.s_species))
        cat_traits[:] = np.nan
        if n_cat_traits > 0:
            for y in range(n_cat_traits):
                #cat_traits_Q[y] = dT * cat_traits_Q[y]  # Only for anagenetic evolution of categorical traits
                pi = self.get_stationary_distribution(cat_traits_Q[y])
                for i in range(self.s_species):
                    cat_traits[-1,y,i] = np.random.choice(cat_states[y], 1, p = pi)

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

            no = np.random.uniform(0, 1)  # draw a random number
            no_extant_lineages = len(te_extant)  # the number of currently extant species
            mass_extinction_prob = self.p_mass_extinction/self.scale
            if no < mass_extinction_prob and no_extant_lineages > 10:  # mass extinction condition
                # print("Mass extinction", t / self.scale)
                # increased loss of species: increased ext probability for this time bin
                m = np.random.uniform(self.magnitude_mass_ext[0], self.magnitude_mass_ext[1])

            for j in te_extant:  # extant lineages
                # continuous trait evolution
                if n_cont_traits > 0:
                    cont_traits[t_abs, :, j] = self.evolve_cont_traits(cont_traits[t_abs + 1, :, j], n_cont_traits, cont_traits_alpha, cont_traits_Theta1, cont_traits_varcov)

                # categorical trait evolution
                if n_cat_traits > 0:
                    for y in range(n_cat_traits):
                        #cat_traits[t_abs, y, j] = self.evolve_cat_traits_ana(cat_traits_Q[y], cat_traits[t_abs + 1, y, j], ran_vec_cat_trait[j], cat_states[y])
                        cat_traits[t_abs, y, j] = cat_traits[t_abs + 1, y, j] # No change along branches

                ran = ran_vec[j]
                # speciation
                if ran < l:
                    te.append(0)  # add species
                    ts.append(t)  # sp time
                    desc = np.array([len(ts)])
                    anc_desc[j] = np.concatenate((anc_desc[j], desc))
                    anc = np.random.choice(anc_desc[j], 1)  # If a lineage already has multiple descendents
                    anc_desc.append(anc)
                    # Inherit traits
                    if n_cont_traits > 0:
                        cont_traits_new_species = self.empty_traits(root_plus_1, n_cont_traits)
                        cont_traits_new_species[t_abs,:] = cont_traits[t_abs,:,j]
                        cont_traits = np.dstack((cont_traits, cont_traits_new_species))
                    if n_cat_traits > 0:
                        cat_traits_new_species = self.empty_traits(root_plus_1, n_cat_traits)
                        # cat_traits_new_species[t_abs,] = cat_traits[t_abs,:,j] # inherit state at speciation
                        for y in range(n_cat_traits):
                            # Change of categorical trait at speciation
                            cat_traits_new_species[t_abs, y] = self.evolve_cat_traits_clado(cat_traits_Q[y], cat_traits[t_abs,y,j], cat_states[y])
                        cat_traits = np.dstack((cat_traits, cat_traits_new_species))
                # extinction
                elif ran > l and ran < (l + m):
                    te[j] = t


        return -np.array(ts) / self.scale, -np.array(te) / self.scale, anc_desc, cont_traits, cat_traits


    def get_random_settings(self, root):
        root = np.abs(root)
        root_scaled = int(root * self.scale)
        timesL_temp = [root_scaled, 0.]
        timesM_temp = [root_scaled, 0.]

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
        if n_cont_traits > 0:
            cont_traits_varcov = self.make_cont_traits_varcov(n_cont_traits)
            cont_traits_Theta1 = self.make_cont_traits_Theta1(n_cont_traits)
            cont_traits_alpha = self.make_cont_traits_alpha(n_cont_traits, root_scaled)

        # categorical traits
        n_cat_traits = int(np.random.choice(np.arange(min(self.n_cat_traits), max(self.n_cat_traits) + 1), 1))
        cat_traits_Q = []
        n_cat_traits_states = np.zeros(n_cat_traits, dtype = int)
        cat_states = []
        for i in range(n_cat_traits):
            n_cat_traits_states[i] = np.random.choice(np.arange(min(self.n_cat_traits_states), max(self.n_cat_traits_states) + 1), 1)
            Qi = self.make_cat_traits_Q(n_cat_traits_states[i])
            cat_traits_Q.append(Qi)
            cat_states_i = np.arange(n_cat_traits_states[i])
            cat_states.append(cat_states_i)

        return L_shifts, M_shifts, L, M, timesL, timesM, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, n_cat_traits, cat_states, cat_traits_Q


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
            cont_traits = cont_traits + cont_traits_alpha * (cont_traits_Theta1 - cont_traits) + np.random.normal(0.0, cont_traits_varcov, 1)
        elif n_cont_traits > 1:
            cont_traits = cont_traits + cont_traits_alpha * (cont_traits_Theta1 - cont_traits) + np.random.multivariate_normal(np.zeros(n_cont_traits), cont_traits_varcov, 1)

        return cont_traits


    def evolve_cat_traits_ana(self, Q, s, ran, cat_states):
        s = int(s)
        state = s
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


    def run_simulation(self, verbose = False):
        LOtrue = [0]
        n_extinct = -0
        while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP:
            root = -np.random.uniform(np.min(self.root_r), np.max(self.root_r))  # ROOT AGES
            L_shifts, M_shifts, L, M, timesL, timesM, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, n_cat_traits, cat_states, cat_traits_Q = self.get_random_settings(root)
            L_tt, M_tt, linL, linM = self.add_linear_time_effect(L_shifts, M_shifts)

            FAtrue, LOtrue, anc_desc, cont_traits, cat_traits = self.simulate(L_tt, M_tt, root, n_cont_traits, cont_traits_varcov, cont_traits_Theta1, cont_traits_alpha, n_cat_traits, cat_states, cat_traits_Q)

            n_extinct = len(LOtrue[LOtrue > 0])

            if verbose:
                print('N. species', len(LOtrue))
                print('Range speciation rate', np.round(np.min(L_tt), 3), np.round(np.max(L_tt), 3))
                print('Range extinction rate', np.round(np.min(M_tt), 3), np.round(np.max(M_tt), 3))

        ts_te = np.array([FAtrue, LOtrue]).T

        res_bd = {'lambda': L,
                  'tshift_lambda': timesL / self.scale,
                  'mu': M,
                  'tshift_mu': timesM / self.scale,
                  'linear_time_lambda': linL,
                  'linear_time_mu': linM,
                  'N_species': len(LOtrue),
                  'anc_desc': anc_desc,
                  'cont_traits': cont_traits,
                  'cat_traits': cat_traits,
                  'ts_te': ts_te,
                  'cat_traits_Q': cat_traits_Q}
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
                if is_alive[y] and y == len_q:
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
                 delta_time = 1.0):
        self.output_wd = output_wd
        self.delta_time = delta_time


    def write_occurrences(self, sim_fossil, name_file):
        fossil_occ = sim_fossil['fossil_occurrences']
        taxon_names = sim_fossil['taxon_names']
        py = "%s/%s.py" % (self.output_wd, name_file)
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
        file_q_epochs = '%s/%s_q_epochs.txt' % (self.output_wd, name_file)
        np.savetxt(file_q_epochs, sim_fossil['shift_time'], delimiter='\t')


    def get_mean_cont_traits_per_taxon(self, sim_fossil, res_bd):
        cont_traits = res_bd['cont_traits']
        cont_traits = cont_traits[:,:,sim_fossil['taxa_sampled']]
        means_cont_traits = np.nanmean(cont_traits, axis = 0)
        means_cont_traits = means_cont_traits.transpose()

        return means_cont_traits


    def center_and_scale_unitvar(self, cont_traits):
        cont_traits -= np.mean(cont_traits, axis = 0)
        cont_traits /= np.std(cont_traits, axis = 0)

        return cont_traits


    def get_majority_cat_trait_per_taxon(self, sim_fossil, res_bd):
        cat_traits = res_bd['cat_traits']
        n_cat_traits = cat_traits.shape[1]
        n_taxa_sampled = len(sim_fossil['taxa_sampled'])
        maj_cat_traits = np.zeros(n_taxa_sampled * n_cat_traits, dtype = int).reshape((n_taxa_sampled, n_cat_traits))
        for i in range(n_cat_traits):
            cat_traits_i = cat_traits[:, i, sim_fossil['taxa_sampled']]
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
        time_vector = np.arange(0.0, root_age, self.delta_time) # Should time include 0 ?

        return time_vector


    def write_time_vector(self, res_bd, name_file):
        time_vector = self.make_time_vector(res_bd)
        file_time = '%s/%s_time.txt' % (self.output_wd, name_file)
        np.savetxt(file_time, time_vector, delimiter='\t')


    def run_writter(self, sim_fossil, res_bd):
        name_file = ''.join(random.choices(string.ascii_lowercase, k=10))

        self.write_occurrences(sim_fossil, name_file)
        self.write_q_epochs(sim_fossil, name_file)

        traits = pd.DataFrame(data = sim_fossil['taxon_names'], columns = ['scientificName'])

        if res_bd['cont_traits'] is not None:
            mean_cont_traits_taxon = self.get_mean_cont_traits_per_taxon(sim_fossil, res_bd)
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
            trait_file = "%s/%s_traits.csv" % (self.output_wd, name_file)
            traits.to_csv(trait_file, header = True, sep = '\t', index = False)

        self.write_time_vector(res_bd, name_file)

        return name_file



