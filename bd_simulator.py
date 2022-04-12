import sys
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
from collections.abc import Iterable
#from .extract_properties import *
SMALL_NUMBER = 1e-10


class bd_simulator():
    def __init__(self,
                 s_species = 1,  # number of starting species
                 rangeSP = [100, 1000],  # min/max size data set
                 minEX_SP = 0,  # minimum number of extinct lineages allowed
                 root_r = [30., 100],  # range root ages
                 rangeL = [0.2, 0.5],
                 rangeM = [0.2, 0.5],
                 scale = 100., # root * scale = steps for the simulation
                 p_mass_extinction = 0.00924,
                 magnitude_mass_ext = [0.8, 0.95],
                 poiL = 3, # Number of rate shifts expected according to a Poisson distribution
                 poiM = 3, # Number of rate shifts expected according to a Poisson distribution
                 range_linL = None, # Or a range (e.g. [-0.2, 0.2])
                 range_linM = None, # Or a range (e.g. [-0.2, 0.2])
                 cont_traits_varcov = None, # variance-covariance matrix for evolving continuous traits
                 cont_traits_Theta1 = None, # morphological optima
                 cont_traits_alpha = 0.0, # strength of attraction towards Theta1
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
        self.cont_traits_varcov = cont_traits_varcov
        self.cont_traits_Theta1 = cont_traits_Theta1
        self.cont_traits_alpha = cont_traits_alpha
        self.s_species = s_species
        if seed:
            np.random.seed(seed)


    def simulate(self, L, M, root):
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
        n_cont_traits = 0
        if self.cont_traits_varcov is not None:
            n_cont_traits = len(self.cont_traits_varcov)
            self.cont_traits_varcov = dT * np.array(self.cont_traits_varcov) # Does this work for > 1 trait?
            self.cont_traits_varcov = np.sqrt(self.cont_traits_varcov) # variance of traits = BM rate sigma2
        if self.cont_traits_Theta1 is None: # same Theta1 as Theta0
            self.cont_traits_Theta1 = np.repeat(0.0, n_cont_traits)
        root_plus_1 = np.abs(root) + 2
        cont_traits = np.empty((root_plus_1, n_cont_traits, self.s_species))
        cont_traits[:] = np.nan
        if n_cont_traits > 0:
            for i in range(self.s_species):
                Theta0 = np.zeros(n_cont_traits)
                cont_traits[-1,:,i] = self.evolve_cont_traits(Theta0, n_cont_traits) # from past to present


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

            no = np.random.uniform(0, 1)  # draw a random number
            no_extant_lineages = len(te_extant)  # the number of currently extant species
            mass_extinction_prob = self.p_mass_extinction/self.scale
            if no < mass_extinction_prob and no_extant_lineages > 10:  # mass extinction condition
                # print("Mass extinction", t / self.scale)
                # increased loss of species: increased ext probability for this time bin
                m = np.random.uniform(self.magnitude_mass_ext[0], self.magnitude_mass_ext[1])

            for j in te_extant:  # extant lineages
                # trait evolution
                if n_cont_traits > 0:
                    cont_traits[t_abs, :, j] = self.evolve_cont_traits(cont_traits[t_abs + 1, :, j], n_cont_traits)
                ran = ran_vec[j]
                # speciation
                if ran < l:
                    te.append(0)  # add species
                    ts.append(t)  # sp time
                    #a = np.random.choice(te_extant, 1) # from which extant species the new species branches off
                    #a = int(a)
                    desc = np.array([len(ts)])
                    anc_desc[j] = np.concatenate((anc_desc[j], desc))
                    anc = np.random.choice(anc_desc[j], 1)  # If a lineage already has multiple descendents
                    anc_desc.append(anc)
                    if n_cont_traits > 0:
                        cont_traits_new_species = self.init_cont_traits(root_plus_1, n_cont_traits)
                        cont_traits_new_species[t_abs,:] = cont_traits[t_abs,:,j] # inherit traits at time t from ancestor
                        cont_traits = np.dstack((cont_traits, cont_traits_new_species))
                # extinction
                elif ran > l and ran < (l + m):
                    te[j] = t


        return -np.array(ts) / self.scale, -np.array(te) / self.scale, anc_desc, cont_traits


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

        return L_shifts, M_shifts, L, M, timesL, timesM


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

        t_vec = np.linspace(0.0, 1.0, len(L_shifts))

        L_tt = L_shifts + linL * t_vec
        M_tt = M_shifts + linM * t_vec

        L_tt[L_tt < 0.0] = 1e-10
        M_tt[M_tt < 0.0] = 1e-10

        return L_tt, M_tt, linL, linM


    def init_cont_traits(self, past, n_cont_traits):
        tr = np.empty((past, n_cont_traits))
        tr[:] = np.nan

        return tr


    def evolve_cont_traits(self, cont_traits, n_cont_traits):
        if n_cont_traits == 1:
            # Not possible to vectorize; sd needs to have the same size than mean
            cont_traits = self.cont_traits_alpha * (self.cont_traits_Theta1 - cont_traits) + np.random.normal(0.0, self.cont_traits_varcov, 1)
        elif n_cont_traits > 1:
            cont_traits = cont_traits + np.random.multivariate_normal(cont_traits, self.cont_traits_varcov, 1)

        return cont_traits


    def run_simulation(self, print_res = False):
        LOtrue = [0]
        n_extinct = -0
        while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP:
            root = -np.random.uniform(np.min(self.root_r), np.max(self.root_r))  # ROOT AGES
            L_shifts, M_shifts, L, M, timesL, timesM = self.get_random_settings(root)
            L_tt, M_tt, linL, linM = self.add_linear_time_effect(L_shifts, M_shifts)
            FAtrue, LOtrue, anc_desc, cont_traits = self.simulate(L_tt, M_tt, root)
            n_extinct = len(LOtrue[LOtrue > 0])

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
                  'ts_te': ts_te}
        if print_res:
            print("N. species", len(LOtrue))
            ltt = ""
            for i in range(int(max(FAtrue))):
                n = len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i])
                ltt += "\n%s\t%s\t%s" % (i, n, "*" * n)
            print(ltt)
        return res_bd



class fossil_simulator():
    def __init__(self,
                 q = 5.,
                 alpha = 100.,
                 seed = 0):
        self.q = q
        self.alpha = alpha
        if seed:
            np.random.seed(seed)


    def get_duration(self, sp_x):
        return np.abs(np.diff(sp_x))


    def get_is_alive(self, sp_x):
        return sp_x[:,1] == 0.0


    def get_fossil_occurrences(self, sp_x, is_alive):
        dur = self.get_duration(sp_x)
        poi_rate_occ = self.q * dur
        exp_occ = np.round(np.random.poisson(poi_rate_occ)).flatten()
        lineages_sampled = np.arange(sp_x.shape[0])[exp_occ > 0]

        occ = []
        for i in lineages_sampled:
            occ_i = np.random.uniform(sp_x[i, 1], sp_x[i, 0], exp_occ[i])
            if is_alive[i]:
                occ_i = np.concatenate((occ_i, np.zeros(1, dtype = 'float')))
            occ.append(occ_i)

        return occ, lineages_sampled


    def get_taxon_names(self, lineages_sampled):
        num_taxa = len(lineages_sampled)
        taxon_names = []
        for i in range(num_taxa):
            taxon_names.append('T%s' % lineages_sampled[i])

        return taxon_names


    def run_simulation(self, sp_x):
        is_alive = self.get_is_alive(sp_x)
        fossil_occ, lineages_sampled = self.get_fossil_occurrences(sp_x, is_alive)
        taxon_names = self.get_taxon_names(lineages_sampled)

        d = {'fossil_occurrences': fossil_occ,
             'taxon_names': taxon_names}

        return d


def write_PyRate_file(sim_fossil, output_wd, name_file):
    fossil_occ = sim_fossil['fossil_occurrences']
    taxon_names = sim_fossil['taxon_names']
    py = "%s/%s.py" % (output_wd, name_file)
    pyfile = open(py, "w")
    pyfile.write('#!/usr/bin/env python')
    pyfile.write('\n')
    pyfile.write('from numpy import *')
    pyfile.write('\n')
    pyfile.write('data_1 = [') # Open block with fossil occurrences
    pyfile.write('\n')
    for i in range(len(fossil_occ)):
        pyfile.write('array(')
        pyfile.write(str(fossil_occ[i].tolist()))
        pyfile.write(')')
        if i != (len(fossil_occ)-1):
            pyfile.write(',')
        pyfile.write('\n')
    pyfile.write(']') # End block with fossil occurrences
    pyfile.write('\n')
    pyfile.write('d = [data_1]')
    pyfile.write('\n')
    pyfile.write('names = ["Caminalcules"]')
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