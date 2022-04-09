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
                 range_linL = [-0.2, 0.2],
                 range_linM = [-0.2, 0.2],
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
        self.s_species = s_species
        if seed:
            np.random.seed(seed)


    def simulate(self, L, M, root):
        ts = list()
        te = list()
        #L, M, root = L / self.scale, M / self.scale, int(self.root * self.scale)
        root = int(root * self.scale)

        for i in range(self.s_species):
            ts.append(root)
            te.append(0)

        for t in range(root, 0):  # time i.e. integers self.root * self.scale
            #for j in range(len(timesL) - 1):
            #    if -t / self.scale <= timesL[j] and -t / self.scale > timesL[j + 1]:
            #        l = L[j]
            #for j in range(len(timesM) - 1):
            #    if -t / self.scale <= timesM[j] and -t / self.scale > timesM[j + 1]:
            #        m = M[j]
            l = L[t]
            m = M[t]

            # if t % 100 ==0: print t/scale, -times[j], -times[j+1], l, m
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
                # if te[j] == 0:
                ran = ran_vec[j]
                if ran < l:
                    te.append(0)  # add species
                    ts.append(t)  # sp time
                elif ran > l and ran < (l + m):  # extinction
                    te[j] = t

        return -np.array(ts) / self.scale, -np.array(te) / self.scale


    def get_random_settings(self, root):
        root = np.abs(root)
        root_scaled = int(root * self.scale)
        timesL_temp = [root_scaled, 0.]
        timesM_temp = [root_scaled, 0.]

        #Lbase = np.random.uniform(np.min(self.range_base_L), np.max(self.range_base_L), 1) / self.scale
        #Mbase = np.random.uniform(np.min(self.range_base_M), np.max(self.range_base_M), 1) / self.scale

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
        linL = np.random.uniform(np.min(self.range_linL), np.max(self.range_linL), 1)
        linM = np.random.uniform(np.min(self.range_linM), np.max(self.range_linM), 1)

        t_vec = np.linspace(0.0, 1.0, len(L_shifts))

        L_tt = L_shifts + linL * t_vec
        M_tt = M_shifts + linM * t_vec

        L_tt[L_tt < 0.0] = 1e-10
        M_tt[M_tt < 0.0] = 1e-10

        return L_tt, M_tt, linL, linM


    def run_simulation(self, print_res=False):
        LOtrue = [0]
        n_extinct = -0
        while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP:
            root = -np.random.uniform(np.min(self.root_r), np.max(self.root_r))  # ROOT AGES
            L_shifts, M_shifts, L, M, timesL, timesM = self.get_random_settings(root)
            L_tt, M_tt, linL, linM = self.add_linear_time_effect(L_shifts, M_shifts)
            FAtrue, LOtrue = self.simulate(L_tt, M_tt, root)
            n_extinct = len(LOtrue[LOtrue > 0])

        ts_te = np.array([FAtrue, LOtrue])
        if print_res:
            print("L", L * self.scale)
            print("tL", timesL / self.scale)
            print("M", M * self.scale)
            print("tM", timesM / self.scale)
            print("linL", linL)
            print("linM", linM)
            print("N. species", len(LOtrue))
            ltt = ""
            for i in range(int(max(FAtrue))):
                n = len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i])
                ltt += "\n%s\t%s\t%s" % (i, n, "*" * n)
            print(ltt)
        return ts_te.T
