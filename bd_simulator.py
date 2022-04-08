import sys
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
from collections.abc import Iterable
#from .extract_properties import *
SMALL_NUMBER = 1e-10

class bd_simulator():
    def __init__(self,
                 s_species=1,  # number of starting species
                 rangeSP=[100, 1000],  # min/max size data set
                 minEX_SP=0,  # minimum number of extinct lineages allowed
                 root_r=[30., 100],  # range root ages
                 range_base_L=[0.2, 0.5],
                 range_base_M=[0.2, 0.5],
                 scale=100.,
                 p_mass_extinction=0.00924,
                 magnitude_mass_ext=[0.8, 0.95],
                 poiL=3, # Number of rate shifts expected according to a Poisson distribution
                 poiM=3, # Number of rate shifts expected according to a Poisson distribution
                 magL = 10., # Magnitude of shift in speciation rate
                 magM = 10., # Magnitude of shift in extinction rate
                 seed=0):
        self.s_species = s_species
        self.rangeSP = rangeSP
        self.minSP = np.min(rangeSP)
        self.maxSP = np.max(rangeSP)
        self.minEX_SP = minEX_SP
        self.root_r = root_r
        self.range_base_L = range_base_L
        self.range_base_M = range_base_M
        self.scale = scale
        self.p_mass_extinction = p_mass_extinction
        self.magnitude_mass_ext = np.sort(magnitude_mass_ext)
        self.poiL = poiL
        self.poiM = poiM
        self.magL = magL
        self.magM = magM,
        self.s_species = s_species
        if seed:
            np.random.seed(seed)

    def simulate(self, L, M, timesL, timesM, root):
        ts = list()
        te = list()
        L, M, root = L / self.scale, M / self.scale, int(root * self.scale)

        for i in range(self.s_species):
            ts.append(root)
            te.append(0)

        for t in range(root, 0):  # time
            for j in range(len(timesL) - 1):
                if -t / self.scale <= timesL[j] and -t / self.scale > timesL[j + 1]:
                    l = L[j]
            for j in range(len(timesM) - 1):
                if -t / self.scale <= timesM[j] and -t / self.scale > timesM[j + 1]:
                    m = M[j]

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


    def get_rate_shift_magnitude(mag):
        m = np.random.uniform(1, mag, 1)
        incr = np.random.choice(np.arange(2), 1)
        if incr == 1:
            m = 1./m
        return m


    def get_random_settings(self, root):
        root = np.abs(root)
        timesL_temp = [root, 0.]
        timesM_temp = [root, 0.]

        Lbase = np.random.uniform(np.min(self.range_base_L), np.max(self.range_base_L), 1)
        Mbase = np.random.uniform(np.min(self.range_base_M), np.max(self.range_base_M), 1)

        # Number of rate shifts expected according to a Poisson distribution
        nL = np.random.poisson(self.poiL)
        nM = np.random.poisson(self.poiM)

        shift_time_L = np.random.uniform(0, root, nL)
        shift_time_M = np.random.uniform(0, root, nM)

        timesL = np.sort(np.concatenate((timesL_temp, shift_time_L), axis=0))[::-1]
        timesM = np.sort(np.concatenate((timesM_temp, shift_time_M), axis=0))[::-1]

        L = np.zeros(nL + 1, dtype = 'float')
        for i in range(nL + 1):
            m = get_rate_shift_magnitude(self.magL)
            L[i] = Lbase * m
        M = np.zeros(nM + 1, dtype='float')
        for i in range(nM + 1):
            m = get_rate_shift_magnitude(self.magM)
            M[i] = Mbase * m

        return timesL, timesM, L, M

    def run_simulation(self, print_res=False):
        LOtrue = [0]
        n_extinct = -0
        while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP:
            root = -np.random.uniform(np.min(self.root_r), np.max(self.root_r))  # ROOT AGES
            timesL, timesM, L, M = self.get_random_settings(root)
            FAtrue, LOtrue = self.simulate(L, M, timesL, timesM, root)
            n_extinct = len(LOtrue[LOtrue > 0])

        ts_te = np.array([FAtrue, LOtrue])
        if print_res:
            print("L", L, "M", M, "tL", timesL, "tM", timesM)
            print("N. species", len(LOtrue))
            ltt = ""
            for i in range(int(max(FAtrue))):
                n = len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i])
                ltt += "\n%s\t%s\t%s" % (i, n, "*" * n)
            print(ltt)
        return ts_te.T
