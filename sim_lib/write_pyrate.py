import copy
import sys
import os
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


class WritePyRate():
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


    def write_baseline_qtt(self, sim_fossil, name_file):
        file_baseline_qtt = '%s/%s/%s_true_baseline_qtt.txt' % (self.output_wd, name_file, name_file)
        sim_fossil['q_baseline'].to_csv(file_baseline_qtt, na_rep='NA', index=False, sep='\t', float_format="%.3f")


    def write_qtt(self, sim_fossil, name_file):
        file_qtt = '%s/%s/%s_true_qtt.txt' % (self.output_wd, name_file, name_file)
        sim_fossil['qtt'].to_csv(file_qtt, na_rep='NA', index=False, sep='\t', float_format="%.3f")


    def write_qtt_per_taxon(self, sim_fossil, name_file):
        file_qtt = '%s/%s/%s_true_qtt_per_taxon.txt' % (self.output_wd, name_file, name_file)
        sim_fossil['qtt_taxa'].to_csv(file_qtt, na_rep='NA', index=True, sep='\t', float_format="%.3f")


    def write_qmtt_per_taxon(self, sim_fossil, name_file):
        file_qmtt = '%s/%s/%s_true_qmutlitt_per_taxon.txt' % (self.output_wd, name_file, name_file)
        sim_fossil['qmultitt_taxa'].to_csv(file_qmtt, na_rep='NA', index=True, sep='\t', float_format="%.3f")


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


    # def get_majority_cat_trait_per_taxon(self, sim_fossil, res_bd):
    #     cat_traits = res_bd['cat_traits']
    #     n_cat_traits = cat_traits.shape[1]
    #     taxa_sampled = sim_fossil['taxa_sampled']
    #     n_taxa_sampled = len(taxa_sampled)
    #     maj_cat_traits = np.zeros(n_taxa_sampled * n_cat_traits, dtype = int).reshape((n_taxa_sampled, n_cat_traits))
    #     for i in range(n_cat_traits):
    #         cat_traits_i = cat_traits[:, i, taxa_sampled]
    #         with warnings.catch_warnings():
    #             warnings.simplefilter('ignore', category=Warning)
    #             maj_cat_traits_i = mode(cat_traits_i, nan_policy='omit')[0]#[0] # Did this change with newer scipy?
    #         maj_cat_traits[:, i] = maj_cat_traits_i.astype(int)
    #
    #     return maj_cat_traits


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


    def run_writter(self, sim_fossil, res_bd, num_pvr=0, write_tree=False, write_taxon_q=False):
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
        self.write_baseline_qtt(sim_fossil, name_file)
        self.write_qtt(sim_fossil, name_file)
        if write_taxon_q:
            self.write_qtt_per_taxon(sim_fossil, name_file)
            self.write_qmtt_per_taxon(sim_fossil, name_file)

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
