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


class WriteFbdTree():
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
