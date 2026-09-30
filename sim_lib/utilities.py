import sys
import os
from scipy.stats import mode
import numpy as np
import string
import dendropy
import warnings
SMALL_NUMBER = 1e-10


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


def get_majority_cat_trait_per_taxon(res_bd, sim_fossil=None, upper=-np.inf, lower=np.inf):
    cat_traits = res_bd['cat_traits']
    n_cat_traits = cat_traits.shape[1]
    n_taxa_sampled = cat_traits.shape[2]
    taxa_sampled = np.arange(n_taxa_sampled, dtype=int)
    if not sim_fossil is None:
        taxa_sampled = sim_fossil['taxa_sampled']
        n_taxa_sampled = len(taxa_sampled)
    maj_cat_traits = np.zeros(n_taxa_sampled * n_cat_traits, dtype = int).reshape((n_taxa_sampled, n_cat_traits))
    # larger time value: more distant past; negative value: future
    time = res_bd['true_rates_through_time']['time']
    time = np.concatenate((-np.inf, time, np.inf), axis=None)
    trait_idx = np.logical_and(time < lower, time >= upper)
    for i in range(n_cat_traits):
        cat_traits_i = cat_traits[:, i, taxa_sampled]
        cat_traits_i = cat_traits_i[trait_idx, :]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            maj_cat_traits_i = mode(cat_traits_i, nan_policy='omit')[0]
        maj_cat_traits_i[np.isnan(maj_cat_traits_i)] = 0
        maj_cat_traits[:, i] = maj_cat_traits_i.astype(int)

    return maj_cat_traits


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
