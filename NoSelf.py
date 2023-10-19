import subprocess
import copy
import sys
import numpy as np

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
#import bdnn_simulator as bdnnsim
from bdnn_simulator import *


# cont_traits_cov = np.array([[0.3, 0.2],[0.2, 0.3]]) # Colinearity ~0.67

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])

# rnd_seed = 38995818

bd_sim = bdnn_simulator(s_species = 1,  # number of starting species
                        rangeSP = [100., 1000.],  # min/max size data set
                        #minEX_SP = 0,  # minimum number of extinct lineages allowed
                        #maxExtant_SP = 0, # minimum number of extant lineages
                        minExtant_SP = 1,
                        # timewindow_rangeSP = [45., 21.],
                        root_r = [60., 60.],  # range root ages
                        rangeL = [0.15, 0.15],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 1000.0,
                        # cont_traits_sigma_clado = [0.1, 0.1],
                        # poiL = 3,  # expected number of birth rate shifts
                        # poiM = 3,  # expected number of death rate shift
                        # range_linL = [0.6, 0.0],
                        # range_linM = [0.0, 0.6],
                        # Shifts
                        # fixed_Ltt = np.array([[60., 0.20], [37.001, 0.20], [37., 0.6], [30.001, 0.6], [30., 0.0001], [0.0, 0.0001]]),
                        # fixed_Mtt = np.array([[60., 0.05], [40.001, 0.05], [40., 0.5], [33.001, 0.5], [33., 0.15], [0.0, 0.15]]),
                        # fixed_Ltt = np.array([[60., 0.1], [48.001, 0.1], [48., 0.8], [42.001, 0.8],  [42., 0.1], [18.001, 0.1], [18., 0.01], [0., 0.01]]),
                        # fixed_Mtt = np.array([[60., 0.05], [24.001, 0.05], [24., 0.3], [18.001, 0.3], [18., 0.2], [0., 0.2]]),
                        # fixed_Ltt = np.array([[60., 0.10], [40.001, 0.10], [40., 0.4], [30.001, 0.4], [30., 0.05], [0.0, 0.05]]), # Tree FBD
                        # fixed_Mtt = np.array([[60., 0.05], [20.001, 0.05], [20., 0.1], [0.0, 0.1]]),
                        # Linear change
                        # fixed_Ltt = np.array([[60., 0.3], [0.0, 0.01]]),
                        # fixed_Mtt = np.array([[60., 0.01], [0.0, 0.3]]),
                        seed = rnd_seed)  # if > 0 fixes the random seed to make simulations reproducible

scenario = 'Shifts_15'
scenario = 'Constant_03'
scenario = 'Linear_02'
scenario = 'Phylogeny'

# Set directory
base_dir = '/home/torsten/Work/EdgeEffect/Simulations'
output_dir = os.path.join(base_dir, scenario)

# Birth-death simulation
########################
res_bd = bd_sim.run_simulation(verbose = True)


# Sampling simulation
#####################
fossil_sim = fossil_simulator(range_q = [0.5, 0.8],
                              range_alpha = [1000.0, 1000.0],
                              poi_shifts = 0,
                              seed = rnd_seed)
sim_fossil = fossil_sim.run_simulation(res_bd['ts_te'])
print(sim_fossil['q'])
print(sim_fossil['shift_time'])
print(sim_fossil['alpha'])


# Complete data
################
# Write input files for PyRate analysis
write_PyRate = write_PyRate_files(output_wd = output_dir,
                                  delta_time = 1.0,
                                  name = 'Complete')
name_file = write_PyRate.run_writter(sim_fossil, res_bd)

write_occurrence_table(sim_fossil,
                       output_wd = output_dir,
                       name_file = 'Complete')

ltt_true_file = os.path.join(output_dir, 'Complete', 'LTT_true.csv')
np.savetxt(ltt_true_file, res_bd['LTTtrue'], delimiter = '\t', fmt = '%f')


write_FBD = write_FBD_files(output_wd = output_dir,
                            name_file =  name_file,
                            interval_ages = None)
write_FBD.run_FBD_writter(sim_fossil)


# Truncate fossil record for edge effect
########################################
# Keep information on whether a species is extant
# keep_in_interval = np.array([ [np.inf, 20.0], [10.0, 5.0] ])
keep_in_interval = np.array([ [45.0, 21.0] ])
sim_fossil_deepcopy = copy.deepcopy(sim_fossil)


# truncate data and remove information on extant lineages
# Do not translate fossil occurrences by keep_in_interval[1]
############################################################
trunc_fossil = keep_fossils_in_interval(sim_fossil_deepcopy,
                                        keep_in_interval = keep_in_interval,
                                        keep_extant = False)

write_trunc = write_PyRate_files(output_wd = output_dir,
                                 delta_time = 1.0,
                                 name = 'Truncated')
name_trunc = write_trunc.run_writter(trunc_fossil, res_bd)

write_occurrence_table(trunc_fossil,
                       output_wd = output_dir,
                       name_file = 'Truncated')

# Write sampling epochs
# sampling_epochs = '/home/torsten/Work/EdgeEffect/Simulations/%s/Truncated/sampling_epochs.csv' % scenario
# np.savetxt(sampling_epochs, np.transpose(keep_in_interval), delimiter = '\t', fmt = '%f')


write_FBD_trunc = write_FBD_files(output_wd = output_dir,
                                  name_file = name_trunc,
                                  translate = keep_in_interval[0,1])
write_FBD_trunc.run_FBD_writter(trunc_fossil)


# pad the counts by an additional column before and after the truncation boundaries
###################################################################################
write_padded = write_PyRate_files(output_wd = output_dir,
                                  delta_time = 1.0,
                                  name = 'TruncatedPadded')
name_padded = write_padded.run_writter(trunc_fossil, res_bd)

write_FBD_padded = write_FBD_files(output_wd = output_dir,
                                   name_file = name_padded,
                                   padding = keep_in_interval[0,:])
write_FBD_padded.run_FBD_writter(trunc_fossil)


# truncate data
# Do not translate fossil occurrences by keep_in_interval[1]
# but pad the counts by an additional column before and after the truncation boundaries
# and center horseshoe prior
########################################################################################
# write_padded_center = write_PyRate_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
#                                   delta_time = 1.0,
#                                   name = 'TruncPaddedCenter')
# name_padded_center = write_padded_center.run_writter(trunc_fossil_exclExt, res_bd)
#
# write_FBD_padded_center = write_FBD_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
#                                           name_file = name_padded_center,
#                                           padding = keep_in_interval[0,:],
#                                           center_HSMRF = True)
# write_FBD_padded_center.run_FBD_writter(trunc_fossil_exclExt)

# truncate data
# Do not translate fossil occurrences by keep_in_interval[1]
# but pad the counts by an additional column before and after the truncation boundaries
# and fix speciation for most recent bin and extinction of the earliest bin
########################################################################################
write_fix = write_PyRate_files(output_wd = output_dir,
                               delta_time = 1.0,
                               name = 'TruncatedPaddedFix')
name_fix = write_fix.run_writter(trunc_fossil, res_bd)

write_FBD_fix = write_FBD_files(output_wd = output_dir,
                                name_file = name_fix,
                                padding = keep_in_interval[0,:],
                                fix_fake_bin = True)
write_FBD_fix.run_FBD_writter(trunc_fossil)


# Writing tree-based FBD analysis
#################################
FBDtree_complete = write_FBD_tree(fossils = sim_fossil,
                                  res_bd = res_bd,
                                  output_wd = output_dir,
                                  name = 'Complete')
FBDtree_complete.run_writter()

FBDtree_trunc = write_FBD_tree(fossils = sim_fossil,
                               res_bd = res_bd,
                               output_wd = output_dir,
                               name = 'Truncated',
                               edges = keep_in_interval)
FBDtree_trunc.run_writter()



res_bd['tree'].write(path = '/home/torsten/Work/EdgeEffect/Simulations/Phylogeny/FBD/data/PhyloComplete.tre', schema = 'newick')

tree = res_bd['tree']
tree_extant = prune_extinct(tree)
tree_extant.write(path = '/home/torsten/Work/EdgeEffect/Simulations/Phylogeny/Extant/data/PhyloExtant.tre', schema = 'newick')