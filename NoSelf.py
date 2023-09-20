import subprocess
import copy
import sys
import numpy as np

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
#import bdnn_simulator as bdnnsim
from bdnn_simulator import *


# cont_traits_cov = np.array([[0.3, 0.2],[0.2, 0.3]]) # Colinearity ~0.67

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])

# rnd_seed = 94649591

bd_sim = bdnn_simulator(s_species = 1,  # number of starting species
                        rangeSP = [30, 70],  # min/max size data set
                        minEX_SP = 0,  # minimum number of extinct lineages allowed
                        #maxExtant_SP = 1, # minimum number of extant lineages
                        minExtant_SP = 5,
                        timewindow_rangeSP = [25., 15.],
                        root_r = [35., 35.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.1, 0.1],  # range of death rates
                        scale = 100.,
                        p_mass_extinction = 0.0,
                        magnitude_mass_ext = [0.0, 0.0],
                        # cont_traits_sigma_clado = [0.1, 0.1],
                        # poiL = 3,  # expected number of birth rate shifts
                        # poiM = 3,  # expected number of death rate shift
                        # range_linL = [0.0, 0.0],
                        # range_linM = [0.0, 0.0],
                        # Shifts
                        # fixed_Ltt = np.array([[60., 0.20], [37.001, 0.20], [37., 0.6], [30.001, 0.6], [30., 0.0001], [0.0, 0.0001]]),
                        # fixed_Mtt = np.array([[60., 0.05], [40.001, 0.05], [40., 0.5], [33.001, 0.5], [33., 0.1], [0.0, 0.1]]),
                        # fixed_Ltt = np.array([[60., 0.25], [30.001, 0.25], [30., 0.02], [0.0, 0.02]]),
                        # fixed_Mtt = np.array([[60., 0.10], [40.001, 0.10], [40., 0.025], [0.0, 0.025]]),
                        # Linear change
                        # fixed_Ltt = np.array([[60., 0.25], [0.0, 0.01]]),
                        # fixed_Mtt = np.array([[60., 0.01], [0.0, 0.25]]),
                        seed = rnd_seed)  # if > 0 fixes the random seed to make simulations reproducible

scenario = 'Shifts_15'
# scenario = 'Linear_01'

# Birth-death simulation
res_bd = bd_sim.run_simulation(verbose = True)
print(res_bd['lambda'])
print(res_bd['tshift_lambda'])
print(res_bd['mu'])
print(res_bd['tshift_mu'])

# Sampling simulation
fossil_sim = fossil_simulator(range_q = [0.1, 1.0],
                              range_alpha = [1000.0, 1000.0],
                              poi_shifts = 2,
                              seed = rnd_seed)
sim_fossil = fossil_sim.run_simulation(res_bd['ts_te'])
print(sim_fossil['q'])
print(sim_fossil['shift_time'])
print(sim_fossil['alpha'])


# Complete data
################
# Write input files for PyRate analysis
write_PyRate = write_PyRate_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                                  delta_time = 1.0,
                                  name = 'Complete')
name_file = write_PyRate.run_writter(sim_fossil, res_bd)

write_occurrence_table(sim_fossil,
                       output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                       name_file = 'Complete')


# if len(sim_fossil['shift_time']) > 0:
#     sampl = '-qShift', '/home/torsten/Work/BDNN/%s_q_epochs.txt' % name_file
# else:
#     sampl = '-mHPP'

# PyRate_run = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
#                              '/home/torsten/Work/EdgeEffect/Simulations/%s/%s/%s.py' % (scenario, name_file, name_file),
#                              #sampl,
#                              #'-qShift', '/home/torsten/Work/BDNN/%s/%s_q_epochs.txt' % (name_file, name_file),
#                              '-A 4',
#                              #'-mG',
#                              '-n 500001', '-s 5000', '-p 100000'])
#
# PyRate_plot = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
#                               '-plotRJ', '/home/torsten/Work/EdgeEffect/Simulations/%s/%s/pyrate_mcmc_logs' % (scenario, name_file),
#                               '-b 20'])

# Create inpute files for FBD analysis
# interval_ages = np.array([[np.inf, 27.0],
#                           [27.0, 26.0],
#                           [26.0, 25.0]])
write_FBD = write_FBD_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                            name_file =  name_file,
                            interval_ages = None)
write_FBD.run_FBD_writter(sim_fossil)


# Truncate fossil record for edge effect
########################################
# Keep information on whether a species is extant
# keep_in_interval = np.array([ [np.inf, 20.0], [10.0, 5.0] ])
keep_in_interval = np.array([ [44.0, 21.0] ])
sim_fossil_deepcopy = copy.deepcopy(sim_fossil)

# trunc_fossil_inclExt = keep_fossils_in_interval(sim_fossil_deepcopy,
#                                                 keep_in_interval = keep_in_interval,
#                                                 keep_extant = True)
# interval_exceedings = get_interval_exceedings(sim_fossil, res_bd['ts_te'], keep_in_interval)
#
# write_inclExt = write_PyRate_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
#                                    delta_time = 1.0,
#                                    name = 'TruncInclExt')
# name_inclExt = write_inclExt.run_writter(trunc_fossil_inclExt, res_bd)
#
# PyRate_run = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
#                              '/home/torsten/Work/EdgeEffect/Simulations/%s/%s/%s.py' % (scenario, name_inclExt, name_inclExt),
#                              '-A 4',
#                              #'-mG',
#                              '-n 500001', '-s 5000', '-p 100000'])
# PyRate_plot = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
#                               '-plotRJ', '/home/torsten/Work/EdgeEffect/Simulations/%s/%s/pyrate_mcmc_logs' % (scenario, name_inclExt),
#                               '-b 20'])
#
# write_FBD_inclExt = write_FBD_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
#                                     name_file =  name_inclExt)
# write_FBD_inclExt.run_FBD_writter(trunc_fossil_inclExt)

# truncate data and remove information on extant lineages
# Do not translate fossil occurrences by keep_in_interval[1]
############################################################
trunc_fossil = keep_fossils_in_interval(sim_fossil_deepcopy,
                                        keep_in_interval = keep_in_interval,
                                        keep_extant = False)

write_trunc = write_PyRate_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                                 delta_time = 1.0,
                                 name = 'Truncated')
name_trunc = write_trunc.run_writter(trunc_fossil, res_bd)

write_occurrence_table(trunc_fossil,
                       output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                       name_file = 'Truncated')

# PyRate_run = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
#                              '/home/torsten/Work/EdgeEffect/Simulations/%s/%s/%s.py' % (scenario, name_exclExt, name_exclExt),
#                              '-A 4',
#                              #' -translate -%s' % keep_in_interval[0,1], # Why is PyRate complaining about no argument translate?
#                              #'-mG',
#                              '-n 500001', '-s 5000', '-p 100000'])
# PyRate_plot = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
#                               '-plotRJ', '/home/torsten/Work/EdgeEffect/Simulations/%s/%s/pyrate_mcmc_logs' % (scenario, name_exclExt),
#                               '-b 20'])

write_FBD_trunc = write_FBD_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                                  name_file = name_trunc,
                                  translate = keep_in_interval[0,1])
write_FBD_trunc.run_FBD_writter(trunc_fossil)


# pad the counts by an additional column before and after the truncation boundaries
###################################################################################
write_padded = write_PyRate_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                                  delta_time = 1.0,
                                  name = 'TruncatedPadded')
name_padded = write_padded.run_writter(trunc_fossil, res_bd)

write_FBD_padded = write_FBD_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
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
write_fix = write_PyRate_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                               delta_time = 1.0,
                               name = 'TruncatedPaddedFix')
name_fix = write_fix.run_writter(trunc_fossil, res_bd)

write_FBD_fix = write_FBD_files(output_wd = '/home/torsten/Work/EdgeEffect/Simulations/%s' % scenario,
                                name_file = name_fix,
                                padding = keep_in_interval[0,:],
                                fix_fake_bin = True)
write_FBD_fix.run_FBD_writter(trunc_fossil)


# Writting tree-based FBD analysis
FBD_tree = write_FBD_tree(fossils = sim_fossil,
                          res_bd = res_bd,
                          output_wd = '/home/torsten/Work/EdgeEffect/Simulations/Phylogeny',
                          name_file = 'FBD')
FBD_tree.trim_tree_by_lad()
FBD_tree.tree_pruned.write(path = '/home/torsten/Work/EdgeEffect/Simulations/Phylogeny/PhyloPruned.tre', schema = 'newick')
FBD_tree.tree_trimmed.write(path = '/home/torsten/Work/EdgeEffect/Simulations/Phylogeny/PhyloTrimmed.tre', schema = 'newick')
FBD_tree.write_ranges()


tree_extant = prune_extinct(res_bd['tree'])
tree_extant.write(path = '/home/torsten/Work/EdgeEffect/Simulations/Phylogeny/Extant/data/PhyloExtant.tre', schema = 'newick')