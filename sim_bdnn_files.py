import subprocess
import sys
import numpy as np

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
#import bdnn_simulator as bdnnsim
from bdnn_simulator import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1))

#rnd_seed = 54574560


# State-dependent effect of a single continuous traits
######################################################
n_cont_traits = [1, 1] # Range of number of continuous traits
n_cat_traits = [1, 1] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
# 4 dimensions: 1st axis: time; 2nd axis: n_cont_traits; 3rd axis: n_cat_traits; 4th axis: trait effect, min effect, max effect
cont_traits_effect_sp = np.array([[[ [0.1, 0.1], [0.3, 0.3] ]]])
cont_traits_effect_ex = np.array([[[ [0.05, 0.05], [0.3, 0.3] ]]])
cont_traits_effect_bellu_sp = np.array([[[ [1, 1], [-1, -1] ]]])
cont_traits_effect_bellu_ex = np.array([[[ [1, 1], [-1, -1] ]]])
cont_traits_effect_optimum_sp = np.array([[[ [0.0, 0.0], [2.0, 2.0] ]]])
cont_traits_effect_optimum_ex = np.array([[[ [2.0, 2.0], [0.0, 0.0] ]]])


# 3 continuous traits
#####################
n_cont_traits = [2, 2] # Range of number of continuous traits
n_cat_traits = [0, 0] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
cont_traits_effect_sp = np.array([[ [[0.2, 0.2]], [[0.1, 0.1]] ]])
cont_traits_effect_ex = np.array([[ [[0.2, 0.2]], [[0.1, 0.1]] ]])
cont_traits_effect_bellu_sp = np.array([[ [[1, 1]], [[1, 1]] ]])
cont_traits_effect_bellu_ex = np.array([[ [[1, 1]], [[1, 1]] ]])
cont_traits_effect_optimum_sp = np.array([[ [[0.0, 0.0]], [[2.0, 2.0]] ]])
cont_traits_effect_optimum_ex = np.array([[ [[2.0, 2.0]], [[0.0, 0.0]] ]])


bd_sim = bdnn_simulator(s_species = 1,  # number of starting species
                        rangeSP = [200, 300],  # min/max size data set
                        minEX_SP = 0,  # minimum number of extinct lineages allowed
                        minExtant_SP = 2, # minimum number of extant lineages
                        root_r = [35., 35.],  # range root ages
                        rangeL = [0.2, 0.3],  # range of birth rates
                        rangeM = [0.1, 0.2],  # range of death rates
                        scale = 100.,
                        p_mass_extinction = 0.0,
                        magnitude_mass_ext = [0.0, 0.0],
                        poiL = 0,  # expected number of birth rate shifts
                        poiM = 0,  # expected number of death rate shift
                        range_linL = [0.0, 0.0],
                        range_linM = [0.0, 0.0],
                        n_cont_traits = n_cont_traits, # number of continuous traits
                        cont_traits_sigma = [0.3, 0.3], # evolutionary rates for continuous traits
                        cont_traits_cor = [0.0, 0.0], # evolutionary correlation between continuous traits
                        cont_traits_Theta1 = [0.0, 0.0], # morphological optima; 0 is no directional change from the ancestral values
                        cont_traits_alpha = [0.0, 0.0],
                        cont_traits_effect_sp = cont_traits_effect_sp, # np.array([[0.1, 0.5]]), np.array([[0.1, 0.5], [0.0, 0.0]])
                        cont_traits_effect_ex = cont_traits_effect_ex,
                        cont_traits_effect_optimum_sp = cont_traits_effect_optimum_sp,
                        cont_traits_effect_optimum_ex = cont_traits_effect_optimum_ex,
                        cont_traits_effect_bellu_sp = cont_traits_effect_bellu_sp,
                        cont_traits_effect_bellu_ex = cont_traits_effect_bellu_ex,
                        n_cat_traits = n_cat_traits,
                        n_cat_traits_states = n_cat_traits_states, # range number of states for categorical trait
                        cat_traits_ordinal = [False, False],
                        cat_traits_dir = 2,
                        cat_traits_diag = 0.9,
                        cat_traits_effect = np.array([[1., 1.],[1, 1]]),
                        cat_traits_effect_decr_incr = np.array([[True, False],[True, False]]),
                        # n_areas = [1, 1],
                        # dispersal = [0.005, 0.01],
                        # extirpation = [0.05, 0.2],
                        # sp_env_file = '/home/torsten/Work/BDNN/temp_Westerhold.txt',
                        # sp_env_eff = [-0.02, -0.01],
                        # ex_env_file = '/home/torsten/Work/BDNN/temp_Westerhold.txt',
                        # ex_env_eff = [-0.02, -0.01],
                        seed = rnd_seed)  # if > 0 fixes the random seed to make simulations reproducible

fossil_sim = fossil_simulator(range_q = [0.1, 1.0],
                              range_alpha = [1000.0, 1000.0],
                              poi_shifts = 0,
                              seed = rnd_seed)


write_PyRate = write_PyRate_files(output_wd = '/home/torsten/Work/BDNN',
                                  delta_time = 1.0,
                                  name = 'Cont')

# Birth-death simulation
res_bd = bd_sim.run_simulation(verbose = True)
print(res_bd['lambda'])
#print(res_bd['tshift_lambda'])
#print(res_bd['mu'])
#print(res_bd['tshift_mu'])
#print(res_bd['mass_ext_time'])
#print(res_bd['true_rates_through_time'][['speciation', 'extinction']])
#print(res_bd['linear_time_lambda'])
#print(res_bd['linear_time_mu'])
#print(res_bd['cat_traits_Q'])
#print(res_bd['cat_traits_effect'])
print(res_bd['cont_traits_effect_sp'])
print(res_bd['cont_traits_effect_ex'])
#print(res_bd['lineage_rates'][:3,:])
#print(np.min(res_bd['lineage_rates'][:,2]), np.max(res_bd['lineage_rates'][:,2]))
# print(np.unique(res_bd['lineage_rates'][1:,6], return_counts = True)[1])

# np.savetxt('/home/torsten/Work/BDNN/Biogeography.txt', res_bd['geographic_range'][:,0,:], delimiter = '\t')
# np.nanmax(res_bd['geographic_range'][:,0,:])
# print(np.unique(res_bd['geographic_range'][:,0,:]))

# Sampling simulation
sim_fossil = fossil_sim.run_simulation(res_bd['ts_te'])
# print(sim_fossil['q'])
# print(sim_fossil['shift_time'])
# print(sim_fossil['alpha'])

# Write input files for PyRate analysis
name_file = write_PyRate.run_writter(sim_fossil, res_bd)

PyRate_run = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
                             '/home/torsten/Work/BDNN/%s/%s.py' % (name_file, name_file),
                             #sampl,
                             #'-qShift', '/home/torsten/Work/BDNN/%s/%s_q_epochs.txt' % (name_file, name_file),
                             '-A 4',
                             #'-FBDrange 3',
                             '-mG', '-n 200001', '-s 5000', '-p 100000'])

PyRate_plot = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
                              '-plotRJ', '/home/torsten/Work/BDNN/%s/pyrate_mcmc_logs' % name_file, '-b 10'])


write_FBD = write_FBD_files(output_wd = '/home/torsten/Work/BDNN',
                            name_file =  name_file,
                            interval_ages = None)
write_FBD.run_FBD_writter(sim_fossil)


np.savetxt('/home/torsten/Work/BDNN/CatTraits.txt', res_bd['cat_traits'][:,0,:], delimiter = '\t')
np.savetxt('/home/torsten/Work/BDNN/LineageRates.txt', res_bd['lineage_rates'], delimiter = '\t')
np.savetxt('/home/torsten/Work/BDNN/ContTraits.txt', res_bd['cont_traits'][:,0,:], delimiter = '\t')

print(np.nanvar(res_bd['cont_traits'][1,0,:]))
print(res_bd['cont_traits_effect'][0][0,1]**2) # Should be similar to the variance