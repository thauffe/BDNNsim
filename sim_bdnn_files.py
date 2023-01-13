import subprocess
import sys
import numpy as np

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1))




# State-independent effect of a single continuous traits
########################################################
rangeL = [0.5, 0.5]
rangeM = [0.4, 0.4]
n_cont_traits = [1, 1] # Range of number of continuous traits
n_cat_traits = [1, 1] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
# 4 dimensions: 1st axis: time; 2nd axis: n_cont_traits; 3rd axis: n_cat_traits; 4th axis: trait effect, min effect, max effect
cont_traits_effect_sp = np.array([[[ [0.8, 0.8] ]]])
cont_traits_effect_ex = np.array([[[ [0.8, 0.8] ]]])
cont_traits_effect_bellu_sp = np.array([[[ [1, 1] ]]])
cont_traits_effect_bellu_ex = np.array([[[ [1, 1] ]]])
cont_traits_effect_optimum_sp = np.array([[[ [0.0, 0.0] ]]])
cont_traits_effect_optimum_ex = np.array([[[ [0.0, 0.0] ]]])
cont_traits_effect_shift_sp = None
cont_traits_effect_shift_ex = None

# State-dependent effect of a single continuous traits
######################################################
n_cont_traits = [1, 1] # Range of number of continuous traits
n_cat_traits = [1, 1] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
# 4 dimensions: 1st axis: time; 2nd axis: n_cont_traits; 3rd axis: n_cat_states; 4th axis: trait effect, min effect, max effect
cont_traits_effect_sp = np.array([[[ [0.8, 0.8], [0.8, 0.8] ]]])
cont_traits_effect_ex = np.array([[[ [0.8, 0.8], [0.8, 0.8] ]]])
cont_traits_effect_bellu_sp = np.array([[[ [1, 1], [-1, -1] ]]])
cont_traits_effect_bellu_ex = np.array([[[ [1, 1], [-1, -1] ]]])
cont_traits_effect_optimum_sp = np.array([[[ [0.0, 0.0], [0.0, 0.0] ]]])
cont_traits_effect_optimum_ex = np.array([[[ [0.0, 0.0], [0.0, 0.0] ]]])
cont_traits_effect_shift_sp = None
cont_traits_effect_shift_ex = None

# Time-dependence of a single continuous trait (without categorical trait)
##########################################################################
n_cont_traits = [1, 1] # Range of number of continuous traits
n_cat_traits = [0, 0] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
cont_traits_effect_sp = np.array([ [[[0.8, 0.8]]],
                                   [[[0.8, 0.8]]] ])
cont_traits_effect_ex = np.array([ [[[0.8, 0.8]]],
                                   [[[0.8, 0.8]]] ])
cont_traits_effect_bellu_sp = np.array([ [[[1, 1]]],
                                         [[[1, 1]]] ])
cont_traits_effect_bellu_ex = np.array([ [[[1, 1]]],
                                         [[[1, 1]]] ])
cont_traits_effect_optimum_sp = np.array([ [[[0, 0]]],
                                           [[[0, 0]]] ])
cont_traits_effect_optimum_ex = np.array([ [[[0, 0]]],
                                           [[[0, 0]]] ])
cont_traits_effect_shift_sp = np.array([15.0])
cont_traits_effect_shift_ex = np.array([15.0])

# Time-dependence of a single continuous trait (with categorical which does not influence diversification)
##########################################################################################################
n_cont_traits = [1, 1] # Range of number of continuous traits
n_cat_traits = [1, 1] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
cont_traits_effect_sp = np.array([ [[ [0.00001, 0.00001], [0.00001, 0.00001] ]],
                                   [[ [3.0, 3.0], [3.0, 3.0]] ] ])
cont_traits_effect_ex = np.array([ [[ [0.00001, 0.00001], [0.00001, 0.00001] ]],
                                   [[ [3.0, 3.0], [3.0, 3.0]] ] ])
cont_traits_effect_bellu_sp = np.array([ [[ [ 1,  1], [ 1,  1] ]],
                                         [[ [ 1,  1], [ 1,  1] ]] ])
cont_traits_effect_bellu_ex = np.array([ [[ [ 1,  1], [ 1,  1] ]],
                                         [[ [ -1,  -1], [ -1,  -1] ]] ])
cont_traits_effect_optimum_sp = np.array([ [[[0, 0]]] ])
cont_traits_effect_optimum_ex = np.array([ [[[0, 0]]] ])
cont_traits_effect_shift_sp = np.array([15.0])
cont_traits_effect_shift_ex = np.array([15.0])

# Time-dependence of a single continuous trait plus the influence of a categorical trait
########################################################################################
n_cont_traits = [1, 1] # Range of number of continuous traits
n_cat_traits = [1, 1] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
cont_traits_effect_sp = np.array([ # Time bin 1
                                   [[ [1.0, 2.0], [1.0, 2.0] ]],   # Trait 1 State 1 & 2
                                   # Time bin 2
                                   [[ [0.5, 1.0], [0.5, 1.0] ]] ]) # Trait 1 State 1 & 2
cont_traits_effect_ex = np.array([ # Time bin 1
                                   [[ [0.5, 1.0], [0.5, 1.0] ]],   # Trait 1 State 1 & 2
                                   # Time bin 2
                                   [[ [1.0, 2.0], [1.0, 2.0] ]] ]) # Trait 1 State 1 & 2
cont_traits_effect_bellu_sp = np.array([ [[ [ 1,  1], [1, 1] ]],
                                         [[ [1, 1], [ 1,  1] ]] ])
cont_traits_effect_bellu_ex = np.array([ [[ [1, 1], [ 1,  1] ]],
                                         [[ [ 1,  1], [1, 1] ]] ])
cont_traits_effect_optimum_sp = np.array([ [[[0, 0]]] ])
cont_traits_effect_optimum_ex = np.array([ [[[0, 0]]] ])
cont_traits_effect_shift_sp = np.array([15.0])
cont_traits_effect_shift_ex = np.array([15.0])

# Two continuous traits
#######################
rangeL = [0.5, 0.5]
rangeM = [0.4, 0.4]
n_cont_traits = [2, 2] # Range of number of continuous traits
n_cat_traits = [0, 0] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
cont_traits_effect_sp = np.array([[ [[0.8, 0.8]], [[0.1, 0.1]] ]])
cont_traits_effect_ex = np.array([[ [[0.8, 0.8]], [[0.1, 0.1]] ]])
cont_traits_effect_bellu_sp = np.array([[ [[1, 1]], [[1, 1]] ]])
cont_traits_effect_bellu_ex = np.array([[ [[1, 1]], [[1, 1]] ]])
cont_traits_effect_optimum_sp = np.array([[ [[0.0, 0.0]], [[0.0, 0.0]] ]])
cont_traits_effect_optimum_ex = np.array([[ [[0.0, 0.0]], [[0.0, 0.0]] ]])
cont_traits_effect_shift_sp = None
cont_traits_effect_shift_ex = None

# Time-dependence of two continuous traits
##########################################
n_cont_traits = [2, 2] # Range of number of continuous traits
n_cat_traits = [0, 0] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
cont_traits_effect_sp = np.array([
                                 # Time bin 1
                                 [ [[0.2, 0.2]], [[0.15, 0.15]] ], # Trait 1 & 2
                                 # Time bin 2
                                 [ [[0.1, 0.1]], [[0.05, 0.05]] ]  # Trait 1 & 2
                                 ])
cont_traits_effect_ex = np.array([
                                 # Time bin 1
                                 [ [[0.1, 0.1]], [[0.05, 0.05]] ], # Trait 1 & 2
                                 # Time bin 2
                                 [ [[0.2, 0.2]], [[0.15, 0.15]] ]  # Trait 1 & 2
                                 ])
cont_traits_effect_bellu_sp = np.array([ [[[1, 1]]] ])
cont_traits_effect_bellu_ex = np.array([ [[[1, 1]]] ])
cont_traits_effect_optimum_sp = np.array([ [[[0, 0]]] ])
cont_traits_effect_optimum_ex = np.array([ [[[0, 0]]] ])
cont_traits_effect_shift_sp = np.array([15.0])
cont_traits_effect_shift_ex = np.array([15.0])

# Time- and state-dependence of two continuous traits
#####################################################
n_cont_traits = [2, 2] # Range of number of continuous traits
n_cat_traits = [1, 1] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
cont_traits_effect_sp = np.array([
                                 # Time bin 1
                                 [ [[0.30, 0.30], [0.30, 0.30]],   # Trait 1 State 1 & 2
                                   [[0.00, 0.00], [0.00, 0.00]] ], # Trait 2 State 1 & 2
                                 # Time bin 2
                                 [ [[0.15, 0.15], [0.15, 0.15]],   # Trait 1 State 1 & 2
                                   [[0.03, 0.03], [0.03, 0.03]] ]  # Trait 2 State 1 & 2
                                 ])
cont_traits_effect_ex = np.array([[[[0.0, 0.0]]]])
cont_traits_effect_bellu_sp = np.array([
                                 # Time bin 1
                                 [ [[ 1,  1], [-1, -1]],   # Trait 1 State 1 & 2
                                   [[ 1,  1], [-1, -1]] ], # Trait 2 State 1 & 2
                                 # Time bin 2
                                 [ [[-1, -1], [ 1,  1]],   # Trait 1 State 1 & 2
                                   [[-1, -1], [ 1,  1]] ]  # Trait 2 State 1 & 2
                                 ])
cont_traits_effect_bellu_ex = np.array([ [[[1, 1]]] ])
cont_traits_effect_optimum_sp = np.array([
                                 # Time bin 1
                                 [ [[ 0,  0], [ 2,  2]],   # Trait 1 State 1 & 2
                                   [[ 0,  0], [ 2,  2]] ], # Trait 2 State 1 & 2
                                 # Time bin 2
                                 [ [[ 0,  0], [ 2,  2]],   # Trait 1 State 1 & 2
                                   [[ 0,  0], [ 2,  2]] ]  # Trait 2 State 1 & 2
                                 ])
cont_traits_effect_optimum_ex = np.array([ [[[0, 0]]] ])
cont_traits_effect_shift_sp = np.array([15.0])
cont_traits_effect_shift_ex = np.array([15.0])


# Diversity-dependent speciation
################################
rangeL = [0.5, 0.5]
rangeM = [0.4, 0.4]
n_cont_traits = [1, 1] # Range of number of continuous traits
n_cat_traits = [1, 1] # Range of number of categorical traits
n_cat_traits_states = [2, 2] # States for categorical traits
# 4 dimensions: 1st axis: time; 2nd axis: n_cont_traits; 3rd axis: n_cat_traits; 4th axis: trait effect, min effect, max effect
cont_traits_effect_sp = np.array([[[ [SMALL_NUMBER, SMALL_NUMBER] ]]])
cont_traits_effect_ex = np.array([[[ [SMALL_NUMBER, SMALL_NUMBER] ]]])
cont_traits_effect_bellu_sp = np.array([[[ [1, 1] ]]])
cont_traits_effect_bellu_ex = np.array([[[ [1, 1] ]]])
cont_traits_effect_optimum_sp = np.array([[[ [0.0, 0.0] ]]])
cont_traits_effect_optimum_ex = np.array([[[ [0.0, 0.0] ]]])
cont_traits_effect_shift_sp = None
cont_traits_effect_shift_ex = None




bd_sim = bdnn_simulator(s_species = 1,  # number of starting species
                        rangeSP = [200, 300],  # min/max size data set
                        minEX_SP = 0,  # minimum number of extinct lineages allowed
                        minExtant_SP = 2, # minimum number of extant lineages
                        root_r = [35., 35.],  # range root ages
                        rangeL = rangeL,  # range of birth rates
                        rangeM = rangeM,  # range of death rates
                        scale = 100.,
                        p_mass_extinction = 0.0,
                        magnitude_mass_ext = [0.0, 0.0],
                        poiL = 0,  # expected number of birth rate shifts
                        poiM = 0,  # expected number of death rate shift
                        range_linL = [0.0, 0.0],
                        range_linM = [0.0, 0.0],
                        n_cont_traits = n_cont_traits, # number of continuous traits
                        cont_traits_sigma_clado = [0.2, 0.2],
                        cont_traits_sigma = [0.02, 0.02], # evolutionary rates for continuous traits
                        cont_traits_cor = [0.0, 0.0], # evolutionary correlation between continuous traits
                        cont_traits_Theta1 = [0.0, 0.0], # morphological optima; 0 is no directional change from the ancestral values
                        cont_traits_alpha = [0.0, 0.0],
                        cont_traits_effect_sp = cont_traits_effect_sp, # np.array([[0.1, 0.5]]), np.array([[0.1, 0.5], [0.0, 0.0]])
                        cont_traits_effect_ex = cont_traits_effect_ex,
                        cont_traits_effect_optimum_sp = cont_traits_effect_optimum_sp,
                        cont_traits_effect_optimum_ex = cont_traits_effect_optimum_ex,
                        cont_traits_effect_bellu_sp = cont_traits_effect_bellu_sp,
                        cont_traits_effect_bellu_ex = cont_traits_effect_bellu_ex,
                        cont_traits_effect_shift_sp = cont_traits_effect_shift_sp,
                        cont_traits_effect_shift_ex = cont_traits_effect_shift_ex,
                        n_cat_traits = n_cat_traits,
                        n_cat_traits_states = n_cat_traits_states, # range number of states for categorical trait
                        cat_traits_min_freq = [0.3],
                        cat_traits_ordinal = [False, False],
                        cat_traits_dir = 2,
                        cat_traits_diag = 0.9,
                        cat_traits_effect = np.array([[1., 1.],[1, 1]]),
                        cat_traits_effect_decr_incr = np.array([[True, False],[True, False]]),
                        # n_areas = [1, 1],
                        # dispersal = [0.005, 0.01],
                        # extirpation = [0.05, 0.2],
                        sp_env_file = '/home/torsten/Work/BDNN/temp.txt',
                        sp_env_eff = [1.2, 1.2], # [-0.08, -0.05],
                        ex_env_file = '/home/torsten/Work/BDNN/temp.txt',
                        ex_env_eff = [1.2, 1.2],
                        env_effect_cat_trait = [[1, -1],[1, -1]],
                        #K_lam = 100.0,
                        #K_mu = 100.0,
                        #fixed_K_lam = np.array([[35., 100.], [15.001, 100.], [15., 50.], [0.0, 50.]]),
                        seed = rnd_seed)  # if > 0 fixes the random seed to make simulations reproducible

fossil_sim = fossil_simulator(range_q = [0.5, 1.5],
                              range_alpha = [1000.0, 1000.0],
                              poi_shifts = 0,
                              seed = rnd_seed)


write_PyRate = write_PyRate_files(output_wd = '/home/torsten/Work/BDNN',
                                  delta_time = 1.0,
                                  name = 'DivDep')

# Birth-death simulation
res_bd = bd_sim.run_simulation(verbose = True)
#print(res_bd['ts_te'])
#print(res_bd['lambda'])
#print(res_bd['tshift_lambda'])
#print(res_bd['mu'])
#print(res_bd['tshift_mu'])
#print(res_bd['mass_ext_time'])
#print(res_bd['true_rates_through_time'][['speciation', 'extinction']])
print(res_bd['true_rates_through_time'][['time', 'trait_weighted_speciation', 'trait_weighted_extinction']])
#print(res_bd['linear_time_lambda'])
#print(res_bd['linear_time_mu'])
#print(res_bd['cat_traits_Q'])
#print(res_bd['cat_traits_effect'])
print(res_bd['cont_traits_effect_sp'])
print(res_bd['cont_traits_effect_ex'])
print(res_bd['expected_sd_cont_traits'])
# print(res_bd['lineage_rates'][:3,:])
# print(res_bd['cont_traits'])
# np.savetxt('/home/torsten/Work/BDNN/cont_traits.txt', res_bd['cont_traits'][:,0,:], delimiter='\t')
#print(np.min(res_bd['lineage_rates'][:,2]), np.max(res_bd['lineage_rates'][:,2]))
print(np.unique(res_bd['lineage_rates'][1:,8], return_counts = True)[1])

# np.savetxt('/home/torsten/Work/BDNN/Biogeography.txt', res_bd['geographic_range'][:,0,:], delimiter = '\t')
# np.nanmax(res_bd['geographic_range'][:,0,:])
# print(np.unique(res_bd['geographic_range'][:,0,:]))

# Sampling simulation
sim_fossil = fossil_sim.run_simulation(res_bd['ts_te'])
print(sim_fossil['q'])
print(sim_fossil['shift_time'])
print(sim_fossil['alpha'])

# Write input files for PyRate analysis
name_file = write_PyRate.run_writter(sim_fossil, res_bd)


RJMCMC_run = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
                             '/home/torsten/Work/BDNN/%s/%s.py' % (name_file, name_file),
                             '-A', '4',
                             '-mHPP',
                             '-n', '50001', '-s', '5000', '-p', '1000'])

RJMCMC_plot = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
                              '-plotRJ', '/home/torsten/Work/BDNN/%s/pyrate_mcmc_logs' % name_file, '-b', '10'])







write_FBD = write_FBD_files(output_wd = '/home/torsten/Work/BDNN',
                            name_file =  name_file,
                            interval_ages = None)
write_FBD.run_FBD_writter(sim_fossil)


np.savetxt('/home/torsten/Work/BDNN/CatTraits.txt', res_bd['cat_traits'][:,0,:], delimiter = '\t')
np.savetxt('/home/torsten/Work/BDNN/LineageRates.txt', res_bd['lineage_rates'], delimiter = '\t')
np.savetxt('/home/torsten/Work/BDNN/ContTraits.txt', res_bd['cont_traits'][:,0,:], delimiter = '\t')

print(np.nanvar(res_bd['cont_traits'][1,0,:]))
print(res_bd['cont_traits_effect'][0][0,1]**2) # Should be similar to the variance