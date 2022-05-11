import subprocess
import copy

# cont_traits_cov = np.array([[0.3, 0.2],[0.2, 0.3]]) # Colinearity ~0.67

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1))

# rnd_seed = 33464544

bd_sim = bdnn_simulator(s_species = 1,  # number of starting species
                        rangeSP = [200, 300],  # min/max size data set
                        minEX_SP = 0,  # minimum number of extinct lineages allowed
                        minExtant_SP = 0, # minimum number of extant lineages
                        #maxExtant_SP = 0,
                        root_r = [60., 60.],  # range root ages
                        rangeL = [0.2, 0.3],  # range of birth rates
                        rangeM = [0.1, 0.2],  # range of death rates
                        scale = 100.,
                        p_mass_extinction = 0.0,
                        magnitude_mass_ext = [0.0, 0.0],
                        poiL = 0,  # expected number of birth rate shifts
                        poiM = 0,  # expected number of death rate shift
                        range_linL = [0.0, 0.0],
                        range_linM = [0.0, 0.0],
                        fixed_Ltt = np.array([[60., 0.20], [37.001, 0.20], [37., 0.6], [30.001, 0.6], [30., 0.00001], [0.0, 0.00001]]),
                        fixed_Mtt = np.array([[60., 0.05], [40.001, 0.05], [40., 0.5], [33.001, 0.5], [33., 0.1], [0.0, 0.1]]),
                        n_cont_traits = [1, 1],  # number of continuous traits
                        cont_traits_sigma = [0.3, 0.3],  # evolutionary rates for continuous traits
                        cont_traits_cor = [-1, 1],  # evolutionary correlation between continuous traits
                        cont_traits_Theta1 = [0.0, 0.0], # morphological optima; 0 is no directional change from the ancestral values
                        cont_traits_alpha = [0.0, 0.0],
                        cont_traits_effect = [0.0, 0.0], # [0.001, 0.005],
                        n_cat_traits = [1, 1],
                        n_cat_traits_states = [2, 2], # range number of states for categorical trait
                        cat_traits_ordinal = [False, False],
                        cat_traits_dir = 2,
                        cat_traits_diag = 0.9,
                        cat_traits_effect = np.array([[1., 1.],[1, 1]]),
                        cat_traits_effect_decr_incr = np.array([[True, False],[True, False]]),
                        n_areas = [1, 1],
                        dispersal = [0.005, 0.01],
                        extirpation = [0.05, 0.2],
                        seed = rnd_seed)  # if > 0 fixes the random seed to make simulations reproducible


fossil_sim = fossil_simulator(range_q = [0.5, 3.0],
                              range_alpha = [1.0, 3.0],
                              poi_shifts = 0,
                              seed = rnd_seed)


write_PyRate = write_PyRate_files(output_wd = '/home/torsten/Work/BDNN',
                                  delta_time = 1.0,
                                  name = 'FBD')

# Birth-death simulation
res_bd = bd_sim.run_simulation(verbose = True)
print(res_bd['lambda'])
print(res_bd['tshift_lambda'])
print(res_bd['mu'])
print(res_bd['tshift_mu'])
#print(res_bd['mass_ext_time'])
print(res_bd['true_rates_through_time'][['speciation', 'extinction']])
print(res_bd['linear_time_lambda'])
print(res_bd['linear_time_mu'])
print(res_bd['cat_traits_Q'])
print(res_bd['cat_traits_effect'])
#print(res_bd['cont_traits_effect'])
print(res_bd['lineage_rates'][:3,:])
print(np.min(res_bd['lineage_rates'][:,2]), np.max(res_bd['lineage_rates'][:,2]))
print(np.unique(res_bd['lineage_rates'][1:,6], return_counts = True)[1])

# np.savetxt('/home/torsten/Work/BDNN/Biogeography.txt', res_bd['geographic_range'][:,0,:], delimiter = '\t')
# np.nanmax(res_bd['geographic_range'][:,0,:])
# print(np.unique(res_bd['geographic_range'][:,0,:]))

# Sampling simulation
# 2x with rnd_seed = 33464544
sim_fossil = fossil_sim.run_simulation(res_bd['ts_te'])
print(sim_fossil['q'])
print(sim_fossil['shift_time'])
print(sim_fossil['alpha'])

# Write input files for PyRate analysis
name_file = write_PyRate.run_writter(sim_fossil, res_bd)


if len(sim_fossil['shift_time']) > 0:
    sampl = '-qShift', '/home/torsten/Work/BDNN/%s_q_epochs.txt' % name_file
else:
    sampl = '-mHPP'

PyRate_run = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
                             '/home/torsten/Work/BDNN/%s/%s.py' % (name_file, name_file),
                             #sampl,
                             #'-qShift', '/home/torsten/Work/BDNN/%s/%s_q_epochs.txt' % (name_file, name_file),
                             '-A 4',
                             #'-FBDrange 3',
                             '-mG', '-n 200001', '-s 5000', '-p 100000'])

PyRate_plot = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
                              '-plotRJ', '/home/torsten/Work/BDNN/%s/pyrate_mcmc_logs' % name_file, '-b 10'])



np.savetxt('/home/torsten/Work/BDNN/CatTraits.txt', res_bd['cat_traits'][:,0,:], delimiter = '\t')
np.savetxt('/home/torsten/Work/BDNN/LineageRates.txt', res_bd['lineage_rates'], delimiter = '\t')
np.savetxt('/home/torsten/Work/BDNN/ContTraits.txt', res_bd['cont_traits'][:,0,:], delimiter = '\t')

print(np.nanvar(res_bd['cont_traits'][1,0,:]))
print(res_bd['cont_traits_effect'][0][0,1]**2) # Should be similar to the variance



# Create inpute files for FBD analysis
######################################
# root_height = np.ceil(np.max(res_bd['ts_te']))
# interval_ages = np.stack((np.arange(1, root_height, 1)[::-1], np.arange(0, root_height - 1, 1)[::-1]), axis = 1)
# interval_ages[0,0] = np.inf
interval_ages = np.array([[np.inf, 27.0],
                          [27.0, 26.0],
                          [26.0, 25.0]])
write_FBD = write_FBD_files(output_wd = '/home/torsten/Work/BDNN',
                            name_file =  name_file,
                            interval_ages = None)
write_FBD.run_FBD_writter(sim_fossil)


# Truncate fossil record for edge effect
########################################
# keep_in_interval = np.array([ [np.inf, 20.0], [10.0, 5.0] ])
keep_in_interval = np.array([ [44.0, 21.0] ])
sim_fossil_deepcopy = copy.deepcopy(sim_fossil)
trunc_fossil_inclExt = keep_fossils_in_interval(sim_fossil_deepcopy,
                                                keep_in_interval = keep_in_interval,
                                                keep_extant = True)
# interval_exceedings = get_interval_exceedings(sim_fossil, res_bd['ts_te'], keep_in_interval)