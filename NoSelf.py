import subprocess

# cont_traits_cov = np.array([[0.3, 0.2],[0.2, 0.3]]) # Colinearity ~0.67

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1))

# rnd_seed = 11

bd_sim = bdnn_simulator(s_species = 1,  # number of starting species
                        rangeSP = [100, 300],  # min/max size data set
                        minEX_SP = 0,  # minimum number of extinct lineages allowed
                        minExtant_SP = 2, # minimum number of extant lineages
                        root_r = [25., 30.],  # range root ages
                        rangeL = [0.2, 0.3],  # range of birth rates
                        rangeM = [0.1, 0.2],  # range of death rates
                        scale = 100.,
                        p_mass_extinction = 0.0,
                        magnitude_mass_ext = [0.001, 0.002],
                        poiL = 0,  # expected number of birth rate shifts
                        poiM = 0,  # expected number of death rate shift
                        range_linL = [0.0, 0.0],
                        range_linM = [0.0, 0.0],
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
                        cat_traits_effect = [1., 1.],
                        n_areas = [4, 4],
                        dispersal = [0.05, 0.1],
                        extirpation = [0.05, 0.2],
                        seed = rnd_seed)  # if > 0 fixes the random seed to make simulations reproducible


fossil_sim = fossil_simulator(range_q = [0.5, 3.0],
                              range_alpha = [1.0, 3.0],
                              poi_shifts = 2,
                              seed = rnd_seed)


write_PyRate = write_PyRate_files(output_wd = '/home/torsten/Work/BDNN',
                                  delta_time = 1.0)

# Birth-death simulation
res_bd = bd_sim.run_simulation(verbose = True)
print(res_bd['lambda'])
#print(res_bd['tshift_lambda'])
print(res_bd['mu'])
#print(res_bd['tshift_mu'])
#print(res_bd['true_rates_through_time'][['speciation', 'extinction']])
#print(res_bd['linear_time_lambda'])
#print(res_bd['linear_time_mu'])
#print(res_bd['cat_traits_effect'])
print(res_bd['cont_traits_effect'])
print(res_bd['lineage_rates'][:3,:])
print(np.min(res_bd['lineage_rates'][:,2]), np.max(res_bd['lineage_rates'][:,2]))
print(np.unique(res_bd['lineage_rates'][:,6], return_counts = True)[1])
np.savetxt('/home/torsten/Work/BDNN/Biogeography.txt', res_bd['biogeography'][:,0,:], delimiter = '\t')
np.nanmax(res_bd['biogeography'][:,0,:])

# Sampling simulation
sim_fossil = fossil_sim.run_simulation(res_bd['ts_te'])
print(sim_fossil['q'])
print(sim_fossil['shift_time'])
print(sim_fossil['alpha'])

# Truncate fossil record for edge effect
# keep_in_interval = np.array([ [np.inf, 20.0], [10.0, 5.0] ])
# sim_fossil = keep_fossils_in_interval(sim_fossil,
#                                       keep_in_interval = keep_in_interval)
# interval_exceedings = get_interval_exceedings(sim_fossil, res_bd['ts_te'], keep_in_interval)

# Write input files for PyRate analysis
name_file = write_PyRate.run_writter(sim_fossil, res_bd)


if len(sim_fossil['shift_time']) > 0:
    sampl = '-qShift', '/home/torsten/Work/BDNN/%s_q_epochs.txt' % name_file
else:
    sampl = '-mHPP'

PyRate_run = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py',
                             '/home/torsten/Work/BDNN/%s/%s.py' % (name_file, name_file),
                             #sampl,
                             '-qShift', '/home/torsten/Work/BDNN/%s/%s_q_epochs.txt' % (name_file, name_file),
                             '-A 4',
                             '-mG', '-n 500001', '-s 5000', '-p 100000'])

PyRate_plot = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py', '-plotRJ', '/home/torsten/Work/BDNN/%s/pyrate_mcmc_logs' % name_file, '-b 10'])



np.savetxt('/home/torsten/Work/BDNN/CatTraits.txt', res_bd['cat_traits'][:,0,:], delimiter = '\t')
np.savetxt('/home/torsten/Work/BDNN/LineageRates.txt', res_bd['lineage_rates'], delimiter = '\t')
np.savetxt('/home/torsten/Work/BDNN/ContTraits.txt', res_bd['cont_traits'][:,0,:], delimiter = '\t')

print(np.nanvar(res_bd['cont_traits'][1,0,:]))
print(res_bd['cont_traits_effect'][0][0,1]**2) # Should be similar to the variance



# Create inpute files for FBD analysis
######################################
write_FBD = write_FBD_files(output_wd = '/home/torsten/Work/BDNN',
                            name_file =  name_file,
                            interval_ages = np.array([[np.inf, 15.0],
                                                      [15.0, 7.0],
                                                      [7.0, 0.0]]))
write_FBD.run_FBD_writter(sim_fossil)