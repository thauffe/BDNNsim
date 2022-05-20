import subprocess
import copy

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1))

# rnd_seed = 89651236

bd_sim = bdnn_simulator(s_species = 1,  # number of starting species
                        rangeSP = [300, 500],  # min/max size data set
                        minEX_SP = 0,  # minimum number of extinct lineages allowed
                        minExtant_SP = 2, # minimum number of extant lineages
                        root_r = [35., 35.],  # range root ages
                        rangeL = [0.1, 0.1],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 100.,
                        p_mass_extinction = 0.0,
                        magnitude_mass_ext = [0.0, 0.0],
                        poiL = 0,  # expected number of birth rate shifts
                        poiM = 0,  # expected number of death rate shift
                        range_linL = [0.0, 0.0],
                        range_linM = [0.0, 0.0],
                        fixed_Ltt = np.array([[35.0, 0.6],[15.0, 0.1], [15.001, 0.6], [0.0, 0.01]]),
                        fixed_Mtt = np.array([[35.0, 0.1],[20.0, 0.1],[0.0, 0.7]]),
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
                                  name = 'scenario4')

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
sim_fossil = fossil_sim.run_simulation(res_bd['ts_te'])
print(sim_fossil['q'])
print(sim_fossil['shift_time'])
print(sim_fossil['alpha'])

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