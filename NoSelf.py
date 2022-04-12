import subprocess

rnd_seed = 23

cont_traits_cov = np.array([[0.3, 0.2],[0.2, 0.3]]) # Colinearity ~0.67

bd_sim = bd_simulator(s_species=1,  # number of starting species
                      rangeSP=[200, 300],  # min/max size data set
                      minEX_SP=0,  # minimum number of extinct lineages allowed
                      root_r=[10., 15.],  # range root ages
                      rangeL=[0.05, 0.5],  # range of birth rates
                      rangeM=[0.05, 0.3],  # range of death rates
                      scale = 100.,
                      p_mass_extinction=0.0,
                      magnitude_mass_ext=[0.001, 0.002],
                      poiL=2,  # expected number of birth rate shifts
                      poiM=2,  # expected number of death rate shift
                      range_linL = None,
                      range_linM = [-0.05, 0.05],
                      cont_traits_varcov = cont_traits_cov, # a list of length 1, 2D nd.array, or None
                      seed = rnd_seed)  # if > 0 fixes the random seed to make simulations reproducible


fossil_sim = fossil_simulator(q = 3.,
                              alpha = 100,
                              seed = rnd_seed)

# Birth-death simulation
res_bd = bd_sim.run_simulation(print_res=True)

#sp_x[:,0]

sim_fossil = fossil_sim.run_simulation(res_bd['ts_te'])


write_PyRate_file(sim_fossil, '/home/torsten/Work/BDNN', 'Test')


PyRate_run = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py', '/home/torsten/Work/BDNN/Test.py', '-A 4', '-mHPP', '-n 1000001', '-s 5000', '-p 100000'])

PyRate_plot = subprocess.run(['python3', '/home/torsten/Work/Software/PyRate/PyRate.py', '-plotRJ', '/home/torsten/Work/BDNN/pyrate_mcmc_logs', '-b 50'])


np.savetxt('/home/torsten/Work/BDNN/Traits.txt', res_bd['cont_traits'][:,0,:], delimiter = '\t')
np.var(res_bd['cont_traits'][1,0,:])