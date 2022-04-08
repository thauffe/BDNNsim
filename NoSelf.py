rnd_seed = 23

bd_sim = bd_simulator(s_species=1,  # number of starting species
                      rangeSP=[20, 500],  # min/max size data set
                      minEX_SP=0,  # minimum number of extinct lineages allowed
                      root_r=[40., 50],  # range root ages
                      range_base_L=[0.1, 0.4],  # range of birth rates
                      range_base_M=[0.05, 0.2],  # range of death rates
                      p_mass_extinction=0.0,
                      magnitude_mass_ext=[0.001, 0.002],
                      poiL=1,  # expected number of birth rate shifts
                      poiM=1,  # expected number of death rate shifts
                      magL=10.,  # Magnitude of shift in speciation rate
                      magM=10.,  # Magnitude of shift in extinction rate
                      seed=rnd_seed)  # if > 0 fixes the random seed to make simulations reproducible

# Birth-death simulation
sp_x_i = bd_sim.run_simulation(print_res=False)


##################################################################


root = -20.
scale = 100.
s_species = 1 # Number of starting species
timesL = np.array([-root, 0.0])
timesM = np.array([-root, 0.0])
L = np.array([0.2])
M = np.array([0.0])
maxSP = 10000
p_mass_extinction = 0.0
magnitude_mass_ext = [0.0001, 0.0002]

ts = list()
te = list()
L, M, root = L / scale, M / scale, int(root * scale)

for i in range(s_species):
    ts.append(root)
    te.append(0)

    for t in range(root, 0):  # time
        for j in range(len(timesL) - 1):
            if -t / scale <= timesL[j] and -t / scale > timesL[j + 1]:
                l = L[j]
        for j in range(len(timesM) - 1):
            if -t / scale <= timesM[j] and -t / scale > timesM[j + 1]:
                m = M[j]

        # if t % 100 ==0: print t/scale, -times[j], -times[j+1], l, m
        TE = len(te)
        if TE > maxSP:
            break
        ran_vec = np.random.random(TE)
        te_extant = np.where(np.array(te) == 0)[0]

        no = np.random.uniform(0, 1)  # draw a random number
        no_extant_lineages = len(te_extant)  # the number of currently extant species
        mass_extinction_prob = p_mass_extinction/scale
        if no < mass_extinction_prob and no_extant_lineages > 10:  # mass extinction condition
            # print("Mass extinction", t / scale)
            # increased loss of species: increased ext probability for this time bin
            m = np.random.uniform(magnitude_mass_ext[0], magnitude_mass_ext[1])

        for j in te_extant:  # extant lineages
            # if te[j] == 0:
            ran = ran_vec[j]
            if ran < l:
                te.append(0)  # add species
                ts.append(t)  # sp time
            elif ran > l and ran < (l + m):  # extinction
                te[j] = t

np.array([ts]) / scale