#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

import sim_lib as sl


p = argparse.ArgumentParser()

p.add_argument("-seed", type=int, help="random seed", default=-1, metavar=-1)
# Target diversity
p.add_argument("-taxa", type=int, help="range of taxa to simulate", default=[1, np.inf], metavar=[1, np.inf], nargs=2)
p.add_argument("-start_taxa", type=int, help="taxa to start birth-death simulation", default=1, metavar=1)
p.add_argument("-extant", type=int, help="minimum and maximum of targeted extant taxa", default=[1, np.inf], metavar=[1, np.inf], nargs=2)
p.add_argument("-extinct", type=int, help="minimum targeted extinct taxa", default=0, metavar=0)

# Birth-death
p.add_argument("-root", type=float, help="root age", default=[30.0, 30.0], metavar=[30.0, 30.0], nargs=2)
p.add_argument("-lam", type=float, help="random range of birth rate", default=[0.1, 0.2], metavar=[0.1, 0.2], nargs=2)
p.add_argument("-mu", type=float, help="random range of death rate", default=[0.05, 0.1], metavar=[0.05, 0.1], nargs=2)
# Trait evolution
p.add_argument("-cont_traits", type=int, help="random range of continuous traits", default=[1, 1], metavar=[1, 1], nargs=2)
p.add_argument("-cat_traits", type=int, help="random range of categorical traits", default=[1, 1], metavar=[1, 1], nargs=2)
p.add_argument("-cat_states", type=int, help="random range of states for categorical traits", default=[2, 2], metavar=[2, 2], nargs=2)
# Fossil sampling
p.add_argument("-q", type=float, help="random range of sampling rate", default=[0.5, 5.0], metavar=[0.5, 5.0], nargs=2)
p.add_argument("-alpha", type=float, help="random range of alpha parameter heterogeneity in sampling across taxa", default=[0.5, 5.0], metavar=[0.5, 5.0], nargs=2)
p.add_argument("-q_loguniform", help="draw q from loguniform range", action='store_true', default=False)
p.add_argument("-alpha_loguniform", help="draw random alpha from loguniform range", action='store_true', default=False)
p.add_argument("-q_fixed", type=float, help="fixed sampling rate from past to present.", default=[], metavar=[], nargs="+")
p.add_argument("-q_shift", type=float, help="shift times for sampling rate from past to present. If used with q_fixed, one value less then that.", default=[], metavar=[], nargs="+")
# Output
p.add_argument("-wd", type=str, help="path to working directory", default="")
p.add_argument("-name", type=str, help="filename", default="")
# Print messages
p.add_argument("-verbose", help="show messages", action='store_true', default=False)


args = p.parse_args()

def main():
    # Birth-death simulation
    bd_sim = sl.BdnnSimulator(s_species = args.start_taxa,
                              rangeSP=args.taxa,
                              root_r=args.root,
                              minExtant_SP=args.extant[0],
                              maxExtant_SP=args.extant[1],
                              minEX_SP=args.extinct,
                              rangeL=args.lam,
                              rangeM=args.mu,
                              n_cont_traits=args.cont_traits,
                              cont_traits_sigma_clado=[0.2, 0.2],
                              cont_traits_sigma=[0.02, 0.02],
                              n_cat_traits=args.cat_traits,
                              n_cat_traits_states=args.cat_states,
                              cat_traits_diag=0.9,
                              cat_traits_min_freq=[0.0],
                              seed=args.seed)

    res_bd = bd_sim.run_simulation(verbose=args.verbose)

    # Fossil sampling
    fossil_sim = sl.FossilSimulator(range_q=args.q,
                                    range_alpha=[0.5, 5.0],
                                    fixed_shift_times=args.q_shift,
                                    fixed_q=np.array(args.q_fixed),
                                    q_loguniform=args.q_loguniform,
                                    alpha_loguniform=args.alpha_loguniform,
                                    seed=args.seed)
    sim_fossil = fossil_sim.run_simulation(res_bd)

    # Write files for PyRate
    wd = args.wd
    if wd == "":
        wd = os.getcwd()
    write_pyrate = sl.WritePyRate(output_wd=wd, name=args.name)
    _ = write_pyrate.run_writter(sim_fossil, res_bd, num_pvr=0, write_tree=False, write_taxon_q=False)

if __name__ == '__main__':
    main()
