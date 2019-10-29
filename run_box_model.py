#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:31:53 2019

@author: jdesk
"""

"""
NOTES:
the difference between kernel method
analytic (which gives the same as kernel_grid_m)
AND Ecol_grid_R might be explained by the assumption that
the velocity is held constant during each timestep for Ecol_grid_R,
while it is updated on the fly in the other two methods...
# this should be reduced when reducing the timestep ;P
"""

#%% IMPORTS AND DEFS

import os
import math
import numpy as np
#from numba import njit

#import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick

#incl_path = "/home/jdesk/CloudMP"
#import sys
#if os.path.exists(incl_path):
#    sys.path.append(incl_path)
    
import constants as c

from microphysics import compute_radius_from_mass_vec
#from microphysics import compute_radius_from_mass_jit

from collision.box_model import simulate_collisions,analyze_sim_data
from collision.box_model import plot_for_given_kappa, plot_moments_kappa_var

#from kernel import compute_terminal_velocity_Beard
from collision.kernel import compute_terminal_velocity_Beard_vec
from collision.kernel import generate_and_save_kernel_grid_Long_Bott
from collision.kernel import generate_and_save_E_col_grid_R

#from init_SIPs import conc_per_mass_expo
from init_SIPs import generate_and_save_SIP_ensembles_SingleSIP_prob
from init_SIPs import analyze_ensemble_data
#from init_SIPs import generate_myHisto_SIP_ensemble_np
#from init_SIPs import plot_ensemble_data

from generate_SIP_ensemble_dst import gen_mass_ensemble_weights_SinSIP_expo
from generate_SIP_ensemble_dst import gen_mass_ensemble_weights_SinSIP_lognormal

import sys

#%% SET PARAMETERS 

set_log_file = True

sim_data_path = "/Users/bohrer/sim_data_col_box_mod_new/"

#OS = "LinuxDesk"
##OS = "Mac"
##OS = "TROPOS_server"
##OS = "LinuxNote"
#
#if OS == "Mac":
##    sim_data_path = "/Users/bohrer/sim_data_cloudMP/"
#    sim_data_path = "/Users/bohrer/sim_data_col_box_mod/"
#elif OS == "LinuxDesk" or OS == "LinuxNote":
#    sim_data_path = "/mnt/D/sim_data_col_box_mod/"
#elif OS == "TROPOS_server":
#    sim_data_path = "/vols/fs1/work/bohrer/sim_data_col_box_mod/" 

############################################################################################
# SET args for SIP ensemble generation AND Kernel-grid generation
# Kernel grid or Ecol grid must be generated at least once
#args_gen = [1,1,1,1,1]
#args_gen = [1,0,0,0,0]
#args_gen = [0,0,0,0,1]
#args_gen = [0,0,0,0,0]
args_gen = [1,1,1,0,1]

act_gen_SIP = bool(args_gen[0])
act_analysis_ensembles = bool(args_gen[1])
# NOTE: plotting needs analyze first (or in buffer)
act_plot_ensembles = bool(args_gen[2])
act_gen_kernel_grid = bool(args_gen[3])
act_gen_Ecol_grid = bool(args_gen[4])

# SET args for simulation
#args_sim = [0,0,0,0]
args_sim = [1,0,0,0]
#args_sim = [0,1,1,1]
#args_sim = [0,0,0,1]
#args_sim = [1,1,1,1]

act_sim = bool(args_sim[0])
act_analysis = bool(args_sim[1])
act_plot = bool(args_sim[2])
act_plot_moments_kappa_var = bool(args_sim[3])

############################################################################################
### SET PARAMETERS FOR SIMULATION OF COLLISION BOX MODEL

# the number of SIPs Nsip will be approx. 5 * kappa, where kappa is
# taken from the kappa_list
# all kappa stated in kappa_lista are simulated no_sims times
kappa_list=[2]
#kappa_list=[200]
#kappa_list=[5,10]
#kappa_list=[5,10,20,40,60]
#kappa_list=[5,10,20,40,60,100]
#kappa_list=[5,10,20,40,60,100,200,400]
#kappa_list=[3,3.5,5,10,20,40,60,100,200,400]
#kappa_list=[800]
#kappa_list=[5,10,20,40,60,100,200,400,600,800]
#kappa_list=[5,10,20,40,60,100,200,400,600,800,1000,1500,2000,3000]
#kappa_list=[200,400,600,800]

#no_sims = 100
#no_sims = 500
no_sims = 10
#no_sims = 400
#start_seed = 5711
start_seed = 1111
#start_seed = 8711
#start_seed = 94711

seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

#kernel_name = "Golovin"
kernel_name = "Long_Bott"
#kernel_name = "Hall_Bott"

#kernel_method = "kernel_grid_m"
kernel_method = "Ecol_grid_R"
#kernel_method = "analytic"

#dt = 1.0
dt = 10.0
# dt = 20.0

# dt_save = 40.0
#dt_save = 300.0
dt_save = 150.0
# t_end = 200.0
t_end = 3600.0

# NOTE that only "auto_bin" is possible for the analysis of the sim data for now
bin_method_sim_data = "auto_bin"

###############################################################################
### SET GENERAL PARAMETERS

no_bins = 50

mass_density = 1E3 # approx for water
#mass_density = c.mass_density_water_liquid_NTP
#mass_density = c.mass_density_NaCl_dry

###############################################################################
### SET PARAMETERS FOR SIP ENSEMBLES

dist = "expo"
#dist = "lognormal"
gen_method = "SinSIP"

eta = 1.0E-9

eta_threshold = "weak"
#eta_threshold = "fix"

dV = 1.0

## for EXPO dist: set r0 and LWC0 analog to Unterstr.
# -> DNC0 is calc from that
if dist == "expo":
    LWC0 = 1.0E-3 # kg/m^3
    R_mean = 9.3 # in mu
    r_critmin = 0.6 # mu
    m_high_over_m_low = 1.0E6

### for lognormal dist: set DNC0 and mu_R and sigma_R parameters of
# lognormal distr (of the radius) -> see distr. def above
if dist == "lognormal":
    r_critmin = 0.001 # mu
    m_high_over_m_low = 1.0E8
    
    # set DNC0 manually only for lognormal distr. NOT for expo
    DNC0 = 60.0E6 # 1/m^3
    #DNC0 = 2.97E8 # 1/m^3
    #DNC0 = 3.0E8 # 1/m^3
    mu_R = 0.02 # in mu
    sigma_R = 1.4 #  no unit

## PARAMS FOR DATA ANALYSIS OF INITIAL SIP ENSEMBLES

## the "bin_method_init_ensembles" is not necessary anymore
# both methods are applied and plotted by default
# use the same bins as for generation
#bin_method_init_ensembles = "given_bins"
# for auto binning: set number of bins
#bin_method_init_ensembles = "auto_bins"

# only bin_mode = 1 is available for now..
# bin_mode = 1 for bins equal dist on log scale
bin_mode = 1

## for myHisto generation (additional parameters)
spread_mode = 0 # spreading based on lin scale
# spread_mode = 1 # spreading based on log scale
# shift_factor = 1.0
# the artificial bins before the first and after the last one get multiplied
# by this factor (since they only get part of the half of the first/last bin)
overflow_factor = 2.0
# scale factor for the first correction
scale_factor = 1.0
# shift factor for the second correction.
# For shift_factor = 0.5 only half of the effect:
# bin center_new = 0.5 (bin_center_lin_first_corr + shifted bin center)
shift_factor = 0.5

############################################################################################
### SET PARAMETERS FOR KERNEL/ECOL GRID GENERATION AND LOADING

#R_low_kernel, R_high_kernel, no_bins_10_kernel = 0.6, 6E3, 200
R_low_kernel, R_high_kernel, no_bins_10_kernel = 1E-2, 301., 100

save_dir_kernel_grid = sim_data_path + f"{dist}/kernel_grid_data/"
save_dir_Ecol_grid = sim_data_path + f"{dist}/Ecol_grid_data/{kernel_name}/"

############################################################################################
### DERIVED PARAMETERS
# constant converts radius in mu to mass in kg (m = 4/3 pi rho R^3)
c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
c_mass_to_radius = 1.0 / c_radius_to_mass

if eta_threshold == "weak":
    weak_threshold = True
else: weak_threshold = False

if dist == "expo":
    m_mean = c_radius_to_mass * R_mean**3 # in kg
    DNC0 = LWC0 / m_mean # in 1/m^3
    LWC0_over_DNC0 = LWC0 / DNC0
    DNC0_over_LWC0 = DNC0 / LWC0
    print("dist = expo", f"DNC0 = {DNC0:.3e}", "LWC0 =", LWC0, "m_mean = ", m_mean)
    dist_par = (DNC0, DNC0_over_LWC0)

elif dist =="lognormal":
    mu_R_log = np.log(mu_R)
    sigma_R_log = np.log(sigma_R)
    
    # derive parameters of lognormal distribution of mass f_m(m)
    # assuming mu_R in mu and density in kg/m^3
    mu_m_log = 3.0 * mu_R_log + np.log(1.0E-18 * c.four_pi_over_three * mass_density)
    sigma_m_log = 3.0 * sigma_R_log
    print("dist = lognormal", f"DNC0 = {DNC0:.3e}",
          "mu_R =", mu_R, "sigma_R = ", sigma_R)
    dist_par = (DNC0, mu_m_log, sigma_m_log)

no_rpc = DNC0 * dV

###

ensemble_path_add =\
f"{dist}/{gen_method}/eta_{eta:.0e}_{eta_threshold}/ensembles/"
result_path_add =\
f"{dist}/{gen_method}/eta_{eta:.0e}_{eta_threshold}\
/results/{kernel_name}/{kernel_method}/"


#%% KERNEL/ECOL GRID GENERATION
if act_gen_kernel_grid:
    if not os.path.exists(save_dir_kernel_grid):
        os.makedirs(save_dir_kernel_grid)        
    mg_ = generate_and_save_kernel_grid_Long_Bott(
              R_low_kernel, R_high_kernel, no_bins_10_kernel,
              mass_density, save_dir_kernel_grid )[2]
    print(f"generated kernel grid,",
          f"R_low = {R_low_kernel}, R_high = {R_high_kernel}",
          f"number of bins = {len(mg_)}")

if act_gen_Ecol_grid:
    if not os.path.exists(save_dir_Ecol_grid):
        os.makedirs(save_dir_Ecol_grid)        
    E_col__, Rg_ = generate_and_save_E_col_grid_R(
              R_low_kernel, R_high_kernel, no_bins_10_kernel, kernel_name,
              save_dir_Ecol_grid)
    print(f"generated Ecol grid,",
          f"R_low = {R_low_kernel}, R_high = {R_high_kernel}",
          f"number of bins = {len(Rg_)}")

#%% SIP ENSEMBLE GENERATION
if act_gen_SIP:
    no_SIPs_avg = []
    for kappa in kappa_list:
        no_SIPs_avg_ = 0
        ensemble_dir =\
            sim_data_path + ensemble_path_add + f"kappa_{kappa}/"
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)
        for i,seed in enumerate(seed_list):
            if dist == "expo":
                ensemble_parameters = [dV, DNC0, DNC0_over_LWC0, r_critmin,
                       kappa, eta, no_sims, start_seed]

                masses, weights, m_low, bins = \
                    gen_mass_ensemble_weights_SinSIP_expo(
                            1.0E18*m_mean, mass_density,
                            dV, kappa, eta, weak_threshold, r_critmin,
                            m_high_over_m_low,
                            seed, setseed=True)

            elif dist == "lognormal":
                mu_m_log = 3.0 * mu_R_log \
                           + np.log(c.four_pi_over_three * mass_density)
                mu_m_log2 = 3.0 * mu_R_log \
                           + np.log(1E-18 * c.four_pi_over_three * mass_density)
                sigma_m_log = 3.0 * sigma_R_log
                ensemble_parameters = [dV, DNC0, mu_m_log2, sigma_m_log, mass_density,
                               r_critmin, kappa, eta, no_sims, start_seed]

                masses, weights, m_low, bins = \
                    gen_mass_ensemble_weights_SinSIP_lognormal(
                        mu_m_log, sigma_m_log,
                        mass_density,
                        dV, kappa, eta, weak_threshold, r_critmin,
                        m_high_over_m_low,
                        seed, setseed=True)
            xis = no_rpc * weights
            no_SIPs_avg_ += xis.shape[0]
            bins_rad = compute_radius_from_mass_vec(bins, mass_density)
            radii = compute_radius_from_mass_vec(masses, mass_density)

            np.save(ensemble_dir + f"masses_seed_{seed}", 1.0E-18*masses)
            np.save(ensemble_dir + f"radii_seed_{seed}", radii)
            np.save(ensemble_dir + f"xis_seed_{seed}", xis)
            
            if i == 0:
                np.save(ensemble_dir + f"bins_mass", 1.0E-18*bins)
                np.save(ensemble_dir + f"bins_rad", bins_rad)
                np.save(ensemble_dir + "ensemble_parameters",
                        ensemble_parameters)                    
        no_SIPs_avg.append(no_SIPs_avg_/no_sims)
        print(kappa, no_SIPs_avg_/no_sims)
    np.savetxt(sim_data_path + ensemble_path_add + f"no_SIPs_vs_kappa.txt",
               (kappa_list,no_SIPs_avg), fmt = "%-6.5g")
#               , delimiter="\t")
#    np.savetxt()
#### WORKING VERSION -> generates mass ensemble in kg        
#        generate_and_save_SIP_ensembles_SingleSIP_prob(
#            dist, dist_par, mass_density, dV, kappa, eta, weak_threshold,
#            r_critmin, m_high_over_m_low, no_sims, start_seed, ensemble_dir)

#%% SIP ENSEMBLE ANALYSIS AND PLOTTING
if act_analysis_ensembles:
    for kappa in kappa_list:
        ensemble_dir =\
            sim_data_path + ensemble_path_add + f"kappa_{kappa}/"
        
        analyze_ensemble_data(dist, mass_density, kappa, no_sims, ensemble_dir,
                          no_bins, bin_mode,
                          spread_mode, shift_factor, overflow_factor,
                          scale_factor, act_plot_ensembles)

#### OLD WORKING      
#        bins_mass, bins_rad, bins_rad_log, \
#        bins_mass_width, bins_rad_width, bins_rad_width_log, \
#        bins_mass_centers, bins_rad_centers, \
#        masses, xis, radii, f_m_counts, f_m_ind,\
#        f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled,\
#        m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, \
#        f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, \
#        g_ln_r_num_avg, g_ln_r_num_std, \
#        m_min, m_max, R_min, R_max, no_SIPs_avg, \
#        moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
#        moments_an, \
#        f_m_num_avg_my_ext, \
#        f_m_num_avg_my, f_m_num_std_my, \
#        g_m_num_avg_my, g_m_num_std_my, \
#        h_m_num_avg_my, h_m_num_std_my, \
#        bins_mass_my, bins_mass_width_my, \
#        bins_mass_centers_my, bins_mass_center_lin_my, lin_par, aa = \
#            analyze_ensemble_data(dist, mass_density, kappa, no_sims,
#                                  ensemble_dir,
#                                  bin_method_init_ensembles, no_bins, bin_mode,
#                                  spread_mode, shift_factor, overflow_factor,
#                                  scale_factor)

### SIP ensemble plotting is now included in SIP analysis above        
#        if act_plot_ensembles:
#            plot_ensemble_data(kappa, mass_density, eta, r_critmin,
#                dist, dist_par, no_sims, no_bins, bin_method_init_ensembles,
#                bins_mass, bins_rad, bins_rad_log, 
#                bins_mass_width, bins_rad_width, bins_rad_width_log, 
#                bins_mass_centers, bins_rad_centers, 
#                masses, xis, radii, f_m_counts, f_m_ind,
#                f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled, 
#                m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, 
#                f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, 
#                g_ln_r_num_avg, g_ln_r_num_std, 
#                m_min, m_max, R_min, R_max, no_SIPs_avg, 
#                moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,
#                moments_an, lin_par,
#                f_m_num_avg_my_ext,
#                f_m_num_avg_my, f_m_num_std_my, g_m_num_avg_my, g_m_num_std_my, 
#                h_m_num_avg_my, h_m_num_std_my, 
#                bins_mass_my, bins_mass_width_my, 
#                bins_mass_centers_my, bins_mass_center_lin_my,
#                ensemble_dir)

#%% SIMULATE COLLISIONS
if act_sim:
    if set_log_file:
        if not os.path.exists(sim_data_path + result_path_add):
            os.makedirs(sim_data_path + result_path_add)
        sys.stdout = open(sim_data_path + result_path_add
                          + f"std_out_kappa_{kappa_list[0]}_{kappa_list[-1]}_dt_{int(dt)}"
                          + ".log", 'w')
#%% SIMULATION DATA LOAD
    if kernel_method == "kernel_grid_m":
        # convert to 1E-18 kg if mass grid is given in kg...
        mass_grid = \
            1E18*np.load(save_dir_kernel_grid + "mass_grid_out.npy")
        kernel_grid = \
            np.load(save_dir_kernel_grid + "kernel_grid.npy" )
        m_kernel_low = mass_grid[0]
        bin_factor_m = mass_grid[1] / mass_grid[0]
        m_kernel_low_log = math.log(m_kernel_low)
        bin_factor_m_log = math.log(bin_factor_m)
        no_kernel_bins = len(mass_grid)
    
    if kernel_method == "Ecol_grid_R":
        if kernel_name == "Hall_Bott":
            radius_grid = \
                np.load(save_dir_Ecol_grid + "Hall_Bott_R_col_Unt.npy")
            E_col_grid = \
                np.load(save_dir_Ecol_grid + "Hall_Bott_E_col_Unt.npy" )        
        else:        
            radius_grid = \
                np.load(save_dir_Ecol_grid + "radius_grid_out.npy")
            E_col_grid = \
                np.load(save_dir_Ecol_grid + "E_col_grid.npy" )        
        R_kernel_low = radius_grid[0]
        bin_factor_R = radius_grid[1] / radius_grid[0]
        R_kernel_low_log = math.log(R_kernel_low)
        bin_factor_R_log = math.log(bin_factor_R)
        no_kernel_bins = len(radius_grid)
    
### SIMULATIONS
    for kappa in kappa_list:
        no_cols = np.array((0,0))
        print("simulation for kappa =", kappa)
        # SIP ensembles are already stored in directory
        # LINUX desk
        ensemble_dir =\
            sim_data_path + ensemble_path_add + f"kappa_{kappa}/"
        save_dir =\
            sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sim_params = [dt, dV, no_sims, kappa, start_seed]
        
        seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
        for cnt,seed in enumerate(seed_list):
            # LOAD ENSEMBLE DATA
            # NOTE that enesemble masses are given in kg, but we need 1E-18 kg
            # for the collision algorithms
            masses = 1E18 * np.load(ensemble_dir + f"masses_seed_{seed}.npy")
            xis = np.load(ensemble_dir + f"xis_seed_{seed}.npy")                    
            if kernel_name == "Golovin":
                SIP_quantities = (xis, masses)
                kernel_quantities = None                
            elif kernel_name == "Long_Bott":
                if kernel_method == "Ecol_grid_R":
                    mass_densities = np.ones_like(masses) * mass_density
                    radii = compute_radius_from_mass_vec(masses, mass_densities)
                    vel = compute_terminal_velocity_Beard_vec(radii)
                    SIP_quantities = (xis, masses, radii, vel, mass_densities)
                    kernel_quantities = \
                    (E_col_grid, no_kernel_bins, R_kernel_low_log,
                     bin_factor_R_log)
                        
                elif kernel_method == "kernel_grid_m":
                    SIP_quantities = (xis, masses)
                    kernel_quantities = \
                    (kernel_grid, no_kernel_bins, m_kernel_low_log,
                     bin_factor_m_log)
                        
                elif kernel_method == "analytic":
                    SIP_quantities = (xis, masses, mass_density)
                    kernel_quantities = None
                    
            if kernel_name == "Hall_Bott":
                if kernel_method == "Ecol_grid_R":
                    mass_densities = np.ones_like(masses) * mass_density
                    radii = compute_radius_from_mass_vec(masses, mass_densities)
                    vel = compute_terminal_velocity_Beard_vec(radii)
                    SIP_quantities = (xis, masses, radii, vel, mass_densities)
                    kernel_quantities = \
                    (E_col_grid, no_kernel_bins, R_kernel_low_log,
                     bin_factor_R_log)  
                    
            print(f"kappa {kappa}, sim {cnt}: seed {seed} simulation start")
            simulate_collisions(SIP_quantities,
                                kernel_quantities, kernel_name, kernel_method,
                                dV, dt, t_end, dt_save,
                                no_cols, seed, save_dir)
            print(f"kappa {kappa}, sim {cnt}: seed {seed} simulation finished")

#%% DATA ANALYSIS
### DATA ANALYSIS

if act_analysis:
    print("kappa, time_n, {xi_max/xi_min:.3e}, no_SIPS_avg, R_min, R_max")
    for kappa in kappa_list:
        load_dir =\
            sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        analyze_sim_data(kappa, mass_density, dV, no_sims, start_seed, no_bins, load_dir)

#%% PLOTTING
if act_plot:
    for kappa in kappa_list:
        load_dir =\
            sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        plot_for_given_kappa(kappa, eta, dt, no_sims, start_seed, no_bins,
                         kernel_name, gen_method, bin_method_sim_data, load_dir)

#%% PLOT MOMENTS VS TIME for several kappa
# act_plot_moments_kappa_var = True

TTFS, LFS, TKFS = 14,14,12
if act_plot_moments_kappa_var:
    ref_data_path = "collision/ref_data/" \
                    + f"{kernel_name}/"
    if kernel_name == "Long_Bott" or kernel_name == "Hall_Bott":
        moments_ref = np.loadtxt(ref_data_path + "Wang_2007_moments.txt")
        times_ref = np.loadtxt(ref_data_path + "Wang_2007_times.txt")
    elif kernel_name == "Golovin":
        moments_ref = None
        times_ref = None
    
#    if kernel_name == "Long_Bott":
#        data_Wang_2007 = [
#                            295.4 ,
#                            287.4 ,
#                            278.4 ,
#                            264.4 ,
#                            151.7 ,
#                            13.41 ,
#                            1.212,
#                            0.999989, 
#                            0.999989 ,
#                            0.999989 ,
#                            0.999989 ,
#                            0.999989 ,
#                            0.999989 ,
#                            0.999989 ,
#                            6.739E-9 ,
#                            7.402E-9 ,
#                            8.720E-9 ,
#                            3.132E-7 ,
#                            3.498E-4 ,
#                            1.068E-2 ,
#                            3.199E-2 ,
#                            6.813e-14 ,
#                            9.305e-14 ,
#                            5.710e-13 ,
#                            3.967e-8 ,
#                            1.048e-3 ,
#                            2.542e-1 ,
#                            1.731    
#                            ]
#    elif kernel_name == "Hall_Bott":
#        data_Wang_2007 = [
#                            295.4 ,
#                            287.8 ,
#                            279.9 ,
#                            270.2 ,
#                            231.7 ,
#                            124.5 ,
#                            73.99 ,
#                            0.999989, 
#                            0.999989 ,
#                            0.999989 ,
#                            0.999989 ,
#                            0.999989 ,
#                            0.999989 ,
#                            0.999989 ,
#                            6.739e-9 ,
#                            7.184e-9 ,
#                            7.999e-9 ,
#                            7.827e-8 ,
#                            1.942e-5 ,
#                            7.928e-4 ,
#                            6.997e-3 ,
#                            6.813e-14 ,
#                            8.282e-14 ,
#                            3.801e-13 ,
#                            2.531e-9 ,
#                            6.107e-6 ,
#                            2.108e-3 ,
#                            1.221e-1  ]               
        
    fig_dir = sim_data_path + result_path_add
    plot_moments_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
                           kernel_name, gen_method,
                           dist, start_seed,
                           moments_ref, times_ref,
                           sim_data_path,
                           result_path_add,
                           fig_dir, TTFS, LFS, TKFS)
