#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:07:21 2019

@author: jdesk
"""

# 1. init()
# 2. spinup()
# 3. simulate()

### output:
## full saves:
# grid_parameters, grid_scalars, grid_vectors
# pos, cells, vel, masses, xi
## data:
# grid:
# initial: p, T, Theta, S, r_v, r_l, rho_dry, e_s
# continuous
# p, T, Theta, S, r_v, r_l, 
# particles:
# pos, vel, masses

# set:
# files for full save
# files for data output -> which data?

# 1. generate grid and particles + save initial to file
# grid, pos, cells, vel, masses, xi = init()

# 2. changes grid, pos, cells, vel, masses and save to file after spin up
# data output during spinup if desired
# spinup(grid, pos, cells, vel, masses, xi)
# in here:
# advection(grid) (grid, dt_adv) spread over particle time step h
# propagation(pos, vel, masses, grid) (particles) # switch gravity on/off here!
# condensation() (particle <-> grid) maybe do together with collision 
# collision() maybe do together with condensation

# 3. changes grid, pos, cells, vel, masses,
# data output to file and 
# save to file in intervals chosen
# need to set start time, end time, integr. params, interval of print,
# interval of full save, ...)
# simulate(grid, pos, vel, masses, xi)

#%% MODULE IMPORTS

import os
#os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
# import math
#import matplotlib.pyplot as plt
import sys
# from datetime import datetime
# import timeit

from grid import compute_no_grid_cells_from_step_sizes

# import constants as c
from init import initialize_grid_and_particles_SinSIP
#, dst_log_normal

#from generate_SIP_ensemble_dst import \
#    gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl

#from analysis import plot_pos_vel_pt

# from grid import Grid
# from grid import interpolate_velocity_from_cell_bilinear,\
#                  compute_cell_and_relative_position
                 # interpolate_velocity_from_position_bilinear,\
from microphysics import compute_mass_from_radius_vec
from microphysics import compute_mass_from_radius_jit
import constants as c
#os.environ["OMP_NUM_THREADS"] = "1"
#                          compute_R_p_w_s_rho_p
#                          compute_initial_mass_fraction_solute_NaCl,\
#                          compute_density_particle,\
#                          compute_dml_and_gamma_impl_Newton_full,\
                         
# from atmosphere import compute_kappa_air_moist,\
#                        compute_pressure_vapor,\
#                        compute_pressure_ideal_gas,\
#                        epsilon_gc, compute_surface_tension_water,\
#                        kappa_air_dry,\
#                        compute_beta_without_liquid,\
#                        compute_temperature_from_potential_temperature_moist                
                       # compute_diffusion_constant,\
                       # compute_thermal_conductivity_air,\
                       # compute_specific_heat_capacity_air_moist,\
                       # compute_heat_of_vaporization,\
                       # compute_saturation_pressure_vapor_liquid,\
# from file_handling import save_grid_and_particles_full
                          # load_grid_and_particles_full
# from file_handling import dump_particle_data, save_grid_scalar_fields                          
#                         save_particles_to_files,\
# from analysis import compare_functions_run_time

#%% storage directories -> need to assign "simdata_path" and "fig_path"
#my_OS = "Linux_desk"
#my_OS = "Mac"
#my_OS = "TROPOS_server"

simdata_path = "/mnt/D/sim_data_cloudMP/"

if len(sys.argv) > 1:
    simdata_path = sys.argv[1]
#    print("my OS entered = ", my_OS)
#else:
#    if(my_OS == "Linux_desk"):
#        home_path = '/home/jdesk/'
#        simdata_path = "/mnt/D/sim_data_cloudMP/"
#    #    sim_data_path = home_path + "OneDrive/python/sim_data/"
#    #    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
#    elif (my_OS == "Mac"):
#        simdata_path = "/Users/bohrer/sim_data_cloudMP/"
#    elif (my_OS == "TROPOS_server"):
#        simdata_path = "/vols/fs1/work/bohrer/sim_data_cloudMP/"
    #    home_path = "/Users/bohrer/"
    #    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
    #    fig_path =\
    #        home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'



#%% I. GENERATE GRID AND PARTICLES

# 1a. generate grid and particles + save initial to file


#%% PARTICLE PARAMETERS

reseed = False
seed_SIP_gen = 2001

if len(sys.argv) > 2:
    seed_SIP_gen = int(sys.argv[2])

#solute_type = "NaCl"
solute_type = "AS"

if len(sys.argv) > 3:
    solute_type = sys.argv[3]

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([2, 3])
#no_spcm = np.array([4, 4])
#no_spcm = np.array([6, 8])
no_spcm = np.array([26, 38])
#no_spcm = np.array([20, 30])
#no_spcm = np.array([20, 20])

if len(sys.argv) > 4:
    no_spcm[0] = int(sys.argv[4])
if len(sys.argv) > 5:
    no_spcm[1] = int(sys.argv[5])

#%% GRID PARAMETERS
# domain size
x_min = 0.0
x_max = 1500.0
z_min = 0.0
z_max = 1500.0

# grid steps
dx = 20.0
dy = 1.0
dz = 20.0

#dx = 50.0
#dy = 1.0
#dz = 50.0

#dx = 100.0
#dy = 1.0
#dz = 100.0

#dx = 150.0
#dy = 1.0
#dz = 150.0

#dx = 500.0
#dy = 1.0
#dz = 500.0

if len(sys.argv) > 6:
    dx = float(sys.argv[6])
if len(sys.argv) > 7:
    dz = float(sys.argv[7])

no_cells = compute_no_grid_cells_from_step_sizes(
               ((x_min, x_max),(z_min, z_max)), (dx, dz) ) 

dV = dx*dy*dz

p_0 = 1015E2 # surface pressure in Pa
p_ref = 1.0E5 # ref pressure for potential temperature in Pa
r_tot_0 = 7.5E-3 # kg water / kg dry air (constant over whole domain in setup)
# r_tot_0 = 22.5E-3 # kg water / kg dry air
# r_tot_0 = 7.5E-3 # kg water / kg dry air
Theta_l = 289.0 # K
   
#%%
grid_folder =\
    f"{solute_type}/" \
    + f"grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
    + f"{seed_SIP_gen}/"

grid_path = simdata_path + grid_folder
if not os.path.exists(grid_path):
    os.makedirs(grid_path)
    
sys.stdout = open(grid_path + "std_out.log", 'w')

dist = "lognormal"
#dist = "expo"

if dist == "lognormal":
    r_critmin = np.array([1., 3.]) * 1E-3 # mu, # mode 1, mode 2, ..
    m_high_over_m_low = 1.0E8
    
    # droplet number concentration 
    # set DNC0 manually only for lognormal distr.
    # number density of particles mode 1 and mode 2:
    DNC0 = np.array([60.0E6, 40.0E6]) # 1/m^3
    # parameters of radial lognormal distribution -> converted to mass
    # parameters below
    mu_R = 0.5 * np.array( [0.04, 0.15] )
    sigma_R = np.array( [1.4,1.6] )
    
elif dist == "expo":
    LWC0 = 1.0E-3 # kg/m^3
    R_mean = 9.3 # in mu
    r_critmin = 0.6 # mu
    m_high_over_m_low = 1.0E6    

    rho_w = 1E3
    m_mean = compute_mass_from_radius_jit(R_mean, rho_w) # in 1E-18 kg
    DNC0 = 1E18 * LWC0 / m_mean # in 1/m^3
    # we need to hack here a little because of the units of m_mean and LWC0
    LWC0_over_DNC0 = m_mean
    DNC0_over_LWC0 = 1.0/m_mean

    dst_par = (DNC0, DNC0_over_LWC0)

#%% SINGLE SIP INITIALIZATION PARAMETERS

#eta = 6E-10
eta = 1E-10
#eta_threshold = "weak"
eta_threshold = "fix"

#%% SATURATION ADJUSTMENT PHASE PARAMETERS

S_init_max = 1.04
dt_init = 0.1 # s
# number of iterations for the root finding
# Newton algorithm in the implicit method
Newton_iterations = 2
# maximal allowed iter counts in initial particle water take up to equilibrium
iter_cnt_limit = 1000

#%% DERIVED AND PATH CREATION

########### OLD WORKING ###################    
# in log(mu)
# par_sigma = np.log( [1.0,1.0] )
# in mu
#dst_par = []
#for i,sig in enumerate(par_sigma):
#    dst_par.append([mu_R[i],sig])
#dst_par = np.array(dst_par)
# parameters of the quantile-integration
#P_min = 0.001
#P_max = 0.999
#dr = 1E-4
#r0 = dr
#r1 = 10 * par_sigma
########### OLD WORKING ###################   

if eta_threshold == "weak":
    weak_threshold = True
else: weak_threshold = False



idx_mode_nonzero = np.nonzero(no_spcm)[0]

no_modes = len(idx_mode_nonzero)

if no_modes == 1:
    no_spcm = no_spcm[idx_mode_nonzero][0]
else:    
    no_spcm = no_spcm[idx_mode_nonzero]


if dist == "lognormal":
    if no_modes == 1:
        sigma_R = sigma_R[idx_mode_nonzero][0]
        mu_R = mu_R[idx_mode_nonzero][0]
#        no_rpcm = no_rpcm[idx_mode_nonzero][0]
        r_critmin = r_critmin[idx_mode_nonzero][0] # mu
        DNC0 = DNC0[idx_mode_nonzero][0] # mu
        
    else:    
        sigma_R = sigma_R[idx_mode_nonzero]
        mu_R = mu_R[idx_mode_nonzero]
#        no_rpcm = no_rpcm[idx_mode_nonzero]
        r_critmin = r_critmin[idx_mode_nonzero] # mu
        DNC0 = DNC0[idx_mode_nonzero] # mu

    mu_R_log = np.log( mu_R )
    sigma_R_log = np.log( sigma_R )    
    # derive parameters of lognormal distribution of mass f_m(m)
    # assuming mu_R in mu and density in kg/m^3
    # mu_m in 1E-18 kg
    mu_m_log = np.log(compute_mass_from_radius_vec(mu_R,
                                                   c.mass_density_NaCl_dry))
    sigma_m_log = 3.0 * sigma_R_log
    dst_par = (mu_m_log, sigma_m_log)



#folder = "grid_75_75_spcm_4_4/"
# folder = "grid_75_75_spcm_20_20/"


if len(sys.argv) > 1:
    print("storage path entered = ", simdata_path)
if len(sys.argv) > 2:
    print("seed SIP gen entered = ", seed_SIP_gen)

if dist == "expo":
    print("dist = expo", f"DNC0 = {DNC0:.3e}", "LWC0 =", LWC0,
          "m_mean = ", m_mean)
elif dist == "lognormal":
    print("dist = lognormal", "DNC0 =", DNC0,
          "mu_R =", mu_R, "sigma_R =", sigma_R,
           "r_critmin =", r_critmin,
           "m_high_over_m_low =", m_high_over_m_low)

print("no_modes, idx_mode_nonzero:", no_modes, ",", idx_mode_nonzero)

#%% GENERATE GRID AND PARTICLES

#if act_gen_grid:
grid, pos, cells, cells_comb, vel, m_w, m_s, xi, active_ids  = \
    initialize_grid_and_particles_SinSIP(
        x_min, x_max, z_min, z_max, dx, dy, dz,
        p_0, p_ref, r_tot_0, Theta_l, solute_type,
        DNC0, no_spcm, no_modes, dist, dst_par,
        eta, eta_threshold, r_critmin, m_high_over_m_low,
        seed_SIP_gen, reseed,
        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, grid_path)

#%% PLOT PARTICLES WITH VELOCITY VECTORS

#fig_name = grid_path + "pos_vel_t0.png"
#plot_pos_vel_pt(pos, vel, grid,
#                    figsize=(8,8), no_ticks = [11,11],
#                    MS = 1.0, ARRSCALE=10, fig_name=fig_name)

#grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids =\
#    initialize_grid_and_particles(
#        x_min, x_max, z_min, z_max, dx, dy, dz,
#        p_0, p_ref, r_tot_0, Theta_l,
#        n_p, no_spcm, dst, dst_par, 
#        P_min, P_max, r0, r1, dr, rnd_seed, reseed,
#        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, path)

#%% simple plots
# IN WORK: ADD AUTOMATIC CREATION OF DRY SIZE SPECTRA COMPARED WITH EXPECTED PDF
# FOR ALL MODES AND SAVE AS FILE 
# grid.plot_thermodynamic_scalar_fields_grid()