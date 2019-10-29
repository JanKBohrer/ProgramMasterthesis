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
#from collision.box_model import plot_for_given_kappa
from collision.box_model import plot_moments_kappa_var

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

from golovin import dist_vs_time_golo_exp

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
mpl.rcParams.update(plt.rcParamsDefault)
#mpl.use("pdf")
mpl.use("pgf")

#import numpy as np

from microphysics import compute_mass_from_radius_vec
import constants as c

#import numba as 

from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict, pdf_dict
plt.rcParams.update(pgf_dict)
#plt.rcParams.update(pdf_dict)

from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

#%% SET DEFAULT PLOT PARAMETERS
# (can be changed lateron for specific elements directly)
# TITLE, LABEL (and legend), TICKLABEL FONTSIZES
TTFS = 10
LFS = 10
TKFS = 8

#TTFS = 16
#LFS = 12
#TKFS = 12

# LINEWIDTH, MARKERSIZE
LW = 1.2
MS = 2

# raster resolution for e.g. .png
DPI = 600

mpl.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))

#%% SET PARAMETERS 

set_log_file = True

OS = "LinuxDesk"
#OS = "Mac"
#OS = "TROPOS_server"
#OS = "LinuxNote"

if OS == "Mac":
#    sim_data_path = "/Users/bohrer/sim_data_cloudMP/"
    sim_data_path = "/Users/bohrer/sim_data_col_box_mod/"
elif OS == "LinuxDesk":
    sim_data_path = "/mnt/D/sim_data_col_box_mod/"
    home_path = "/home/jdesk/"
elif OS == "LinuxNote":    
    sim_data_path = "/mnt/D/sim_data_col_box_mod/"
elif OS == "TROPOS_server":
    sim_data_path = "/vols/fs1/work/bohrer/sim_data_col_box_mod/" 

############################################################################################
# SET args for SIP ensemble generation AND Kernel-grid generation
#args_gen = [1,1,1,1,1]
#args_gen = [1,0,0,0,0]
#args_gen = [0,0,0,0,1]
args_gen = [0,0,0,0,0]
#args_gen = [1,1,1,0,0]

act_gen_SIP = bool(args_gen[0])
act_analysis_ensembles = bool(args_gen[1])
# NOTE: plotting needs analyze first (or in buffer)
act_plot_ensembles = bool(args_gen[2])
act_gen_kernel_grid = bool(args_gen[3])
act_gen_Ecol_grid = bool(args_gen[4])

# SET args for simulation AND plotting
args_sim = [0,0,0,0,1]

#args_sim = [0,0,0,0]
#args_sim = [1,0,0,0]
#args_sim = [0,1,1,1]
#args_sim = [0,0,1,1]
#args_sim = [0,0,1,1]
#args_sim = [0,0,1,0]
#args_sim = [0,0,1,0]
#args_sim = [0,0,0,1]
#args_sim = [1,1,1,1]

print_reduction = False
#print_reduction = True

#setup = "Golovin"
#setup = "Long_Bott"
setup = "Hall_Bott"

act_sim = bool(args_sim[0])
act_analysis = bool(args_sim[1])
act_plot = bool(args_sim[2])
act_plot_moments_kappa_var = bool(args_sim[3])
act_plot_convergence_t_fix = bool(args_sim[4])


############################################################################################
### SET PARAMETERS FOR SIMULATION OF COLLISION BOX MODEL

# this is modified below for the chosen operations...
kappa_list=[5,10,20,40,100,400,1000,2000,3000]

#kappa_list=[3.5]
#kappa_list=[200]
if act_plot:
    kappa_list=[5]
#kappa_list=[5,10,20,40,60]

# Golovin
if act_plot_moments_kappa_var:
    if setup == "Golovin":
        kappa_list=[5,10,20,40,60,100,200]
    else:
#        kappa_list=[5,10,20,40,100,200,400,1000,3000]
        kappa_list=[5,10,20,40,100,400,1000,2000,3000]

#kappa_list=[5,10,20,40,60,100,200,400,600,800,1000,1500,2000,3000]

#kappa_list=[5,10,20,40,60,100,200,400]

#kappa_list=[3,3.5,5,10,20,40,60,100,200,400]
#kappa_list=[800]
#kappa_list=[5,10,20,40,60,100,200,400,600,800]


#kappa_list=[5,10,20,40,100,200,400,1000,2000,3000]

#kappa_list=[5,10,20,40,60,100,200,400,600,800,1000,1500,2000,3000]
#kappa_list=[200,400,600,800]

# for plotting of g_ln_R
# Golovin
if setup == "Golovin":
    kappa1 = 10
    kappa2 = 200
    start_seed = 5711
    kernel_name = "Golovin"
    kernel_method = "analytic"
    dt = 1.0
    eta_threshold = "weak"
# Long and Hall
else:
    kappa1 = 10
    kappa2 = 3000
    start_seed = 3711
    kernel_name = setup
    kernel_name = setup
    #kernel_method = "kernel_grid_m"
    kernel_method = "Ecol_grid_R"
    dt = 10.0
    eta_threshold = "fix"

#no_sims = 100
no_sims = 500
#no_sims = 10
#no_sims = 400
# LOng and Hall
# Golovin
#start_seed = 8711
#start_seed = 94711

seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

# dt = 20.0

# dt_save = 40.0
dt_save = 300.0
#dt_save = 150.0
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

#%% MOMENTS OF INITIAL EXPO DIST

def moments_expo(n, DNC0, LWC0):
    if n == 0:
        return DNC0
    elif n == 1:
        return LWC0
    else:    
        return math.factorial(n) * DNC0 * (LWC0/DNC0)**n

#%% FUNCTION DEF: PLOTTING OF g_ln_R
# figname is full path
# time_idx: chosen time indices to plot referring to "save_times"
#  
def plot_g_ln_R_for_given_kappa(kappa, eta, dt, no_sims, start_seed, no_bins,
                                DNC0, m_mean,
                                kernel_name, gen_method, bin_method, time_idx,
                                load_dir, load_dir_k1, load_dir_k2,
                                figsize, figname,
                                plot_compare,
                                figsize2, figname2):
#    bG = 1.5E-18 # K = b * (m1 + m2) # b in m^3/(fg s)
    bG = 1.5 # K = b * (m1 + m2) # b in m^3/(fg s)
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    no_times = len(save_times)
    # bins_mass_centers =
    bins_mass_centers = np.load(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy")
    bins_rad_centers = np.load(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy")
#    print(no_bins)
#    no_masses = bins_mass_centers[time_n]
    
    add_masses = 4
    
#    print(save_times)
#    print(bins_mass_centers[0])
    
    f_m_num_avg_vs_time = np.load(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_m_num_avg_vs_time = np.load(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_ln_r_num_avg_vs_time = np.load(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_ln_r_num_std_vs_time = np.load(load_dir + f"g_ln_r_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")


    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")

#    fig_name = "fm_gm_glnr_vs_t"
#    fig_name += f"_kappa_{kappa}_dt_{int(dt)}_no_sims_{no_sims}_no_bins_{no_bins}.png"
    no_rows = 1
    fig, axes = plt.subplots(nrows=no_rows, figsize=figsize)
    
    ax = axes
    ax.set_xscale("log", nonposx="mask")    
    ax.set_yscale("log", nonposy="mask")    
#    ax.set_yscale("log")
#    ax.set_xscale("log")    
#    for time_n in range(no_times):
    for time_n in time_idx:
        mask = g_ln_r_num_avg_vs_time[time_n]*1000.0 > 1E-6
#        print("g_ln_r_num_avg_vs_time[time_n]*1000.0")
#        print(g_ln_r_num_avg_vs_time[time_n]*1000.0)
        ax.plot(bins_rad_centers[time_n][0][mask],
                g_ln_r_num_avg_vs_time[time_n][mask]*1000.0,
                label = f"{int(save_times[time_n]//60)}", zorder=50)
        
#        below_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0        

        above_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
                      + g_ln_r_num_std_vs_time[time_n]*1000.0
        above_curve = \
            np.where(above_curve <= 1E-4, 1E-4, above_curve)
#        above_curve = \
#            np.where(g_ln_r_num_avg_vs_time[time_n]*1000.0 <= 1E-4,
#                     g_ln_r_num_avg_vs_time[time_n]*1000.0,
#                     above_curve)
        
        below_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
                      - g_ln_r_num_std_vs_time[time_n]*1000.0
        below_curve = \
            np.where(below_curve <= 1E-4, 1E-4, below_curve)
#        below_curve = \
#            np.where(g_ln_r_num_avg_vs_time[time_n]*1000.0 <= 1E-4,
#                     g_ln_r_num_avg_vs_time[time_n]*1000.0, below_curve)
        
#        print(bins_rad_centers[time_n][0][mask])
        
        ax.fill_between(bins_rad_centers[time_n][0][mask],
                        below_curve[mask],
                        above_curve[mask],
#                        g_ln_r_num_avg_vs_time[time_n]*1000.0 \
#                        + g_ln_r_num_std_vs_time[time_n]*1000.0,
#                                rel_dev[0,row_n,col_n] - std_rel[0,0,row_n,col_n],
#                                rel_dev[0,row_n,col_n] + std_rel[1,0,row_n,col_n],
#                                alpha=0.15, facecolor="green")
                        alpha=0.4, lw=1
#                                edgecolor=face_colors[var_n],
#                                label=data_labels_mom[var_n]
                        )              
    ax.set_prop_cycle(None)
    for j,time_n in enumerate(time_idx):
        # t in s
        # x0 (mass mean) in 1E-18 kg
        # n0 (DNC0) in 1/m^3
#        print(time_n, bins_mass_centers[time_n])
        if kernel_name == "Golovin":
            scale_g=1000.      
            no_bins_ref = 2*no_bins
            ref_masses = np.zeros(no_bins_ref + add_masses)
            
            bin_factor = np.sqrt(bins_mass_centers[time_n][0][-1]/bins_mass_centers[time_n][0][-2])
#            bin_factor = bins_mass_centers[time_n][0][-1]/bins_mass_centers[time_n][0][-2]
            
#            ref_masses[:no_bins] = bins_mass_centers[time_n][0]
            ref_masses[0] = bins_mass_centers[time_n][0][0]

            for n in range(1,no_bins_ref + add_masses):
                ref_masses[n] = ref_masses[n-1]*bin_factor
#            for n in range(add_masses):
#                ref_masses[no_bins+n] = ref_masses[no_bins+n-1]*bin_factor
            
            # double the number of ref masses on a log scale
            
    #        print(len(ref_masses))
    #        print(ref_masses)
            f_m_golo = dist_vs_time_golo_exp(ref_masses,
                                             save_times[time_n],m_mean,DNC0,bG)
    #        print(len(bins_mass_centers[time_n][0]))
            g_ln_r_golo = f_m_golo * 3 * ref_masses**2
            g_ln_r_ref = g_ln_r_golo
            ref_radii = 1E6 * (3. * ref_masses / (4. * math.pi * 1E3))**(1./3.)
            
#        elif kernel_name == "Long_Bott":
        else:
            scale_g = 1.
            dp = f"collision/ref_data/{kernel_name}/"
            ref_radii = np.loadtxt(dp + "Wang_2007_radius_bin_centers.txt")[j][::5]
            g_ln_r_ref = np.loadtxt(dp + "Wang_2007_g_ln_R.txt")[j][::5]
        fmt="o"    
#        print(g_ln_r_golo)
        print(j)
        print(ref_radii.shape)
        ax.plot(ref_radii,
                g_ln_r_ref*scale_g,
#                fmt,
#                ":",
                "o",
                fillstyle='none',
                linewidth = 2,
                markersize = 3, mew=0.4)                

#                linestyle=":", linewidth = 2.5)
#        ax.plot(ref_radii,
#                g_ln_r_ref, linestyle=":", linewidth = 2.5)
    # ax.plot(bins_rad_centers[0][0], g_ln_r_num_avg_vs_time[0]*1000.0, "x")
    # ax.plot(R_, g_ln_r_ana_)
    
    
    ax.set_xlabel("Radius ($\si{\micro\meter}$)")
    ax.set_ylabel(r"$g_{\ln(R)}$ $\mathrm{(g \; m^{-3})}$")
    # ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),18))
    # ax.xaxis.set_ticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
    # ax.set_xticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
    # import matplotlib
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    # ax.yaxis.set_ticks(np.logspace(-11,0,12))
    if kernel_name == "Golovin":
        ax.set_xticks( np.logspace(0,3,4) )
        # ax.set_yticks( np.logspace(-4,1,6) )
        ax.set_xlim([1.0,2.0E3])
        ax.set_ylim([1.0E-4,10.0])
    elif kernel_name == "Long_Bott":
        ax.set_xlim([1.0,5.0E3])
        ax.set_ylim([1.0E-4,10.0])
    elif kernel_name == "Hall_Bott":
        ax.set_xlim([1.0,5.0E3])
        ax.set_yticks( np.logspace(-4,2,7) )
        ax.set_ylim([1.0E-4,10.0])        



    ax.grid(which="major")
#    ax.legend(ncol=4, prop={'size': 6}, fontsize=20)
#    ax.legend(ncol=4, handlelength=0.5)
    ax.legend(ncol=7, handlelength=0.8, handletextpad=0.2,
              columnspacing=0.8, borderpad=0.2, loc="upper center")
#    for ax in axes:
    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)

#    fig.suptitle(
#f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
#gen_method={gen_method}, kernel={kernel_name}, bin_method={bin_method}")
#    fig.tight_layout()
#    plt.subplots_adjust(top=0.95)
    
#    fig.savefig(figname)
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.065
                )    
############################################# PLOT 2 KAPPAS FOR COMARISON
############################################# PLOT 2 KAPPAS FOR COMARISON
    if plot_compare:
        print("PLOT COMPARE STARTED")
        no_rows = 2
        fig, axes = plt.subplots(nrows=no_rows, figsize=figsize2)
        
        load_dirs = (load_dir_k1, load_dir_k2)
        
        for n,kappa in enumerate((kappa1,kappa2)):
            load_dir = load_dirs[n]
            bins_mass_centers = np.load(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy")
            bins_rad_centers = np.load(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy")
        
            add_masses = 4
            
            f_m_num_avg_vs_time = np.load(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
            g_m_num_avg_vs_time = np.load(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
            g_ln_r_num_avg_vs_time = np.load(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
            g_ln_r_num_std_vs_time = np.load(load_dir + f"g_ln_r_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#            g_ln_r_num_std_vs_time_no_sims_500_no_bins_50        
            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")        
            
            LW_compare = 1.0
#            LW_compare = 0.8
            ax = axes[n]
            
            ax.set_xscale("log", nonposx="mask")    
            ax.set_yscale("log", nonposy="mask")                
        
        #    for time_n in range(no_times):
            for time_n in time_idx:
                mask = g_ln_r_num_avg_vs_time[time_n]*1000.0 > 1E-6
        #        print("g_ln_r_num_avg_vs_time[time_n]*1000.0")
        #        print(g_ln_r_num_avg_vs_time[time_n]*1000.0)
                ax.plot(bins_rad_centers[time_n][0][mask],
                        g_ln_r_num_avg_vs_time[time_n][mask]*1000.0,
                        label = f"{int(save_times[time_n]//60)}", zorder=50)
                
        #        below_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0        
        
                above_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
                              + g_ln_r_num_std_vs_time[time_n]*1000.0
                above_curve = \
                    np.where(above_curve <= 1E-4, 1E-4, above_curve)
        #        above_curve = \
        #            np.where(g_ln_r_num_avg_vs_time[time_n]*1000.0 <= 1E-4,
        #                     g_ln_r_num_avg_vs_time[time_n]*1000.0,
        #                     above_curve)
                
                below_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
                              - g_ln_r_num_std_vs_time[time_n]*1000.0
                below_curve = \
                    np.where(below_curve <= 1E-4, 1E-4, below_curve)
        #        below_curve = \
        #            np.where(g_ln_r_num_avg_vs_time[time_n]*1000.0 <= 1E-4,
        #                     g_ln_r_num_avg_vs_time[time_n]*1000.0, below_curve)
                
        #        print(bins_rad_centers[time_n][0][mask])
                
                ax.fill_between(bins_rad_centers[time_n][0][mask],
                                below_curve[mask],
                                above_curve[mask],
        #                        g_ln_r_num_avg_vs_time[time_n]*1000.0 \
        #                        + g_ln_r_num_std_vs_time[time_n]*1000.0,
        #                                rel_dev[0,row_n,col_n] - std_rel[0,0,row_n,col_n],
        #                                rel_dev[0,row_n,col_n] + std_rel[1,0,row_n,col_n],
        #                                alpha=0.15, facecolor="green")
                                alpha=0.4, lw=1
        #                                edgecolor=face_colors[var_n],
        #                                label=data_labels_mom[var_n]
                                )                          
#                below_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0

#                below_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
#                              - g_ln_r_num_std_vs_time[time_n]*1000.0
#                below_curve = \
#                    np.where(below_curve <= 1E-4, 1E-4, below_curve)
#                    
#                ax.plot(bins_rad_centers[time_n][0],
#                        g_ln_r_num_avg_vs_time[time_n]*1000.0,
#                        label = f"{int(save_times[time_n]//60)}",
#                        linewidth=LW_compare,
#                        zorder=50
#                        )
##                ax.plot(bins_rad_centers[time_n][0],
##                        below_curve, "-", c="k")
#                ax.fill_between(bins_rad_centers[time_n][0],
#                                below_curve,
#                                g_ln_r_num_avg_vs_time[time_n]*1000.0 + g_ln_r_num_std_vs_time[time_n]*1000.0,
##                                rel_dev[0,row_n,col_n] - std_rel[0,0,row_n,col_n],
##                                rel_dev[0,row_n,col_n] + std_rel[1,0,row_n,col_n],
##                                alpha=0.15, facecolor="green")
#                                alpha=0.4,
#                                zorder=48
##                                lw=1
##                                edgecolor=face_colors[var_n],
##                                label=data_labels_mom[var_n]
#                                )                
            ax.set_prop_cycle(None)
#            for time_n in time_idx:
#                # t in s
#                # x0 (mass mean) in 1E-18 kg
#                # n0 (DNC0) in 1/m^3
#        #        print(time_n, bins_mass_centers[time_n])
#                ref_masses = np.zeros(no_bins+add_masses)
#                bin_factor = bins_mass_centers[time_n][0][-1]/bins_mass_centers[time_n][0][-2]
#                ref_masses[:no_bins] = bins_mass_centers[time_n][0]
#                for n in range(add_masses):
#                    ref_masses[no_bins+n] = ref_masses[no_bins+n-1]*bin_factor
#        #        print(len(ref_masses))
#        #        print(ref_masses)
#                f_m_golo = dist_vs_time_golo_exp(ref_masses,
#                                                 save_times[time_n],m_mean,DNC0,bG)
#        #        print(len(bins_mass_centers[time_n][0]))
#                g_ln_r_golo = f_m_golo * 3 * ref_masses**2
#                ref_radii = 1E6 * (3. * ref_masses / (4. * math.pi * 1E3))**(1./3.)
#        #        print(g_ln_r_golo)
#                ax.plot(ref_radii,
#                        g_ln_r_golo*1000, linestyle=":", linewidth = 2.5)
            for j,time_n in enumerate(time_idx):
                # t in s
                # x0 (mass mean) in 1E-18 kg
                # n0 (DNC0) in 1/m^3
        #        print(time_n, bins_mass_centers[time_n])
#                if kernel_name == "Golovin":
#                    scale_g=1000.
#                    ref_masses = np.zeros(no_bins+add_masses)
#                    bin_factor = bins_mass_centers[time_n][0][-1]/bins_mass_centers[time_n][0][-2]
#                    ref_masses[:no_bins] = bins_mass_centers[time_n][0]
#                    for n in range(add_masses):
#                        ref_masses[no_bins+n] = ref_masses[no_bins+n-1]*bin_factor
#            #        print(len(ref_masses))
#            #        print(ref_masses)
#                    f_m_golo = dist_vs_time_golo_exp(ref_masses,
#                                                     save_times[time_n],m_mean,DNC0,bG)
#            #        print(len(bins_mass_centers[time_n][0]))
#                    g_ln_r_golo = f_m_golo * 3 * ref_masses**2
#                    g_ln_r_ref = g_ln_r_golo
#                    ref_radii = 1E6 * (3. * ref_masses / (4. * math.pi * 1E3))**(1./3.)
                if kernel_name == "Golovin":
                    scale_g=1000.      
                    no_bins_ref = 2*no_bins
                    ref_masses = np.zeros(no_bins_ref+add_masses)
                    
                    bin_factor = np.sqrt(bins_mass_centers[time_n][0][-1]/bins_mass_centers[time_n][0][-2])
        #            bin_factor = bins_mass_centers[time_n][0][-1]/bins_mass_centers[time_n][0][-2]
                    
        #            ref_masses[:no_bins] = bins_mass_centers[time_n][0]
                    ref_masses[0] = bins_mass_centers[time_n][0][0]
        
                    for n in range(1,no_bins_ref+add_masses):
                        ref_masses[n] = ref_masses[n-1]*bin_factor
        #            for n in range(add_masses):
        #                ref_masses[no_bins+n] = ref_masses[no_bins+n-1]*bin_factor
                    
                    # double the number of ref masses on a log scale
                    
            #        print(len(ref_masses))
            #        print(ref_masses)
                    f_m_golo = dist_vs_time_golo_exp(ref_masses,
                                                     save_times[time_n],m_mean,DNC0,bG)
            #        print(len(bins_mass_centers[time_n][0]))
                    g_ln_r_golo = f_m_golo * 3 * ref_masses**2
                    g_ln_r_ref = g_ln_r_golo
                    ref_radii = 1E6 * (3. * ref_masses / (4. * math.pi * 1E3))**(1./3.)                
                else:
                    scale_g=1.
#                elif kernel_name == "Long_Bott":
                    dp = f"collision/ref_data/{kernel_name}/"
                    ref_radii = np.loadtxt(dp + "Wang_2007_radius_bin_centers.txt")[j][::5]
                    g_ln_r_ref = np.loadtxt(dp + "Wang_2007_g_ln_R.txt")[j][::5]
                    
        #        print(g_ln_r_golo)
                fmt="o"    
        #        print(g_ln_r_golo)
                ax.plot(ref_radii,
                        g_ln_r_ref*scale_g,
                        fmt,
                        fillstyle='none',
                        linewidth = 2,
                        markersize = 2.3,
                        mew=0.3, zorder=40)             
#                ax.plot(ref_radii,
#                        g_ln_r_ref, linestyle=(0, (1, 1)), linewidth = 2.5)                
#                        g_ln_r_ref, linestyle=":", linewidth = 2.5)                
            # ax.plot(bins_rad_centers[0][0], g_ln_r_num_avg_vs_time[0]*1000.0, "x")
            # ax.plot(R_, g_ln_r_ana_)
            
#            ax.set_yscale("log")
#            ax.set_xscale("log")

#            if n==1:
            
            # ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),18))
            # ax.xaxis.set_ticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
            # ax.set_xticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
            # import matplotlib
            # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
            # ax.yaxis.set_ticks(np.logspace(-11,0,12))
            if kernel_name == "Golovin":
                ax.set_xticks( np.logspace(0,3,4) )
                ax.set_yticks( np.logspace(-4,0,5) )
                ax.set_xlim([1.0,2.0E3])
                ax.set_ylim([1.0E-4,10.0])
            elif kernel_name == "Long_Bott":
                ax.set_xlim([1.0,5.0E3])
                ax.set_yticks( np.logspace(-4,2,7) )
                ax.set_ylim([1.0E-4,10.0])
            elif kernel_name == "Hall_Bott":
                ax.set_xlim([1.0,5.0E3])
                ax.set_yticks( np.logspace(-4,2,7) )
                ax.set_ylim([1.0E-4,10.0])
            ax.grid(which="major")
        #    ax.legend(ncol=4, prop={'size': 6}, fontsize=20)
        #    ax.legend(ncol=4, handlelength=0.5)
#        if n==0:    
#        for ax in axes:
            
        axes[1].set_xlabel("Radius ($\si{\micro\meter}$)")
        axes[0].set_ylabel(r"$g_{\ln(R)}$ $\mathrm{(g \; m^{-3})}$")            
        axes[1].set_ylabel(r"$g_{\ln(R)}$ $\mathrm{(g \; m^{-3})}$")            
        
        axes[0].legend(ncol=7, handlelength=0.8, handletextpad=0.2,
                          columnspacing=0.8, borderpad=0.15, loc="upper center",
                          bbox_to_anchor=(0.5,1.02))
    #    for ax in axes:
#        if n==0:
        axes[0].tick_params(which="both", bottom=True, top=True,
                           left=True, right=True, labelbottom=False)
#        else:
        axes[1].tick_params(which="both", bottom=True, top=True,
                           left=True, right=True)
    #        ax.tick_params(axis='both', which='minor', labelsize=TKFS,
    #                       width=0.6, size=2, labelleft=False)
    #    fig.suptitle(
    #f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
    #gen_method={gen_method}, kernel={kernel_name}, bin_method={bin_method}")
    #    fig.tight_layout()
    #    plt.subplots_adjust(top=0.95)
        
    #    fig.savefig(figname)
        xpos_ = -0.054
        ypos_ = 0.86
        fig.text(xpos_, ypos_ , r"\textbf{(a)}", fontsize=LFS)    
        fig.text(xpos_, ypos_*0.51, r"\textbf{(b)}", fontsize=LFS)    
#        fig.text(-0.05, 0.85, "(b)", fontsize=LFS)    
        fig.savefig(figname2,
#                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.05
                    )    
#    ### PLOT MOMENTS VS TIME
#    t_Unt = [0,10,20,30,35,40,50,55,60]
#    lam0_Unt = [2.97E8, 2.92E8, 2.82E8, 2.67E8, 2.1E8, 1.4E8,  1.4E7, 4.0E6, 1.2E6]
#    t_Unt2 = [0,10,20,30,40,50,60]
#    lam2_Unt = [8.0E-15, 9.0E-15, 9.5E-15, 6E-13, 2E-10, 7E-9, 2.5E-8]
#    
##    fig_name = "moments_vs_time"
##    fig_name += f"_kappa_{kappa}_dt_{int(dt)}_no_sims_{no_sims}_no_bins_{no_bins}.png"
#    no_rows = 4
#    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
#    for i,ax in enumerate(axes):
#        ax.plot(save_times/60, moments_vs_time_avg[:,i],"x-")
#        if i != 1:
#            ax.set_yscale("log")
#        ax.grid()
#        ax.set_xticks(save_times/60)
#        ax.set_xlim([save_times[0]/60, save_times[-1]/60])
#        # ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
#        #                  bottom=True, top=False, left=False, right=False)
#        ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
#    if kernel_name == "Golovin":
#        axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
#        axes[2].set_yticks( np.logspace(-15,-9,7) )
#        axes[3].set_yticks( np.logspace(-26,-15,12) )
#    
#    axes[0].plot(t_Unt,lam0_Unt, "o")
#    axes[2].plot(t_Unt2,lam2_Unt, "o")
#    fig.suptitle(
#f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
#gen_method={gen_method}, kernel={kernel_name}, bin_method={bin_method}")
#    fig.tight_layout()
#    plt.subplots_adjust(top=0.95)
#    fig.savefig(load_dir + fig_name)
        
    plt.close("all")


#%% FUNCTION DEF: MOMENTS VS TIME FOR DIF KAPPA

def compute_moments_Golovin(t, n, DNC, LWC, b):
    if n == 0:
        mom = DNC * np.exp(-b*LWC*t)
    elif n == 1:
        mom = LWC
    elif n == 2:
        mom = 2* LWC * LWC / DNC * np.exp(2*b*LWC*t)
    elif n == 3:
        mom =  LWC**3 / (DNC**2)\
               * (12 * np.exp(4*b*LWC*t) - 6 * np.exp(3*b*LWC*t))
#        mom =  LWC**3 / (DNC**2)\
#               * (36 * np.exp(4*b*LWC*t) - 30 * np.exp(3*b*LWC*t))
    return mom

def plot_moments_vs_time_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
                           kernel_name, gen_method,
                           dist, start_seed,
                           moments_ref, times_ref,
                           sim_data_path,
                           result_path_add,
                           figsize, figname, figsize2, figname2,
                           figsize3, figname3, figsize4, figname4,
                           TTFS, LFS, TKFS):
#    mom0_last_time_Unt = np.array([1.0E7,5.0E6,1.8E6,1.0E6,8.0E5,
#                                   5.0E5,5.0E5,5.0E5,5.0E5,5.0E5])
    
#    t_Unt = [0,10,20,30,35,40,50,55,60]
#    lam0_Unt = [2.97E8, 2.92E8, 2.82E8, 2.67E8, 2.1E8, 1.4E8,  1.4E7, 4.0E6, 1.2E6]
#    t_Unt2 = [0,10,20,30,40,50,60]
#    lam2_Unt = [8.0E-15, 9.0E-15, 9.5E-15, 6E-13, 2E-10, 7E-9, 2.5E-8]

    # Wang 2007: s = 16 -> kappa = 53.151 = s * log_2(10)
#    t_Wang = np.linspace(0,60,7)
##    moments_vs_time_Wang = np.loadtxt(ref_data_path)
#    moments_vs_time_Wang = np.array(ref_data_list)
##        sim_data_path + f"col_box_mod/results/{dist}/{kernel_name}/Wang2007_results2.txt")
#    moments_vs_time_Wang = np.reshape(moments_vs_time_Wang,(4,7)).T
#    moments_vs_time_Wang[:,0] *= 1.0E6
#    moments_vs_time_Wang[:,2] *= 1.0E-6
#    moments_vs_time_Wang[:,3] *= 1.0E-12

    no_kappas = len(kappa_list)
    
#    fig_name = f"moments_vs_time_kappa_var_{no_kappas}"
#    # fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.png"
##    fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.png"
#    fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.pdf"
    no_rows = 3
#    no_rows = 4
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(figsize), sharex=True)
    
#    mom0_last_time = np.zeros(len(kappa_list),dtype=np.float64)
    
#    marker_cycle = ["o"  ]
    
#    if kernel_name == "Golovin":
#        pass
    
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#        moments_vs_time_avg[:,1] *= 1.0E3
#        mom0_last_time[kappa_n] = moments_vs_time_avg[-1,0]
#        print(moments_vs_time_avg.shape)
        if kappa_n < 10: fmt = "-"
        else: fmt = "x--"            
        
        for ax_n,i in enumerate((0,2,3)):
#            if ax_n == 0:
#            lab = f"{float(kappa*5):.2}"
            if kappa*5 < 100000:
                lab = f"{kappa*5}"
            else:
                lab = f"{float(kappa*5):.2}"
#            if kernel_name == "Golovin":
#                axes[ax_n].plot(save_times/60, moments_vs_time_avg[:,i],fmt,
#    #                            label=r"\num{5}",
#                                label=f"\\num{{{lab}}}",
#                                lw=1.2,
#                                ms=5,                        
#                                mew=0.8,
#                                zorder=98)
#            else:
            axes[ax_n].plot(save_times/60, moments_vs_time_avg[:,i],fmt,
#                            label=r"\num{5}",
                            label=f"\\num{{{lab}}}",
                            lw=1.2,
                            ms=5,                        
                            mew=0.8,
                            zorder=98)
#            else:
#                axes[ax_n].plot(save_times/60, moments_vs_time_avg[:,i],fmt
#                                )
    if kernel_name == "Golovin":
        DNC = 296799076.3
        LWC = 1E-3
        bG = 1.5
            
    for ax_n,i in enumerate((0,2,3)):
#    for i,ax in enumerate(axes):
#        if kernel_name == "Long_Bott" or kernel_name == "Hall_Bott":
#            ax.plot(times_ref/60, moments_ref[i],
#                    "o", c = "k",fillstyle='none', markersize = 8, mew=1.0, label="Wang")
        ax = axes[ax_n]
        if kernel_name == "Golovin":
            
            fmt = "o"
#            bG = 1.5E-18
            times_ref = np.linspace(0.,3600.,19)
#            fmt = ":"
#            moments_log = np.log(moments_ref[i])
#            bref = moments_log[0]
#            if i == 0: aref = -bG * moments_ref[1][0]
#            else: aref = (moments_log[1] - moments_log[0]) / times_ref[-1]
#            def linFitRef(t):
#                return bref + aref * t
#            moments_ref_i = np.exp( bref + aref * times_ref )
            moments_ref_i = compute_moments_Golovin(times_ref, i, DNC, LWC, bG)
#            moments_ref_at_save_times = compute_moments_Golovin(save_times, i, DNC, LWC, bG)
#            print("moments ", i)
#            print(moments_vs_time_avg[:,i])
#            print("rel dev moment ", i)
#            print((moments_ref_at_save_times - moments_vs_time_avg[:,i])/moments_ref_at_save_times)
        else:
            fmt = "o"
            moments_ref_i = moments_ref[i]
        ax.plot(times_ref/60, moments_ref_i,
                fmt, c = "k",
                fillstyle='none',
                linewidth = 2,
                markersize = 3, mew=0.4,
                label="Ref",
                zorder=99)
        if i != 1:
            ax.set_yscale("log")
        if kernel_name == "Long_Bott":
            ax.grid(which="major")
        else:
            ax.grid(which="both")
        # if i ==0: ax.legend()
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()      
            
            if kernel_name == "Golovin":
#                pass
                ax.legend(
                        np.reshape(handles, (2,4)).T.flatten(),
                        np.reshape(labels, (2,4)).T.flatten(),
    #                    np.concatenate((handles[::2],handles[1::2]),axis=0),
    #                      np.concatenate((labels[::2],labels[1::2]),axis=0),
                          ncol=4, handlelength=0.8, handletextpad=0.2,
                          columnspacing=0.5, borderpad=0.2, loc="upper right",
                          bbox_to_anchor=(1.015, 1.04))            
        if i == 2:
            if kernel_name == "Long_Bott":
                ax.legend(
#                        np.reshape(handles, (5,2)).T.flatten(),
#                        np.reshape(labels, (5,2)).T.flatten(),
    #                    np.concatenate((handles[::2],handles[1::2]),axis=0),
    #                      np.concatenate((labels[::2],labels[1::2]),axis=0),
                          ncol=2, handlelength=0.8, handletextpad=0.2,
                          columnspacing=0.5, borderpad=0.2, loc="lower left",
                          bbox_to_anchor=(0.0, 0.6)).set_zorder(100)            
            if kernel_name == "Hall_Bott":
                ax.legend(
#                        np.reshape(handles, (5,2)).T.flatten(),
#                        np.reshape(labels, (5,2)).T.flatten(),
    #                    np.concatenate((handles[::2],handles[1::2]),axis=0),
    #                      np.concatenate((labels[::2],labels[1::2]),axis=0),
                          ncol=2, handlelength=0.8, handletextpad=0.2,
                          columnspacing=0.5, borderpad=0.2, loc="lower left",
                          bbox_to_anchor=(0.0, 0.5)).set_zorder(100)            
#        if i == 1:
#            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.05),
#                      fontsize=TKFS)
        ax.set_xticks(save_times[::2]/60)
        ax.set_xlim([save_times[0]/60, save_times[-1]/60])
        # ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
        #                  bottom=True, top=False, left=False, right=False)
        ax.tick_params(which="both", bottom=True, top=True,
                       left=True, right=True
                       )
        ax.tick_params(axis='both', which='major', labelsize=TKFS,
                       width=0.8, size=3)
        ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                       width=0.6, size=2, labelleft=False)
#        set_yticklabels(ha = 'left')
#        ax.yaxis.major_ticklabels.set_ha("left")
#        ax.tick_params(axis='both', which='minor', )
    axes[-1].set_xlabel("Time (min)",fontsize=LFS)
    axes[0].set_ylabel(r"$\lambda_0$ = DNC $(\mathrm{m^{-3}})$ ",
                       fontsize=LFS)
#    axes[1].set_ylabel(r"$\lambda_1$ = LWC $(\mathrm{kg \, m^{-3}})$ ",
#                       fontsize=LFS)
    axes[1].set_ylabel(r"$\lambda_2$ $(\mathrm{kg^2 \, m^{-3}})$ ",
                       fontsize=LFS)
    axes[2].set_ylabel(r"$\lambda_3$ $(\mathrm{kg^3 \, m^{-3}})$ ",
                       fontsize=LFS)
    if kernel_name == "Golovin":
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
        axes[0].set_yticks([5.0E6,5.0E7,5.0E8], minor=True)
#        axes[0].set_yticks(np.logspace(6.5,8.5,3), minor=True)
        axes[1].set_yticks( np.logspace(-15,-9,7) )
        axes[2].set_yticks( np.logspace(-26,-15,12)[::2] )
        axes[2].set_yticks( np.logspace(-26,-15,12)[1::2], minor=True )
#        axes[2].get_yaxis().horizontalalignment="left"
#        axes[2].set_xticklabels(None, minor=True)
    elif kernel_name == "Long_Bott":
        # axes[0].set_yticks([1.0E6,1.0E7,1.0E8,3.0E8,4.0E8])
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8])
        axes[0].set_yticks(
                np.concatenate((
                        np.linspace(2E6,9E6,8),
                        np.linspace(2E7,9E7,8),
                        np.linspace(2E8,3E8,2),
                                )),
                minor=True)
        axes[0].set_ylim([1.0E6,4.0E8])
        # axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        axes[1].set_yticks( np.logspace(-15,-7,9) )
        axes[1].set_ylim([1.0E-15,1.0E-7])
        axes[2].set_yticks( np.logspace(-26,-11,16)[::2])
        axes[2].set_yticks( np.logspace(-26,-11,16)[1::2],minor=True)
        axes[2].set_ylim([1.0E-26,1.0E-11])
    elif kernel_name == "Hall_Bott":
        # axes[0].set_yticks([1.0E6,1.0E7,1.0E8,3.0E8,4.0E8])
#        axes[0].set_yticks([1.0E8])
        axes[0].set_yticks([7E7,8E7,9E7,1E8,2.0E8,3E8])
#        axes[0].set_yticks([1.0E8])
#        axes[0].set_yticks([7E7,8E7,9E7,2.0E8,3E8], minor=True)

        axes[0].set_ylim([7.0E7,4.0E8])
#        axes[0].tick_params(axis='y', which='minor')
#        axes[0].yaxis.set_ticklabels([r'$2\times10^4$','','',r'$5\times10^4$','','','','', 
#                r'$2\times 10^4$','','',r'$5\times10^4$','','','','',
#                r'$2\times 10^5$','','',r'$5\times10^5$','','','','',
#                ],minor=True)        
        axes[0].yaxis.set_ticklabels(
                [r'$7\times10^7$','','',r'$1\times10^8$',
                r'$2\times 10^8$',r'$3\times10^8$','','','','',
                r'$2\times 10^5$','','',r'$5\times10^5$','','','','',
                ])        
#        plt.tick_params(axis='y', which='minor')
#        from matplotlib.ticker import FormatStrFormatter
#        from matplotlib.ticker import LogFormatter
#        formatter = LogFormatter(labelOnlyBase=False,
##                                 minor_thresholds=(2, 0.4)
#                                 )
#        axes[0].get_yaxis().set_minor_formatter(formatter)
#        axes[0].yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
#        axes[0].set_yticklabels(["7E7","1E8","2E8","3E8"])
        # axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        axes[1].set_yticks( np.logspace(-15,-8,8) )
        axes[1].set_ylim([1.0E-15,1.0E-8])
        axes[2].set_yticks( np.logspace(-26,-12,8) )
        axes[2].set_ylim([1.0E-26,1.0E-12])
    # axes[0].plot(t_Unt,lam0_Unt, "o", c = "k")
    # axes[2].plot(t_Unt2,lam2_Unt, "o", c = "k")
        
#    for mom0_last in mom0_last_time:
#    print(mom0_last_time/mom0_last_time.min())
#    if len(mom0_last_time) >= 3:
#        print(mom0_last_time/mom0_last_time[-2])
#        print(mom0_last_time_Unt/mom0_last_time_Unt.min())
#        print()
#    title=\
#f"Moments of the distribution for various $\kappa$ (see legend)\n\
#dt={dt:.1e}, eta={eta:.0e}, r_critmin=0.6, no_sims={no_sims}, \
#gen_method={gen_method}, kernel={kernel_name}"
#     title=\
# f"Moments of the distribution for various $\kappa$ (see legend)\n\
# dt={dt}, eta={eta:.0e}, no_sims={no_sims}, \
# gen_method={gen_method}, kernel=LONG"
#    fig.suptitle(title, fontsize=TTFS, y = 0.997)
#    fig.tight_layout()
    # fig.subplots_adjust()
#    plt.subplots_adjust(top=0.965)
#    fig.savefig(fig_dir + fig_name)
    
    if kernel_name == "Hall_Bott":
        xpos_ = -0.19
        ypos_ = 0.86
    elif kernel_name == "Long_Bott":
        xpos_ = -0.14
        ypos_ = 0.86
    elif kernel_name == "Golovin":
        xpos_ = -0.14
        ypos_ = 0.86
    fig.text(xpos_, ypos_ , r"\textbf{(c)}", fontsize=LFS)    
#    fig.text(xpos_, ypos_*0.51, r"\textbf{(b)}", fontsize=LFS)       
    
    
    
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )      
    
    no_rows = len(kappa_list)
#    no_rows = 4


### REFERENCE REL DEV PLOTS 
### REFERENCE REL DEV PLOTS 
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=2, figsize=(figsize2), sharex=True)    
    
#    if kernel_name == "Golovin":
#        for kappa_n,kappa in enumerate((5,200)):
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        
        if kernel_name == "Golovin":
            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
        else:
#            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
#            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")[::2,:]
            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")[::2,:]
#            print ("moments_vs_time_avg.shape")
#            print (moments_vs_time_avg.shape)
            
        
        for cnt_m,mom_n in enumerate((0,2,3)):
            if kernel_name == "Golovin":
                moments_ref_i = compute_moments_Golovin(save_times, mom_n, DNC, LWC, bG)
                save_times0 = save_times
            else:
                moments_ref_i = moments_ref[mom_n][::3]
#                print("moments_ref_i.shape")
#                print(moments_ref_i.shape)
                save_times0 = save_times[::2]
#                print(save_times)
                
            
            rel_dev = np.abs(moments_vs_time_avg[:,mom_n]-moments_ref_i)/moments_ref_i
            
            rel_err = moments_vs_time_std[:,mom_n] / moments_ref_i
#            rel_err = moments_vs_time_std[:,mom_n] / moments_vs_time_avg[:,mom_n]
            
            ax = axes[kappa_n,0]
            ax.errorbar(save_times0/60, rel_dev, rel_err, label=str(mom_n))
            ax.grid()
#                ax.annotate(
##                        r"NS={}".format(kappa*5),
#                        "NS",
#                            (0.9,0.02),
#                            xycoords="axes fraction")                
          
#                ax.annotate()

            ax = axes[kappa_n,1]
#                ax.errorbar(save_times/60, rel_dev, rel_err, label=str(mom_n))
            ax.plot(save_times0/60, rel_err, label=str(mom_n))
            ax.grid()
        axes[kappa_n,1].annotate(
#                        r"NS={}".format(kappa*5),
                            r"$N_\mathrm{{SIP}} = {}$".format(kappa*5),
                            (0.01,0.82),
                            xycoords="axes fraction")                 
        
        ax = axes[kappa_n,0]
        ax.set_yscale("log")
        if kernel_name == "Golovin":
            ax.set_yticks(np.logspace(-5,0,6))
        else:
            ax.set_yticks(np.logspace(-4,1,6))

        ax = axes[kappa_n,1]
        ax.set_yscale("log")
        if kernel_name == "Golovin":
            ax.set_yticks(np.logspace(-5,0,6))
        else:
            ax.set_yticks(np.logspace(-6,0,4))
        
#    if kernel_name == "Golovin":            
#        axes[0,0].legend()
#        axes[0,1].legend(loc = "upper center")
#    else:
    axes[0,0].legend(ncol=3)
    axes[0,1].legend(ncol=3)
#        axes[0,1].legend(loc = "upper center", ncol=3)
#    axes[0,1].legend(loc = "best")
    
    axes[0,0].set_title("rel. dev. $(\lambda-\lambda_\mathrm{ref})/\lambda_\mathrm{ref}$")
    axes[0,1].set_title("rel. error $\mathrm{SD}(\lambda)/\lambda_\mathrm{ref}$ ")
    
    
    axes[-1,0].set_xticks(np.linspace(0,60,7))
    axes[-1,1].set_xticks(np.linspace(0,60,7))
#        axes[0].set_yticks(np.logspace(-4,0,5))
    axes[-1,0].set_xlim((0,60))
    axes[-1,1].set_xlim((0,60))
    
    axes[-1,0].set_xlabel("Time (min)")
    axes[-1,1].set_xlabel("Time (min)")
#        axes[-1].set_ylim((1E-5,2E-1))
#        axes[0].set_yticks(np.linspace(-1,0.4,8))
#        axes[1].set_yticks(np.linspace(-0.15,0.4,8))
    
    fig.savefig(figname2,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )   
    
### CONVERGENCE REL DEV PLOTS 
### CONVERGENCE REL DEV PLOTS 
    no_rows = 1
    fig, axes = plt.subplots(nrows=no_rows, ncols=2,
                             figsize=(figsize3)
                             )    
    
    kappa = kappa_list[-1]
    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    moments_vs_time_avg_ref = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    moments_vs_time_std_ref = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    
#    if kernel_name == "Golovin":
#        for kappa_n,kappa in enumerate((5,200)):
#    for kappa_n,kappa in enumerate([kappa_list[-2]]):
    kappa = kappa_list[-2]
    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
        
#        if kernel_name == "Golovin":
#        else:
##            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
##            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
#            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")
##            print ("moments_vs_time_avg.shape")
##            print (moments_vs_time_avg.shape)
#        
#        for cnt_m,mom_n in enumerate((0,2,3)):
##            if kernel_name == "Golovin":
##                moments_ref_i = compute_moments_Golovin(save_times, mom_n, DNC, LWC, bG)
##                save_times0 = save_times
##            else:
##                moments_ref_i = moments_ref[mom_n][::3]
###                print("moments_ref_i.shape")
###                print(moments_ref_i.shape)
##                save_times0 = save_times[::2]
###                print(save_times)
    save_times0 = save_times
#    moments_ref_i = 
    
    for cnt_m,mom_n in enumerate((0,2,3)):
        moments_ref_i = moments_vs_time_avg_ref[:,mom_n]
        rel_dev = np.abs(moments_vs_time_avg[:,mom_n]-moments_ref_i)/moments_ref_i
        
        rel_err = moments_vs_time_std[:,mom_n] / moments_ref_i
        
        ax = axes[0]
        ax.errorbar(save_times0/60, rel_dev, rel_err, label=str(mom_n))
        ax.grid()
    #                ax.annotate(
    ##                        r"NS={}".format(kappa*5),
    #                        "NS",
    #                            (0.9,0.02),
    #                            xycoords="axes fraction")                
      
    #                ax.annotate()
    
        ax.set_yscale("log")

        ax = axes[1]
    #                ax.errorbar(save_times/60, rel_dev, rel_err, label=str(mom_n))
        ax.plot(save_times0/60, rel_err, label=str(mom_n))
        ax.grid()
    
#        axes[kappa_n,1].annotate(
##                        r"NS={}".format(kappa*5),
#                            r"$N_\mathrm{{SIP}} = {}$".format(kappa*5),
#                            (0.01,0.82),
#                            xycoords="axes fraction")                 
        ax.set_yscale("log")
        
#        ax = axes[kappa_n,0]
#        if kernel_name == "Golovin":
#            ax.set_yticks(np.logspace(-5,0,6))
#        else:
#            ax.set_yticks(np.logspace(-4,1,6))
#
#        ax = axes[kappa_n,1]
#        if kernel_name == "Golovin":
#            ax.set_yticks(np.logspace(-5,0,6))
#        else:
#            ax.set_yticks(np.logspace(-6,0,4))
        
#    if kernel_name == "Golovin":            
#        axes[0,0].legend()
#        axes[0,1].legend(loc = "upper center")
#    else:
    for ax in axes:
        ax.legend(
    #                        np.reshape(handles, (5,2)).T.flatten(),
    #                        np.reshape(labels, (5,2)).T.flatten(),
    #                    np.concatenate((handles[::2],handles[1::2]),axis=0),
    #                      np.concatenate((labels[::2],labels[1::2]),axis=0),
                  ncol=3, handlelength=0.8, handletextpad=0.2,
                  columnspacing=0.5, borderpad=0.2,
#                  loc="lower left",
#                  bbox_to_anchor=(0.0, 0.6)
                  ).set_zorder(100)           
#    axes[1].legend(
##                        np.reshape(handles, (5,2)).T.flatten(),
##                        np.reshape(labels, (5,2)).T.flatten(),
##                    np.concatenate((handles[::2],handles[1::2]),axis=0),
##                      np.concatenate((labels[::2],labels[1::2]),axis=0),
#              ncol=3, handlelength=0.8, handletextpad=0.2,
#              columnspacing=0.5, borderpad=0.2, loc="lower left",
#              bbox_to_anchor=(0.0, 0.6)).set_zorder(100)           
#    axes[0].legend(ncol=3)
#    axes[1].legend(ncol=3)
#        axes[0,1].legend(loc = "upper center", ncol=3)
#    axes[0,1].legend(loc = "best")
    
    axes[0].set_title("rel. dev. $(\lambda-\lambda_\mathrm{ref})/\lambda_\mathrm{ref}$")
    axes[1].set_title("rel. error $\mathrm{SD}(\lambda)/\lambda_\mathrm{ref}$ ")
    
    
    axes[0].set_xticks(np.linspace(0,60,7))
    axes[1].set_xticks(np.linspace(0,60,7))
#        axes[0].set_yticks(np.logspace(-4,0,5))
    axes[0].set_xlim((0,60))
    axes[1].set_xlim((0,60))
    
    axes[0].set_xlabel("Time (min)")
    axes[1].set_xlabel("Time (min)")
#        axes[-1].set_ylim((1E-5,2E-1))
#        axes[0].set_yticks(np.linspace(-1,0.4,8))
#        axes[1].set_yticks(np.linspace(-0.15,0.4,8))
    
    fig.savefig(figname3,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )   
    
### CONVERGENCE REL DEV PLOTS 2
### CONVERGENCE REL DEV PLOTS 2   
    no_rows = 1
    fig, axes = plt.subplots(nrows=no_rows, ncols=2,
                             figsize=(figsize4)
                             )    
    
    kappa = kappa_list[-1]
    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    moments_vs_time_avg_ref = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    moments_vs_time_std_ref = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    
#    if kernel_name == "Golovin":
#        for kappa_n,kappa in enumerate((5,200)):
#    for kappa_n,kappa in enumerate([kappa_list[-2]]):
    kappa = kappa_list[1]
    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
        
#        if kernel_name == "Golovin":
#        else:
##            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
##            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
#            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")
##            print ("moments_vs_time_avg.shape")
##            print (moments_vs_time_avg.shape)
#        
#        for cnt_m,mom_n in enumerate((0,2,3)):
##            if kernel_name == "Golovin":
##                moments_ref_i = compute_moments_Golovin(save_times, mom_n, DNC, LWC, bG)
##                save_times0 = save_times
##            else:
##                moments_ref_i = moments_ref[mom_n][::3]
###                print("moments_ref_i.shape")
###                print(moments_ref_i.shape)
##                save_times0 = save_times[::2]
###                print(save_times)
    save_times0 = save_times
#    moments_ref_i = 
    
    for cnt_m,mom_n in enumerate((0,2,3)):
        moments_ref_i = moments_vs_time_avg_ref[:,mom_n]
        rel_dev = np.abs(moments_vs_time_avg[:,mom_n]-moments_ref_i)/moments_ref_i
        
        rel_err = moments_vs_time_std[:,mom_n] / moments_ref_i
        
        ax = axes[0]
        ax.errorbar(save_times0/60, rel_dev, rel_err, label=str(mom_n))
        ax.grid()
    #                ax.annotate(
    ##                        r"NS={}".format(kappa*5),
    #                        "NS",
    #                            (0.9,0.02),
    #                            xycoords="axes fraction")                
      
    #                ax.annotate()
    
        ax.set_yscale("log")

        ax = axes[1]
    #                ax.errorbar(save_times/60, rel_dev, rel_err, label=str(mom_n))
        ax.plot(save_times0/60, rel_err, label=str(mom_n))
        ax.grid()
    
#        axes[kappa_n,1].annotate(
##                        r"NS={}".format(kappa*5),
#                            r"$N_\mathrm{{SIP}} = {}$".format(kappa*5),
#                            (0.01,0.82),
#                            xycoords="axes fraction")                 
        ax.set_yscale("log")
        
#        ax = axes[kappa_n,0]
#        if kernel_name == "Golovin":
#            ax.set_yticks(np.logspace(-5,0,6))
#        else:
#            ax.set_yticks(np.logspace(-4,1,6))
#
#        ax = axes[kappa_n,1]
#        if kernel_name == "Golovin":
#            ax.set_yticks(np.logspace(-5,0,6))
#        else:
#            ax.set_yticks(np.logspace(-6,0,4))
        
#    if kernel_name == "Golovin":            
#        axes[0,0].legend()
#        axes[0,1].legend(loc = "upper center")
#    else:
    for ax in axes:
        ax.legend(
    #                        np.reshape(handles, (5,2)).T.flatten(),
    #                        np.reshape(labels, (5,2)).T.flatten(),
    #                    np.concatenate((handles[::2],handles[1::2]),axis=0),
    #                      np.concatenate((labels[::2],labels[1::2]),axis=0),
                  ncol=3, handlelength=0.8, handletextpad=0.2,
                  columnspacing=0.5, borderpad=0.2,
#                  loc="lower left",
#                  bbox_to_anchor=(0.0, 0.6)
                  ).set_zorder(100)           
#    axes[1].legend(
##                        np.reshape(handles, (5,2)).T.flatten(),
##                        np.reshape(labels, (5,2)).T.flatten(),
##                    np.concatenate((handles[::2],handles[1::2]),axis=0),
##                      np.concatenate((labels[::2],labels[1::2]),axis=0),
#              ncol=3, handlelength=0.8, handletextpad=0.2,
#              columnspacing=0.5, borderpad=0.2, loc="lower left",
#              bbox_to_anchor=(0.0, 0.6)).set_zorder(100)           
#    axes[0].legend(ncol=3)
#    axes[1].legend(ncol=3)
#        axes[0,1].legend(loc = "upper center", ncol=3)
#    axes[0,1].legend(loc = "best")
    
    axes[0].set_title("rel. dev. $(\lambda-\lambda_\mathrm{ref})/\lambda_\mathrm{ref}$")
    axes[1].set_title("rel. error $\mathrm{SD}(\lambda)/\lambda_\mathrm{ref}$ ")
    
    
    axes[0].set_xticks(np.linspace(0,60,7))
    axes[1].set_xticks(np.linspace(0,60,7))
#        axes[0].set_yticks(np.logspace(-4,0,5))
    axes[0].set_xlim((0,60))
    axes[1].set_xlim((0,60))
    
    axes[0].set_xlabel("Time (min)")
    axes[1].set_xlabel("Time (min)")
#        axes[-1].set_ylim((1E-5,2E-1))
#        axes[0].set_yticks(np.linspace(-1,0.4,8))
#        axes[1].set_yticks(np.linspace(-0.15,0.4,8))
    
    fig.savefig(figname4,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )   
    
    plt.close("all")    

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

#figpath = home_path + "/Masterthesis/Figures/04Evaluation/"        
figpath = home_path + "/Masterthesis/Figures/05ColBoxModel/"        
#figpath = home_path + "/Masterthesis/Figures/05ColBoxModel/new/"        
figsize = cm2inch(7.,4.5)
if kernel_name == "Golovin":
    figsize2 = cm2inch(7.,9.0)
else:
    figsize2 = cm2inch(7.,11.0)
#figsize2 = cm2inch(7.,11.0)
# for golovin: 3600 s sim time. frame every 150 s -> 24 + 1 = 25 frames
if kernel_name == "Golovin":
    time_idx = np.arange(0,25,4)
else: time_idx = np.arange(0,13,2)    
#kappa2 = 3000
if act_plot:
    for n,kappa in enumerate(kappa_list):
        if n==0:
            plot_compare=True
#            plot_compare=False
        else: plot_compare=False
        figname = figpath + f"{kernel_name}_g_ln_R_vs_R_kappa_{kappa}.pdf"
        figname2 = figpath + f"{kernel_name}_g_ln_R_vs_R_kappa_{kappa1}_{kappa2}.pdf"
        load_dir =\
            sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        load_dir_k1 =\
            sim_data_path + result_path_add + f"kappa_{kappa1}/dt_{int(dt)}/"
        load_dir_k2 =\
            sim_data_path + result_path_add + f"kappa_{kappa2}/dt_{int(dt)}/"
#        plot_for_given_kappa(kappa, eta, dt, no_sims, start_seed, no_bins,
#                         kernel_name, gen_method, bin_method_sim_data, load_dir)
        plot_g_ln_R_for_given_kappa(kappa, eta, dt, no_sims, start_seed,
                                    no_bins, DNC0, m_mean,
                                     kernel_name, gen_method, bin_method_sim_data,
                                     time_idx,
                                     load_dir,
                                     load_dir_k1,
                                     load_dir_k2,
                                     figsize, figname,
                                     plot_compare,
                                     figsize2, figname2)
        
#%% PLOT MOMENTS VS TIME for several kappa
# act_plot_moments_kappa_var = True

if kernel_name == "Golovin":
    figsize = cm2inch(6.5,9.0)
else:
    figsize = cm2inch(6.5,11.0)
figsize2 = cm2inch(17.5,23.0)
figsize3 = cm2inch(17.5,5)
figsize4 = figsize3
#figsize4 = cm2inch(17.5,9.0*5/8)

if print_reduction:
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#    print(moments_vs_time_avg.shape)
#        moments_vs_time_avg[:,1] *= 1.0E3
    
    ### print reduction of the init concentration with time
    
        print()
        print("kappa = ", kappa)
        print()
        for time_n, t_ in enumerate(save_times):
            rel_dev = 100*(moments_vs_time_avg[0,0] - moments_vs_time_avg[time_n,0])\
                      / moments_vs_time_avg[0,0]
            print(t_, rel_dev)
    
#TTFS, LFS, TKFS = 14,14,12
if act_plot_moments_kappa_var:
    ref_data_path = "collision/ref_data/" \
                    + f"{kernel_name}/"
#    if kernel_name == "Long_Bott" or kernel_name == "Hall_Bott":
#        moments_ref = np.loadtxt(ref_data_path + "Wang_2007_moments.txt")
#        times_ref = np.loadtxt(ref_data_path + "Wang_2007_times.txt")
#    elif kernel_name == "Golovin":
#        moments_ref = None
#        times_ref = None
    moments_ref = np.loadtxt(ref_data_path + "Wang_2007_moments.txt")
    times_ref = np.loadtxt(ref_data_path + "Wang_2007_times.txt")
    
    figname = figpath \
            + f"{kernel_name}_moments_vs_time_kappa_{kappa_list[0]}_{kappa_list[-1]}.pdf"    
    figname2 = figpath \
            + f"{kernel_name}_moments_vs_time_rel_dev_kappa_{kappa_list[0]}_{kappa_list[-1]}.pdf"    
    figname3 = figpath \
            + f"{kernel_name}_moments_vs_time_converge_kappa_{kappa_list[0]}_{kappa_list[-1]}.pdf"    
    figname4 = figpath \
            + f"{kernel_name}_moments_vs_time_converge_50_kappa_{kappa_list[0]}_{kappa_list[-1]}.pdf"    
    fig_dir = sim_data_path + result_path_add
#    plot_moments_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
#                           kernel_name, gen_method,
#                           dist, start_seed,
#                           moments_ref, times_ref,
#                           sim_data_path,
#                           result_path_add,
#                           TTFS, LFS, TKFS)
    plot_moments_vs_time_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
                               kernel_name, gen_method,
                               dist, start_seed,
                               moments_ref, times_ref,
                               sim_data_path,
                               result_path_add,
                               figsize, figname, figsize2, figname2,
                               figsize3, figname3,  figsize4, figname4,
                               TTFS, LFS, TKFS)

#%% PLOT CONVERGENCE BEHAVIOR AT A FIXED TIME

figsize_conv = cm2inch(18,7.5)

idx_t_conv = 12
t_conv = 60
LW_conv = 1
    
#TTFS, LFS, TKFS = 14,14,12
if act_plot_convergence_t_fix:
    MS_mom = 5
    MEW_mom = 0.5
    ELW_mom = 0.8
    capsize_mom = 2    
    
    if setup == "Golovin":
        kappa_list=[5,10,20,40,60,100,200]
    else:
        kappa_list=[5,10,20,40,60,100,200,400,600,800,1000,1500,2000,3000]    
    
    figname = figpath \
            + f"{kernel_name}_convergence_t_{t_conv}_{kappa_list[0]}_{kappa_list[-1]}.pdf"    
#    figdir = sim_data_path + result_path_add    
    
    no_rows = 1
    no_cols = 3

    # load reference with highest SIP number    
    kappa = kappa_list[-1]

    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    moments_vs_time_avg_ref = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    moments_vs_time_std_ref = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    
    moments_ref = moments_vs_time_avg_ref[idx_t_conv,np.array((0,2,3))][:,None]
    moments_ref_std = moments_vs_time_std_ref[idx_t_conv,np.array((0,2,3))][:,None]
    
    print(moments_vs_time_avg_ref.shape)
    
    no_data = len(kappa_list) - 1
    # [k,[Nsip,lambda_k]]
    moments_vs_Nsip = np.zeros((3, no_data), dtype = np.float64)
    moments_vs_Nsip_std = np.zeros((3, no_data), dtype = np.float64)
    Nsip = []
    for kappa_n,kappa in enumerate(kappa_list[:-1]):
        load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
#        save_times0 = save_times
        Nsip.append(kappa*5)
        for cnt_m,mom_n in enumerate((0,2,3)):
            moments_vs_Nsip[cnt_m,kappa_n] = \
                moments_vs_time_avg[idx_t_conv,mom_n]
            moments_vs_Nsip_std[cnt_m,kappa_n] = \
                moments_vs_time_std[idx_t_conv,mom_n]
    print(Nsip)
    print(moments_vs_Nsip)
    
    moments_vs_Nsip_rel_dev = np.abs(moments_vs_Nsip - moments_ref) / moments_ref
#    moments_vs_Nsip_rel_dev = np.abs(moments_vs_Nsip) / moments_ref
    moments_vs_Nsip_rel_err = moments_vs_Nsip_std / moments_ref
    
    moments_vs_Nsip_rel_dev_log = np.log(moments_vs_Nsip_rel_dev)
    moments_vs_Nsip_rel_err_log = np.log(moments_vs_Nsip_rel_err)
    
    Nsip_log = np.log(Nsip)
    
    from scipy.optimize import curve_fit
    
    def lin_fit(x,a,b):
        return a*x+b
        
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                             figsize=(figsize_conv)
                             )  
    
    for cnt_m,mom_n in enumerate((0,2,3)):
        ax = axes[cnt_m]
        
        rel_dev = moments_vs_Nsip_rel_dev[cnt_m]
        rel_err = moments_vs_Nsip_rel_err[cnt_m]
        rel_dev_log = moments_vs_Nsip_rel_dev_log[cnt_m]
        rel_err_log = moments_vs_Nsip_rel_err_log[cnt_m]
        print(setup, "fit values")
        if setup == "Long_Bott":
            if cnt_m == 0:
                p, cov = curve_fit(lin_fit, Nsip_log,
                                   rel_dev_log,
                                   sigma=rel_err_log)   
                
                print(mom_n, p, np.sqrt(np.diag(cov)))        
                print(mom_n, np.exp(p), np.exp(np.sqrt(np.diag(cov)))        )
                
                fitted = lin_fit(Nsip_log,*p)
                fitted = np.exp(fitted)
            elif cnt_m == 1:
                
                idx_cut = 7
                p, cov = curve_fit(lin_fit, Nsip_log[:idx_cut],
                                   rel_dev[:idx_cut],
                                   sigma=rel_err[:idx_cut])   
                print(mom_n, p, np.sqrt(np.diag(cov)))        
                
#                fitted = lin_fit(Nsip_log,*p)
                fitted = lin_fit(Nsip_log[:idx_cut+2],*p)
                
                p2, cov2 = curve_fit(lin_fit, Nsip_log[idx_cut:],
                                   rel_dev[idx_cut:],
                                   sigma=rel_err[idx_cut:])   
                print(mom_n, p2, np.sqrt(np.diag(cov2)))        
                
                fitted2 = lin_fit(Nsip_log[:],*p2)
                
#                fitted = np.exp(fitted)
            elif cnt_m == 2:
                
                p, cov = curve_fit(lin_fit, Nsip_log,
                                   rel_dev,
                                   sigma=rel_err)   
                
                print(mom_n, p, np.sqrt(np.diag(cov)))        
                
                fitted = lin_fit(Nsip_log,*p)
#                fitted = np.exp(fitted)
            if cnt_m == 0:
                yerr_low = np.where(moments_vs_Nsip_rel_dev[cnt_m]-moments_vs_Nsip_rel_err[cnt_m] <= 0,
                                    moments_vs_Nsip_rel_dev[cnt_m]*0.999,
                                    moments_vs_Nsip_rel_err[cnt_m])
            else: 
                yerr_low = moments_vs_Nsip_rel_err[cnt_m]
            yerr_high = moments_vs_Nsip_rel_err[cnt_m]
        elif setup == "Hall_Bott":
#            if cnt_m == 0:
            if cnt_m == 0:
                p, cov = curve_fit(lin_fit, Nsip_log,
                                   rel_dev_log,
                                   sigma=rel_err_log)   
                print(mom_n, p, np.sqrt(np.diag(cov)))        
                print(mom_n, np.exp(p), np.exp(np.sqrt(np.diag(cov)))        )
                
                fitted = lin_fit(Nsip_log,*p)
                fitted = np.exp(fitted)
            elif cnt_m == 1:
                
                idx_cut = 8
                p, cov = curve_fit(lin_fit, Nsip_log[:idx_cut],
                                   rel_dev[:idx_cut],
                                   sigma=rel_err[:idx_cut])   
                print(mom_n, p, np.sqrt(np.diag(cov)))        
                
#                fitted = lin_fit(Nsip_log,*p)
                fitted = lin_fit(Nsip_log[:idx_cut+2],*p)
                
                p2, cov2 = curve_fit(lin_fit, Nsip_log[idx_cut:],
                                   rel_dev[idx_cut:],
                                   sigma=rel_err[idx_cut:])   
                print(mom_n, p2, np.sqrt(np.diag(cov2)))        
                
                fitted2 = lin_fit(Nsip_log[:],*p2)
                
#                fitted = np.exp(fitted)
            elif cnt_m == 2:
                
                p, cov = curve_fit(lin_fit, Nsip_log,
                                   rel_dev,
                                   sigma=rel_err)   
                
                print(mom_n, p, np.sqrt(np.diag(cov)))        
                
                fitted = lin_fit(Nsip_log,*p)
            if cnt_m == 0:
                yerr_low = np.where(moments_vs_Nsip_rel_dev[cnt_m]-moments_vs_Nsip_rel_err[cnt_m] <= 0,
                                    moments_vs_Nsip_rel_dev[cnt_m]*0.999,
                                    moments_vs_Nsip_rel_err[cnt_m])
            else: 
                yerr_low = moments_vs_Nsip_rel_err[cnt_m]
            yerr_high = moments_vs_Nsip_rel_err[cnt_m]
            
#        print(yerr_low.shape, yerr_high.shape)
#        ax.plot(Nsip,moments_vs_Nsip[cnt_m])
#        ax.plot(Nsip,moments_vs_Nsip_rel_dev[cnt_m], "x")
        ax.errorbar(Nsip,moments_vs_Nsip_rel_dev[cnt_m],
#                    yerr=moments_vs_Nsip_rel_err[cnt_m],
                    yerr=(yerr_low,yerr_high),
                    fmt = "x", ms = MS_mom, mew=MEW_mom,
                    fillstyle="none", elinewidth = ELW_mom,
                    capsize=capsize_mom, label = "sim"
                    )        
        if setup == "Long_Bott":
            if cnt_m == 1:
                ax.plot(Nsip[:idx_cut+2], fitted, label = "fit", linewidth= LW_conv)
                ax.plot(Nsip, fitted2, label = "fit 2", linewidth= LW_conv)
            else:
                ax.plot(Nsip, fitted, label = "fit", linewidth= LW_conv)
        if setup == "Hall_Bott":
            if cnt_m == 1:
                ax.plot(Nsip[:idx_cut+2], fitted, label = "fit", linewidth= LW_conv)
                ax.plot(Nsip, fitted2, label = "fit 2", linewidth= LW_conv)
            else:
                ax.plot(Nsip, fitted, label = "fit", linewidth= LW_conv)
#        if cnt_m == 0:
#            ax.set_xscale("log")
        ax.set_xscale("log")
#        ax.set_yscale("log")
#        ax.set_xticks(Nsip)
#        ax.set_xticklabels(Nsip)
        ax.grid(axis="y")
        ax.legend()
#        ax.set_xlim(10,2E4)
        ax.set_xlabel(r"$N_\mathrm{SIP}$")
        ax.set_title(r"$|\lambda_{0} - \lambda_\mathrm{{maxSIP}}| / \lambda_\mathrm{{maxSIP}}$".format(mom_n))
    
    axes[0].set_yscale("log")
    if setup == "Long_Bott":
    #    axes[1].set_yscale("log")
        axes[0].set_ylim(5E-3,5E1)
    if setup == "Hall_Bott":
    #    axes[1].set_yscale("log")
#        pass
        axes[0].set_ylim(8E-4,1)
#    
#    axes[0].set_title("rel. dev. $(\lambda-\lambda_\mathrm{ref})/\lambda_\mathrm{ref}$")
#    axes[1].set_title("rel. error $\mathrm{SD}(\lambda)/\lambda_\mathrm{ref}$ ")
#    
#    
#    axes[0].set_xticks(np.linspace(0,60,7))
#    axes[1].set_xticks(np.linspace(0,60,7))
##        axes[0].set_yticks(np.logspace(-4,0,5))
#    axes[0].set_xlim((0,60))
#    axes[1].set_xlim((0,60))
#    
#    axes[0].set_xlabel("Time (min)")
#    axes[1].set_xlabel("Time (min)")
#        axes[-1].set_ylim((1E-5,2E-1))
#        axes[0].set_yticks(np.linspace(-1,0.4,8))
#        axes[1].set_yticks(np.linspace(-0.15,0.4,8))
    pad_ax_h = 0.1
    pad_ax_v = 0.22
    fig.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
    fig.subplots_adjust(wspace=pad_ax_v)  
    
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )   
    
    plt.close("all")    
