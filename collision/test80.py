#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:18:13 2019

@author: jdesk
"""

import numpy as np

OS = "LinuxDesk"
# OS = "MacOS"

# args for SIP ensemble generation
args_gen = [1,1,1]
#args_gen = [1,1,1]

act_gen = bool(args_gen[0])
act_analysis_ensembles = bool(args_gen[1])
# NOTE: plotting needs analyze first (or in buffer)
act_plot_ensembles = bool(args_gen[2])

args_sim = [0,0,0,0]
#args_sim = [1,0,0,0]
#args_sim = [0,1,1,1]
# args_sim = [1,1,1,1]

act_sim = bool(args_sim[0])
act_analysis = bool(args_sim[1])
act_plot = bool(args_sim[2])
act_plot_moments_kappa_var = bool(args_sim[3])

dist = "expo"
gen_method = "SinSIP"

# kappa = 40
# kappa = 800
kappa_list=[10]
#kappa_list=[5]
#kappa_list=[5,10,20,40]
# kappa_list=[5,10,20,40,60,100,200]
# kappa_list=[5,10,20,40,60,100,200,400]
#kappa_list=[5,10,20,40,60,100,200,400]
# kappa_list=[5,10,20,40,60,100,200,400,600,800]
# kappa_list=[600]
# kappa_list=[800]

eta = 1.0E-9

eta_threshold = "weak"
#eta_threshold = "fix"

if eta_threshold == "weak":
    weak_threshold = True
else: weak_threshold = False

no_sims = 10

start_seed = 3711

no_bins = 50
bin_method = "auto_bin"

# kernel_name = "Golovin"
kernel_name = "Long_Bott"
#kernel_method = "Ecol_grid_R"
#kernel_method = "kernel_grid_m"
kernel_method = "analytic"

# dt = 1.0
dt = 10.0
# dt = 20.0
dV = 1.0

mass_density = c.mass_density_water_liquid_NTP

# dt_save = 40.0
dt_save = 300.0
# t_end = 200.0
t_end = 3600.0

## set for expo AND lognormal
LWC0 = 1.0E-3 # kg/m^3
# this constant converts radius in mu to mass in kg (m = 4/3 pi rho R^3)
c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
c_mass_to_radius = 1.0 / c_radius_to_mass

### for EXPO dist: set r0 and LWC0 analog to Unterstr.
# -> DNC0 is calc from that
if dist == "expo":

    r_critmin = 0.6 # mu
    m_high_over_m_low = 1.0E6
    
    R_mean = 9.3 # in mu
    m_mean = c_radius_to_mass * R_mean**3 # in kg
    DNC0 = LWC0 / m_mean # in 1/m^3
    print("DNC0 = ", DNC0, "LWC0 =", LWC0, "m_mean = ", m_mean)

### for lognormal dist: set DNC0 and mu_R and sigma_R parameters of
# lognormal distr (of the radius) -> see distr. def above
if dist == "lognormal":
    
    r_critmin = 0.001 # mu
    m_high_over_m_low = 1.0E8
    
    DNC0 = 60.0E6 # 1/m^3
    #DNC0 = 2.97E8 # 1/m^3
    #DNC0 = 3.0E8 # 1/m^3
    mu_R = 0.02 # in mu
    sigma_R = 1.4 #  no unit
    
    mu_R_log = np.log(mu_R)
    sigma_R_log = np.log(sigma_R)
    
    # derive parameters of lognormal distribution of mass f_m(m)
    # assuming mu_R in mu and density in kg/m^3
    mu_m_log = 3.0 * mu_R_log + np.log(1.0E-18 * c.four_pi_over_three * mass_density)
    sigma_m_log = 3.0 * sigma_R_log

### DATA ANALYSIS SET PARAMETERS

# use the same bins as for generation
sample_mode = "given_bins"
# for auto binning: set number of bins
#sample_mode = "auto_bins"

# no_bins = 10
# no_bins = 25
# no_bins = 30
no_bins = 50
# no_bins = 100
# bin_mode = 1 for bins equal dist on log scale
bin_mode = 1

# for myHisto generation (additional parameters)
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

sim_data_path = "/mnt/D/sim_data_unif/col_box_mod/"
ensemble_path = f"{dist}/{gen_method}/eta_{eta:.0e}_{eta_threshold}/ensembles/"

kappa = 10
ensemble_dir = sim_data_path + ensemble_path + f"kappa_{kappa}/"

seed = 3711

m = np.load( ensemble_dir + f"masses_seed_{seed}.npy")