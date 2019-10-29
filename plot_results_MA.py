#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:43:10 2019

@author: jdesk
"""

#%% MODULE IMPORTS
import os
import numpy as np
# from datetime import datetime
# import timeit

import constants as c
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


import matplotlib.ticker as mticker
#import numpy as np

from microphysics import compute_mass_from_radius_vec
#import constants as c

#import numba as 
from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict, pdf_dict

#plt.rcParams.update(pdf_dict)

#from file_handling import load_grid_and_particles_full,\
#                          load_grid_scalar_fields\

from analysis import sample_masses, sample_radii
from analysis import avg_moments_over_boxes
from analysis import sample_masses_per_m_dry , sample_radii_per_m_dry
from analysis import plot_size_spectra_R_Arabas, generate_size_spectra_R_Arabas
from plotting_fcts_MA import plot_scalar_field_frames_extend_avg_MA
from plotting_fcts_MA import plot_size_spectra_R_Arabas_MA
from plotting_fcts_MA import plot_scalar_field_frames_std_MA

mpl.rcParams.update(plt.rcParamsDefault)
mpl.use("pgf")
#mpl.use("pdf")
#mpl.use("agg")
mpl.rcParams.update(pgf_dict)
#mpl.rcParams.update(pdf_dict)

#%%
def gen_data_paths(solute_type_var, kernel_var, seed_SIP_gen_var, seed_sim_var,
                   DNC0_var, no_spcm_var, no_seeds_var, dt_col_var):
    grid_paths = []
    data_paths = []
    
    no_variations = len(solute_type_var)
    for var_n in range(no_variations):
        solute_type = solute_type_var[var_n]
        no_cells = no_cells_var[var_n]
        no_spcm = no_spcm_var[var_n]
        no_seeds = no_seeds_var[var_n]
        seed_SIP_gen = seed_SIP_gen_var[var_n]
        seed_sim = seed_sim_var[var_n]
        
        data_folder = \
            f"{solute_type}" \
            + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
            + f"eval_data_avg_Ns_{no_seeds}_" \
            + f"sg_{seed_SIP_gen}_ss_{seed_sim}_t_{int(t_start)}_{int(t_end)}/"    
        data_path = simdata_path + data_folder
        
        grid_path = simdata_path + data_folder + "grid_data/" \
                    + f"{seed_SIP_gen}_{seed_sim}/"
    
        data_paths.append(data_path)
        grid_paths.append(grid_path)

    return grid_paths, data_paths

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

#%% STORAGE DIRECTORIES
##my_OS = "Linux_desk"
#my_OS = "Mac"
##my_OS = "TROPOS_server"
#
#if(my_OS == "Linux_desk"):
#    home_path = '/home/jdesk/'
#    simdata_path = "/mnt/D/sim_data_cloudMP/"
##    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
#elif (my_OS == "Mac"):
#    home_path = '/Users/bohrer/'
#    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
##    fig_path = home_path \
##               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'
#elif (my_OS == "TROPOS_server"):
#    simdata_path = "/vols/fs1/work/bohrer/sim_data_cloudMP/"

simdata_path = "/Users/bohrer/sim_data_cloudMP/"
figpath = "/Users/bohrer/" + "testingSubmit/"
#figpath = home_path + "Masterthesis/Figures/06TestCase/"

#%% CHOOSE OPERATIONS

#args_plot = [0,0,0,0,0,0,0,0,0,0]
#args_plot = [1,0,0,0,0,0,0,0,0,0]
args_plot = [0,1,0,0,0,0,0,0,0,0]
#args_plot = [0,0,1,0,0,0,0,0,0,0]
#args_plot = [0,0,0,0,1,0,0,0,0,0]

#args_plot = [1,1,1,1,1,1,1,1,1,1]

act_plot_grid_frames_init = args_plot[0]
act_plot_grid_frames_avg = args_plot[1]
act_plot_grid_frames_std = args_plot[2]
act_plot_grid_frames_abs_dev = args_plot[3]
act_plot_spectra_avg_Arabas = args_plot[4]
act_plot_moments_vs_z = args_plot[5]
act_plot_moments_diff_vs_z = args_plot[6]
act_plot_moments_norm_vs_t = args_plot[7]
act_plot_SIP_convergence = args_plot[8]
act_compute_CFL = args_plot[8]

# this one is not adapted for MA plots...
#act_plot_grid_frames_avg_shift = args_plot[2]
act_plot_grid_frames_avg_shift = False

#%% Simulations done
### ADD THE PARAMETERS OF THE EVALUATED SIMULATIONS
### IN THE LISTS AND CHOOSE BY SETTING SIM_N BELOW
gseedl=(3711, 3811, 3811, 3411, 3311, 3811 ,4811 ,4811 ,3711, 3711, 3811,2101)
sseedl=(6711 ,6811 ,6811 ,6411 ,6311 ,7811 ,6811 ,7811 ,6711, 6711, 8811,4101)

ncl=(75, 75, 75, 75, 75, 75, 75, 75, 75, 150, 75,75)
solute_typel=["AS", "AS" ,"AS", "AS" ,"AS" ,"AS", "AS", "AS", "NaCl","AS","AS","AS"]

DNC1 = np.array((60,60,60,30,120,60,60,60,60,60,60,60))
DNC2 = DNC1 * 2 // 3
no_spcm0l=(13, 26 ,52 ,26 ,26 ,26 ,26 ,26 ,26,26,26,1)
no_spcm1l=(19, 38 ,76 ,38 ,38 ,38 ,38 ,38 ,38,38,38,2)
no_col_per_advl=(2, 2, 2 ,2 ,2 ,2 ,2 ,2 ,2,2,10,2)

no_sims_list = (50,50,50,50,50,50,50,50,50,50,50,2)

kernel_l = ["Long","Long","Long","Long","Long",
            "Hall","NoCol","Ecol05",
            "Long", "Long", "Long","Long"]

#%% SET SIMULATION PARAS
# choose the setup, which shall be plotted. this is the index of the lists above
#SIM_N = 1  # default
#SIM_N = 2 # Nsip 128
#SIM_N = 3 # pristine
#SIM_N = 4 # polluted
#SIM_N = 7 # Ecol 0.5
#SIM_N = 10 # dt_col 0.1
SIM_N = 11 # added for test

shift_cells_x = 18

#Ns=1
Ns=no_sims_list[SIM_N]
t_grid = 0.
#t_grid = 14400.
t_start=300.0
#t_start=7200.0
t_end=600.0
#t_end=14400.0
dt=1. # advection TS

# how long was the spin up?
t_spin_up=300

#simulation_mode="with_collision"
simulation_mode = "spin_up"

#%% SET PLOTTING PARAMETERS

figsize_spectra = cm2inch(16.8,24)
figsize_tg_cells = cm2inch(6.6,7)
figsize_scalar_fields = cm2inch(16.8,20)

figsize_scalar_fields_init = cm2inch(7.4,7.4)
#figsize_scalar_fields = cm2inch(100,60)


### GRID FRAMES
if SIM_N == 9:
    show_target_cells = False
else:
    show_target_cells = True

### SPECTRA
# if set to "None" the values are loaded from stored files
#no_rows = 5
#no_cols = 3
    
if SIM_N == 7:
    no_rows = 5
    no_cols = 2
else:    
    no_rows = None
    no_cols = None

# if set to "None" the values are loaded from stored files
#no_bins_R_p = 30
#no_bins_R_s = 30
no_bins_R_p = None
no_bins_R_s = None

# indices for the possible fields
#0 \Theta
#1 S
#2 r_v
#3 r_l
#4 r_\mathrm{aero}
#5 r_c
#6 r_r
#7 n_\mathrm{aero}
#8 n_c
#9 n_r
#10 R_\mathrm{avg}
#11 R_{2/1}
#12 R_\mathrm{eff}

#if fields_type == 0:
#    idx_fields_plot = np.array((0,2,3,7))
#    fields_name_add = "Th_rv_rl_na_"
#
#if fields_type == 1:
#    idx_fields_plot = np.array((5,6,9,12))
#    fields_name_add = "rc_rr_nr_Reff_"    
#
#if fields_type == 3:
#    idx_fields_plot = np.array((7,5,6,12))
#    fields_name_add = "Naero_rc_rr_Reff_"  
#if fields_type == 4:
#    idx_fields_plot = np.array((7,8,9,5,6))
#    fields_name_add = "Naero_Nc_Nr_rc_rr_"  

fields_types_avg = [4]
#fields_types_avg = [0,1]

#if fields_type == 0:
#    idx_fields_plot = np.array((0,2,3,7))
#    fields_name_add = "Th_rv_rl_na_"
#
#if fields_type == 1:
#    idx_fields_plot = np.array((5,6,9,12))
#    fields_name_add = "rc_rr_nr_Reff_"  
#
#if fields_type == 2:
#    idx_fields_plot = np.array((2,7,5,6))
#    fields_name_add = "rv_na_rc_rr_"  
#if fields_type == 3:
#    idx_fields_plot = np.array((7,5,6,12))
#    fields_name_add = "Naero_rc_rr_Reff_"  

fields_types_std = [3]

plot_abs = True
plot_rel = True

# time indices for scalar field frames
# [ 7200  7800  8400  9000  9600 10200 10800]
#idx_times_plot = np.array((0,2,3,6))
idx_times_plot = np.array((0,1))

#%% SET PARAS FOR PLOTTING MOMENTS FOR COMPARISON OF NSIP

figsize_moments = cm2inch(17,23)
figname_moments0 = "moments_vs_z_Nsip_var_" 
figname_moments_rel_dev0 = "moments_vs_z_rel_dev_Nsip_var_"
figname_conv0 = "moments_convergence_SIP_"
figname_conv_err0 = "moments_convergence_SIP_ERR_"

no_boxes_z = 25
no_cells_per_box_x = 3
target_cells_x = np.array((16, 58, 66))    

no_moments = 4
idx_t = [3]

no_variations = 3

no_cells_var = [[75,75],[75,75],[75,75]] 
solute_type_var = ["AS","AS","AS"]
kernel_var = ["Long","Long","Long"]
seed_SIP_gen_var = [3711,3811,3811]
seed_sim_var = [6711,6811,6811]
DNC0_var = [[60,40],[60,40],[60,40]]
no_spcm_var = [[13, 19],[26, 38],[52, 76]]
no_seeds_var = [50]*3
dt_col_var = [0.5]*3
#dt_col_var = [[0.5,0.5,0.5]]

#no_cells_var = [[75,75],[75,75],[75,75]] 
#solute_type_var = ["AS","AS","AS"]
#kernel_var = ["Long","Long","Long"]
#seed_SIP_gen_var = [3711,3711,3711]
#seed_sim_var = [4711,4711,4711]
#DNC0_var = [[60,40],[60,40],[60,40]]
#no_spcm_var = [[20, 30],[20, 30],[20, 30]]
#no_seeds_var = [4,4,4]
#dt_col_var = [[0.5,0.5,0.5]]

#var_type = "solute"
#var_type = "DNC"
var_type = "no_spcm"
#var_type = "dt"
#var_type = "no_cells"

MS_mom = 5
MEW_mom = 0.5
ELW_mom = 0.8
capsize_mom = 2

data_labels_mom = []
for var_n in range(no_variations):
    if var_type == "no_spcm":
        data_labels_mom.append( str(np.sum(no_spcm_var[var_n])) )
    if var_type == "dt":
        data_labels_mom.append( f"{dt_col_var[var_n]:.1f} s" )
    if var_type == "no_cells":
        data_labels_mom.append( f"{no_cells_var[var_n][0]} s" )
    
grid_paths, data_paths = gen_data_paths(solute_type_var, kernel_var,
                                        seed_SIP_gen_var, seed_sim_var,
                                        DNC0_var, no_spcm_var,
                                        no_seeds_var, dt_col_var)

#%% SET PARAS FOR ABSOLUTE DIFFERENCE PLOTS

#%% DERIVED PARAS
#%% DERIVED GRID PARAMETERS

# needed for filename
no_cells = (ncl[SIM_N], ncl[SIM_N])
#no_cells = (75, 75)
#no_cells = (3, 3)

#shift_cells_x = 56

#%% DERIVED PARTICLE PARAMETERS

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = solute_typel[SIM_N]
kernel = kernel_l[SIM_N]
#kernel = "Ecol05"
seed_SIP_gen = gseedl[SIM_N]
seed_sim = sseedl[SIM_N]
DNC0 = [DNC1[SIM_N], DNC2[SIM_N]]
no_spcm = np.array([no_spcm0l[SIM_N], no_spcm1l[SIM_N]])

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([13, 19])

no_seeds = Ns

dt_col = dt / no_col_per_advl[SIM_N]

figname_base =\
f"{solute_type}_{kernel}_dim_{no_cells[0]}_{no_cells[1]}"\
+ f"_SIP_{no_spcm[0]}_{no_spcm[1]}_Ns_{no_seeds}_DNC_{DNC0[0]}_{DNC0[1]}_dtcol_{int(dt_col*10)}"

figname_moments = figname_moments0 + figname_base + ".pdf"
figname_moments_rel_dev = figname_moments_rel_dev0 + figname_base + ".pdf"
figname_conv = figname_conv0 + figname_base + ".pdf"
figname_conv_err = figname_conv_err0 + figname_base + ".pdf"

#%% LOAD GRID AND SET PATHS

data_folder = \
    f"{solute_type}" \
    + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
    + f"eval_data_avg_Ns_{no_seeds}_" \
    + f"sg_{seed_SIP_gen}_ss_{seed_sim}_t_{int(t_start)}_{int(t_end)}/"

data_path = simdata_path + data_folder

grid_path = simdata_path + data_folder + "grid_data/" \
            + f"{seed_SIP_gen}_{seed_sim}/"

from file_handling import load_grid_from_files

grid = load_grid_from_files(grid_path + f"grid_basics_{int(t_grid)}.txt",
                            grid_path + f"arr_file1_{int(t_grid)}.npy",
                            grid_path + f"arr_file2_{int(t_grid)}.npy")

if act_compute_CFL:
    # \Delta t \, \max(|u/\Delta x| + |v/\Delta y|)
    v_abs_max_err = np.abs(grid.velocity[0]) / grid.steps[0] \
            + np.abs(grid.velocity[1]) / grid.steps[1]
    
    v_abs = np.sqrt( grid.velocity[0]**2 + grid.velocity[1]**2 )
    CFL = np.amax(v_abs_max_err) * dt
    
    print("CFL", CFL)
    print("v_abs", np.amax(v_abs))

scale_x = 1E-3    
grid.steps *= scale_x
grid.ranges *= scale_x
grid.corners[0] *= scale_x
grid.corners[1] *= scale_x
grid.centers[0] *= scale_x
grid.centers[1] *= scale_x  

seed_SIP_gen_list = np.load(data_path + "seed_SIP_gen_list.npy" )
seed_sim_list = np.load(data_path + "seed_sim_list.npy")

target_cell_list = np.load(data_path
        + f"target_cell_list_avg_Ns_{no_seeds}_"
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")

no_neighbor_cells = np.load(data_path
        + f"neighbor_cells_list_avg_Ns_{no_seeds}_"
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
        )       
no_cells_x = no_neighbor_cells[0]
no_cells_z = no_neighbor_cells[1]

if no_cells_x % 2 == 0: no_cells_x += 1
if no_cells_z % 2 == 0: no_cells_z += 1  

no_tg_cells = len(target_cell_list[0])   

#%% PLOT INITIAL CONFIG

if act_plot_grid_frames_init:
    from plotting_fcts_MA import plot_scalar_field_frames_init_avg_MA
    
#    for fields_type in fields_types_avg:
    
#        if fields_type == 0:
#            idx_fields_plot = np.array((0,2,3,7))
#            fields_name_add = "Th_rv_rl_na_"
#        
#        if fields_type == 1:
#            idx_fields_plot = np.array((5,6,9,12))
#            fields_name_add = "rc_rr_nr_Reff_"    
        
    fields_with_time = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    save_times_out_fr = np.load(data_path
            + f"save_times_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    field_names_out = np.load(data_path
            + f"field_names_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    units_out = np.load(data_path
            + f"units_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    scales_out = np.load(data_path
            + f"scales_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    
#    particle_data = 
    
#    fields_with_time = fields_with_time[idx_times_plot][:,idx_fields_plot]
#    save_times_out_fr = save_times_out_fr[idx_times_plot]-7200
#    field_names_out = field_names_out[idx_fields_plot]
#    units_out = units_out[idx_fields_plot]
#    scales_out = scales_out[idx_fields_plot]

#    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
#    fig_name = \
#               f"scalar_fields_avg_" \
#               + f"t_{save_times_out_fr[0]}_" \
#               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
#               + f"Nfie_{len(field_names_out)}_" \
#               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
#               + f"ss_{seed_sim_list[0]}_" \
#               + fields_name_add + ".pdf"
    
    fig_name = "fields_INIT_" + figname_base + ".pdf"

               
    if not os.path.exists(figpath):
            os.makedirs(figpath)    
    plot_scalar_field_frames_init_avg_MA(grid, fields_with_time,
                                        save_times_out_fr,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path=figpath+fig_name,
                                        figsize = figsize_scalar_fields_init,
                                        no_ticks=[6,6], 
                                        alpha = 1.0,
                                        TTFS = 10, LFS = 10, TKFS = 8,
                                        cbar_precision = 2,
                                        show_target_cells = show_target_cells,
                                        target_cell_list = target_cell_list,
                                        no_cells_x = no_cells_x,
                                        no_cells_z = no_cells_z)     
    plt.close("all")   


#%% PLOT AVG GRID FRAMES

if act_plot_grid_frames_avg:
    
    for fields_type in fields_types_avg:
    
        if fields_type == 0:
            idx_fields_plot = np.array((0,2,3,7))
            fields_name_add = "Th_rv_rl_na_"
        
        if fields_type == 1:
            idx_fields_plot = np.array((5,6,9,12))
            fields_name_add = "rc_rr_nr_Reff_"    
        
        if fields_type == 3:
            idx_fields_plot = np.array((7,5,6,12))
            fields_name_add = "Naero_rc_rr_Reff_"    

        if fields_type == 4:
            idx_fields_plot = np.array((7,8,9,5,6))
            fields_name_add = "Naero_Nc_Nr_rc_rr_" 
            show_target_cells = False
        
#        if fields_type == 3:
#            idx_fields_plot = np.array((7,5,6,12))
#            fields_name_add = "Naero_rc_rr_Reff_"    
        
        fields_with_time = np.load(data_path
                + f"fields_vs_time_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        save_times_out_fr = np.load(data_path
                + f"save_times_out_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        field_names_out = np.load(data_path
                + f"field_names_out_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        units_out = np.load(data_path
                + f"units_out_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        scales_out = np.load(data_path
                + f"scales_out_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )    
    
        fields_with_time = fields_with_time[idx_times_plot][:,idx_fields_plot]
        
        save_times_out_fr = save_times_out_fr[idx_times_plot] - t_spin_up
        field_names_out = field_names_out[idx_fields_plot]
        units_out = units_out[idx_fields_plot]
        scales_out = scales_out[idx_fields_plot]
    
    #    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
        fig_path = figpath
    #    fig_name = \
    #               f"scalar_fields_avg_" \
    #               + f"t_{save_times_out_fr[0]}_" \
    #               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
    #               + f"Nfie_{len(field_names_out)}_" \
    #               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
    #               + f"ss_{seed_sim_list[0]}_" \
    #               + fields_name_add + ".pdf"
        
        fig_name = "fields_avg_" + fields_name_add + figname_base + ".pdf"
    
        if not os.path.exists(fig_path):
                os.makedirs(fig_path)    
        plot_scalar_field_frames_extend_avg_MA(grid, fields_with_time,
                                            save_times_out_fr,
                                            field_names_out,
                                            units_out,
                                            scales_out,
                                            solute_type,
                                            simulation_mode, # for time in label
                                            fig_path=fig_path+fig_name,
                                            figsize = figsize_scalar_fields,
                                            SIM_N = SIM_N,
                                            no_ticks=[6,6], 
                                            alpha = 1.0,
                                            TTFS = 10, LFS = 10, TKFS = 8,
                                            cbar_precision = 2,
                                            show_target_cells = show_target_cells,
                                            target_cell_list = target_cell_list,
                                            no_cells_x = no_cells_x,
                                            no_cells_z = no_cells_z)     
    plt.close("all")   

#%% PLOT STD GRID FRAMES STD

if act_plot_grid_frames_std:
    
    for fields_type in fields_types_std:
    
        if fields_type == 0:
            idx_fields_plot = np.array((0,2,3,7))
            fields_name_add = "Th_rv_rl_na_"
        
        if fields_type == 1:
            idx_fields_plot = np.array((5,6,9,12))
            fields_name_add = "rc_rr_nr_Reff_"  
        
        if fields_type == 2:
            idx_fields_plot = np.array((2,7,5,6))
            fields_name_add = "rv_na_rc_rr_"  

        if fields_type == 3:
            idx_fields_plot = np.array((7,5,6,12))
            fields_name_add = "Naero_rc_rr_Reff_"  

        fields_with_time = np.load(data_path
                + f"fields_vs_time_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        fields_with_time_std = np.load(data_path
                + f"fields_vs_time_std_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        save_times_out_fr = np.load(data_path
                + f"save_times_out_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        field_names_out = np.load(data_path
                + f"field_names_out_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        units_out = np.load(data_path
                + f"units_out_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        scales_out = np.load(data_path
                + f"scales_out_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )    
    
        fields_with_time = fields_with_time[idx_times_plot][:,idx_fields_plot]
        fields_with_time_std = fields_with_time_std[idx_times_plot][:,idx_fields_plot]
        save_times_out_fr = save_times_out_fr[idx_times_plot] - t_spin_up
        field_names_out = field_names_out[idx_fields_plot]
        units_out = units_out[idx_fields_plot]
        scales_out = scales_out[idx_fields_plot]
    
    #    fig_name = \
    #               f"scalar_fields_avg_" \
    #               + f"t_{save_times_out_fr[0]}_" \
    #               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
    #               + f"Nfie_{len(field_names_out)}_" \
    #               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
    #               + f"ss_{seed_sim_list[0]}_" \
    #               + fields_name_add + ".pdf"
        
        fig_name_abs = "fields_std_ABS_ERROR_" + fields_name_add + figname_base + ".pdf"
        fig_name_rel = "fields_std_REL_ERROR_" + fields_name_add + figname_base + ".pdf"
                   
        if not os.path.exists(figpath):
                os.makedirs(figpath)    
        plot_scalar_field_frames_std_MA(grid, fields_with_time,
                                        fields_with_time_std,
                                            save_times_out_fr,
                                            field_names_out,
                                            units_out,
                                            scales_out,
                                            solute_type,
                                            simulation_mode, # for time in label
                                            fig_path_abs=figpath+fig_name_abs,
                                            fig_path_rel=figpath+fig_name_rel,
                                            figsize = figsize_scalar_fields,
                                            SIM_N = SIM_N,
                                            plot_abs=plot_abs,
                                            plot_rel=plot_rel,                                            
                                            no_ticks=[6,6], 
                                            alpha = 1.0,
                                            TTFS = 10, LFS = 10, TKFS = 8,
                                            cbar_precision = 2,
                                            show_target_cells = show_target_cells,
                                            target_cell_list = target_cell_list,
                                            no_cells_x = no_cells_x,
                                            no_cells_z = no_cells_z)     
    plt.close("all")   

#%% PLOT ABSOLUTE DEVIATIONS BETWEEN TWO GRIDS

#compare_type_abs_dev = "dt_col"
compare_type_abs_dev = "Ncell"
#compare_type_abs_dev = "solute"
#compare_type_abs_dev = "Kernel"
#compare_type_abs_dev = "Nsip"

figsize_abs_dev = figsize_scalar_fields

if act_plot_grid_frames_abs_dev:
    
    from plotting_fcts_MA import plot_scalar_field_frames_abs_dev_MA
    
    # n_aero, r_c, r_r, R_eff
#    idx_fields_plot = (7, 5, 6, 12)
    
    ###
#    idx_fields_plot = np.array((7,5,6,12))
#    fields_name_add = "Naero_rc_rr_Reff_"
    
    ###
    idx_fields_plot = np.array((7,8,9,10))
    fields_name_add = "Naero_Nc_Nr_Ravg_"
    
    idx_times_plot = np.array((0,2,3,6))
    
    if compare_type_abs_dev == "dt_col":
        SIM_Ns = [1,10]
    elif compare_type_abs_dev == "Ncell":
        SIM_Ns = [1,9]
    elif compare_type_abs_dev == "solute":
        SIM_Ns = [1,8]
    elif compare_type_abs_dev == "Kernel":
        SIM_Ns = [1,5]
    elif compare_type_abs_dev == "Nsip":
        SIM_Ns = [1,2]
#        fname_abs_dev_add = ""
    
    SIM_N = SIM_Ns[0]
    
    no_cells = (ncl[SIM_N], ncl[SIM_N])
    solute_type = solute_typel[SIM_N]
    kernel = kernel_l[SIM_N]
    seed_SIP_gen = gseedl[SIM_N]
    seed_sim = sseedl[SIM_N]
    DNC0 = [DNC1[SIM_N], DNC2[SIM_N]]
    no_spcm = np.array([no_spcm0l[SIM_N], no_spcm1l[SIM_N]])
    Ns=no_sims_list[SIM_N]
    no_seeds = Ns
    dt_col = dt / no_col_per_advl[SIM_N]
    
#    figname_base =\
#    f"{solute_type}_{kernel}_dim_{no_cells[0]}_{no_cells[1]}"\
#    + f"_SIP_{no_spcm[0]}_{no_spcm[1]}_Ns_{no_seeds}_DNC_{DNC0[0]}_{DNC0[1]}_dtcol_{int(dt_col*10)}"
    
    data_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen}_ss_{seed_sim}/"
    
    grid_path = simdata_path + data_folder + "grid_data/" \
                + f"{seed_SIP_gen}_{seed_sim}/"
    
    grid = load_grid_from_files(grid_path + f"grid_basics_{int(t_grid)}.txt",
                                grid_path + f"arr_file1_{int(t_grid)}.npy",
                                grid_path + f"arr_file2_{int(t_grid)}.npy")
    scale_x = 1E-3    
    grid.steps *= scale_x
    grid.ranges *= scale_x
    grid.corners[0] *= scale_x
    grid.corners[1] *= scale_x
    grid.centers[0] *= scale_x
    grid.centers[1] *= scale_x  
    
    data_path = simdata_path + data_folder
    seed_SIP_gen_list = np.load(data_path + "seed_SIP_gen_list.npy" )
    seed_sim_list = np.load(data_path + "seed_sim_list.npy")
    
    fields_with_time1 = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    fields_with_time_std1 = np.load(data_path
            + f"fields_vs_time_std_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    save_times_out_fr = np.load(data_path
            + f"save_times_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    field_names_out = np.load(data_path
            + f"field_names_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    units_out = np.load(data_path
            + f"units_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    scales_out = np.load(data_path
            + f"scales_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )   
    
    ####################
    SIM_N = SIM_Ns[1]
    
    no_cells = (ncl[SIM_N], ncl[SIM_N])
    solute_type = solute_typel[SIM_N]
    kernel = kernel_l[SIM_N]
    seed_SIP_gen = gseedl[SIM_N]
    seed_sim = sseedl[SIM_N]
    DNC0 = [DNC1[SIM_N], DNC2[SIM_N]]
    no_spcm = np.array([no_spcm0l[SIM_N], no_spcm1l[SIM_N]])
    Ns=no_sims_list[SIM_N]
    no_seeds = Ns
    dt_col = dt / no_col_per_advl[SIM_N]
    
    figname_base =\
    f"{solute_type}_{kernel}_dim_{no_cells[0]}_{no_cells[1]}"\
    + f"_SIP_{no_spcm[0]}_{no_spcm[1]}_Ns_{no_seeds}_DNC_{DNC0[0]}_{DNC0[1]}_dtcol_{int(dt_col*10)}"
    
    data_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen}_ss_{seed_sim}/"
    
    data_path = simdata_path + data_folder
    seed_SIP_gen_list = np.load(data_path + "seed_SIP_gen_list.npy" )
    seed_sim_list = np.load(data_path + "seed_sim_list.npy")
    
    fields_with_time2 = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    fields_with_time_std2 = np.load(data_path
            + f"fields_vs_time_std_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )   

    
    ####################
    
#    print(fields_with_time2.shape)
    
    fields_with_time1 = fields_with_time1[idx_times_plot][:,idx_fields_plot]
    fields_with_time_std1 = fields_with_time_std1[idx_times_plot][:,idx_fields_plot]
    fields_with_time2 = fields_with_time2[idx_times_plot][:,idx_fields_plot]
    fields_with_time_std2 = fields_with_time_std2[idx_times_plot][:,idx_fields_plot]
    
    if compare_type_abs_dev == "Ncell":
        fields_with_time2 = (  fields_with_time2[:,:,::2,::2] 
                             + fields_with_time2[:,:,1::2,::2] 
                             + fields_with_time2[:,:,::2,1::2] 
                             + fields_with_time2[:,:,1::2,1::2] ) / 4.
        fields_with_time_std2 = (  fields_with_time_std2[:,:,::2,::2] 
                                 + fields_with_time_std2[:,:,1::2,::2] 
                                 + fields_with_time_std2[:,:,::2,1::2] 
                                 + fields_with_time_std2[:,:,1::2,1::2] ) / 4.
#    print(fields_with_time_x.shape)
    
    print(field_names_out)
    
    save_times_out_fr = save_times_out_fr[idx_times_plot] - t_spin_up
    field_names_out = field_names_out[idx_fields_plot]
    units_out = units_out[idx_fields_plot]
    scales_out = scales_out[idx_fields_plot]

#    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
#    fig_path = figpath
#    fig_name = \
#               f"scalar_fields_avg_" \
#               + f"t_{save_times_out_fr[0]}_" \
#               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
#               + f"Nfie_{len(field_names_out)}_" \
#               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
#               + f"ss_{seed_sim_list[0]}_" \
#               + fields_name_add + ".pdf"
    
    fig_name = "fields_abs_dev_" + compare_type_abs_dev + "_" \
               + fields_name_add + figname_base + ".pdf"
    fig_name_abs_err = "fields_abs_dev_ABS_ERR_" + compare_type_abs_dev + "_" \
               + fields_name_add + figname_base + ".pdf"
               
    if not os.path.exists(figpath):
            os.makedirs(figpath)    
            
    plot_scalar_field_frames_abs_dev_MA(grid,
                                        fields_with_time1,
                                        fields_with_time_std1,
                                        fields_with_time2,
                                        fields_with_time_std2,
                                        save_times_out_fr,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        compare_type=compare_type_abs_dev,
                                        fig_path=figpath + fig_name,
                                        fig_path_abs_err=figpath + fig_name_abs_err,
                                        figsize=figsize_abs_dev,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 10, LFS = 10, TKFS = 8,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        )   
    ### CAN BE ACTIVATED!!!
#    if compare_type_abs_dev == "Ncell":       
#        fig_name2 = "fields_coarse_grain_" + compare_type_abs_dev + "_" \
#           + fields_name_add + figname_base + ".pdf"
#        plot_scalar_field_frames_extend_avg_MA(grid, fields_with_time2,
#                                            save_times_out_fr,
#                                            field_names_out,
#                                            units_out,
#                                            scales_out,
#                                            solute_type,
#                                            simulation_mode, # for time in label
#                                            fig_path=figpath+fig_name2,
#                                            figsize = figsize_scalar_fields,
#                                            no_ticks=[6,6], 
#                                            alpha = 1.0,
#                                            TTFS = 10, LFS = 10, TKFS = 8,
#                                            cbar_precision = 2,
#                                            show_target_cells = show_target_cells,
#                                            target_cell_list = target_cell_list,
#                                            no_cells_x = no_cells_x,
#                                            no_cells_z = no_cells_z)     
#      
#        fig_name3 = "fields_default_compare_" + compare_type_abs_dev + "_" \
#           + fields_name_add + figname_base + ".pdf"
#        plot_scalar_field_frames_extend_avg_MA(grid, fields_with_time1,
#                                            save_times_out_fr,
#                                            field_names_out,
#                                            units_out,
#                                            scales_out,
#                                            solute_type,
#                                            simulation_mode, # for time in label
#                                            fig_path=figpath+fig_name3,
#                                            figsize = figsize_scalar_fields,
#                                            no_ticks=[6,6], 
#                                            alpha = 1.0,
#                                            TTFS = 10, LFS = 10, TKFS = 8,
#                                            cbar_precision = 2,
#                                            show_target_cells = show_target_cells,
#                                            target_cell_list = target_cell_list,
#                                            no_cells_x = no_cells_x,
#                                            no_cells_z = no_cells_z)     
    plt.close("all")   

#%% PLOT AVG GRID FRAMES SHIFT IN X DIRECTION

if act_plot_grid_frames_avg_shift:
    from analysis import plot_scalar_field_frames_extend_avg_shift

    fields_with_time = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    save_times_out_fr = np.load(data_path
            + f"save_times_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    field_names_out = np.load(data_path
            + f"field_names_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    units_out = np.load(data_path
            + f"units_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    scales_out = np.load(data_path
            + f"scales_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    

    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
    fig_name = \
               f"scalar_fields_avg_shift_{shift_cells_x}_" \
               + f"t_{save_times_out_fr[0]}_" \
               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
               + f"Nfie_{len(field_names_out)}_" \
               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
               + f"ss_{seed_sim_list[0]}.png"
    if not os.path.exists(fig_path):
            os.makedirs(fig_path)    
    plot_scalar_field_frames_extend_avg_shift(grid, fields_with_time,
                                        save_times_out_fr,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path=fig_path+fig_name,
                                        no_ticks=[6,6], 
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
#                                        show_target_cells = False,
                                        show_target_cells = show_target_cells,
                                        target_cell_list = target_cell_list,
                                        no_cells_x = no_cells_x,
                                        no_cells_z = no_cells_z,
                                        shift_cells_x = shift_cells_x)     
#    plt.close("all")   

#%% PLOT SPECTRA AVG 

if act_plot_spectra_avg_Arabas:
#    grid = load_grid_from_files(grid_path + f"grid_basics_{int(t_grid)}.txt",
#                            grid_path + f"arr_file1_{int(t_grid)}.npy",
#                            grid_path + f"arr_file2_{int(t_grid)}.npy")
#    from analysis import plot_size_spectra_R_Arabas
    
    f_R_p_list = np.load(data_path
            + f"f_R_p_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    f_R_s_list = np.load(data_path
            + f"f_R_s_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    bins_R_p_list = np.load(data_path
            + f"bins_R_p_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    bins_R_s_list = np.load(data_path
            + f"bins_R_s_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    save_times_out_spectra = np.load(data_path
            + f"save_times_out_spectra_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    grid_r_l_list = np.load(data_path
            + f"grid_r_l_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    R_min_list = np.load(data_path
            + f"R_min_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    R_max_list = np.load(data_path
            + f"R_max_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    
    # if not given manually above:
    if no_rows is None:
        no_rowcol = np.load(data_path
                + f"no_rows_no_cols_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )    
        no_rows = no_rowcol[0]
        no_cols = no_rowcol[1]
        
    print(no_rows, no_cols)    
    if no_bins_R_p is None:
        no_bins_p_s = np.load(data_path
                + f"no_bins_p_s_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )        
        no_bins_R_p = no_bins_p_s[0]
        no_bins_R_s = no_bins_p_s[1]
        
        
        print("no_bins_R_p")
        print(no_bins_R_p)
        print("no_bins_R_s")
        print(no_bins_R_s)
    
    j_low = target_cell_list[1].min()
    j_high = target_cell_list[1].max()    
    t_low = save_times_out_spectra.min()
    t_high = save_times_out_spectra.max()    

    no_tg_cells = no_rows * no_cols

#    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"    
#    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"  
    fig_path = figpath
    if not os.path.exists(fig_path):
        os.makedirs(fig_path) 

    fig_path_spectra =\
        fig_path \
        + f"spectra_Nc_{no_tg_cells}_" + figname_base + ".pdf"
#        f"spectra_at_tg_cells_j_from_{j_low}_to_{j_high}_" \
#        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
#        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    fig_path_tg_cells =\
        fig_path \
        + f"target_c_pos_Nc_{no_tg_cells}_" + figname_base + ".pdf"
#        + f"tg_cell_posi_j_from_{j_low}_to_{j_high}_" \
#        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
#        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    fig_path_R_eff =\
        fig_path \
        + f"R_eff_Nc_{no_tg_cells}_" + figname_base + ".pdf"
#        + f"R_eff_{j_low}_to_{j_high}_" \
#        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
#        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    
    plot_size_spectra_R_Arabas_MA(
            f_R_p_list, f_R_s_list,
            bins_R_p_list, bins_R_s_list,
            grid_r_l_list,
            R_min_list, R_max_list,
            save_times_out_spectra,
            solute_type,
            grid,
            target_cell_list,
            no_cells_x, no_cells_z,
            no_bins_R_p, no_bins_R_s,
            no_rows, no_cols,
            t_spin_up,
            SIM_N,
            TTFS=10, LFS=10, TKFS=8, LW = 2.0, MS = 0.4,
            figsize_spectra = figsize_spectra,
            figsize_trace_traj = figsize_tg_cells,
            fig_path = fig_path_spectra,
            show_target_cells = True,
            fig_path_tg_cells = fig_path_tg_cells   ,
            fig_path_R_eff = fig_path_R_eff,
            trajectory = None,
            show_textbox=False)    
#    plot_size_spectra_R_Arabas_MA(f_R_p_list, f_R_s_list,
#                               bins_R_p_list, bins_R_s_list,
#                               grid_r_l_list,
#                               R_min_list, R_max_list,
#                               save_times_out_spectra,
#                               solute_type,
#                               grid,
#                               target_cell_list,
#                               no_cells_x, no_cells_z,
#                               no_bins_R_p, no_bins_R_s,
#                               no_rows, no_cols,
#                               TTFS=12, LFS=10, TKFS=10, LW = 2.0,
#                               fig_path = fig_path + fig_name,
#                               show_target_cells = True,
#                               fig_path_tg_cells = fig_path_tg_cells   ,
#                               fig_path_R_eff = fig_path_R_eff
#                               )        
    plt.close("all")    
    

#%% PLOT MOMENTS VS Z

fmt_list = ["x-", "o--", "d:"]

units_mom = [
        r"\si{1/m^3}",
        r"\si{\micro\meter/m^3}",
        r"\si{\micro\meter^2/m^3}",
        r"\si{\micro\meter^3/m^3}"]
#        r"$1/m^3$",
#        r"$\si{\micro\meter/m^3}$",
#        r"$\si{\micro\meter^2/m^3}$",
#        r"$\si{\micro\meter^3/m^3}$"]

#no_cells_var = [[75,75],[75,75],[75,75]] 
#solute_type_var = ["AS","AS","AS"]
#kernel_var = ["Long","Long","Long"]
#seed_SIP_gen_var = [3711,3811,3811]
#seed_sim_var = [6711,6811,6811]
#DNC0_var = [[60,40],[60,40],[60,40]]
#no_spcm_var = [[13, 19],[26, 38],[52, 76]]
#no_seeds_var = [50]*3
#dt_col_var = [0.5]*3

if act_plot_moments_vs_z:
    print(grid_paths)
    
    no_target_cells_x = len(target_cells_x)
    no_target_cells_z = no_boxes_z

    no_rows = no_moments
    no_cols = no_target_cells_x
    
    
    fig, axes = plt.subplots(no_rows, no_cols, figsize=figsize_moments,
                             sharex=True, sharey="row")
    
    for var_n in range(no_variations):
        grid_path_ = grid_paths[var_n]
        data_path_ = data_paths[var_n]
    
        grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
                                grid_path_ + f"arr_file1_{int(t_grid)}.npy",
                                grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
        
        g_seed = seed_SIP_gen_var[var_n]
        s_seed = seed_sim_var[var_n]
        Ns = no_seeds_var[var_n]
        
    #    np.save(simdata_path + output_folder
    #            + f"moments_vs_time_avg_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            moments_vs_time_avg)
    #    np.save(simdata_path + output_folder
    #            + f"moments_vs_time_std_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            moments_vs_time_std)
    #    np.save(simdata_path + output_folder
    #            + f"save_times_out_avg_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            save_times_out)
    
        moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
                       + f"moments_vs_time_all_seeds_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")
    #    moments_vs_time_avg = np.load(data_path + "moments/"
    #                   + f"moments_vs_time_avg_Ns_{no_seeds}_"
    #                   + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
    #    moments_vs_time_std = np.load(data_path + "moments/"
    #                   + f"moments_vs_time_std_Ns_{no_seeds}_"
    #                   + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
        save_times_out = np.load(data_path_ + "moments/"
                       + f"save_times_out_avg_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")
        
#        np.load(data_path_ + "moments/"
#                       + f"save_times_out_avg_Ns_{no_seeds}_"
#                       + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
        
        Nz = grid.no_cells[1]
        no_cells_per_box_z = Nz // no_boxes_z
        
        print("no_cells_per_box_z")
        print(no_cells_per_box_z)
        
        start_cell_z = no_cells_per_box_z // 2
        
        target_cells_z = np.arange( start_cell_z, Nz, no_cells_per_box_z )
        
        print("target_cells_z")
        print(target_cells_z)
        
#        print(target_cells_z)        
        
        no_seeds = moments_vs_time_all_seeds.shape[0]
#        no_moments = moments_vs_time_all_seeds.shape[2]
        
        moments_vs_z = np.zeros( (no_target_cells_x,
                                  no_target_cells_z,
                                  no_moments
                                  ),
                                dtype = np.float64)
        
        times = save_times_out[ np.array(idx_t) ]
        no_times_eval = len(times)
        
    #    from numba import njit
    #    @njit()    
                                
        moments_at_boxes_all_seeds = avg_moments_over_boxes(
                moments_vs_time_all_seeds, no_seeds, idx_t, no_moments,
                target_cells_x, target_cells_z,
                no_cells_per_box_x, no_cells_per_box_z )
        
        moments_at_boxes_avg = np.average(moments_at_boxes_all_seeds, axis=0)
        moments_at_boxes_std = \
            np.std(moments_at_boxes_all_seeds, axis=0, ddof=1) / np.sqrt(no_seeds)
        
        for row_n in range(no_moments):
            for col_n in range(no_target_cells_x):
                ax = axes[row_n, col_n]
                z = ((target_cells_z + 0.5) * grid.steps[1]) / 1000.
                x = ((target_cells_x[col_n] + 0.5) * grid.steps[0]) / 1000.
                ax.errorbar(z, moments_at_boxes_avg[0,row_n,col_n],
                            yerr=moments_at_boxes_std[0,row_n,col_n],
                            fmt = fmt_list[var_n], ms = MS_mom, mew=MEW_mom,
                            fillstyle="none", elinewidth = ELW_mom,
                            capsize=capsize_mom,
                            label=data_labels_mom[var_n])
    #            ax.plot( z, moments_at_boxes_std[0,row_n,col_n] )
                if var_n == 0:
                    if row_n == no_rows - 1:
                        ax.set_xlabel("$z$ (km)")
                    if row_n == 0:
                        ax.set_title(f"$x={x:.2}$ km")
                    if col_n == 0: 
                        ax.set_ylabel(f"$\\lambda_{row_n}$ (${units_mom[row_n]}$)")
    for row_n in range(no_moments):
        for col_n in range(no_target_cells_x):                    
            ax = axes[row_n, col_n]
            axes[row_n, col_n].grid()
            
            if col_n == 0:
                if row_n == 0:
                    axes[row_n, col_n].legend(ncol=3,
                           handlelength=1.8, handletextpad=0.3,
                          columnspacing=0.8, borderpad=0.25) 
                else:
                    ax.set_yscale("log")
                    axes[row_n, col_n].legend(ncol=1,
                           handlelength=1.6, handletextpad=0.3,
                          columnspacing=0.8, borderpad=0.25) 
                
    
#    axes[1,0].set_ylim((1E8,1.8E9))
    axes[no_rows-1, no_cols-1].set_xticks(np.linspace(0,1.5,6))                
    axes[no_rows-1, no_cols-1].set_xlim((0.0,1.5))
    pad_ax_h = 0.15
    pad_ax_v = 0.1
    fig.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
    fig.subplots_adjust(wspace=pad_ax_v)                  
    fig.savefig(figpath + figname_moments,
                bbox_inches = 'tight',
                pad_inches = 0.04,
                dpi=600
                )          
    
#%% PLOT MOMENTS DIFFERENCES VS Z

#fmt_list = ["", "x-", "o--", "d:"]
#fmt_list = ["x-", "o--", "d:"]
fmt_list = ["x", "o", "d"]

#units_mom = [ "a","b","c","d"]
#units_mom = [
#        r"\si{1/m^3}",
#        r"\si{\micro\meter/m^3}",
#        r"\si{\micro\meter^2/m^3}",
#        r"\si{\micro\meter^3/m^3}"]

#        r"$1/m^3$",
#        r"$\si{\micro\meter/m^3}$",
#        r"$\si{\micro\meter^2/m^3}$",
#        r"$\si{\micro\meter^3/m^3}$"]

#no_cells_var = [[75,75],[75,75],[75,75]] 
#solute_type_var = ["AS","AS","AS"]
#kernel_var = ["Long","Long","Long"]
#seed_SIP_gen_var = [3711,3811,3811]
#seed_sim_var = [6711,6811,6811]
#DNC0_var = [[60,40],[60,40],[60,40]]
#no_spcm_var = [[13, 19],[26, 38],[52, 76]]
#no_seeds_var = [50]*3
#dt_col_var = [0.5]*3

if act_plot_moments_diff_vs_z:
    print(grid_paths)
    
    no_target_cells_x = len(target_cells_x)
    no_target_cells_z = no_boxes_z

    no_rows = no_moments
    no_cols = no_target_cells_x
    
    
    ### load reference curve at last var_n (last in grid_path list)
    var_n = no_variations-1
    grid_path_ = grid_paths[var_n]
    data_path_ = data_paths[var_n]

    grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
                            grid_path_ + f"arr_file1_{int(t_grid)}.npy",
                            grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
    
    g_seed = seed_SIP_gen_var[var_n]
    s_seed = seed_sim_var[var_n]
    Ns = no_seeds_var[var_n]
    
#    np.save(simdata_path + output_folder
#            + f"moments_vs_time_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            moments_vs_time_avg)
#    np.save(simdata_path + output_folder
#            + f"moments_vs_time_std_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            moments_vs_time_std)
#    np.save(simdata_path + output_folder
#            + f"save_times_out_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            save_times_out)

    moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
                   + f"moments_vs_time_all_seeds_Ns_{Ns}_"
                   + f"sg_{g_seed}_ss_{s_seed}.npy")
#    moments_vs_time_avg = np.load(data_path + "moments/"
#                   + f"moments_vs_time_avg_Ns_{no_seeds}_"
#                   + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
#    moments_vs_time_std = np.load(data_path + "moments/"
#                   + f"moments_vs_time_std_Ns_{no_seeds}_"
#                   + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
    save_times_out = np.load(data_path_ + "moments/"
                   + f"save_times_out_avg_Ns_{Ns}_"
                   + f"sg_{g_seed}_ss_{s_seed}.npy")
    
#        np.load(data_path_ + "moments/"
#                       + f"save_times_out_avg_Ns_{no_seeds}_"
#                       + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
    
    Nz = grid.no_cells[1]
    no_cells_per_box_z = Nz // no_boxes_z
    
    print("no_cells_per_box_z")
    print(no_cells_per_box_z)
    
    start_cell_z = no_cells_per_box_z // 2
    
    target_cells_z = np.arange( start_cell_z, Nz, no_cells_per_box_z )
    
    print("target_cells_z")
    print(target_cells_z)
    
#        print(target_cells_z)        


    
    no_seeds = moments_vs_time_all_seeds.shape[0]
#        no_moments = moments_vs_time_all_seeds.shape[2]
    
    moments_vs_z = np.zeros( (no_target_cells_x,
                              no_target_cells_z,
                              no_moments
                              ),
                            dtype = np.float64)
    
    times = save_times_out[ np.array(idx_t) ]
    no_times_eval = len(times)
    
#    from numba import njit
#    @njit()    
                            
            
        
    moments_at_boxes_all_seeds = avg_moments_over_boxes(
            moments_vs_time_all_seeds, no_seeds, idx_t, no_moments,
            target_cells_x, target_cells_z,
            no_cells_per_box_x, no_cells_per_box_z )
    
    
    moments_at_boxes_avg_ref = np.average(moments_at_boxes_all_seeds, axis=0)
    moments_at_boxes_std_ref = \
        np.std(moments_at_boxes_all_seeds, axis=0, ddof=1) / np.sqrt(no_seeds)
#        np.std(moments_at_boxes_all_seeds, axis=0, ddof=1)  

    fig, axes = plt.subplots(no_rows, no_cols, figsize=figsize_moments,
                             sharex=True, sharey=False)
    
    ylim = np.zeros( (no_rows, no_cols, 2))
    ylim[:,:,::2] = 1
    ### load and plot other curves
    for var_n in range(0,no_variations-1):
        grid_path_ = grid_paths[var_n]
        data_path_ = data_paths[var_n]
    
        grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
                                grid_path_ + f"arr_file1_{int(t_grid)}.npy",
                                grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
        
        g_seed = seed_SIP_gen_var[var_n]
        s_seed = seed_sim_var[var_n]
        Ns = no_seeds_var[var_n]
        
    #    np.save(simdata_path + output_folder
    #            + f"moments_vs_time_avg_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            moments_vs_time_avg)
    #    np.save(simdata_path + output_folder
    #            + f"moments_vs_time_std_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            moments_vs_time_std)
    #    np.save(simdata_path + output_folder
    #            + f"save_times_out_avg_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            save_times_out)
    
        moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
                       + f"moments_vs_time_all_seeds_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")
    #    moments_vs_time_avg = np.load(data_path + "moments/"
    #                   + f"moments_vs_time_avg_Ns_{no_seeds}_"
    #                   + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
    #    moments_vs_time_std = np.load(data_path + "moments/"
    #                   + f"moments_vs_time_std_Ns_{no_seeds}_"
    #                   + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
        save_times_out = np.load(data_path_ + "moments/"
                       + f"save_times_out_avg_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")
        
#        np.load(data_path_ + "moments/"
#                       + f"save_times_out_avg_Ns_{no_seeds}_"
#                       + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
        
        Nz = grid.no_cells[1]
        no_cells_per_box_z = Nz // no_boxes_z
        
        print("no_cells_per_box_z")
        print(no_cells_per_box_z)
        
        start_cell_z = no_cells_per_box_z // 2
        
        target_cells_z = np.arange( start_cell_z, Nz, no_cells_per_box_z )
        
        print("target_cells_z")
        print(target_cells_z)
        
#        print(target_cells_z)        
        
        no_seeds = moments_vs_time_all_seeds.shape[0]
#        no_moments = moments_vs_time_all_seeds.shape[2]
        
        moments_vs_z = np.zeros( (no_target_cells_x,
                                  no_target_cells_z,
                                  no_moments
                                  ),
                                dtype = np.float64)
        
        times = save_times_out[ np.array(idx_t) ]
        no_times_eval = len(times)
        
    #    from numba import njit
    #    @njit()    
            
        moments_at_boxes_all_seeds = avg_moments_over_boxes(
                moments_vs_time_all_seeds, no_seeds, idx_t, no_moments,
                target_cells_x, target_cells_z,
                no_cells_per_box_x, no_cells_per_box_z )
        
        moments_at_boxes_avg = np.average(moments_at_boxes_all_seeds, axis=0)
        moments_at_boxes_std = \
            np.std(moments_at_boxes_all_seeds, axis=0, ddof=1) / np.sqrt(no_seeds)
        
        rel_dev = np.abs((moments_at_boxes_avg - moments_at_boxes_avg_ref) \
                         / moments_at_boxes_avg_ref)
        std_rel0 = moments_at_boxes_std / moments_at_boxes_avg_ref
        
        rel_dev_thresh = 1E-3
        
        err_low = np.where( rel_dev >= rel_dev_thresh, 
                 np.minimum( rel_dev-rel_dev_thresh, std_rel0 ),
                 0.0
                 )
        std_rel = [err_low,
                   std_rel0]
        std_rel = np.array(std_rel)
        
        std_rel_ref = moments_at_boxes_std_ref / moments_at_boxes_avg_ref
        
        data_colors = ["blue","k"]
        face_colors = ["blue","darkorange"]
        
        for row_n in range(no_moments):
            for col_n in range(no_target_cells_x):
#                if row_n > 0:
                m_min = rel_dev[0,row_n,col_n].min()
#                if row_n == 0:
                m_max = (rel_dev[0,row_n,col_n] + std_rel[1,0,row_n,col_n]).max()
#                else:
#                    m_max = rel_dev[0,row_n,col_n].max()
                if m_min < ylim[row_n][col_n][0]: ylim[row_n][col_n][0] = m_min
                if m_max > ylim[row_n][col_n][1]: ylim[row_n][col_n][1] = m_max
                ax = axes[row_n, col_n]
                z = ((target_cells_z + 0.5) * grid.steps[1]) / 1000.
#                print(z)
                x = ((target_cells_x[col_n] + 0.5) * grid.steps[0]) / 1000.
#                if var_n == 0: color = "blue"
#                if var_n == 1: color = "k"
                ax.plot(z, rel_dev[0,row_n,col_n],
#                            yerr=std_rel[0,row_n,col_n],
#                            yerr=std_rel[:,0,row_n,col_n],
                            fmt_list[var_n],
                            ms = MS_mom, mew=MEW_mom,
                            fillstyle="none",
                            c = data_colors[var_n],
#                            elinewidth = ELW_mom,
#                            capsize=capsize_mom,
                            label=data_labels_mom[var_n], zorder=51)
#                ax.errorbar(z, rel_dev[0,row_n,col_n],
##                            yerr=std_rel[0,row_n,col_n],
#                            yerr=std_rel[:,0,row_n,col_n],
#                            fmt = fmt_list[var_n], ms = MS_mom, mew=MEW_mom,
#                            fillstyle="none", elinewidth = ELW_mom,
#                            capsize=capsize_mom,
#                            label=data_labels_mom[var_n], zorder=51)
                ax.fill_between(z,
                                rel_dev[0,row_n,col_n] - std_rel[0,0,row_n,col_n],
                                rel_dev[0,row_n,col_n] + std_rel[1,0,row_n,col_n],
#                                alpha=0.15, facecolor="green")
                                alpha=0.3, lw=1,
                                edgecolor=face_colors[var_n],
                                label=data_labels_mom[var_n])
#                                facecolor="orange")
                if var_n == no_variations-2:
                    ax.fill_between(z,
                                    np.ones_like(std_rel_ref[0,row_n, col_n])*rel_dev_thresh,
                                    std_rel_ref[0,row_n, col_n],
#                                    alpha=0.15, facecolor="green")
                                    alpha=0.5,
                                    facecolor="lightgreen",
#                                    facecolor="grey",
                                    edgecolor="green", lw=1,
                                    zorder=50,
                                    label=data_labels_mom[no_variations-1])
    #            ax.plot( z, moments_at_boxes_std[0,row_n,col_n] )
#                ax.set_yscale("log")
                if var_n == 0:
                    if row_n == no_rows - 1:
                        ax.set_xlabel("$z$ (km)")
                    if row_n == 0:
                        ax.set_title(f"$x={x:.2}$ km")
                    if col_n == 0: 
                        ax.set_ylabel(f"$\\lambda_{row_n}$ (rel. dev.)")
    annotations = ["A", "B", "C", "D", "E", "F",
                   "G", "H", "I", "J", "K", "L",
                   "M", "N", "O", "P", "Q", "R"]
    cnt_ = -1
    for row_n in range(no_moments):
        for col_n in range(no_target_cells_x):                    
            cnt_ += 1
            ax = axes[row_n, col_n]
            ax.annotate(f"({annotations[cnt_]})", (0.03,0.92),
                        xycoords="axes fraction")
            ax.grid()
#            if col_n == 0:
#                ax.legend()
            if row_n == 0:
                ax.set_ylim((rel_dev_thresh, ylim[row_n,col_n][1]*1.1))
            if row_n > 0:
                ax.set_ylim((rel_dev_thresh, ylim[row_n,col_n][1]*1.1))
#                ax.set_ylim((ylim[row_n,col_n][0]/2, ylim[row_n,col_n][1]*2))
#                ax.set_yscale("symlog")
            ax.set_yscale("log")
#                ax.set_yscale("log", nonposy="mask")
#                ax.set_yscale("log", nonposy="mask")
#                if np.abs(ylim[row_n, col_n]).max() > 0.5:
#                    ax.set_yscale("log")
    
    axes[no_rows-1, no_cols-1].set_xticks(np.linspace(0,1.5,6))                                
    axes[no_rows-1, no_cols-1].set_xlim((0.0,1.5))
    axes[2,1].legend(loc="upper right", bbox_to_anchor=(1.06,1.1))
#    axes[1,0].set_ylim((1E8,1.8E9))
#    axes[1,0].set_yscale("log")
    pad_ax_h = 0.1
    pad_ax_v = 0.26
    fig.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
    fig.subplots_adjust(wspace=pad_ax_v)                    
    
    fig.savefig(figpath + figname_moments_rel_dev,
                bbox_inches = 'tight',
                pad_inches = 0.04,
                dpi=600
                )          
    plt.close("all")    
    

#%% PLOT n_tot, R_avg, Var_R, skewness vs t

#idx_t_conv = ()

figsize_conv = cm2inch(15,15)

#figname_conv = ""

time_idx_conv = np.arange(0,7)

if act_plot_moments_norm_vs_t:

    no_rows = 2
    no_cols = 2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize_conv, sharex=True)
    
    for var_n in range(no_variations):
        grid_path_ = grid_paths[var_n]
        data_path_ = data_paths[var_n]
        
        grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
                                grid_path_ + f"arr_file1_{int(t_grid)}.npy",
                                grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
        
        g_seed = seed_SIP_gen_var[var_n]
        s_seed = seed_sim_var[var_n]
        Ns = no_seeds_var[var_n]
        
        #    np.save(simdata_path + output_folder
        #            + f"moments_vs_time_avg_Ns_{no_seeds}_"
        #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
        #            moments_vs_time_avg)
        #    np.save(simdata_path + output_folder
        #            + f"moments_vs_time_std_Ns_{no_seeds}_"
        #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
        #            moments_vs_time_std)
        #    np.save(simdata_path + output_folder
        #            + f"save_times_out_avg_Ns_{no_seeds}_"
        #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
        #            save_times_out)
        
        save_times_out = np.load(data_path_ + "moments/"
                       + f"save_times_out_avg_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")    
        
        moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
                       + f"moments_vs_time_all_seeds_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")
        
        print(var_n, moments_vs_time_all_seeds.shape)
    
    #    moments_vs_time_seed_avg = np.average(moments_vs_time_all_seeds, axis=0)
        
    #    moments_vs_time_seed_std =\
    #        np.std(moments_vs_time_all_seeds, axis=0,ddof=1) / np.sqrt(Ns)
    
    #    print(var_n, moments_vs_time_seed_avg.shape)
        
        moments_vs_time_grid_avg = np.average(moments_vs_time_all_seeds, axis=(3,4))
    #    moments_vs_time_grid_std = np.average(moments_vs_time_seed_std, axis=(2,3))
        
        print(var_n, moments_vs_time_grid_avg.shape)
        
        ### go to normalized values: mean radius = mom1/mom0, variance
        
        moments_norm_vs_time = np.zeros_like(moments_vs_time_grid_avg)
        moments_norm_vs_time[:,:,0] = moments_vs_time_grid_avg[:,:,0]
        moments_norm_vs_time[:,:,1] = moments_vs_time_grid_avg[:,:,1] \
                                    / moments_vs_time_grid_avg[:,:,0]
        moments_norm_vs_time[:,:,2] = moments_vs_time_grid_avg[:,:,2] \
                                    / moments_vs_time_grid_avg[:,:,0] \
                                    - moments_norm_vs_time[:,:,1]**2
        std_xx = np.sqrt( moments_norm_vs_time[:,:,2] )
        moments_norm_vs_time[:,:,3] = (moments_vs_time_grid_avg[:,:,3] \
                                     / moments_vs_time_grid_avg[:,:,0] \
                                     - 3 * moments_norm_vs_time[:,:,1] \
                                         * std_xx**2 \
                                     - moments_norm_vs_time[:,:,1]**3) / std_xx**3
    #    moments_norm_vs_time = np.zeros_like(moments_vs_time_grid_avg)
    #    moments_norm_vs_time[:,0] = moments_vs_time_grid_avg[:,0]
    #    moments_norm_vs_time[:,1] = moments_vs_time_grid_avg[:,1] \
    #                                / moments_vs_time_grid_avg[:,0]
    #    moments_norm_vs_time[:,2] = moments_vs_time_grid_avg[:,2] \
    #                                / moments_vs_time_grid_avg[:,0] \
    #                                - moments_norm_vs_time[:,1]**2
    #    std_xx = np.sqrt( moments_norm_vs_time[:,2] )
    #    moments_norm_vs_time[:,3] = (moments_vs_time_grid_avg[:,3] \
    #                                 / moments_vs_time_grid_avg[:,0] \
    #                                 - 3 * moments_norm_vs_time[:,1] \
    #                                     * std_xx**2 \
    #                                 - moments_norm_vs_time[:,1]**3) / std_xx**3
    
        moments_norm_vs_time_avg = np.average(moments_norm_vs_time, axis=0)
        
        moments_norm_vs_time_std =\
            np.std(moments_norm_vs_time, axis=0, ddof=1) / np.sqrt(Ns)
        print(var_n, moments_norm_vs_time_avg.shape)
                            
        mom_n = 0
        for row_n in range(no_rows):
            for col_n in range(no_cols):
                ax = axes[row_n, col_n]
                print(mom_n, row_n, col_n)
    #            print(moments_vs_time_grid_avg[:,mom_n])
    #            ax.plot(save_times_out/60-120, moments_vs_time_grid_avg[:,mom_n])
    #            ax.errorbar(save_times_out/60-120,
    #                        moments_vs_time_grid_avg[:,mom_n],
    #                        moments_vs_time_grid_std[:,mom_n])
                ax.errorbar(save_times_out[time_idx_conv]/60-t_spin_up//60,
                            moments_norm_vs_time_avg[time_idx_conv,mom_n],
                            moments_norm_vs_time_std[time_idx_conv,mom_n],fmt="x")
    #            ax.plot(save_times_out[time_idx_conv]/60-120,
    #                    moments_norm_vs_time_avg[time_idx_conv,mom_n])
                
                ax.set_xticks(np.linspace(0,60,7))
                ax.set_xlim((0,60))
    #            ax.set_yscale("log")
                ax.grid()
                mom_n += 1
        
    #    pad_ax_h = 0.1
    #    pad_ax_v = 0.26
    #    fig.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
    #    fig.subplots_adjust(wspace=pad_ax_v)                    
        
    fig.savefig(figpath + figname_conv,
                    bbox_inches = 'tight',
                    pad_inches = 0.04,
                    dpi=600
                    )          
        
    plt.close("all")        


#%% PLOT Nsip CONVERGENCE 

#idx_t_conv = ()

#figsize_conv = cm2inch(12,8)
figsize_conv = cm2inch(18,6)

#figname_conv = ""

time_idx_conv = np.arange(0,7)

ax_titles_conv =\
    [
     r"avg. $|\lambda_\mathrm{0} - \lambda_\mathrm{ref}| / \lambda_\mathrm{ref}$ (\%)",
     r"avg. $|\lambda_\mathrm{1} - \lambda_\mathrm{ref}| / \lambda_\mathrm{ref}$ (\%)",
     r"avg. $|\lambda_\mathrm{2} - \lambda_\mathrm{ref}| / \lambda_\mathrm{ref}$",
     r"avg. $|\lambda_\mathrm{3} - \lambda_\mathrm{ref}| / \lambda_\mathrm{ref}$",
    ]
ax_titles_conv_err =\
    [
     r"avg. $\mathrm{SD}(\lambda_\mathrm{0}) / \lambda_\mathrm{ref}$ (\%)",
     r"avg. $\mathrm{SD}(\lambda_\mathrm{1}) / \lambda_\mathrm{ref}$ (\%)",
     r"avg. $\mathrm{SD}(\lambda_\mathrm{2}) / \lambda_\mathrm{ref}$",
     r"avg. $\mathrm{SD}(\lambda_\mathrm{3}) / \lambda_\mathrm{ref}$"
    ]
#    [
#     r"grid avg. $|\lambda_\mathrm{0} - \lambda_\mathrm{ref}| / \lambda_\mathrm{ref}$ (\%)",
#     r"grid avg. $|\lambda_\mathrm{1} - \lambda_\mathrm{ref}| / \lambda_\mathrm{ref}$ (\%)",
#     r"grid avg. $|\lambda_\mathrm{2} - \lambda_\mathrm{ref}| / \lambda_\mathrm{ref}$",
#     r"grid avg. $|\lambda_\mathrm{3} - \lambda_\mathrm{ref}| / \lambda_\mathrm{ref}$",
#    ]
#ax_titles_conv =\
#    [
#     r"grid avg. $|\lambda_\mathrm{avg,0} - \lambda_\mathrm{avg,ref}| / \lambda_\mathrm{avg,ref}$",
#     r"$|\lambda_\mathrm{avg,1} - \lambda_\mathrm{avg,ref}| / \lambda_\mathrm{avg,ref}$",
#     r"$|\lambda_\mathrm{avg,2} - \lambda_\mathrm{avg,ref}| / \lambda_\mathrm{avg,ref}$",
#     r"$|\lambda_\mathrm{avg,3} - \lambda_\mathrm{avg,ref}| / \lambda_\mathrm{avg,ref}$",
#    ]

if act_plot_SIP_convergence:

    var_n = no_variations-1
    
    grid_path_ = grid_paths[var_n]
    data_path_ = data_paths[var_n]
    
    grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
                            grid_path_ + f"arr_file1_{int(t_grid)}.npy",
                            grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
    
    g_seed = seed_SIP_gen_var[var_n]
    s_seed = seed_sim_var[var_n]
    Ns = no_seeds_var[var_n]
    
    save_times_out = np.load(data_path_ + "moments/"
                   + f"save_times_out_avg_Ns_{Ns}_"
                   + f"sg_{g_seed}_ss_{s_seed}.npy")    
    
    moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
                   + f"moments_vs_time_all_seeds_Ns_{Ns}_"
                   + f"sg_{g_seed}_ss_{s_seed}.npy")
    
    print(var_n, moments_vs_time_all_seeds.shape)

#    moments_vs_time_grid_avg = np.average(moments_vs_time_all_seeds, axis=(3,4))
    
    
    moments_vs_time_seed_avg_ref = np.average(moments_vs_time_all_seeds, axis=0)
    moments_vs_time_seed_std_ref = np.std(moments_vs_time_all_seeds, axis=0) \
                                / np.sqrt(Ns)
    
    print(var_n, moments_vs_time_seed_avg_ref.shape)
    
    moments_devs = []
    
    no_rows = 1
    no_cols = 4
#    no_rows = 2
#    no_cols = 2
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                             figsize=figsize_conv, sharex=True)
    
    for var_n in range(no_variations-1):
        grid_path_ = grid_paths[var_n]
        data_path_ = data_paths[var_n]
        
        grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
                                grid_path_ + f"arr_file1_{int(t_grid)}.npy",
                                grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
        
        g_seed = seed_SIP_gen_var[var_n]
        s_seed = seed_sim_var[var_n]
        Ns = no_seeds_var[var_n]
        
        save_times_out = np.load(data_path_ + "moments/"
                       + f"save_times_out_avg_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")    
        
        moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
                       + f"moments_vs_time_all_seeds_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")
        
        print(var_n, moments_vs_time_all_seeds.shape)
    
#        moments_vs_time_grid_avg = np.average(moments_vs_time_all_seeds, axis=(3,4))
        
        # get rel dev and error in EACH CELL!!!
        moments_vs_time_seed_avg =\
            np.average(moments_vs_time_all_seeds, axis=0) \
        
        moments_vs_time_seed_std = np.std(moments_vs_time_all_seeds, axis=0) \
                                    / np.sqrt(Ns)
        
        moments_vs_time_rel_dev_avg =\
            np.abs(moments_vs_time_seed_avg - moments_vs_time_seed_avg_ref)\
            / moments_vs_time_seed_avg_ref
        
        moments_vs_time_rel_dev_std =\
            np.sqrt(moments_vs_time_seed_std**2
                    + moments_vs_time_seed_std_ref**2)\
            / moments_vs_time_seed_avg_ref
        
        print(var_n, moments_vs_time_rel_dev_avg.shape)
        
        
        for mom_n in range(4):
            print(var_n)
            print(mom_n, np.amax(moments_vs_time_rel_dev_avg[:,mom_n], axis=(1,2)))
        
        # now average over all cells:
        moments_vs_time_rel_dev_avg =\
            np.average(moments_vs_time_rel_dev_avg, axis=(2,3))
        moments_vs_time_rel_dev_std =\
            np.average(moments_vs_time_rel_dev_std, axis=(2,3))
        
        moments_devs.append(moments_vs_time_rel_dev_avg)
        
        print(var_n, moments_vs_time_rel_dev_avg.shape)
        
        scale_ = 100
        
        mom_n = 0
#        for row_n in range(no_rows):
        for col_n in range(no_cols):
#                ax = axes[row_n, col_n]
            ax = axes[col_n]
            print(mom_n, col_n)
            if mom_n in [0,1]:
                scale_ = 100
            else:
                scale_ = 1
#                if row_n == 0:
#                    scale_ = 100
#                else:
#                    scale_ = 1
                
            ax.errorbar(save_times_out[time_idx_conv]/60-t_spin_up//60,
                        scale_*moments_vs_time_rel_dev_avg[time_idx_conv,mom_n],
                        scale_*moments_vs_time_rel_dev_std[time_idx_conv,mom_n],
                        fmt = fmt_list[var_n], ms = MS_mom, mew=MEW_mom,
                        fillstyle="none", elinewidth = ELW_mom,
                        capsize=capsize_mom,
                        label=data_labels_mom[var_n]                            
                        )
#            ax.errorbar(save_times_out[time_idx_conv]/60-120,
#                        moments_vs_time_rel_dev_avg[time_idx_conv,mom_n],
#                        moments_vs_time_rel_dev_std[time_idx_conv,mom_n],
#                        fmt = fmt_list[var_n], ms = MS_mom, mew=MEW_mom,
#                        fillstyle="none", elinewidth = ELW_mom,
#                        capsize=capsize_mom,
#                        label=data_labels_mom[var_n]                            
#                        )
            mom_n += 1
    mom_n = 0                
#    for row_n in range(no_rows):
    for col_n in range(no_cols):
#            ax = axes[row_n, col_n]
        ax = axes[col_n]
        ax.set_xticks(np.linspace(0,60,7))
#            ax.set_xlim((-1,61))
        ax.set_xlim((-3,63))
#            ax.set_yscale("log")
        ax.grid(axis='y')
        if mom_n == 0:
            ax.legend()
#        if row_n == 0:
        ax.set_xlabel("Time (min)")
#            if row_n == 1:
#                ax.set_xlabel("Time (min)")
        ax.set_title(ax_titles_conv[mom_n])
        mom_n += 1
    axes[2].set_yscale("log")
    axes[3].set_yscale("log")
    pad_ax_h = 0.1
    pad_ax_v = 0.33
    fig.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
    fig.subplots_adjust(wspace=pad_ax_v)                    
    
    fig.savefig(figpath + figname_conv,
                bbox_inches = 'tight',
                pad_inches = 0.04,
                dpi=600
                ) 
    
    # plot improvement factors, when going from 32 to 64 SIPs
    fig3, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                             figsize=figsize_conv, sharex=True)    
    rel_dev_rel = moments_devs[0]/moments_devs[1]
    
    print(" np.average(rel_dev_rel), axis = 0 ")
    print( np.average(rel_dev_rel[0:7],axis = 0) )
    
    mom_n = 0
    for col_n in range(no_cols):
#            ax = axes[row_n, col_n]
        ax = axes[col_n]
        ax.plot(save_times_out[time_idx_conv]/60-t_spin_up//60, rel_dev_rel[time_idx_conv,mom_n])
        ax.set_xticks(np.linspace(0,60,7))
#            ax.set_xlim((-1,61))
        ax.set_xlim((-3,63))
#            ax.set_yscale("log")
        ax.grid(axis='y')
#        if mom_n == 0:
#            ax.legend()
#        if row_n == 0:
        ax.set_xlabel("Time (min)")
#            if row_n == 1:
#                ax.set_xlabel("Time (min)")
        ax.set_title(ax_titles_conv[mom_n])
        mom_n += 1
#    axes[2].set_yscale("log")
#    axes[3].set_yscale("log")
    pad_ax_h = 0.1
    pad_ax_v = 0.33
    fig3.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
    fig3.subplots_adjust(wspace=pad_ax_v)                    
    
    fig3.savefig(figpath + "REL_DEV_REL_" + figname_conv,
                bbox_inches = 'tight',
                pad_inches = 0.04,
                dpi=600
                )     
    
    ########################################################################
    ########################################################################
    ### plot rel errors in second plot
    fig2, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                             figsize=figsize_conv, sharex=True)
    
    for var_n in range(no_variations):
        grid_path_ = grid_paths[var_n]
        data_path_ = data_paths[var_n]
        
        grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
                                grid_path_ + f"arr_file1_{int(t_grid)}.npy",
                                grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
        
        g_seed = seed_SIP_gen_var[var_n]
        s_seed = seed_sim_var[var_n]
        Ns = no_seeds_var[var_n]
        
        save_times_out = np.load(data_path_ + "moments/"
                       + f"save_times_out_avg_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")    
        
        moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
                       + f"moments_vs_time_all_seeds_Ns_{Ns}_"
                       + f"sg_{g_seed}_ss_{s_seed}.npy")
        
        print(var_n, moments_vs_time_all_seeds.shape)
    
#        moments_vs_time_grid_avg = np.average(moments_vs_time_all_seeds, axis=(3,4))
        
        # get rel dev and error in EACH CELL!!!
        moments_vs_time_seed_avg =\
            np.average(moments_vs_time_all_seeds, axis=0) \
        
        moments_vs_time_seed_std = np.std(moments_vs_time_all_seeds, axis=0) \
                                    / np.sqrt(Ns)
        
        moments_vs_time_rel_dev_avg =\
            np.abs(moments_vs_time_seed_avg - moments_vs_time_seed_avg_ref)\
            / moments_vs_time_seed_avg_ref
        
        moments_vs_time_rel_dev_std =\
            np.sqrt(moments_vs_time_seed_std**2
                    + moments_vs_time_seed_std_ref**2)\
            / moments_vs_time_seed_avg_ref
        
        print(var_n, moments_vs_time_rel_dev_avg.shape)

        for mom_n in range(4):
            print(var_n)
            print(mom_n, np.amax(moments_vs_time_rel_dev_avg[:,mom_n], axis=(1,2)))
        
        # now average over all cells:
        moments_vs_time_rel_dev_avg =\
            np.average(moments_vs_time_rel_dev_avg, axis=(2,3))
        moments_vs_time_rel_dev_std =\
            np.average(moments_vs_time_rel_dev_std, axis=(2,3))
        
        print(var_n, moments_vs_time_rel_dev_avg.shape)
        
        scale_ = 100
        
        mom_n = 0
#        for row_n in range(no_rows):
        for col_n in range(no_cols):
#                ax = axes[row_n, col_n]
            ax = axes[col_n]
            print(mom_n, col_n)
            if mom_n in [0,1]:
                scale_ = 100
            else:
                scale_ = 1
#                if row_n == 0:
#                    scale_ = 100
#                else:
#                    scale_ = 1
                
            ax.plot(save_times_out[time_idx_conv]/60-t_spin_up//60,
                        scale_*moments_vs_time_rel_dev_std[time_idx_conv,mom_n],
                        fmt_list[var_n], ms = MS_mom, mew=MEW_mom,
                        fillstyle="none",
#                        elinewidth = ELW_mom,
#                        capsize=capsize_mom,
                        label=data_labels_mom[var_n]                            
                        )
#            ax.errorbar(save_times_out[time_idx_conv]/60-120,
#                        moments_vs_time_rel_dev_avg[time_idx_conv,mom_n],
#                        moments_vs_time_rel_dev_std[time_idx_conv,mom_n],
#                        fmt = fmt_list[var_n], ms = MS_mom, mew=MEW_mom,
#                        fillstyle="none", elinewidth = ELW_mom,
#                        capsize=capsize_mom,
#                        label=data_labels_mom[var_n]                            
#                        )
            mom_n += 1
    mom_n = 0                
#    for row_n in range(no_rows):
    for col_n in range(no_cols):
#            ax = axes[row_n, col_n]
        ax = axes[col_n]
        ax.set_xticks(np.linspace(0,60,7))
#            ax.set_xlim((-1,61))
        ax.set_xlim((-3,63))
#            ax.set_yscale("log")
        ax.grid(axis='y')
        if mom_n == 0:
            ax.legend()
#        if row_n == 0:
        ax.set_xlabel("Time (min)")
#            if row_n == 1:
#                ax.set_xlabel("Time (min)")
        ax.set_title(ax_titles_conv_err[mom_n])
        mom_n += 1
    axes[2].set_yscale("log")
    axes[3].set_yscale("log")
    pad_ax_h = 0.1
    pad_ax_v = 0.33
    fig2.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
    fig2.subplots_adjust(wspace=pad_ax_v)                    
    
    fig2.savefig(figpath + figname_conv_err,
                bbox_inches = 'tight',
                pad_inches = 0.04,
                dpi=600
                )          
        
    plt.close("all")        

#%% SAVE Nsip CONVERGENCE 

#idx_t_conv = ()

#figsize_conv = cm2inch(15,15)
#
##figname_conv = ""
#
#time_idx_conv = np.arange(0,7)
#
#ax_titles_conv =\
#    [
#     r"$|\lambda_\mathrm{avg,0} - \lambda_\mathrm{avg,ref}| / \lambda_\mathrm{avg,ref}$",
#     r"$|\lambda_\mathrm{avg,1} - \lambda_\mathrm{avg,ref}| / \lambda_\mathrm{avg,ref}$",
#     r"$|\lambda_\mathrm{avg,2} - \lambda_\mathrm{avg,ref}| / \lambda_\mathrm{avg,ref}$",
#     r"$|\lambda_\mathrm{avg,3} - \lambda_\mathrm{avg,ref}| / \lambda_\mathrm{avg,ref}$",
#    ]
#
#if act_plot_SIP_convergence:
#
#    var_n = no_variations-1
#    
#    grid_path_ = grid_paths[var_n]
#    data_path_ = data_paths[var_n]
#    
#    grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
#                            grid_path_ + f"arr_file1_{int(t_grid)}.npy",
#                            grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
#    
#    g_seed = seed_SIP_gen_var[var_n]
#    s_seed = seed_sim_var[var_n]
#    Ns = no_seeds_var[var_n]
#    
#    save_times_out = np.load(data_path_ + "moments/"
#                   + f"save_times_out_avg_Ns_{Ns}_"
#                   + f"sg_{g_seed}_ss_{s_seed}.npy")    
#    
#    moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
#                   + f"moments_vs_time_all_seeds_Ns_{Ns}_"
#                   + f"sg_{g_seed}_ss_{s_seed}.npy")
#    
#    print(var_n, moments_vs_time_all_seeds.shape)
#
#    moments_vs_time_grid_avg = np.average(moments_vs_time_all_seeds, axis=(3,4))
#    
#    
#    moments_vs_time_seed_avg_ref = np.average(moments_vs_time_grid_avg, axis=0)
#    moments_vs_time_seed_std_ref = np.std(moments_vs_time_grid_avg, axis=0) \
#                                / np.sqrt(Ns)
#    
#    print(var_n, moments_vs_time_seed_avg_ref.shape)
#    
#    no_rows = 2
#    no_cols = 2
#    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize_conv, sharex=True)
#    
#    for var_n in range(no_variations-1):
#        grid_path_ = grid_paths[var_n]
#        data_path_ = data_paths[var_n]
#        
#        grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
#                                grid_path_ + f"arr_file1_{int(t_grid)}.npy",
#                                grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
#        
#        g_seed = seed_SIP_gen_var[var_n]
#        s_seed = seed_sim_var[var_n]
#        Ns = no_seeds_var[var_n]
#        
#        save_times_out = np.load(data_path_ + "moments/"
#                       + f"save_times_out_avg_Ns_{Ns}_"
#                       + f"sg_{g_seed}_ss_{s_seed}.npy")    
#        
#        moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
#                       + f"moments_vs_time_all_seeds_Ns_{Ns}_"
#                       + f"sg_{g_seed}_ss_{s_seed}.npy")
#        
#        print(var_n, moments_vs_time_all_seeds.shape)
#    
#        moments_vs_time_grid_avg = np.average(moments_vs_time_all_seeds, axis=(3,4))
#        
#        
#        moments_vs_time_seed_avg_rel =\
#            np.average(moments_vs_time_grid_avg, axis=0) \
#        
#        moments_vs_time_seed_avg_rel =\
#            np.abs(moments_vs_time_seed_avg_rel - moments_vs_time_seed_avg_ref)\
#            / moments_vs_time_seed_avg_ref
#        
#        moments_vs_time_seed_std_rel = np.std(moments_vs_time_grid_avg, axis=0) \
#                                    / np.sqrt(Ns)
#        
#        moments_vs_time_seed_std_rel = moments_vs_time_seed_std_rel \
#                                        / moments_vs_time_seed_avg_ref
#        
#        print(var_n, moments_vs_time_seed_avg_rel.shape)
#        
#        mom_n = 0
#        for row_n in range(no_rows):
#            for col_n in range(no_cols):
#                ax = axes[row_n, col_n]
#                print(mom_n, row_n, col_n)
#                ax.errorbar(save_times_out[time_idx_conv]/60-120,
#                            100*moments_vs_time_seed_avg_rel[time_idx_conv,mom_n],
#                            100*moments_vs_time_seed_std_rel[time_idx_conv,mom_n],
#                            fmt = fmt_list[var_n], ms = MS_mom, mew=MEW_mom,
#                            fillstyle="none", elinewidth = ELW_mom,
#                            capsize=capsize_mom,
#                            label=data_labels_mom[var_n]                            
#                            )
#                mom_n += 1
#    mom_n = 0                
#    for row_n in range(no_rows):
#        for col_n in range(no_cols):
#            ax = axes[row_n, col_n]
#            ax.set_xticks(np.linspace(0,60,7))
#            ax.set_xlim((-1,61))
##            ax.set_yscale("log")
#            ax.grid()
#            ax.legend()
#            if row_n == 1:
#                ax.set_xlabel("Time (min)")
#            ax.set_title(ax_titles_conv[mom_n])
#            mom_n += 1
##    pad_ax_h = 0.1
##    pad_ax_v = 0.26
##    fig.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
##    fig.subplots_adjust(wspace=pad_ax_v)                    
#    
#    fig.savefig(figpath + figname_conv,
#                bbox_inches = 'tight',
#                pad_inches = 0.04,
#                dpi=600
#                )          
#        
#    plt.close("all")        
#
#
