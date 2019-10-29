#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:28:30 2019

@author: jdesk
"""

#%% MODULE IMPORTS
import os
import numpy as np
# import os
# from datetime import datetime
# import timeit

import constants as c
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\
                          

#                     plot_particle_size_spectra
# from integration import compute_dt_max_from_CFL
#from grid import compute_no_grid_cells_from_step_sizes

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
mpl.rcParams.update(plt.rcParamsDefault)
#mpl.use("pdf")
mpl.use("pgf")

import matplotlib.ticker as mticker


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

from analysis import sample_masses, sample_radii
from analysis import sample_masses_per_m_dry , sample_radii_per_m_dry
from analysis import plot_size_spectra_R_Arabas, generate_size_spectra_R_Arabas

from plotting_fcts_MA import plot_size_spectra_R_Arabas_MA, \
                             plot_particle_trajectories_MA

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
my_OS = "Linux_desk"
#my_OS = "Mac"
#my_OS = "TROPOS_server"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
#    fig_path = home_path \
#               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "TROPOS_server"):
    simdata_path = "/vols/fs1/work/bohrer/sim_data_cloudMP/"


#%% GRID PARAMETERS

no_cells = (75, 75)
#no_cells = (3, 3)

dx = 20.
dy = 1.
dz = 20.
dV = dx*dy*dz

#%% PARTICLE PARAMETERS

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = "AS"

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([10, 10])
#no_spcm = np.array([12, 12])
#no_spcm = np.array([16, 24])
no_spcm = np.array([20, 30])
#no_spcm = np.array([26, 38])

# seed of the SIP generation -> needed for the right grid folder
# 3711, 3713, 3715, 3717
# 3719, 3721, 3723, 3725

#seed_SIP_gen = 4811
seed_SIP_gen = 3711

#seed_SIP_gen_list = [3711, 3713]
#seed_SIP_gen_list = [3711, 3713, 3715, 3717]
#no_sims = 50
no_sims = 4
seed_SIP_gen_list = np.arange(seed_SIP_gen, seed_SIP_gen + no_sims * 2, 2)

# for collisons
seed_sim = 4711
#seed_sim = 6811

#seed_sim_list = [4711, 4711]
#seed_sim_list = [4711, 4711, 4711, 4711]
#seed_sim_list = [6711, 6713, 6715, 6717]
#seed_sim_list = [4711, 4713, 4715, 4717]
#no_sims = 50
seed_sim_list = np.arange(seed_sim, seed_sim + no_sims * 2, 2)

#simulation_mode = "spin_up"
#simulation_mode = "wo_collision"
simulation_mode = "with_collision"

dt_col = 0.5
#dt_col = 1.0

if simulation_mode == "spin_up":
    spin_up_finished = False
else:
    spin_up_finished = True

# path = simdata_path + folder_load_base
#t_grid = 0
#t_grid = 7200
t_grid = 14400

#t_start = 0
t_start = 7200

#t_end = 7200
t_end = 14400

#t_grid = 10800
#t_end = 60
#t_end = 3600
#t_end = 10800

#%% PLOTTING PARAMETERS 

#args_plot = [1,1,1,1,1,1]
#args_plot = [0,1,0,0,0]
#args_plot = [0,0,1,0,0]
#args_plot = [0,0,0,1,0]
#args_plot = [0,0,0,0,1]

#args_plot = [0,0,0,1,0,0,0,0,0,0]
#args_plot = [0,0,1,0,0,0,0,0,0,0]
args_plot = [0,0,0,0,0,0,0,0,0,1]

#args_plot = [0,0,0,0,0,0,0,0,0,1]
#args_plot = [0,0,0,0,0,0,1,1]

act_plot_scalar_fields_once = args_plot[0]
act_plot_spectra = args_plot[1]
act_plot_particle_trajectories = args_plot[2]
act_plot_particle_positions = args_plot[3]
act_plot_scalar_field_frames = args_plot[4]
act_plot_scalar_field_frames_ext = args_plot[5]
#act_gen_grid_frames_avg = args_plot[6]
act_plot_grid_frames_avg = args_plot[6]
#act_gen_spectra_avg_Arabas = args_plot[7]
act_plot_spectra_avg_Arabas = args_plot[7]
act_get_grid_data = args_plot[8]
act_plot_life_cycle = args_plot[9]

### times for grid data
grid_times = [0,7200,14400]

### target cells for spectra analysis
#    i_tg = [20,40,60]
#    j_tg = [20,40,50,60,65,70]

#    i_tg = [20,40]
i_tg = [16,58,66]
#    j_tg = [20]
#    j_tg = [20,40]
#    j_tg = [40,60]
j_tg = [27, 44, 46, 51, 72][::-1]

i_list, j_list = np.meshgrid(i_tg,j_tg, indexing = "xy")
target_cell_list = np.array([i_list.flatten(), j_list.flatten()])

#    print(target_cell_list)
# region range of spectra analysis
# please enter uneven numbers: no_cells_x = 5 =>  [x][x][tg cell][x][x]
no_cells_x = 3
no_cells_z = 3

figpath = home_path + "/Masterthesis/Figures/06TestCase/"

### TRACERS TRAJECTORIES
figsize_pt_trajs = cm2inch(4.9,7)
#figsize_pt_trajs = cm2inch(6.6,7)
#figsize_pt_trajs = cm2inch(20,20)
#figname_pt_trajs = "particle_trajectories_no_col_no_grav_"
figname_pt_trajs = "particle_trajectories_no_col_grav_"
#figname_pt_trajs = "particle_trajectories_col_grav_"

#pt_trajs_ncol_legend =
#MS_pt_trajs = 2
MS_pt_trajs = 1.
#MS_pt_trajs = 0.9
LW_pt_trajs = 0.6

dd = (24,32)
#aa = (6,8,24,32)
bb = (13,37)
#bb = (1,13,33,37)
cc = (38,)
aa = (27,)
#dd = (15,19,27,39)

#selection2 = list(aa + bb + cc + dd)

# this is the selection for 20 30, 3711 4711, AS w_col
#selection2 = (38, 32, 24, 37,13,27)

# this is the selection for 20 30, 3711 4711, AS wo col, wo grav
#selection2 = np.arange(0,40,5)
#selection2 = (1,6,11,16,21,31,19,5)

# this is the selection for 26 38 AS 4811(?) 6811 no col, with grav
# with 80 tracers total
selection2 = np.arange(1,80,10)

if figname_pt_trajs == "particle_trajectories_no_col_no_grav_":
    label_y_traj = True
else:
    label_y_traj = False


#selection2 = list(aa + bb + cc + dd)
#selection2.sort()

if act_plot_life_cycle:
    figsize = cm2inch(6.8,7.5)        
    figsize_spectra = cm2inch(16.8,8)
    figsize_trace_traj = cm2inch(6.6,7)
    figname = figpath + "dropletLifeCycle.pdf"   

#%% LOAD GRID AND PARTICLES AT TIME t_grid

grid_folder =\
    f"{solute_type}" \
    + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
    + f"{seed_SIP_gen}/"

if simulation_mode == "spin_up":
    save_folder = "spin_up_wo_col_wo_grav/"
elif simulation_mode == "wo_collision":
    if spin_up_finished:
        save_folder = "w_spin_up_wo_col/"
    else:
        save_folder = "wo_spin_up_wo_col/"
elif simulation_mode == "with_collision":
    if spin_up_finished:
        save_folder = f"w_spin_up_w_col/{seed_sim}/"
    else:
        save_folder = f"wo_spin_up_w_col/{seed_sim}/"

# load grid and particles full from grid_path at time t
if int(t_grid) == 0:        
    grid_path = simdata_path + grid_folder
else:    
    grid_path = simdata_path + grid_folder + save_folder

load_path = simdata_path + grid_folder + save_folder

#reload = True
#
#if reload:
grid, pos, cells, vel, m_w, m_s, xi, active_ids  = \
    load_grid_and_particles_full(t_grid, grid_path)

if solute_type == "AS":
    compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    mass_density_dry = c.mass_density_AS_dry
elif solute_type == "NaCl":
    compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    mass_density_dry = c.mass_density_NaCl_dry

R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass_vec(m_s, mass_density_dry)


#%% FCT DEF SPECTRA AND LIFE CYCLE


#%% GRID THERMODYNAMIC FIELDS AT t_grid
if act_plot_scalar_fields_once:
    fig_path = load_path
    grid.plot_thermodynamic_scalar_fields(t=t_grid, fig_path = fig_path)

#%% PARTICLE SPECTRA

plt.ioff()

if act_plot_spectra:
    from analysis import plot_particle_size_spectra_tg_list
    t = t_grid
    fig_path = load_path
    
    target_cell = (0,0)
    no_cells_x = 9
    no_cells_z = 1
    
    i_tg = [20,40,60]
    j_tg = [20,40,45,50,55,60,65,70,74]
#    i_tg = [20]
#    j_tg = [20,40]
    i_list, j_list = np.meshgrid(i_tg,j_tg, indexing = "xy")
    target_cell_list = [i_list.flatten(), j_list.flatten()]
    
    no_rows = len(j_tg)
    no_cols = len(i_tg)
#    no_rows = 3
#    no_cols = 2
    filename_ = load_path + "grid_scalar_fields_t_" + str(int(t)) + ".npy"
    fields_ = np.load(filename_)
    grid_temperature = fields_[3]
    
    [m_w, m_s] = np.load(load_path + f"particle_scalar_data_all_{int(t)}.npy")
    pos = np.load(load_path + f"particle_vector_data_all_{int(t)}.npy")[0]
    xi = np.load(load_path + f"particle_xi_data_all_{int(t)}.npy")
    
    # ACTIVATE LATER
    cells = np.load(load_path + f"particle_cells_data_all_{int(t)}.npy")
    # FOR NOW..., DEACTIVATE LATER
#    cells = np.array( [np.floor(pos[0]/dx) , np.floor(pos[1]/dz)] ).astype(int)
#    np.save(load_path + f"particle_cells_data_all_{int(t)}.npy", cells)
    
    plot_particle_size_spectra_tg_list(
        t, m_w, m_s, xi, cells,
        dV,
        dz,
        grid_temperature,
        solute_type,
        target_cell_list, no_cells_x, no_cells_z,
        no_rows, no_cols,
        TTFS=12, LFS=10, TKFS=10,
        fig_path = fig_path)
#    plot_particle_size_spectra_tg_list(t, m_w, m_s, xi, cells, grid, solute_type,
#                          target_cell_list, no_cells_x, no_cells_z,
#                          no_rows, no_cols,
#                          TTFS=12, LFS=10, TKFS=10,
#                          fig_path = fig_path)
    
#    for i in range(0,75,20):
##    for i in range(0,75,20):
#        for j in range(49,75,5):
##        for j in range(60,75,5):
#    #for i in range(0,10,4):
#    #    for j in range(0,10,4):
#    #for i in range(0,3):
#    #    for j in range(0,3):
#            target_cell = (i,j)
#            plot_particle_size_spectra(m_w, m_s, xi, cells, grid, solute_type,
#                                       target_cell, no_cells_x, no_cells_z,
#                                       no_rows=1, no_cols=1,
#                                       TTFS=12, LFS=10, TKFS=10,
#                                       fig_path = fig_path)

#%% PARTICLE TRAJECTORIES
#act_plot_particle_trajectories = True
if act_plot_particle_trajectories:
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    pt_dumps_per_grid_frame = frame_every // dump_every
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    # grid_save_times =\
    #     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    print("grid_save_times")
    print(grid_save_times)
    
    from file_handling import load_particle_data_from_blocks
    vecs, scals, xis = load_particle_data_from_blocks(load_path,
                                                      grid_save_times,
                                                      pt_dumps_per_grid_frame)
    print(vecs.shape)
    from analysis import plot_particle_trajectories
    fig_name = figpath + figname_pt_trajs \
               + f"t_{int(t_start)}_" \
               + f"{int(t_end)}.pdf"
#    selection = np.arange(3,40,4)

    print(selection2)
    
    plot_particle_trajectories_MA(vecs[::1,0], grid, selection2,
                                  MS=MS_pt_trajs, LW=LW_pt_trajs, arrow_every=5,
                               fig_name=fig_name, figsize=figsize_pt_trajs,
                               TTFS=TTFS, LFS=LFS, TKFS=TKFS,
                               ARROW_SCALE=18,ARROW_WIDTH=0.004,
                               t_start=t_start, t_end=t_end,
                               label_y=label_y_traj)
    plt.close("all")
    
#%% PARTICLE POSITIONS AND VELOCITIES

if act_plot_particle_positions:
    
    figsize = cm2inch(7,7)
    
    MS_pt = 0.05
    
    from file_handling import load_particle_data_all
    from file_handling import load_particle_data_all_old
    from analysis import plot_pos_vel_pt_with_time, plot_pos_vel_pt
    from plotting_fcts_MA import plot_pos_vel_pt_MA
    # plot only every x IDs
#    ID_every = 25
#    ID_every = 40
    ID_every = 50
#    ID_every = 100
    # plot a series of "frames" defined by grid_save_times OR just one frame at
    # the time of the current loaded grid and particles
#    plot_time_series = True
    plot_time_series = False
    plot_frame_every = 4
    
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    pt_dumps_per_grid_frame = frame_every // dump_every
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    # grid_save_times =\
    #     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    print("grid_save_times")
    print(grid_save_times)
    fig_name = figpath + "pt_pos_vel_INIT.pdf"
    load_path + "particle_positions_" \
               + f"t_{grid_save_times[0]}_" \
               + f"{grid_save_times[-1]}.png"
    if plot_time_series:
        vec_data, scal_data, xi_data  = \
            load_particle_data_all_old(load_path, grid_save_times)
#        vec_data, cells_data, scal_data, xi_data, active_ids_data = \
#            load_particle_data_all(load_path, grid_save_times)
        pos_data = vec_data[:,0]
        vel_data = vec_data[:,1]
        pos2 = pos_data[::plot_frame_every,:,::ID_every]
        vel2 = vel_data[::plot_frame_every,:,::ID_every]
        plot_pos_vel_pt_with_time(pos2, vel2, grid,
                                  grid_save_times[::plot_frame_every],
                            figsize=(8,8*len(pos2)), no_ticks = [6,6],
                            MS = 1.0, ARRSCALE=20, fig_name=fig_name)
    else:
        pos2 = pos[:,::ID_every]
        vel2 = vel[:,::ID_every]
        
        plot_pos_vel_pt_MA(pos2, vel2, grid,
                            figsize=figsize, no_ticks = [6,6],
                            MS = MS_pt, ARROWSCALE=28,
                            ARROWWIDTH=0.0018,
                            HEADW = 4, HEADL = 8,
                            fig_name=fig_name)
    
#%% PLOT GRID SCALAR FIELD FRAMES OVER TIME

if act_plot_scalar_field_frames:
    from analysis import plot_scalar_field_frames
    #path = simdata_path + load_folder
    plot_frame_every = 2
    # field_ind = np.arange(6)
    field_ind = np.array((0,1,2,5))
    
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    pt_dumps_per_grid_frame = frame_every // dump_every
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    fields = load_grid_scalar_fields(load_path, grid_save_times)
    print("fields.shape")
    print(fields.shape)
    
    time_ind = np.arange(0, len(grid_save_times), plot_frame_every)
    
    # grid_save_times =\
    #     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    print()
    print("plot scalar field frames with times:")
    print("grid_save_times")
    print(grid_save_times)
    
    print("grid_save_times indices and times chosen:")
    for idx_t in time_ind:
        print(idx_t, grid_save_times[idx_t])
    
    no_ticks=[6,6]
    
    fig_name = load_path \
               + f"scalar_fields_" \
               + f"t_{grid_save_times[0]}_" \
               + f"{grid_save_times[-1]}.png"
    # fig_name = None
    
    plot_scalar_field_frames(grid, fields, grid_save_times,
                             field_ind, time_ind, no_ticks, fig_name)

#%% PLOT GRID SCALAR FIELD FRAMES OVER TIME EXTENDED

if act_plot_scalar_field_frames_ext:
    from analysis import plot_scalar_field_frames_extend    
#    from file_handling import load_particle_data_all

    plot_frame_every = 6

    field_ind = np.array((2,5,0,1))
#    field_ind_ext = np.array((0,1,2))
    field_ind_ext = np.array((0,1,2,3,4,5))
    
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    pt_dumps_per_grid_frame = frame_every // dump_every
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
#    time_ind = np.arange(0, len(grid_save_times), plot_frame_every)
    
    time_ind = np.array((0,2,4,6,8))
    
    
    
    fields = load_grid_scalar_fields(load_path, grid_save_times)
    print("fields.shape")
    print(fields.shape)
    
    vec_data, cells_with_time, scal_data, xi_with_time, active_ids_with_time =\
        load_particle_data_all(load_path, grid_save_times)
    
    m_w_with_time = scal_data[:,0]
    m_s_with_time = scal_data[:,1]
    
    print("m_w_with_time.shape")
    print(m_w_with_time.shape)
    print("m_s_with_time.shape")
    print(m_s_with_time.shape)
    
    # grid_save_times =\
    #     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    print()
    print("plot scalar field frames with times:")
    print("grid_save_times")
    print(grid_save_times)
    
    print("grid_save_times indices and times chosen:")
    for idx_t in time_ind:
        print(idx_t, grid_save_times[idx_t])
    
    no_ticks=[6,6]
    
    fig_name = load_path \
               + f"scalar_fields_ext_" \
               + f"t_{grid_save_times[0]}_" \
               + f"{grid_save_times[time_ind[-1]]}_fr_{len(time_ind)}.png"    
    plot_scalar_field_frames_extend(grid, fields,
                                    m_s_with_time, m_w_with_time,
                                    xi_with_time, cells_with_time,
                                    active_ids_with_time,
                                    solute_type,
                                    grid_save_times, field_ind, time_ind,
                                    field_ind_ext,
                                    no_ticks=no_ticks, fig_path=fig_name,
                                    TTFS = 12, LFS = 10, TKFS = 10,
                                    cbar_precision = 2)
    

#%% PLOT AVG GRID SCALAR FIELD FRAMES OVER TIME

if act_plot_grid_frames_avg:
#if act_gen_grid_frames_avg:
    from analysis import generate_field_frame_data_avg
    from analysis import plot_scalar_field_frames_extend_avg

    load_path_list = []    

    plot_frame_every = 6

    field_ind = np.array((2,5,0,1))
#    field_ind_ext = np.array((0,1,2))
    field_ind_deri = np.array((0,1,2,3,4,5,6,7,8))
    
#    time_ind = np.arange(0, len(grid_save_times), plot_frame_every)
    
    time_ind = np.array((0,2,4,6,8,10,12))
#    time_ind = np.array((0,3,6,9))
    
    show_target_cells = True
#    i_tg = [20,40,60]
#    j_tg = [20,40,50,60,65,70]
    
##    i_tg = [20,40]
#    i_tg = [16,58]
##    j_tg = [20]
##    j_tg = [20,40]
##    j_tg = [40,60]
#    j_tg = [27, 44, 46, 51, 73]
#    
#    i_list, j_list = np.meshgrid(i_tg,j_tg, indexing = "xy")
#    target_cell_list = np.array([i_list.flatten(), j_list.flatten()])
#    
##    print(target_cell_list)
#    
#    no_cells_x = 3
#    no_cells_z = 3    
    
    
    no_seeds = len(seed_SIP_gen_list)
    for seed_n in range(no_seeds):
        seed_SIP_gen_ = seed_SIP_gen_list[seed_n]
        seed_sim_ = seed_sim_list[seed_n]
        
        grid_folder_ =\
            f"{solute_type}" \
            + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
            + f"{seed_SIP_gen_}/"
        
        if simulation_mode == "spin_up":
            save_folder_ = "spin_up_wo_col_wo_grav/"
        elif simulation_mode == "wo_collision":
            if spin_up_finished:
                save_folder_ = "w_spin_up_wo_col/"
            else:
                save_folder_ = "wo_spin_up_wo_col/"
        elif simulation_mode == "with_collision":
            if spin_up_finished:
                save_folder_ = f"w_spin_up_w_col/{seed_sim_}/"
            else:
                save_folder_ = f"wo_spin_up_w_col/{seed_sim_}/"        
                load_path_list.append()
        
        load_path_list.append(simdata_path + grid_folder_ + save_folder_)
    print(load_path_list)    
    fields_with_time, save_times_out, field_names_out, units_out, \
           scales_out = generate_field_frame_data_avg(load_path_list,
                                                        field_ind, time_ind,
                                                        field_ind_deri,
                                                        grid.mass_dry_inv,
                                                        grid.no_cells,
                                                        solute_type)
#    output_folder = \
#        f"{solute_type}" \
#        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
#        + f"eval_data_avg_Ns_{no_seeds}_" \
#        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}/"
#    
#    if not os.path.exists(simdata_path + output_folder):
#        os.makedirs(simdata_path + output_folder)    
#    
#    np.save(simdata_path + output_folder
#            + "seed_SIP_gen_list",
#            seed_SIP_gen_list)
#    np.save(simdata_path + output_folder
#            + "seed_sim_list",
#            seed_sim_list)
#    
#    np.save(simdata_path + output_folder
#            + f"fields_vs_time_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            fields_with_time)
#    np.save(simdata_path + output_folder
#            + f"save_times_out_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            save_times_out)
#    np.save(simdata_path + output_folder
#            + f"field_names_out_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            field_names_out)
#    np.save(simdata_path + output_folder
#            + f"units_out_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            units_out)
#    np.save(simdata_path + output_folder
#            + f"scales_out_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            scales_out)
    
    
    grid_folder_ =\
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"plots/{simulation_mode}/dt_col_{dt_col}/"
    
    fig_path = simdata_path + grid_folder_
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path) 
         
    fig_name = fig_path \
               + f"scalar_fields_avg_" \
               + f"t_{save_times_out[0]}_" \
               + f"{save_times_out[-1]}_Nfr_{len(save_times_out)}_" \
               + f"Nfields_{len(field_names_out)}_" \
               + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}.png" 
               
    plot_scalar_field_frames_extend_avg(grid, fields_with_time,
                                        save_times_out,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path=fig_name,
                                        no_ticks=[6,6], 
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = show_target_cells,
                                        target_cell_list = target_cell_list,
                                        no_cells_x = no_cells_x,
                                        no_cells_z = no_cells_z)    
    plt.close("all")
    
#%% PLOT SPECTRA AVG 

if act_plot_spectra_avg_Arabas:
#    figsize_spectra = cm2inch
#    figsize_trace_traj
#if act_gen_spectra_avg_Arabas:
    figsize = cm2inch(6.8,7.5)        
    figsize_spectra = cm2inch(16.8,8)
    figsize_trace_traj = cm2inch(6.6,7)
    figpath = home_path + "/Masterthesis/Figures/05TestCase/"
#    figname = figpath + "dropletLifeCycle.pdf"      
    
    no_bins_R_p = 30
    no_bins_R_s = 30
    
    no_rows = len(j_tg)
    no_cols = len(i_tg)
    #no_rows = 2
    #no_cols = 1
    print(target_cell_list)
    #time_ind = np.array((0,2,4,6,8,10,12))
#    ind_time = np.zeros(len(target_cell_list[0]), dtype = np.int64)
    ind_time = 6 * np.ones(len(target_cell_list[0]), dtype = np.int64)
    
    load_path_list = []    
    no_seeds = len(seed_SIP_gen_list)
    for seed_n in range(no_seeds):
        seed_SIP_gen_ = seed_SIP_gen_list[seed_n]
        seed_sim_ = seed_sim_list[seed_n]
        
        grid_folder_ =\
            f"{solute_type}" \
            + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
            + f"{seed_SIP_gen_}/"
        
        if simulation_mode == "spin_up":
            save_folder_ = "spin_up_wo_col_wo_grav/"
        elif simulation_mode == "wo_collision":
            if spin_up_finished:
                save_folder_ = "w_spin_up_wo_col/"
            else:
                save_folder_ = "wo_spin_up_wo_col/"
        elif simulation_mode == "with_collision":
            if spin_up_finished:
                save_folder_ = f"w_spin_up_w_col/{seed_sim_}/"
            else:
                save_folder_ = f"wo_spin_up_w_col/{seed_sim_}/"        
                load_path_list.append()
        
        load_path_list.append(simdata_path + grid_folder_ + save_folder_)
    
    #print(load_path_list)
    
    f_R_p_list, f_R_s_list, bins_R_p_list, bins_R_s_list, save_times_out,\
    grid_r_l_list, R_min_list, R_max_list = \
        generate_size_spectra_R_Arabas(load_path_list,
                                       ind_time,
                                       grid.mass_dry_inv,
                                       grid.no_cells,
                                       solute_type,
                                       target_cell_list,
                                       no_cells_x, no_cells_z,
                                       no_bins_R_p, no_bins_R_s)  
    
#    output_folder = \
#        f"{solute_type}" \
#        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
#        + f"eval_data_avg_Ns_{no_seeds}_" \
#        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}/"
#    
#    if not os.path.exists(simdata_path + output_folder):
#        os.makedirs(simdata_path + output_folder)    
#    
#    np.save(simdata_path + output_folder
#            + "seed_SIP_gen_list",
#            seed_SIP_gen_list)
#    np.save(simdata_path + output_folder
#            + "seed_sim_list",
#            seed_sim_list)
#    
#    np.save(simdata_path + output_folder
#            + f"f_R_p_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            f_R_p_list)    
#    np.save(simdata_path + output_folder
#            + f"f_R_s_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            f_R_s_list)    
#    np.save(simdata_path + output_folder
#            + f"bins_R_p_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            bins_R_p_list)    
#    np.save(simdata_path + output_folder
#            + f"bins_R_s_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            bins_R_s_list)    
#    np.save(simdata_path + output_folder
#            + f"save_times_out_spectra_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            save_times_out)    
#    np.save(simdata_path + output_folder
#            + f"grid_r_l_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            grid_r_l_list)    
#    np.save(simdata_path + output_folder
#            + f"R_min_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            R_min_list)    
#    np.save(simdata_path + output_folder
#            + f"R_max_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            R_max_list)    
#    np.save(simdata_path + output_folder
#            + f"target_cell_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            target_cell_list)    
#    np.save(simdata_path + output_folder
#            + f"neighbor_cells_list_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            [no_cells_x, no_cells_z])    
    
    
    
#     for fig name
    if no_cells_x % 2 == 0: no_cells_x += 1
    if no_cells_z % 2 == 0: no_cells_z += 1  
    j_low = target_cell_list[1].min()
    j_high = target_cell_list[1].max()
    t_low = save_times_out.min()
    t_high = save_times_out.max()
    no_tg_cells = len(save_times_out)
    
    grid_folder_ =\
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"plots/{simulation_mode}/dt_col_{dt_col}/"
         
#    fig_path = simdata_path + grid_folder_
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path) 
    
    fig_path = figpath
    fig_name =\
        fig_path \
        + f"spectra_at_tg_cells_j_from_{j_low}_to_{j_high}_" \
        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    fig_path_tg_cells =\
        fig_path \
        + f"tg_cell_posi_j_from_{j_low}_to_{j_high}_" \
        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    fig_path_R_eff =\
        fig_path \
        + f"R_eff_{j_low}_to_{j_high}_" \
        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    
#    plot_size_spectra_R_Arabas_MA(f_R_p_list, f_R_s_list,
#                               bins_R_p_list, bins_R_s_list,
#                               grid_r_l_list,
#                               R_min_list, R_max_list,
#                               save_times_out,
#                               solute_type,
#                               grid,
#                               target_cell_list,
#                               no_cells_x, no_cells_z,
#                               no_bins_R_p, no_bins_R_s,
#                               no_rows, no_cols,
#                               TTFS=12, LFS=10, TKFS=10, LW = 2.0,
#                               fig_path = fig_name,
#                               show_target_cells = True,
#                               fig_path_tg_cells = fig_path_tg_cells   ,
#                               fig_path_R_eff = fig_path_R_eff
#                               )        
    plot_size_spectra_R_Arabas_MA(
            f_R_p_list, f_R_s_list,
            bins_R_p_list, bins_R_s_list,
            grid_r_l_list,
            R_min_list, R_max_list,
            save_times_out,
            solute_type,
            grid,
            target_cell_list,
            no_cells_x, no_cells_z,
            no_bins_R_p, no_bins_R_s,
            no_rows, no_cols,
            TTFS=10, LFS=10, TKFS=8, LW = 2.0, MS = 0.4,
            figsize_spectra = figsize_spectra,
            figsize_trace_traj = figsize_trace_traj,
            fig_path = fig_name,
            show_target_cells = True,
            fig_path_tg_cells = fig_path_tg_cells   ,
            fig_path_R_eff = fig_path_R_eff,
            trajectory = None )    
    plt.close("all")

#%% EXTRACT GRID DATA
    
if act_get_grid_data:
    import shutil
    output_path = \
        simdata_path \
        + f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}/"    
    grid_path_base = \
        simdata_path \
        + f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"{seed_SIP_gen_list[0]}/"
#        _ss_{seed_sim_list[0]}
    no_grid_times = len(grid_times)
    
    for gt in grid_times:
        if gt == 0:
            shutil.copy(grid_path_base + "grid_basics_0.txt", output_path)
            shutil.copy(grid_path_base + "arr_file1_0.npy", output_path)
            shutil.copy(grid_path_base + "arr_file2_0.npy", output_path)
        elif gt <= 7200.1:
            shutil.copy(grid_path_base + "spin_up_wo_col_wo_grav/"
                         + f"grid_basics_{int(gt)}.txt",
                         output_path)
            shutil.copy(grid_path_base + "spin_up_wo_col_wo_grav/"
                         + f"arr_file1_{int(gt)}.npy",
                         output_path)
            shutil.copy(grid_path_base + "spin_up_wo_col_wo_grav/"
                         + f"arr_file2_{int(gt)}.npy",
                         output_path)
        elif gt > 7200.1:
            shutil.copy(grid_path_base
                         + f"w_spin_up_w_col/{seed_sim_list[0]}/"
                         + f"grid_basics_{int(gt)}.txt",
                         output_path)
            shutil.copy(grid_path_base
                         + f"w_spin_up_w_col/{seed_sim_list[0]}/"
                         + f"arr_file1_{int(gt)}.npy",
                         output_path)
            shutil.copy(grid_path_base
                         + f"w_spin_up_w_col/{seed_sim_list[0]}/"
                         + f"arr_file2_{int(gt)}.npy",
                         output_path)
            

#%% TRACED PARTICLE ANALYSIS: DROPLET LIFE CYCLE

if act_plot_life_cycle:    
    
    print(load_path)
    
    trace_ids = np.load(load_path + "trace_ids.npy")
    print(trace_ids)
    
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    pt_dumps_per_grid_frame = frame_every // dump_every
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    # grid_save_times =\
    #     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    print("grid_save_times")
    print(grid_save_times)
    
    #traced_vectors[dump_N,0] = pos[:,trace_ids]
    #traced_vectors[dump_N,1] = vel[:,trace_ids]
    #traced_scalars[dump_N,0] = m_w[trace_ids]
    #traced_scalars[dump_N,1] = m_s[trace_ids]
    #traced_xi[dump_N] = xi[trace_ids]
    #traced_water[dump_N] = water_removed[0]
    
    from file_handling import load_particle_data_from_blocks
    vecs, scals, xis = load_particle_data_from_blocks(load_path,
                                                      grid_save_times,
                                                      pt_dumps_per_grid_frame)
    
#    trace_times = np.arange(0, 7201, 10)
    trace_times = np.arange(7200, 14401, 10)
    
    pos_trace = vecs[:,0]
    vel_trace = vecs[:,1]
    m_w_trace = scals[:,0]
    m_s_trace = scals[:,1]
    
    print(pos_trace.shape)
    print(m_w_trace.shape)
    
    act_traj_all = False
    if act_traj_all:
        fig, ax = plt.subplots(figsize=(8,8))
        
        for cnt, trace_id in enumerate(trace_ids):
            ax.plot( pos_trace[::2,0, cnt], pos_trace[::2,1, cnt], "o", markersize=1 )
        #    ax.annotate(f"({cnt} {trace_id})",
            ax.annotate(f"({cnt})",
                        (pos_trace[0,0, cnt], pos_trace[0,1, cnt]))
        
        fig.savefig(load_path + "positions_with_id_annotate.pdf")
    
    # possible ids: 39, 2, 25, 9
    
    # for 20_30: 3711, 4711 try trace_id_n1 = 36
    
    #trace_id_n1 = 9
    
    trace_id_n1 = 36

    save_folder_ =\
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"plots/{simulation_mode}/dt_col_{dt_col}/"\
        + f"gen_{seed_SIP_gen_list[0]}_sim_{seed_sim_list[0]}/"
    
    fig_path = simdata_path + save_folder_
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)     
#    trace_id_n1 = 25
    #trace_id_n1 = 39
    
    m_w1 = m_w_trace[:,trace_id_n1]
    m_s1 = m_s_trace[:,trace_id_n1]
    xi1 = xis[:,trace_id_n1]
    pos1 = pos_trace[:,:,trace_id_n1]
    vel1 = vel_trace[:,:,trace_id_n1]
    
    fields = load_grid_scalar_fields(load_path, grid_save_times)
    
    
    T_grid = fields[:, 3]
    
    # NOTE that for pos = [ pos_t0, pos_t1, .. ]
    # BUT cells1 = [ cells_x_vs_t, cells_z_vs_t ]
    cells1, rel_pos1 = grid.compute_cell_and_relative_location(pos1[:,0],
                                                               pos1[:,1])
    
    # the same as tg_cells_tracer below...
    cells1_at_grid_save_times = cells1[:,::pt_dumps_per_grid_frame]
    scalar_fields_at_tg_cells = np.zeros((len(grid_save_times),
                                         len(fields[0])), dtype = np.float64)
    
    for stime_n, save_time in enumerate(grid_save_times):
        scalar_fields_at_tg_cells[stime_n] = \
            fields[stime_n,:, cells1_at_grid_save_times[0,stime_n],
                   cells1_at_grid_save_times[1,stime_n]]


    T_p = np.zeros_like(xi1)
    
    ttime_n = 0
    for stime_n, save_time in enumerate(grid_save_times[:-1]):
        for cnt in range(30):
            T_p[ttime_n] = T_grid[stime_n, cells1[0,ttime_n], cells1[1,ttime_n] ]
            ttime_n += 1
    T_p[-1] = T_grid[-1, cells1[0,-1], cells1[1,-1] ]

    
    R_p1, w_s1, rho_p1 = compute_R_p_w_s_rho_p_AS(m_w1, m_s1, T_p)
    
    R_s1 = \
    compute_radius_from_mass_vec(m_s1, c.mass_density_AS_dry)
    
    #%% 
    act_plot_part_traj_simple = False
    fig_name = fig_path + f"traj_tracer_{trace_id_n1}.pdf"
#    fig_name = load_path + f"traj_tracer_{trace_id_n1}.pdf"
    from analysis import plot_particle_trajectories
    
    if act_plot_part_traj_simple:
        plot_particle_trajectories( pos1, grid, MS=2.0, arrow_every=5,
                                   ARROW_SCALE=20,ARROW_WIDTH=0.005,
                                   fig_name=fig_name, figsize=(10,10),
                                   TTFS=14, LFS=12, TKFS=12,
                                   t_start=t_start, t_end=t_end)
    
    pos1_shift = np.copy(pos1)

#%% life cycle: data vs time
    act_plot_life_cycle_intern = False
    if act_plot_life_cycle_intern:
        field_names = ["r_v", "r_l", "\Theta", "T", "p", "S"]
    #    scales = [1000, 1000, 1, 1, 0.001, 1]
    #    scales = [1000/7.5, 1000/1.0, 1./289, 1./289, 1E-5, 1.]
        scales = [1000./7.5, 1000./1., 1/2.89, 1., 1E-5, 1.]
    #    field_shifts = [6.5E-3, 0., 289., 273., 0., 0. ]
        field_shifts = [0., 0., 289., 273.15, 0., 0. ]
        units = ["g/kg", "g/kg", "K", "K", "mPa", "-"]    
        
        marker_list = ("o","x", "+", "D", "s", "^", "P"  )
        
        linestyle_list = ("-", "-", "--", ":", "-.", "-" )
    #    linestyle_list = ("-", "-", "--", ":", "-.",  (0, (3, 5, 1, 5, 1, 5)) )
    
        scale_pos = 1E-3
        # not nec for now
    #    if pos1[0,0] < 375.:
    #        pos1_shift[:,0] += 750.
    #        pos1_shift[:,0] = pos1_shift[:,0] % 1500.
        
        ### IN WORK: shift und spiegeln?? -> not nec: found better tracer ;P
        
        MS = 5.
        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors_default = prop_cycle.by_key()['color']
    #    fig_name = load_path + f"R_p_vs_t_tracer_{trace_id_n1}.pdf"
        fig_name = fig_path + f"R_p_vs_t_tracer_{trace_id_n1}.pdf"
        
        ### shift times to 2 h > 0 h
        trace_times -= 7200
        grid_save_times -= 7200
        
        fig, axes = plt.subplots(nrows = 2, figsize=figsize, sharex=True,
                                 gridspec_kw={'height_ratios': [1, 2]})
        
        ax = axes[0]
        ax.plot(trace_times/60, R_p1, c = "k", label = r"$R_p$")
        field_n = 1
        ax.plot(grid_save_times/60,
                (scalar_fields_at_tg_cells[:,field_n])
                * scales[field_n]*10,
                "o-", label = r"${}$".format(field_names[field_n]),
                c = colors_default[0], fillstyle = "none")
        ax2 = ax.twinx()
        ax2.plot(trace_times/60, pos1_shift[:,0]*scale_pos,
                 c = colors_default[1], linestyle = "--", label = "x")
        ax2.plot(trace_times/60, pos1_shift[:,1]*scale_pos,
                 c = colors_default[2], linestyle = ":", label = "z")
        ax.grid()
    #    ax.legend()
        bbox_y_pos = 1.4
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, 
                   loc = "upper right", bbox_to_anchor=(0.5-0.02, bbox_y_pos),
                   ncol = 4,
                   handlelength=1, handletextpad=0.3,
                  columnspacing=0.8, borderpad=0.25)
    #    ax.legend()
    #    ax.legend(loc = "lower left", bbox_to_anchor=(-0.1, 1.1), ncol = 2)
        ax.set_xticks(grid_save_times[::2]/60)
        ax.set_xlim((grid_save_times[0]/60, grid_save_times[-1]/60))
        ax.set_yticks((0,5,10,15,20))
        ax2.set_yticks((0,0.5,1.0,1.5))
    #    ax2.legend(loc='upper right', bbox_to_anchor=(0.8, 1.))
    #    ax2.legend(loc = "upper right", bbox_to_anchor=(1.1, bbox_y_pos), ncol = 2)
    #    ax2.legend(loc = "lower right", bbox_to_anchor=(1.1, 1.1), ncol = 2)
        
        ax.set_ylabel(r"$R_p$ and $r_l$")
    #    ax.set_ylabel(r"$R_p$ (\si{\micro\meter}) and $r_l \times 10$ (g/kg)")
        ax2.set_ylabel(r"$x$ and $z$ (km)")
    
    ############################################################
        ax = axes[1]
        
    #    for field_n in (0,1,2,3,4,5):
    #    for field_n in (0,2,4,5):
        for field_n in (0,4,5):
            if field_n == 1:
                ax.plot(-1,1.0, marker=marker_list[field_n],
                    label = r"${}$".format(field_names[field_n]))
            else:
                ax.plot(grid_save_times/60,
                        (scalar_fields_at_tg_cells[:,field_n] - field_shifts[field_n])
                        * scales[field_n],
    #                    scalar_fields_at_tg_cells[:,field_n] * scales[field_n],
                        marker=marker_list[field_n],
                        label = r"${}$".format(field_names[field_n]),
                        fillstyle = "none", markersize = MS,
                        linestyle = linestyle_list[field_n],
                        c = colors_default[field_n])
        ax2 = ax.twinx()
        for field_n in [3]:
    #    for field_n in (2,3):
            ax2.plot(grid_save_times/60,
    #                        (scalar_fields_at_tg_cells[:,field_n]),
                            (scalar_fields_at_tg_cells[:,field_n] - field_shifts[field_n])
                        * scales[field_n],
        #                    scalar_fields_at_tg_cells[:,field_n] * scales[field_n],
                            marker=marker_list[field_n],
                            label = r"${}$".format(field_names[field_n]),
                            fillstyle = "none", markersize = MS,
                            linestyle = linestyle_list[field_n],
                            c = colors_default[field_n])
        
    #    ax2 = ax.twinx()
    #    ax2.plot(grid_save_times/60,
    #                scalar_fields_at_tg_cells[:,field_n] * scales[field_n],
    #                "x-", label = r"${}$".format(field_names[field_n]),
    #                c = "orange")
    #    bbox_y_pos = 1.5
        ax.set_xticks(grid_save_times[::2]/60)
        ax.set_xlim((grid_save_times[0]/60, grid_save_times[-1]/60))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()    
        
        axes[0].legend(lines + lines2, labels + labels2, loc = "upper left",
                  bbox_to_anchor=(0.5 + 0.0, bbox_y_pos), ncol = 5,
                  handlelength=1.2, handletextpad=0.3,
                  columnspacing=0.8, borderpad=0.3)
    
    #    ax.legend(loc = "lower left", bbox_to_anchor=(-0.1, bbox_y_pos), ncol = 3)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
    #    ax2.legend(loc = "upper right", bbox_to_anchor=(1.1, bbox_y_pos), ncol = 2)
    #    ax2.legend(loc = "lower right", bbox_to_anchor=(1.1, bbox_y_pos), ncol = 2)
    #    ax2.legend(loc = "best")
        
    #    fig.tight_layout()
    #    fig.savefig(fig_name)
        ax.set_xlabel("Time (min)")    
        ax.set_ylabel(r"$r_v$, $p$ and $S$")    
    #    ax.set_ylabel(r"$r_v$, $\Theta$, $p$ and $S$")    
        ax2.set_ylabel(r"$T$ (\si{\celsius})")    
        
        fig.subplots_adjust(hspace=0.05)
    #    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        
        fig.savefig(figname,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.04
                    )       
        plt.close("all")
    
    #%%
    #plt.close("all")
    
    # get the cells at the grid_save_times
    
    act_plot_spectra_Ara = True
    
    if act_plot_spectra_Ara:
    
        # this has to picked individually for each tracer!
#        ind_tracer_grid_times = np.arange(5,len(grid_save_times)-5)
#        ind_tracer_grid_times = np.arange(5,len(grid_save_times)-5)
#        ind_tracer_grid_times = np.array([1,2,4,7,8,12]) + 5
        ind_tracer_grid_times = np.array([1,6,7,9,12,13])
        #    no_rows = len(save_times_out)
        # len(save_times) = 25 for 2h sim with 1 frame per 5 min
        no_rows = 2 
        no_cols = 3
        
        tg_cells_tracer = cells1[:,::pt_dumps_per_grid_frame][:,ind_tracer_grid_times]
        
        ind_traj_first_t = ind_tracer_grid_times[0]*pt_dumps_per_grid_frame
        ind_traj_last_t = ind_tracer_grid_times[-1]*pt_dumps_per_grid_frame
        
    #    ind_time = np.arange(grid_save_times.shape[0])
        ind_time = ind_tracer_grid_times
        
        no_bins_R_p = 30
        no_bins_R_s = 30
        
        load_path_list = []    
    #    no_seeds = len(seed_SIP_gen_list)
        no_seeds = 1
        for seed_n in range(no_seeds):
            seed_SIP_gen_ = seed_SIP_gen_list[seed_n]
            seed_sim_ = seed_sim_list[seed_n]
            
            grid_folder_ =\
                f"{solute_type}" \
                + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
                + f"{seed_SIP_gen_}/"
            
            if simulation_mode == "spin_up":
                save_folder_ = "spin_up_wo_col_wo_grav/"
            elif simulation_mode == "wo_collision":
                if spin_up_finished:
                    save_folder_ = "w_spin_up_wo_col/"
                else:
                    save_folder_ = "wo_spin_up_wo_col/"
            elif simulation_mode == "with_collision":
                if spin_up_finished:
                    save_folder_ = f"w_spin_up_w_col/{seed_sim_}/"
                else:
                    save_folder_ = f"wo_spin_up_w_col/{seed_sim_}/"        
                    load_path_list.append()
            
            load_path_list.append(simdata_path + grid_folder_ + save_folder_)
        
        #print(load_path_list)
        
        #%%
        f_R_p_list, f_R_s_list, bins_R_p_list, bins_R_s_list, save_times_out,\
        grid_r_l_list, R_min_list, R_max_list = \
            generate_size_spectra_R_Arabas(load_path_list,
                                           ind_time,
                                           grid.mass_dry_inv,
                                           grid.no_cells,
                                           solute_type,
                                           tg_cells_tracer,
                                           no_cells_x, no_cells_z,
                                           no_bins_R_p, no_bins_R_s)  
    
        # for fig name
        if no_cells_x % 2 == 0: no_cells_x += 1
        if no_cells_z % 2 == 0: no_cells_z += 1  
        j_low = tg_cells_tracer[1].min()
        j_high = tg_cells_tracer[1].max()
        t_low = save_times_out.min()
        t_high = save_times_out.max()
        no_tg_cells = len(save_times_out)
        
    #    grid_folder_ =\
    #        f"{solute_type}" \
    #        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
    #        + f"plots/{simulation_mode}/dt_col_{dt_col}/tracer_{trace_id_n1}/"
    #         
    #    save_folder_ =\
    #        f"{solute_type}" \
    #        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
    #        + f"plots/{simulation_mode}/dt_col_{dt_col}/"\
    #        + f"gen_{seed_SIP_gen_list[0]}_sim_{seed_sim_list[0]}/"        
            
    #    fig_path = simdata_path + save_folder_
    #    fig_path = simdata_path + grid_folder_
        
    #    if not os.path.exists(fig_path):
    #        os.makedirs(fig_path) 
        
        fig_name =\
            figpath \
            + f"spectra_tracer_{trace_id_n1}_" \
            + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
            + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
        fig_path_tg_cells =\
            figpath \
            + f"tg_cell_posi_tracer_{trace_id_n1}_" \
            + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
            + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
        fig_path_R_eff =\
            figpath \
            + f"R_eff_tracer_{trace_id_n1}_" \
            + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
            + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
        
#        from analysis import plot_size_spectra_R_Arabas
        ind_traj_plot1 = 0 * pt_dumps_per_grid_frame
        ind_traj_plot2 = 19 * pt_dumps_per_grid_frame
        
        scale_x = 1E-3    
        grid.steps *= scale_x
        grid.ranges *= scale_x
        grid.corners[0] *= scale_x
        grid.corners[1] *= scale_x
        grid.centers[0] *= scale_x
        grid.centers[1] *= scale_x  
        
        plot_size_spectra_R_Arabas_MA(
                f_R_p_list, f_R_s_list,
                bins_R_p_list, bins_R_s_list,
                grid_r_l_list,
                R_min_list, R_max_list,
                save_times_out,
                solute_type,
                grid,
                tg_cells_tracer,
                no_cells_x, no_cells_z,
                no_bins_R_p, no_bins_R_s,
                no_rows, no_cols,
                TTFS=10, LFS=10, TKFS=8, LW = 2.0, MS = 0.4,
                figsize_spectra = figsize_spectra,
                figsize_trace_traj = figsize_trace_traj,
                fig_path = fig_name,
                show_target_cells = True,
#                fig_path_tg_cells = fig_path_tg_cells,
#                fig_path_R_eff = fig_path_R_eff,
                fig_path_tg_cells = None,
                fig_path_R_eff = None,
                trajectory = pos1_shift[ind_traj_plot1:ind_traj_plot2] )
#                trajectory = pos1_shift[:] )
#                trajectory = pos1_shift[ind_traj_first_t:ind_traj_last_t] )



    plt.close("all")
    
    
    
    

