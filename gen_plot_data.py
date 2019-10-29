 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:02:06 2019

@author: jdesk
"""

#%% MODULE IMPORTS
import os
import numpy as np

import sys

import constants as c
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full 

#%% STORAGE DIRECTORIES
#my_OS = "Linux_desk"
##my_OS = "Mac"
##my_OS = "TROPOS_server"
#
#if len(sys.argv) > 1:
#    my_OS = sys.argv[1]
#
#if(my_OS == "Linux_desk"):
#    home_path = '/home/jdesk/'
#    simdata_path = "/mnt/D/sim_data_cloudMP/"
##    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
#elif (my_OS == "Mac"):
#    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
##    fig_path = home_path \
##               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'
#elif (my_OS == "TROPOS_server"):
#    simdata_path = "/vols/fs1/work/bohrer/sim_data_cloudMP/"

simdata_path = "/mnt/D/sim_data_cloudMP/"

if len(sys.argv) > 1:
    simdata_path = sys.argv[1]


#%% CHOOSE OPERATIONS

#args_gen = [1,0,0,0]
#args_gen = [0,0,0,1]
#args_gen = [0,1,0,0]
args_gen = [1,1,1,1]
#args_gen = [1,1,1,1]

act_gen_grid_frames_avg = args_gen[0]
act_gen_spectra_avg_Arabas = args_gen[1]
act_get_grid_data = args_gen[2]
act_gen_moments_all_grid_cells = args_gen[3]

#%% GRID PARAMETERS

#no_cells = (75, 75)
no_cells = np.array((75, 75))

if len(sys.argv) > 2:
    no_cells[0] = int(sys.argv[2])
if len(sys.argv) > 3:
    no_cells[1] = int(sys.argv[3])


#%% PARTICLE PARAMETERS

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = "AS"

if len(sys.argv) > 4:
    solute_type = sys.argv[4]

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([16, 24])
no_spcm = np.array([20, 30])
#no_spcm = np.array([26, 38])

if len(sys.argv) > 5:
    no_spcm[0] = int(sys.argv[5])
if len(sys.argv) > 6:
    no_spcm[1] = int(sys.argv[6])

no_seeds = 4
#no_seeds = 50

if len(sys.argv) > 7:
    no_seeds = int(sys.argv[7])

# seed of the SIP generation -> needed for the right grid folder
# start seed and also seed for base grid loading
seed_SIP_gen = 3711

if len(sys.argv) > 8:
    seed_SIP_gen = int(sys.argv[8])

seed_SIP_gen_list = np.arange(seed_SIP_gen, seed_SIP_gen + no_seeds * 2, 2)


# start seed collisions
seed_sim = 4711

if len(sys.argv) > 9:
    seed_sim = int(sys.argv[9])

seed_sim_list = np.arange(seed_sim, seed_sim + no_seeds * 2, 2)

#%% SIM PARAMETERS

#simulation_mode = "spin_up"
#simulation_mode = "wo_collision"
simulation_mode = "with_collision"
if len(sys.argv) > 10:
    simulation_mode = sys.argv[10]
    
spin_up_finished = True
#spin_up_finished = False

# grid load time
# path = simdata_path + folder_load_base
#t_grid = 0
#t_grid = 7200
#t_grid = 10800
t_grid = 14400

if len(sys.argv) > 11:
    t_grid = float(sys.argv[11])

#t_start = 0
t_start = 7200

#t_end = 60
#t_end = 3600
#t_end = 7200
#t_end = 10800
t_end = 14400

if len(sys.argv) > 12:
    t_start = float(sys.argv[12])
if len(sys.argv) > 13:
    t_end = float(sys.argv[13])

dt = 1.0 # s # timestep of advection

### SET THIS
# only even integers possible:
no_cond_per_adv = 10

# number of cond steps per adv step
# possible values: 1, 2 OR no_cond_per_adv
no_col_per_adv = 2
#no_col_per_adv = no_cond_per_adv

if len(sys.argv) > 14:
    no_col_per_adv = int(sys.argv[14])

dt_col = dt / no_col_per_adv

#%% ANALYSIS PARAMETERS

### GRID FRAMES
#plot_frame_every = 6

field_ind = np.array((2,5,0,1))
#    field_ind_ext = np.array((0,1,2))
field_ind_deri = np.array((0,1,2,3,4,5,6,7,8))

#    time_ind = np.arange(0, len(grid_save_times), plot_frame_every)

#time_ind = np.array((0,2,4,6,8,10,12))
time_ind = np.array((0,1))
#time_ind = np.arange(0,25,2)
#    time_ind = np.array((0,3,6,9))

### SPECTRA
# target cells for spectra analysis
i_tg = [16,58,66]
#i_tg = [20,40,60]
j_tg = [27, 37, 44, 46, 51, 72][::-1]
#j_tg = [27, 44, 46, 51, 72][::-1]
#j_tg = [20,40,50,60,65,70]

# target list from ordered mesh grid
# may also set the target list manually as [[i0, i1, ...], [j0, j1, ...]]
i_list, j_list = np.meshgrid(i_tg,j_tg, indexing = "xy")
target_cell_list = np.array([i_list.flatten(), j_list.flatten()])

no_rows = len(j_tg)
no_cols = len(i_tg)
#no_rows = 5
#no_cols = 3

#    print(target_cell_list)
# region range of spectra analysis
# please enter uneven numbers: no_cells_x = 5 =>  [x][x][tg cell][x][x]
no_cells_x = 3
no_cells_z = 3

# time indices may be chosen individually for each spectrum
# where the cell of each spectrum is given in target_cell_list (s.a.)
#time_ind = np.array((0,2,4,6,8,10,12))
#    ind_time = np.zeros(len(target_cell_list[0]), dtype = np.int64)
# choose 1 * ... for spectra plotted after 5 min
# choose 6 * ... for spectra plotted after 30 min etc.
ind_time = 1 * np.ones(len(target_cell_list[0]), dtype = np.int64)
#ind_time = 6 * np.ones(len(target_cell_list[0]), dtype = np.int64)

no_bins_R_p = 30
no_bins_R_s = 30

### TIMES FOR GRID DATA
# gen_seed = list[0], sim_seed = list[0]
#grid_times = [0,7200,14400]
grid_times = [0]

### MOMENTS
no_moments = 4
time_ind_moments = np.array((0,1))
#time_ind_moments = np.arange(0,25,2)

#%% DERIVED    
#no_seeds = len(seed_SIP_gen_list)

#%% LOAD GRID AND PARTICLES AT TIME t_grid

grid_folder = f"{solute_type}"\
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

#%% GENERATE GRID FRAMES AVG

show_target_cells = True

if act_gen_grid_frames_avg:
    from analysis import generate_field_frame_data_avg
#    from analysis import plot_scalar_field_frames_extend_avg

    load_path_list = []    

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
    fields_with_time, fields_with_time_std, save_times_out,\
    field_names_out, units_out, scales_out = \
        generate_field_frame_data_avg(load_path_list,
                                        field_ind, time_ind,
                                        field_ind_deri,
                                        grid.mass_dry_inv,
                                        grid.volume_cell,
                                        grid.no_cells,
                                        solute_type)
    ### create only plotting data output to be transfered
    output_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}_t_{int(t_start)}_{int(t_end)}/"
    
    if not os.path.exists(simdata_path + output_folder):
        os.makedirs(simdata_path + output_folder)    
    
    np.save(simdata_path + output_folder
            + "seed_SIP_gen_list",
            seed_SIP_gen_list)
    np.save(simdata_path + output_folder
            + "seed_sim_list",
            seed_sim_list)
    
    np.save(simdata_path + output_folder
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            fields_with_time)
    np.save(simdata_path + output_folder
            + f"fields_vs_time_std_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            fields_with_time_std)
    np.save(simdata_path + output_folder
            + f"save_times_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            save_times_out)
    np.save(simdata_path + output_folder
            + f"field_names_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            field_names_out)
    np.save(simdata_path + output_folder
            + f"units_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            units_out)
    np.save(simdata_path + output_folder
            + f"scales_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            scales_out)


    
#%% GENERATE SPECTRA AVG 

if act_gen_spectra_avg_Arabas:
    from analysis import generate_size_spectra_R_Arabas
    

    
    print("target_cell_list")
    print(target_cell_list)
    
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
    
    output_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}_t_{int(t_start)}_{int(t_end)}/"
    
    if not os.path.exists(simdata_path + output_folder):
        os.makedirs(simdata_path + output_folder)    
    
    np.save(simdata_path + output_folder
            + "seed_SIP_gen_list",
            seed_SIP_gen_list)
    np.save(simdata_path + output_folder
            + "seed_sim_list",
            seed_sim_list)
    
    np.save(simdata_path + output_folder
            + f"f_R_p_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            f_R_p_list)    
    np.save(simdata_path + output_folder
            + f"f_R_s_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            f_R_s_list)    
    np.save(simdata_path + output_folder
            + f"bins_R_p_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            bins_R_p_list)    
    np.save(simdata_path + output_folder
            + f"bins_R_s_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            bins_R_s_list)    
    np.save(simdata_path + output_folder
            + f"save_times_out_spectra_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            save_times_out)    
    np.save(simdata_path + output_folder
            + f"grid_r_l_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            grid_r_l_list)    
    np.save(simdata_path + output_folder
            + f"R_min_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            R_min_list)    
    np.save(simdata_path + output_folder
            + f"R_max_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            R_max_list)    
    np.save(simdata_path + output_folder
            + f"target_cell_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            target_cell_list)    
    np.save(simdata_path + output_folder
            + f"neighbor_cells_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            [no_cells_x, no_cells_z])    
    np.save(simdata_path + output_folder
            + f"no_rows_no_cols_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            [no_rows, no_cols])    
    np.save(simdata_path + output_folder
            + f"no_bins_p_s_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            [no_bins_R_p, no_bins_R_s])
    
#%% EXTRACT GRID DATA
    
if act_get_grid_data:
    import shutil
    output_path0 = \
        simdata_path \
        + f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}_t_{int(t_start)}_{int(t_end)}/grid_data/"    

    grid_path_base0 = \
        simdata_path \
        + f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"
#        + f"{seed_SIP_gen_list[0]}/"
#        _ss_{seed_sim_list[0]}

    no_grid_times = len(grid_times)

    output_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}_t_{int(t_start)}_{int(t_end)}/"

    np.save(simdata_path + output_folder
            + "seed_SIP_gen_list",
            seed_SIP_gen_list)
    np.save(simdata_path + output_folder
            + "seed_sim_list",
            seed_sim_list)
    
    for seed_n in range(no_seeds):
        s1 = f"{seed_SIP_gen_list[seed_n]}"
        s2 = f"{seed_sim_list[seed_n]}"
        grid_path_base = grid_path_base0 + f"{seed_SIP_gen_list[seed_n]}/"
        output_path = output_path0 + f"{s1}_{s2}/"
        
        if not os.path.exists(output_path):
            os.makedirs(output_path) 
            
        for gt in grid_times:
            if gt == 0:
                shutil.copy(grid_path_base + "grid_basics_0.txt",
                            output_path)
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
                             + f"w_spin_up_w_col/{seed_sim_list[seed_n]}/"
                             + f"grid_basics_{int(gt)}.txt",
                             output_path)
                shutil.copy(grid_path_base
                             + f"w_spin_up_w_col/{seed_sim_list[seed_n]}/"
                             + f"arr_file1_{int(gt)}.npy",
                             output_path)
                shutil.copy(grid_path_base
                             + f"w_spin_up_w_col/{seed_sim_list[seed_n]}/"
                             + f"arr_file2_{int(gt)}.npy",
                             output_path)    

#%% GENERATE MOMENTS FOR ALL GRID CELLS

if act_gen_moments_all_grid_cells:
    from analysis import generate_moments_avg_std
    
    load_path_list = []    

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
    print("load_path_list")    
    print(load_path_list)    
#    moments_vs_time_avg, moments_vs_time_std, save_times_out = \
    moments_vs_time_all_seeds, save_times_out = \
        generate_moments_avg_std(load_path_list,
                               no_moments, time_ind_moments,
                               grid.volume_cell,
                               no_cells, solute_type)
    ### create only plotting data output to be transfered
    output_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}_t_{int(t_start)}_{int(t_end)}/moments/"
    
    if not os.path.exists(simdata_path + output_folder):
        os.makedirs(simdata_path + output_folder)    
    
    np.save(simdata_path + output_folder
            + "seed_SIP_gen_list",
            seed_SIP_gen_list)
    np.save(simdata_path + output_folder
            + "seed_sim_list",
            seed_sim_list)
    
    np.save(simdata_path + output_folder
            + f"moments_vs_time_all_seeds_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            moments_vs_time_all_seeds)
    
#    np.save(simdata_path + output_folder
#            + f"moments_vs_time_avg_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            moments_vs_time_avg)
#    np.save(simdata_path + output_folder
#            + f"moments_vs_time_std_Ns_{no_seeds}_"
#            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
#            moments_vs_time_std)
    np.save(simdata_path + output_folder
            + f"save_times_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
            save_times_out)
