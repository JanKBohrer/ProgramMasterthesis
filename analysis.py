#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:17:55 2019

@author: jdesk
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# in unit (kg/kg)
@njit()
def update_mixing_ratio(mixing_ratio, m_w, xi, cells, mass_dry_inv, 
                        id_list, mask):
    mixing_ratio.fill(0.0)
    for ID in id_list[mask]:
        mixing_ratio[cells[0,ID],cells[1,ID]] += m_w[ID] * xi[ID]
    mixing_ratio *= 1.0E-18 * mass_dry_inv  

# in unit (1/kg)
@njit()
def update_number_concentration_per_dry_mass(conc, xi, cells, mass_dry_inv, 
                        id_list, mask):
    conc.fill(0.0)
    for ID in id_list[mask]:
        conc[cells[0,ID],cells[1,ID]] += xi[ID]
    conc *= mass_dry_inv 

#%% RUNTIME OF FUNCTIONS
# functions is list of strings,
# e.g. ["compute_r_l_grid_field", "compute_r_l_grid_field_np"]
# pars is string,
# e.g. "m_w, xi, cells, grid.mixing_ratio_water_liquid, grid.mass_dry_inv"
# rs is list of repeats (int)
# ns is number of exec per repeat (int)
# example:
# funcs = ["compute_r_l_grid_field_np", "compute_r_l_grid_field"]
# pars = "m_w, xi, cells, grid.mixing_ratio_water_liquid, grid.mass_dry_inv"
# rs = [5,5,5]
# ns = [100,10000,1000]
# compare_functions_run_time(funcs, pars, rs, ns, globals_=globals())
# NOTE that we need to call with globals_=globals() explicitly
# a default argument for globals_ cannot be given in the function definition..
# because in that case, the globals are taken from module "analysis.py" and
# not from the environment of the executed program
def compare_functions_run_time(functions, pars, rs, ns, globals_):
    import timeit
    # import numpy as np
    # print (__name__)
    t = []
    for i,func in enumerate(functions):
        print(func + ": repeats =", rs[i], "no reps = ", ns[i])
    # print(globals_)
    for i,func in enumerate(functions):
        statement = func + "(" + pars + ")"
        t_ = timeit.repeat(statement, repeat=rs[i],
                           number=ns[i], globals=globals_)
        t.append(t_)
        print("best = ", f"{min(t_)/ns[i]*1.0E6:.4}", "us;",
              "worst = ", f"{max(t_)/ns[i]*1.0E6:.4}", "us;",
              "mean =", f"{np.mean(t_)/ns[i]*1.0E6:.4}",
              "+-", f"{np.std(t_, ddof = 1)/ns[i]*1.0E6:.3}", "us" )

#%% PARTICLE TRACKING

# traj = [ pos0, pos1, pos2, .. ]
# where pos0 = [pos_x_0, pos_z_0] is pos at time0
# where pos_x_0 = [x0, x1, x2, ...]
# selection = [n1, n2, ...] --> take only these indic. from the list of traj !!!
def plot_particle_trajectories(traj, grid, selection=None,
                               no_ticks=[6,6], figsize=(8,8),
                               MS=1.0, arrow_every=5,
                               ARROW_SCALE=12,ARROW_WIDTH=0.005, 
                               TTFS = 10, LFS=10,TKFS=10,fig_name=None,
                               t_start=0, t_end=3600):
    centered_u_field = ( grid.velocity[0][0:-1,0:-1]
                         + grid.velocity[0][1:,0:-1] ) * 0.5
    centered_w_field = ( grid.velocity[1][0:-1,0:-1]
                         + grid.velocity[1][0:-1,1:] ) * 0.5
    
    # title font size (pt)
    # TTFS = 10
    # # labelsize (pt)
    # LFS = 10
    # # ticksize (pt)
    # TKFS = 10
    
    # ARROW_SCALE = 12
    # ARROW_WIDTH = 0.005
    # no_major_xticks = 6
    # no_major_yticks = 6
    # tick_every_x = grid.no_cells[0] // (no_major_xticks - 1)
    # tick_every_y = grid.no_cells[1] // (no_major_yticks - 1)
    
    # arrow_every = 5
    
    # part_every = 2
    # loc_every = 1
    
    pos_x = grid.centers[0][arrow_every//2::arrow_every,
                            arrow_every//2::arrow_every]
    pos_z = grid.centers[1][arrow_every//2::arrow_every,
                            arrow_every//2::arrow_every]
    # print(pos_x)
    # pos_x = grid.centers[0][arrow_every//2::arrow_every,
    #                         arrow_every//2::arrow_every]/1000
    # pos_z = grid.centers[1][arrow_every//2::arrow_every,
    #                         arrow_every//2::arrow_every]/1000
    
    # pos_x = grid.centers[0][::arrow_every,::arrow_every]/1000
    # pos_z = grid.centers[1][::arrow_every,::arrow_every]/1000
    
    # pos_x = grid.centers[0][30:40,5:15]
    # pos_z = grid.centers[1][30:40,5:15]
    # pos_x = grid.surface_centers[1][0][30:40,5:15]
    # pos_z = grid.surface_centers[1][1][30:40,5:15]
    
    vel_x = centered_u_field[arrow_every//2::arrow_every,
                             arrow_every//2::arrow_every]
    vel_z = centered_w_field[arrow_every//2::arrow_every,
                             arrow_every//2::arrow_every]
    
    tick_ranges = grid.ranges
    fig, ax = plt.subplots(figsize=figsize)
    if traj[0,0].size == 1:
        ax.plot(traj[:,0], traj[:,1] ,"o", markersize = MS)
    # print(selection)
    else:        
        if selection == None: selection=range(len(traj[0,0]))
        for ID in selection:
            x = traj[:,0,ID]
            z = traj[:,1,ID]
            ax.plot(x,z,"o", markersize = MS)
    ax.quiver(pos_x, pos_z, vel_x, vel_z,
              pivot = 'mid',
              width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=99 )
    ax.set_xticks( np.linspace( tick_ranges[0,0], tick_ranges[0,1],
                                no_ticks[0] ) )
    ax.set_yticks( np.linspace( tick_ranges[1,0], tick_ranges[1,1],
                                no_ticks[1] ) )
    ax.tick_params(axis='both', which='major', labelsize=TKFS)
    ax.set_xlabel('horizontal position (m)', fontsize = LFS)
    ax.set_ylabel('vertical position (m)', fontsize = LFS)
    ax.set_title(
    'Air velocity field and arbitrary particle trajectories\nfrom $t = $'\
    + str(t_start) + " s to " + str(t_end) + " s",
        fontsize = TTFS, y = 1.04)
    ax.grid(color='gray', linestyle='dashed', zorder = 0)    
    # ax.grid()
    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name)

#%% BINNING OF SIPs:

def auto_bin_SIPs(masses, xis, no_bins, dV, no_sims, xi_min=1):

    ind = np.nonzero(xis)
    m_sort = masses[ind]
    xi_sort = xis[ind]
    # print(m_sort.shape)
    # print(xi_sort.shape)
    # print()
    
    ind = np.argsort(m_sort)
    m_sort = m_sort[ind]
    xi_sort = xi_sort[ind]
    # print(m_sort.shape)
    # print(xi_sort.shape)
    
    # plt.plot(masses, "o")
    # plt.plot(m_sort, "o")
    # print(np.nonzero(xis))
    
    # plt.loglog(m_sort, xi_sort, "+")
    # plt.plot(m_sort, xi_sort, "+")
    
    ### merge particles with xi < xi_min
    for i in range(len(xi_sort)-1):
        if xi_sort[i] < xi_min:
            xi = xi_sort[i]
            m = m_sort[i]
            xi_left = 0
            j = i
            while(j > 0 and xi_left==0):
                j -= 1
                xi_left = xi_sort[j]
            if xi_left != 0:
                m1 = m_sort[j]
                dm_left = m-m1
            else:
                dm_left = 1.0E18
            m2 = m_sort[i+1]
            if m2-m < dm_left:
                j = i+1
                # assign to m1 since distance is smaller
                # i.e. increase number of xi[i-1],
                # then reweight mass to total mass
            m_sum = m*xi + m_sort[j]*xi_sort[j]
            # print("added pt ", i, "to", j)
            # print("xi_i bef=", xi)
            # print("xi_j bef=", xi_sort[j])
            # print("m_i bef=", m)
            # print("m_j bef=", m_sort[j])
            # xi_sort[j] += xi_sort[i]
            # m_sort[j] = m_sum / xi_sort[j]
            # print("xi_j aft=", xi_sort[j])
            # print("m_j aft=", m_sort[j])
            xi_sort[i] = 0           
    
    if xi_sort[-1] < xi_min:
        i = -1
        xi = xi_sort[i]
        m = m_sort[-1]
        xi_left = 0
        j = i
        while(xi_left==0):
            j -= 1
            xi_left = xi_sort[j]
        
        m_sum = m*xi + m_sort[j]*xi_sort[j]
        xi_sort[j] += xi_sort[i]
        m_sort[j] = m_sum / xi_sort[j]
        xi_sort[i] = 0           
    
    ind = np.nonzero(xi_sort)
    xi_sort = xi_sort[ind]
    m_sort = m_sort[ind]
    
    # print()
    # print("xis.min()")
    # print("xi_sort.min()")
    # print(xis.min())
    # print(xi_sort.min())
    # print()
    
    # print("xi_sort.shape, xi_sort.dtype")
    # print("m_sort.shape, m_sort.dtype")
    # print(xi_sort.shape, xi_sort.dtype)
    # print(m_sort.shape, m_sort.dtype)
    
    # plt.loglog(m_sort, xi_sort, "x")
    # plt.plot(m_sort, xi_sort, "x")
    
    # resort, if masses have changed "places" in the selection process
    ind = np.argsort(m_sort)
    m_sort = m_sort[ind]
    xi_sort = xi_sort[ind]
    
    ### merge particles, which have masses or xis < m_lim, xi_lim
    no_bins0 = no_bins
    no_bins *= 10
    # print("no_bins")
    # print("no_bins0")
    # print(no_bins)
    # print(no_bins0)
    
    no_spc = len(xi_sort)
    n_save = int(no_spc//1000)
    if n_save < 2: n_save = 2
    
    no_rpc = np.sum(xi_sort)
    total_mass = np.sum(xi_sort*m_sort)
    xi_lim = no_rpc / no_bins
    m_lim = total_mass / no_bins
    
    bin_centers = []
    m_bin = []
    xi_bin = []
    
    n_left = no_rpc
    
    i = 0
    while(n_left > 0 and i < len(xi_sort)-n_save):
        bin_mass = 0.0
        bin_xi = 0
        # bin_center = 0.0
        
        # while(bin_xi < n_lim):
        while(bin_mass < m_lim and bin_xi < xi_lim and n_left > 0
              and i < len(xi_sort)-n_save):
            bin_xi += xi_sort[i]
            bin_mass += xi_sort[i] * m_sort[i]
            n_left -= xi_sort[i]
            i += 1
        bin_centers.append(bin_mass / bin_xi)
        m_bin.append(bin_mass)
        xi_bin.append(bin_xi)
            
    # return m_bin, xi_bin, bin_centers    
    
    xi_bin = np.array(xi_bin)
    bin_centers = np.array(bin_centers)
    m_bin = np.array(m_bin)
    
    ### merge particles, whose masses are close together in log space:
    bin_size_log =\
        (np.log10(bin_centers[-1]) - np.log10(bin_centers[0])) / no_bins0
    
    # print("np.sum(xi_bin*bin_centers), m_bin.sum() before")
    # print(np.sum(xi_bin*bin_centers), m_bin.sum())
    
    i = 0
    while(i < len(xi_bin)-1):
        m_next_bin = bin_centers[i] * 10**bin_size_log
        m = bin_centers[i]
        j = i
        while (m < m_next_bin and j < len(xi_bin)-1):
            j += 1
            m = bin_centers[j]
        if m >= m_next_bin:
            j -= 1
        if (i != j):
            m_sum = 0.0
            xi_sum = 0
            for k in range(i,j+1):
                m_sum += m_bin[k]
                xi_sum += xi_bin[k]
                if k > i:
                    xi_bin[k] = 0
            bin_centers[i] = m_sum / xi_sum
            xi_bin[i] = xi_sum
            m_bin[i] = m_sum
        i = j+1            
    
    
    ind = np.nonzero(xi_bin)
    xi_bin = xi_bin[ind]
    bin_centers = bin_centers[ind]        
    m_bin = m_bin[ind]
    
    # print("np.sum(xi_bin*bin_centers), m_bin.sum() after")
    # print(np.sum(xi_bin*bin_centers), m_bin.sum())
    
    ######
    # bin_size = 0.5 * (bin_centers[-1] - bin_centers[0]) / no_bins0
    # bin_size = (bin_centers[-1] - bin_centers[0]) / no_bins0
    
    # print("len(bin_centers) bef =", len(bin_centers))
    
    # for i, bc in enumerate(bin_centers[:-1]):
    #     if bin_centers[i+1] - bc < bin_size and xi_bin[i] != 0:
    #         m_sum = m_bin[i+1] + m_bin[i]
    #         xi_sum = xi_bin[i+1] + xi_bin[i]
    #         bin_centers[i] = m_sum / xi_sum
    #         xi_bin[i] = xi_sum
    #         xi_bin[i+1] = 0
    #         m_bin[i] = m_sum

    # ind = np.nonzero(xi_bin)
    # xi_bin = xi_bin[ind]
    # m_bin = m_bin[ind]
    # bin_centers = bin_centers[ind]
    ######
    
    # print("len(bin_centers) after =", len(bin_centers))

    # radii = compute_radius_from_mass(m_sort, c.mass_density_water_liquid_NTP)
    radii = compute_radius_from_mass_vec(bin_centers,
                                     c.mass_density_water_liquid_NTP)
    
    ###
    # find the midpoints between the masses/radii
    # midpoints = 0.5 * ( m_sort[:-1] + m_sort[1:] )
    # m_left = 2.0 * m_sort[0] - midpoints[0]
    # m_right = 2.0 * m_sort[-1] - midpoints[-1]
    bins = 0.5 * ( radii[:-1] + radii[1:] )
    # add missing bin borders for m_min and m_max:
    R_left = 2.0 * radii[0] - bins[0]
    R_right = 2.0 * radii[-1] - bins[-1]
    
    bins = np.hstack([R_left, bins, R_right])
    bins_log = np.log(bins)
    # print(midpoints)
       
    # mass_per_ln_R = m_sort * xi_sort
    # mass_per_ln_R *= 1.0E-15/no_sims
    
    m_bin = np.array(m_bin)
    
    g_ln_R = m_bin * 1.0E-15 / no_sims / (bins_log[1:] - bins_log[0:-1]) / dV
    
    return g_ln_R, radii, bins, xi_bin, bin_centers

# masses = masses_vs_time[3]
# xis = xis_vs_time[3]
# radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
# xi_min = 100
# m_bin, xi_bin, bins = auto_bin_SIPs(masses, xis, xi_min)

# r_bin = compute_radius_from_mass(bins, c.mass_density_water_liquid_NTP)

# plt.plot(r_bin, m_bin, "o")

# masses = masses_vs_time[3]
# xis = xis_vs_time[3]
# radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
# print(masses.shape)
# print(xis.shape)
# print()

# xi_min = 100
# no_bins = 40
# g_ln_R, R_sort, bins, xi_bi, m_bins = auto_bin_SIPs(masses,
#                                                     xis, xi_min, no_bins,
#                                                     dV, no_sims)

# fig = plt.figure()
# ax = plt.gca()
# ax.loglog(R_sort, g_ln_R, "x")
# # ax.plot(R_sort, g_ln_R, "x")
# # ax.plot(R_sort, xi_bin)

# ###

# method = "log_R"

# R_min = 0.99*np.amin(radii)
# R_max = 1.01*np.amax(radii)
# # R_max = 3.0*np.amax(radii)
# print("R_min=", R_min)
# print("R_max=", R_max)

# no_bins = 20
# if method == "log_R":
#     bins = np.logspace(np.log10(R_min), np.log10(R_max), no_bins)
# elif method == "lin_R":
#     bins = np.linspace(R_min, R_max, no_bins)
# # print(bins)

# # masses in 10^-15 gram
# mass_per_ln_R, _ = np.histogram(radii, bins, weights=masses*xis)
# # convert to gram
# mass_per_ln_R *= 1.0E-15/no_sims
# # print(mass_per_ln_R)
# # print(mass_per_ln_R.shape, bins.shape)

# bins_log = np.log(bins)
# # bins_mid = np.exp((bins_log[1:] + bins_log[:-1]) * 0.5)
# bins_mid = (bins[1:] + bins[:-1]) * 0.5

# g_ln_R = mass_per_ln_R / (bins_log[1:] - bins_log[0:-1]) / dV

# # print(g_ln_R.shape)
# # print(np.log(bins_mid[1:])-np.log(bins_mid[0:-1]))
# ax.loglog( bins_mid, g_ln_R, "-" )
###

#%% PARTICLE POSITIONS AND VELOCITIES

def plot_pos_vel_pt(pos, vel, grid,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2, fig_name=None):
    # u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
    # v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
    ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
    ax.quiver(*pos, *vel, scale=ARRSCALE, pivot="mid")
    # ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
              # scale=ARRSCALE, pivot="mid", color="red")
    # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
    #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
    #           scale=0.5, pivot="mid", color="red")
    # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
    #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
    #           scale=0.5, pivot="mid", color="blue")
    x_min = grid.ranges[0,0]
    x_max = grid.ranges[0,1]
    y_min = grid.ranges[1,0]
    y_max = grid.ranges[1,1]
    ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
    ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
    # ax.set_xticks(grid.corners[0][:,0])
    # ax.set_yticks(grid.corners[1][0,:])
    ax.set_xticks(grid.corners[0][:,0], minor = True)
    ax.set_yticks(grid.corners[1][0,:], minor = True)
    # plt.minorticks_off()
    # plt.minorticks_on()
    ax.grid()
    fig.tight_layout()
    # ax.grid(which="minor")
    # plt.show()
    if fig_name is not None:
        fig.savefig(fig_name)
        
# pos = [pos0, pos1, pos2, ..] where pos0[0] = [x0, x1, x2, x3, ..] etc
def plot_pos_vel_pt_with_time(pos_data, vel_data, grid, save_times,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2, fig_name=None):
    # u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
    # v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
    no_rows = len(pos_data)
    fig, axes = plt.subplots(nrows=no_rows, figsize=figsize)
    for i,ax in enumerate(axes):
        pos = pos_data[i]
        vel = vel_data[i]
        ax.plot(grid.corners[0], grid.corners[1], "x", color="red",
                markersize=MS)
        ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
        ax.quiver(*pos, *vel, scale=ARRSCALE, pivot="mid")
        # ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
                  # scale=ARRSCALE, pivot="mid", color="red")
        # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
        #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
        #           scale=0.5, pivot="mid", color="red")
        # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
        #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
        #           scale=0.5, pivot="mid", color="blue")
        x_min = grid.ranges[0,0]
        x_max = grid.ranges[0,1]
        y_min = grid.ranges[1,0]
        y_max = grid.ranges[1,1]
        ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
        ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
        # ax.set_xticks(grid.corners[0][:,0])
        # ax.set_yticks(grid.corners[1][0,:])
        ax.set_xticks(grid.corners[0][:,0], minor = True)
        ax.set_yticks(grid.corners[1][0,:], minor = True)
        # plt.minorticks_off()
        # plt.minorticks_on()
        ax.grid()
        ax.set_title("t = " + str(save_times[i]) + " s")
        ax.set_xlabel('x (m)')
        ax.set_xlabel('z (m)')
    fig.tight_layout()
    # ax.grid(which="minor")
    # plt.show()
    if fig_name is not None:
        fig.savefig(fig_name)

#%% PARTICLE SIZE SPECTRA
        
# active ids not necessary: choose target cell and no_cells_x
# such that the region is included in the valid domain
@njit()
def sample_masses(m_w, m_s, xi, cells, id_list, grid_temperature,
                  target_cell, no_cells_x, no_cells_z):
    
#    m_dry = []
#    m_wat = []
#    multi = []
#    
#    i_p = []
#    j_p = []
    
    dx = no_cells_x // 2
    dz = no_cells_z // 2
    
    i_low = target_cell[0] - dx
    i_high = target_cell[0] + dx
    j_low = target_cell[0] - dz
    j_high = target_cell[0] + dz
    
    mask =   ((cells[0] >= i_low) & (cells[0] <= i_high)) \
           & ((cells[1] >= j_low) & (cells[1] <= j_high))
    
    no_masses = mask.sum()
    
    m_s_out = np.zeros(no_masses, dtype = np.float64)
    m_w_out = np.zeros(no_masses, dtype = np.float64)
    xi_out = np.zeros(no_masses, dtype = np.float64)
    T_p = np.zeros(no_masses, dtype = np.float64)
    
    for cnt, ID in enumerate(id_list[mask]):
        m_s_out[cnt] = m_s[ID]
        m_w_out[cnt] = m_w[ID]
        xi_out[cnt] = xi[ID]
        T_p[cnt] = grid_temperature[ cells[0,ID], cells[1,ID] ]
    
    return m_w_out, m_s_out, xi_out, T_p


# active ids not necessary: choose target cell and no_cells_x
# such that the region is included in the valid domain
# weights_out = xi/mass_dry_inv (in that respective cell) 
# weights_out in number/kg_dry_air
@njit()
def sample_masses_per_m_dry(m_w, m_s, xi, cells, id_list, grid_temperature,
                            grid_mass_dry_inv,
                  target_cell, no_cells_x, no_cells_z):
    
#    m_dry = []
#    m_wat = []
#    multi = []
#    
#    i_p = []
#    j_p = []
    
    dx = no_cells_x // 2
    dz = no_cells_z // 2
    
    i_low = target_cell[0] - dx
    i_high = target_cell[0] + dx
    j_low = target_cell[1] - dz
    j_high = target_cell[1] + dz
    
    no_cells_eval = (dx * 2 + 1) * (dz * 2 + 1)
    
    mask =   ((cells[0] >= i_low) & (cells[0] <= i_high)) \
           & ((cells[1] >= j_low) & (cells[1] <= j_high))
    
    no_masses = mask.sum()
    
    m_s_out = np.zeros(no_masses, dtype = np.float64)
    m_w_out = np.zeros(no_masses, dtype = np.float64)
    xi_out = np.zeros(no_masses, dtype = np.float64)
    weights_out = np.zeros(no_masses, dtype = np.float64)
    T_p = np.zeros(no_masses, dtype = np.float64)
    
    for cnt, ID in enumerate(id_list[mask]):
        m_s_out[cnt] = m_s[ID]
        m_w_out[cnt] = m_w[ID]
        xi_out[cnt] = xi[ID]
        weights_out[cnt] = xi[ID] * grid_mass_dry_inv[cells[0,ID], cells[1,ID]]
        T_p[cnt] = grid_temperature[ cells[0,ID], cells[1,ID] ]
    
    return m_w_out, m_s_out, xi_out, weights_out, T_p, no_cells_eval


## @njit()
#def sample_masses(m_w, m_s, xi, cells, target_cell, no_cells_x, no_cells_z):
#    m_dry = []
#    m_wat = []
#    multi = []
#    
#    i_p = []
#    j_p = []
#    
#    dx = no_cells_x // 2
#    dz = no_cells_z // 2
#    
#    i_an = range(target_cell[0] - dx, target_cell[0] + dx + 1)
#    j_an = range(target_cell[1] - dz, target_cell[1] + dz + 1)
#    # print("cells.shape in sample masses")
#    # print(cells.shape)
#    
#    for ID, m_s_ in enumerate(m_s):
#        # print(ID)
#        i = cells[0,ID]
#        j = cells[1,ID]
#        if i in i_an and j in j_an:
#            m_dry.append(m_s_)
#            m_wat.append(m_w[ID])
#            multi.append(xi[ID])
#            i_p.append(i)
#            j_p.append(j)
#    m_wat = np.array(m_wat)
#    m_dry = np.array(m_dry)
#    multi = np.array(multi)
#    i = np.array(i)
#    j = np.array(j)
#    
#    return m_wat, m_dry, multi, i, j

from microphysics import compute_radius_from_mass_vec,\
                         compute_R_p_w_s_rho_p_NaCl,\
                         compute_R_p_w_s_rho_p_AS
import constants as c
# we always assume the only quantities stored are m_s, m_w, xi
def sample_radii(m_w, m_s, xi, cells, solute_type, id_list,
                 grid_temperature, target_cell, no_cells_x, no_cells_z):
    m_w_out, m_s_out, xi_out, T_p = sample_masses(m_w, m_s, xi, cells, id_list,
                                                  grid_temperature,
                                                  target_cell, no_cells_x,
                                                  no_cells_z)
    # print("m_wat")
    # print("m_dry")
    # print("multi")
    # print(m_wat)
    # print(m_dry)
    # print(multi)
#    T_p = grid_temperature[i,j]
    if solute_type == "AS":
        mass_density_dry = c.mass_density_AS_dry
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        mass_density_dry = c.mass_density_NaCl_dry
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    
    R_s = compute_radius_from_mass_vec(m_s_out, mass_density_dry)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w_out, m_s_out, T_p)
    
    return R_p, R_s, xi_out        

# weights_out in number/kg_dry_air
# we always assume the only quantities stored are m_s, m_w, xi
def sample_radii_per_m_dry(m_w, m_s, xi, cells, solute_type, id_list,
                 grid_temperature, grid_mass_dry_inv,
                 target_cell, no_cells_x, no_cells_z):
    m_w_out, m_s_out, xi_out, weights_out, T_p, no_cells_eval = \
        sample_masses_per_m_dry(m_w, m_s, xi, cells, id_list, grid_temperature,
                                grid_mass_dry_inv,
                                target_cell, no_cells_x, no_cells_z)
    # print("m_wat")
    # print("m_dry")
    # print("multi")
    # print(m_wat)
    # print(m_dry)
    # print(multi)
#    T_p = grid_temperature[i,j]
    if solute_type == "AS":
        mass_density_dry = c.mass_density_AS_dry
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        mass_density_dry = c.mass_density_NaCl_dry
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    
    R_s = compute_radius_from_mass_vec(m_s_out, mass_density_dry)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w_out, m_s_out, T_p)
    
    return R_p, R_s, xi_out, weights_out, no_cells_eval        
#def sample_radii(m_w, m_s, xi, cells, grid_temperature,
#                 target_cell, no_cells_x, no_cells_z, solute_type):
#    m_wat, m_dry, multi, i, j = sample_masses(m_w, m_s, xi, cells,
#                                        target_cell, no_cells_x, no_cells_z)
#    # print("m_wat")
#    # print("m_dry")
#    # print("multi")
#    # print(m_wat)
#    # print(m_dry)
#    # print(multi)
#    T_p = grid_temperature[i,j]
#    if solute_type == "AS":
#        mass_density_dry = c.mass_density_AS_dry
#        R, w_s, rho_p = compute_R_p_w_s_rho_p_AS(m_wat, m_dry, T_p)
#    elif solute_type == "NaCl":
#        mass_density_dry = c.mass_density_AS_dry
#        R, w_s, rho_p = compute_R_p_w_s_rho_p_NaCl(m_wat, m_dry, T_p)
#    R_s = compute_radius_from_mass_vec(m_dry, mass_density_dry)
#    return R, R_s, multi        

#%%
def avg_moments_over_boxes(
        moments_vs_time_all_seeds, no_seeds, idx_t, no_moments,
        target_cells_x, target_cells_z,
        no_cells_per_box_x, no_cells_per_box_z):
    no_times_eval = len(idx_t)
    no_target_cells_x = len(target_cells_x)
    no_target_cells_z = len(target_cells_z)
    di_cell = no_cells_per_box_x // 2
    dj_cell = no_cells_per_box_z // 2    
    moments_at_boxes_all_seeds = np.zeros( (no_seeds,no_times_eval,no_moments,
                                  no_target_cells_x,no_target_cells_z),
                                 dtype = np.float64)

    for seed_n in range(no_seeds):
        for time_n, time_ind in enumerate(idx_t):
            for mom_n in range(no_moments):
                for box_n_x, tg_cell_x in enumerate(target_cells_x):
                    for box_n_z , tg_cell_z in enumerate(target_cells_z):
                        moment_box = 0.
                        i_tg_corner = tg_cell_x - di_cell
                        j_tg_corner = tg_cell_z - dj_cell
                        cells_box_x = np.arange(i_tg_corner,
                                                i_tg_corner+no_cells_per_box_x)
                        cells_box_z = np.arange(j_tg_corner,
                                                j_tg_corner+no_cells_per_box_z)
#                        print()
#                        print("cells_box_x")
#                        print(cells_box_x)
#                        print("cells_box_z")
#                        print(cells_box_z)
                        MG = np.meshgrid(cells_box_x, cells_box_z)
                        
                        cells_box_x = MG[0].flatten()
                        cells_box_z = MG[1].flatten()
                        
                        moment_box = moments_vs_time_all_seeds[seed_n, time_n, mom_n,
                                                               cells_box_x, cells_box_z]
                        
                        moment_box = np.average(moment_box)
                        moments_at_boxes_all_seeds[seed_n, time_n, mom_n, box_n_x, box_n_z] = \
                            moment_box
    return moments_at_boxes_all_seeds 

#%% PLOTTING

def simple_plot(x_, y_arr_):
    fig = plt.figure()
    ax = plt.gca()
    for y_ in y_arr_:
        ax.plot (x_, y_)
    ax.grid()

# INWORK: add title and ax labels
def plot_scalar_field_2D( grid_centers_x_, grid_centers_y_, field_,
                         tick_ranges_, no_ticks_=[5,5],
                         no_contour_colors_ = 10, no_contour_lines_ = 5,
                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
    fig, ax = plt.subplots(figsize=(8,8))

    contours = plt.contour(grid_centers_x_, grid_centers_y_,
                           field_, no_contour_lines_, colors = 'black')
    ax.clabel(contours, inline=True, fontsize=8)
    CS = ax.contourf( grid_centers_x_, grid_centers_y_,
                     field_,
                     levels = no_contour_colors_,
                     vmax = field_.max(),
                     vmin = field_.min(),
                    cmap = plt.cm.coolwarm)
    ax.set_xticks( np.linspace( tick_ranges_[0,0], tick_ranges_[0,1],
                                no_ticks_[0] ) )
    ax.set_yticks( np.linspace( tick_ranges_[1,0], tick_ranges_[1,1],
                                no_ticks_[1] ) )
    plt.colorbar(CS, fraction=colorbar_fraction_ , pad=colorbar_pad_)
    
def plot_particle_size_spectra(m_w, m_s, xi, cells, grid, solute_type,
                          target_cell, no_cells_x, no_cells_z,
                          no_rows=1, no_cols=1, TTFS=12, LFS=10, TKFS=10,
                          fig_path = None):
    R, Rs, multi = sample_radii(m_w, m_s, xi, cells, grid.temperature,
                                    target_cell, no_cells_x, no_cells_z,
                                    solute_type)

    log_R = np.log10(R)
    log_Rs = np.log10(Rs)
    multi = np.array(multi)
    if no_cells_x % 2 == 0: no_cells_x += 1
    if no_cells_z % 2 == 0: no_cells_z += 1
    
    V_an = no_cells_x * no_cells_z * grid.volume_cell * 1.0E6
    # V_an = (no_neighbors_x * 2 + 1) * (no_neighbors_z * 2 + 1)
    # * grid.volume_cell * 1.0E6
    
    no_bins = 40
    no_bins_s = 30
    bins = np.empty(no_bins)
    
    R_min = 1E-2
    Rs_max = 0.3
    # R_max = 12.0
    R_max = 120.0
    
    R_min_log = np.log10(R_min)
    Rs_max_log = np.log10(Rs_max)
    R_max_log = np.log10(R_max)
    
    # print(log_R.shape)
    # print(log_Rs.shape)
    # print(multi.shape)
    
    h1, bins1 = np.histogram( log_R, bins=no_bins,  weights=multi/V_an )
    h2, bins2 = np.histogram( log_Rs, bins=no_bins_s,  weights=multi/V_an )
    # h1, bins1 = np.histogram(log_R, bins=no_bins,
    #                          range=(R_min_log,R_max_log), weights=multi/V_an)
    # h2, bins2 = np.histogram( log_Rs, bins=no_bins_s,
    #                           range=(R_min_log, Rs_max_log),
    #                           weights=multi/V_an )
    bins1 = 10 ** bins1
    bins2 = 10 ** bins2
    d_bins1 = np.diff(bins1)
    d_bins2 = np.diff(bins2)
    
    #########################
    
    # # title size (pt)
    # TTFS = 22
    # # labelsize (pt)
    # LFS = 20
    # # ticksize (pt)
    # TKFS = 18
    
    # figsize_x = cm2inch(10.8)
    # figsize_y = cm2inch(10.3)
    # figsize_y = 8
    
    # no_rows = 1
    # no_cols = 1
    
    def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)
    
    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                           figsize = cm2inch(10.8,9.3),
                           # figsize = (figsize_x*no_cols, figsize_y*no_rows),
    #                        sharey=True,
    #                        sharex=True,
                             )
    ax = axes
    # ax.hist(R, bins = bins1, weights = multi/d_bins1)
    ax.bar(bins1[:-1], h1, width = d_bins1, align = 'edge',
    # ax.bar(bins1[:-1], h1/d_bins1, width = d_bins1, align = 'edge',
    #        fill = False,
    #        color = None,
            alpha = 0.05,
           linewidth = 0,
    #        color = (0,0,1,0.05),
    #        edgecolor = (0,0,0,0.0),
           
          )
    ax.bar(bins2[:-1], h2, width = d_bins2, align = 'edge', 
    # ax.bar(bins2[:-1], h2/d_bins2, width = d_bins2, align = 'edge',
    #        fill = False,
    #        color = None,
            alpha = 0.1,
           linewidth = 0,
    #        color = (1,0,0,0.05),
    #        edgecolor = (1,0,0,1.0),
          )
    LW = 2
    ax.plot(np.repeat(bins1,2)[:],
            np.hstack( [[0.001], np.repeat(h1,2)[:], [0.001] ] ),
            linewidth = LW, zorder = 6, label = "wet")
    ax.plot(np.repeat(bins2,2)[:],
            np.hstack( [[0.001], np.repeat(h2,2)[:], [0.001] ] ),
            linewidth = LW, label = "dry")
    
    
    ax.tick_params(axis='both', which='major', labelsize=TKFS, length = 5)
    ax.tick_params(axis='both', which='minor', labelsize=TKFS, length = 3)
    
    height = int(grid.compute_location(*target_cell,0.0,0.0)[1])
    ax.set_xlabel(r"particle radius (mu)", fontsize = LFS)
    ax.set_ylabel(r"concentration (${\mathrm{cm}^{3}}$)", fontsize = LFS)
    # ax.set_xlabel(r"particle radius ($\si{\micro m}$)", fontsize = LFS)
    # ax.set_ylabel(r"concentration ($\si{\# / cm^{3}}$)", fontsize = LFS)
    ax.set_title( f'h = {height} m ' +
                 f"tg cell ({target_cell[0]} {target_cell[1]}) "
                    + f"no cells ({no_cells_x}, {no_cells_z})",
                    fontsize = TTFS )
    
    # X = np.linspace(1E-2,1.0,1000)
    # Y = np.log(X)
    # Z = gaussian(Y, np.log(0.075), np.log(1.6))
    
    # ax.plot(X,Z*11,'--',c="k")
    
    # ax.set_xticks(bins1)
    ax.set_xscale("log")
    # ax.set_xticks(bins1)
    ax.set_yscale("log")
    ax.set_ylim( [0.01,50] )
    ax.grid(linestyle="dashed")
    
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
    ax.legend(handles[::-1], labels[::-1], ncol = 2, prop={'size': TKFS},
              loc='upper center', bbox_to_anchor=(0.5, 1.05), frameon = False)
    fig.tight_layout()
    plt.show()
    if fig_path is not None:
#        fig.savefig(fig_path + "spectrum_" + str(height) +".pdf")
        fig.savefig(fig_path
                    + f"spectrum_cell_{target_cell[0]}_{target_cell[1]}_"
                    + f"no_cells_{no_cells_x}_{no_cells_z}.pdf")

def plot_particle_size_spectra_tg_list(
        t, m_w, m_s, xi, cells,
        grid_volume_cell,
        grid_step_z,
        grid_temperature,
        solute_type,
        target_cell_list, no_cells_x, no_cells_z,
        no_rows, no_cols,
        TTFS=12, LFS=10, TKFS=10,
        fig_path = None):
    
    i_list = target_cell_list[0]
    j_list = target_cell_list[1]
    
#    no_rows = len(i_list)
#    no_cols = len(j_list)
    
    def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)
    
    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                             figsize = (no_cols*5, no_rows*4) )
#    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
#                             figsize = cm2inch(10.8,9.3))
    
    plot_n = -1
    for row_n in range(no_rows)[::-1]:
        for col_n in range(no_cols):
            plot_n += 1
            target_cell = (i_list[plot_n], j_list[plot_n])
            if no_cols == 1:
                if no_rows == 1:
                    ax = axes
                else:                    
                    ax = axes[row_n]
            else:
                ax = axes[row_n, col_n]
            
            R, Rs, multi = sample_radii(m_w, m_s, xi, cells, grid_temperature,
                                            target_cell, no_cells_x, no_cells_z,
                                            solute_type)
        
            log_R = np.log10(R)
            log_Rs = np.log10(Rs)
            multi = np.array(multi)
            if no_cells_x % 2 == 0: no_cells_x += 1
            if no_cells_z % 2 == 0: no_cells_z += 1
            
            V_an = no_cells_x * no_cells_z * grid_volume_cell * 1.0E6
            # V_an = (no_neighbors_x * 2 + 1) * (no_neighbors_z * 2 + 1)
            # * grid.volume_cell * 1.0E6
            
            no_bins = 40
            no_bins_s = 30
            bins = np.empty(no_bins)
            
            R_min = 1E-2
            Rs_max = 0.3
            # R_max = 12.0
            R_max = 120.0
            
            R_min_log = np.log10(R_min)
            Rs_max_log = np.log10(Rs_max)
            R_max_log = np.log10(R_max)
            
            # print(log_R.shape)
            # print(log_Rs.shape)
            # print(multi.shape)
            
            h1, bins1 = np.histogram( log_R, bins=no_bins,  weights=multi/V_an )
            h2, bins2 = np.histogram( log_Rs, bins=no_bins_s,  weights=multi/V_an )
            # h1, bins1 = np.histogram(log_R, bins=no_bins,
            #                          range=(R_min_log,R_max_log), weights=multi/V_an)
            # h2, bins2 = np.histogram( log_Rs, bins=no_bins_s,
            #                           range=(R_min_log, Rs_max_log),
            #                           weights=multi/V_an )
            bins1 = 10 ** bins1
            bins2 = 10 ** bins2
            d_bins1 = np.diff(bins1)
            d_bins2 = np.diff(bins2)
            
            #########################
            
            # # title size (pt)
            # TTFS = 22
            # # labelsize (pt)
            # LFS = 20
            # # ticksize (pt)
            # TKFS = 18
            
            # figsize_x = cm2inch(10.8)
            # figsize_y = cm2inch(10.3)
            # figsize_y = 8
            
            # no_rows = 1
            # no_cols = 1
        
    
        
    
    #    ax = axes
        # ax.hist(R, bins = bins1, weights = multi/d_bins1)
            ax.bar(bins1[:-1], h1, width = d_bins1, align = 'edge',
            # ax.bar(bins1[:-1], h1/d_bins1, width = d_bins1, align = 'edge',
            #        fill = False,
            #        color = None,
                    alpha = 0.05,
                   linewidth = 0,
            #        color = (0,0,1,0.05),
            #        edgecolor = (0,0,0,0.0),
                   
                  )
            ax.bar(bins2[:-1], h2, width = d_bins2, align = 'edge', 
            # ax.bar(bins2[:-1], h2/d_bins2, width = d_bins2, align = 'edge',
            #        fill = False,
            #        color = None,
                    alpha = 0.1,
                   linewidth = 0,
            #        color = (1,0,0,0.05),
            #        edgecolor = (1,0,0,1.0),
                  )
            LW = 2
            ax.plot(np.repeat(bins1,2)[:],
                    np.hstack( [[0.001], np.repeat(h1,2)[:], [0.001] ] ),
                    linewidth = LW, zorder = 6, label = "wet")
            ax.plot(np.repeat(bins2,2)[:],
                    np.hstack( [[0.001], np.repeat(h2,2)[:], [0.001] ] ),
                    linewidth = LW, label = "dry")
            
            
            ax.tick_params(axis='both', which='major', labelsize=TKFS, length = 5)
            ax.tick_params(axis='both', which='minor', labelsize=TKFS, length = 3)
            
#            height = int(grid.compute_location(*target_cell,0.0,0.0)[1])
            height = int((target_cell[1]+0.5)*grid_step_z)
            ax.set_xlabel(r"particle radius (mu)", fontsize = LFS)
            ax.set_ylabel(r"concentration (${\mathrm{cm}^{3}}$)", fontsize = LFS)
            # ax.set_xlabel(r"particle radius ($\si{\micro m}$)", fontsize = LFS)
            # ax.set_ylabel(r"concentration ($\si{\# / cm^{3}}$)", fontsize = LFS)
            ax.set_title( f'h = {height} m ' +
                         f"tg cell ({target_cell[0]} {target_cell[1]}) "
                            + f"no cells ({no_cells_x}, {no_cells_z})",
                            fontsize = TTFS )
            
            # X = np.linspace(1E-2,1.0,1000)
            # Y = np.log(X)
            # Z = gaussian(Y, np.log(0.075), np.log(1.6))
            
            # ax.plot(X,Z*11,'--',c="k")
            
            # ax.set_xticks(bins1)
            ax.set_xscale("log")
            # ax.set_xticks(bins1)
            ax.set_yscale("log")
            ax.set_ylim( [0.01,50] )
            ax.grid(linestyle="dashed")
            
            handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
            ax.legend(handles[::-1], labels[::-1], ncol = 2, prop={'size': TKFS},
                      loc='upper center', bbox_to_anchor=(0.5, 1.04), frameon = False)
    fig.tight_layout()
#    plt.show()
    if fig_path is not None:
#        fig.savefig(fig_path + "spectrum_" + str(height) +".pdf")
        fig.savefig(fig_path 
                    + f"spectrum_cell_list_j_from_{j_list[0]}_to_{j_list[-1]}_" 
                    + f"no_cells_{no_cells_x}_{no_cells_z}_t_{int(t)}.pdf")

#%% PLOT GRID SCALAR FIELDS WITH TIME

# fields = [fields_t0, fields_t1, ...]
# fields_ti =
# (grid.mixing_ratio_water_vapor, grid.mixing_ratio_water_liquid,
#  grid.potential_temperature, grid.temperature,
#  grid.pressure, grid.saturation)
# input:
# field_indices = (idx1, idx2, ...)  (tuple of int)
# time_indices = (idx1, idx2, ...)  (tuple of int)
# title size (pt)
# TTFS = 18
# labelsize (pt)
# LFS = 18
# ticksize (pt)
# TKFS = 18
def plot_scalar_field_frames(grid, fields,
                            save_times, field_indices, time_indices,
                            no_ticks=[6,6], fig_path=None,
                            TTFS = 12, LFS = 10, TKFS = 10,):
    
    no_rows = len(time_indices)
    no_cols = len(field_indices)
    
    field_names = ["r_v", "r_l", "Theta", "T", "p", "S"]
    scales = [1000, 1000, 1, 1, 0.01, 1]
    units = ["g/kg", "g/kg", "K", "K", "hPa", "-"]
    
    tick_ranges = grid.ranges
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize = (4.5*no_cols, 4*no_rows))
    for i in range(no_rows):
        for j in range(no_cols):
            ax = axes[i,j]
            idx_t = time_indices[i]
            idx_f = field_indices[j]
            # print("idx_t")
            # print(idx_t)
            # print("idx_f")
            # print(idx_f)
            field = fields[idx_t, idx_f]*scales[idx_f]
            field_min = field.min()
            if idx_f == 1: field_min = 0.001
            field_max = field.max()
            if idx_f in [2,3,4]: 
                cmap = "coolwarm"
                alpha = None
            else: 
                cmap = "rainbow"
                alpha = 0.7
#                contours = ax[i,j].contour(grid_centers_x_, grid_centers_y_,
#                       field, no_contour_lines_, colors = 'black')
#                ax[i,j].clabel(contours, inline=True, fontsize=8)
            CS = ax.pcolormesh(*grid.corners, field, cmap=cmap, alpha=alpha,
                                    vmin=field_min, vmax=field_max,
                                    edgecolor="face", zorder=1)
            CS.cmap.set_under("white")
            if idx_f == 1:
                cbar = fig.colorbar(CS, ax=ax, extend = "min")
            else: cbar = fig.colorbar(CS, ax=ax)
            cbar.ax.tick_params(labelsize=TKFS)
            ax.set_title(
                field_names[idx_f] + ' (' + units[idx_f] + '), t = '
                + str(int(save_times[idx_t]//60)) + " min", fontsize = TTFS )
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )
            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            if i == no_rows-1:
                ax.set_xlabel(r'x (m)', fontsize = LFS)
            if j == 0:
                ax.set_ylabel(r'z (m)', fontsize = LFS)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
          
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)

#%% PLOT SCALAR FIELD FRAMES WITH r_aero, r_c, r_r

def compute_order_of_magnitude(x):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return b  

import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.colors import hex2color, LinearSegmentedColormap
#import cmocean.cm as cmo
#cmap = "rainbow"
#cmap = "gist_rainbow_r"
#cmap = "nipy_spectral"
#cmap = "nipy_spectral_r"
#cmap = "gist_ncar_r"
#cmap = "cubehelix_r"
#cmap = "viridis_r"
#cmap = "plasma_r"
#cmap = "magma_r"
#cmap = cmo.rain
#alpha = 0.7

colors1 = plt.cm.get_cmap('gist_ncar_r', 256)
#top = plt.cm.get_cmap('gist_rainbow_r', 256)
colors2 = plt.cm.get_cmap('rainbow', 256)
#top = plt.cm.get_cmap('Greys', 128)
#bottom = plt.cm.get_cmap('Blues', 128)

newcolors = np.vstack((colors1(np.linspace(0, 0.16, 24)),
                       colors2(np.linspace(0, 1, 256))))
#newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                       bottom(np.linspace(0, 1, 128))))
cmap_new = mpl.colors.ListedColormap(newcolors, name='my_rainbow')

### CREATE COLORMAP LIKE ARABAS 2015
hex_colors = ['#FFFFFF', '#993399', '#00CCFF', '#66CC00',
              '#FFFF00', '#FC8727', '#FD0000']
rgb_colors = [hex2color(c) + tuple([1.0]) for c in hex_colors]
no_colors = len(rgb_colors)

cdict_lcpp_colors = np.zeros( (3, no_colors, 3) )

for i in range(3):
    cdict_lcpp_colors[i,:,0] = np.linspace(0.0,1.0,no_colors)
    for j in range(no_colors):
        cdict_lcpp_colors[i,j,1] = rgb_colors[j][i]
        cdict_lcpp_colors[i,j,2] = rgb_colors[j][i]

cdict_lcpp = {"red": cdict_lcpp_colors[0],
              "green": cdict_lcpp_colors[1],
              "blue": cdict_lcpp_colors[2]}

cmap_lcpp = LinearSegmentedColormap('testCmap', segmentdata=cdict_lcpp, N=256)

# fields = [fields_t0, fields_t1, ...]
# fields_ti =
# (grid.mixing_ratio_water_vapor, grid.mixing_ratio_water_liquid,
#  grid.potential_temperature, grid.temperature,
#  grid.pressure, grid.saturation)
# input:
# field_indices = (idx1, idx2, ...)  (tuple of int)
# time_indices = (idx1, idx2, ...)  (tuple of int)
# title size (pt)
# TTFS = 18
# labelsize (pt)
# LFS = 18
# ticksize (pt)
# TKFS = 18
        
# need grid for dry air density and ?
# fields are with time
# m_s, w_s, xi       
@njit()
def update_T_p(grid_temp, cells, T_p):
    for ID in range(len(T_p)):
        # T_p_ = grid_temp[cells[0,ID],cells[1,ID]]
        T_p[ID] = grid_temp[cells[0,ID],cells[1,ID]]    
# ind_ext = [i1, i2, ...]
# 0: r_aero
# 1: r_cloud
# 2: r_rain        
def plot_scalar_field_frames_extend(grid, fields, m_s, m_w, xi, cells,
                                    active_ids,
                                    solute_type,
                                    save_times, field_indices, time_indices,
                                    derived_indices,
                                    no_ticks=[6,6], fig_path=None,
                                    TTFS = 12, LFS = 10, TKFS = 10,
                                    cbar_precision = 2):
#    print(save_times)
    if solute_type == "AS":
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    
    no_rows = len(time_indices)
    no_cols = len(field_indices) + len(derived_indices)
    no_fields_orig_chosen = len(field_indices)
    
    no_fields_ext = 6
    
    # load particle data: stored data: m_s, m_w, xi
    # need R_p -> need rho_p -> need w_s and T_p
    # classify droplets per cell by R_p:
    # aerosol: R_p < 0.5 mu
    # cloud drops: 0.5 < R_p < 25 mu
    # rain drops: 25 mu < R_p
    bins_drop_class = [0.5,25.]
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize = (4.5*no_cols, 4*no_rows))
    
#    for t_n, idx_t in enumerate(time_indices):
#        pos_ = pos[idx_t]
#        # DEACTIVATE LATER... -> need active ids and cells from storage...
#        cells = np.array( [np.floor(pos_[0]/grid.steps[0]),
#                           np.floor(pos_[1]/grid.steps[1])] ).astype(int)
#        update_T_p(fields[idx_t, 3], cells, T_p)        
        
    field_names = ["r_v", "r_l", "\Theta", "T", "p", "S"]
    scales = [1000, 1000, 1, 1, 0.01, 1]
    units = ["g/kg", "g/kg", "K", "K", "hPa", "-"]
    
    field_names_ext = ["r_\mathrm{aero}", "r_c", "r_r",
                       "n_\mathrm{aero}", "n_c", "n_r"]
    units_ext = ["g/kg", "g/kg", "g/kg", "1/mg", "1/mg", "1/mg"]
    scales_ext = [1000, 1000, 1000, 1E-6, 1E-6, 1E-6]
    
#    field_names += field_names_ext
    
    tick_ranges = grid.ranges
    for i in range(no_rows):
        idx_t = time_indices[i]
        no_SIPs = len(xi[idx_t])
        T_p = np.zeros(no_SIPs, dtype = np.float64)
        id_list = np.arange(no_SIPs)
        update_T_p(fields[idx_t, 3], cells[idx_t], T_p)
        R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w[idx_t], m_s[idx_t], T_p)
        idx_R_p = np.digitize(R_p, bins_drop_class)
        idx_classification = np.arange(3).reshape((3,1))
        
        masks_R_p = idx_classification == idx_R_p
               
        fields_ext = np.zeros((no_fields_ext, grid.no_cells[0],
                               grid.no_cells[1]),
                              dtype = np.float64)
        
        for mask_n in range(3):
            mask = np.logical_and(masks_R_p[mask_n], active_ids[idx_t])
            update_mixing_ratio(fields_ext[mask_n],
                                m_w[idx_t], xi[idx_t], cells[idx_t],
                                grid.mass_dry_inv, 
                                id_list, mask)   
            update_number_concentration_per_dry_mass(fields_ext[mask_n+3],
                                xi[idx_t], cells[idx_t],
                                grid.mass_dry_inv, 
                                id_list, mask)
        
        for j in range(no_cols):
            ax = axes[i,j]
            if j < no_fields_orig_chosen:
                idx_f = field_indices[j]
                field = fields[idx_t, idx_f]*scales[idx_f]
                ax_title = field_names[idx_f]
                unit = units[idx_f]
                if idx_f in [2,3,4]:
                    cmap = "coolwarm"
                    alpha = 1.0
                else:
                    cmap = "rainbow"
                    alpha = 0.8
            else: 
                idx_f = derived_indices[j-no_fields_orig_chosen]
                field = fields_ext[idx_f]*scales_ext[idx_f]
                ax_title = field_names_ext[idx_f]
                unit = units_ext[idx_f]
                cmap = "rainbow"
                alpha = 0.8
            field_max = field.max()
            field_min = field.min()
    #        oom_max = compute_order_of_magnitude(field_max)
            oom_max = oom = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
            if oom_max > 2 or oom_max < 0:
                my_format = True
                oom_factor = 10**(-oom)
                
                field_min *= oom_factor
                field_max *= oom_factor            
            
            if oom_max in [1,2]: str_format = "%.1f"
            else: str_format = "%.2f"
#            def fmt_cbar(x, pos):
#        #        a, b = '{:.2e}'.format(x).split('e')
#        #        a = float(a)
#        #        b = int(b) - oom_max
#                return r'${0:.{prec}f}$'.format(x, prec=cbar_precision) 
            
            if field_min/field_max < 1E-4:
#                cmap = cmap_new
                # Arabas 2015
                cmap = cmap_lcpp
                alpha = 0.8
            # REMOVE APLHA HERE
            alpha = 1.0
            norm_ = mpl.colors.Normalize 
            if ax_title in ["r_r", "n_r"] and field_max > 1.:
                norm_ = mpl.colors.LogNorm
                field_min = 0.01
                if ax_title == "r_r":
                    field_max = 1.
                elif ax_title == "n_r":
                    field_max = 10.
            else: norm_ = mpl.colors.Normalize   
            
            if ax_title == "r_c":
                field_min = 0.0
                field_max = 1.3
            if ax_title == "n_c":
                field_min = 0.0
                field_max = 150.
            if ax_title == "n_\mathrm{aero}":
                field_min = 0.0
                field_max = 150.
                
            
            CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor="face", zorder=1,
                                norm = norm_(vmin=field_min, vmax=field_max)
                                )
            CS.cmap.set_under("white")
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )
            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_xlabel(r'x (m)', fontsize = LFS)
            ax.set_ylabel(r'z (m)', fontsize = LFS)
            ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
                         int(save_times[idx_t]/60)),
                         fontsize = TTFS)
#            ax.set_title( r"${0}$ ({1})".format(field_names[j], units[j]),
#                         fontsize = TTFS)
            
#            cbar = plt.colorbar(CS, ax=ax, extend = "min",
#                                format=mticker.FuncFormatter(fmt_cbar))
            cbar = plt.colorbar(CS, ax=ax,
                                format=mticker.FormatStrFormatter(str_format))
            if my_format:
                cbar.ax.text(field_min - (field_max-field_min),
                             field_max + (field_max-field_min)*0.01,
                             r'$\times\,10^{{{}}}$'.format(oom_max),
                             va='bottom', ha='left', fontsize = TKFS)
#            cbar.ax.text(-2.3*(field_max-field_min), field_max*1.01,
#                         r'$\times\,10^{{{}}}$'.format(oom_max),
#                         va='bottom', ha='left', fontsize = TKFS)
#            cbar.ax.text(0,1,
#                         r'$\times\,10^{{{}}}$'.format(oom_max),
#                         va='bottom', ha='left', fontsize = TKFS)
            
#            if oom_max == 2:
#                ticks_cbar = np.arange( int(10*field_min)*0.1,
#                                      (int(10*field_max)+1)*0.1,
#                                             int(1*(field_max-field_min)) / 5)
#            else:
#                ticks_cbar = np.arange( int(10*field_min)*0.1,
#                                  (int(10*field_max)+1)*0.1,
#                                         int(10*(field_max-field_min)) / 50)
#            print(i,j,ticks_cbar)
#            cbar.set_ticks(np.arange( int(10*field_min)*0.1,
#                                         (int(10*field_max)+1)*0.1),
#                                         int(10*(field_max-field_min)) / 80 )
            cbar.ax.tick_params(labelsize=TKFS)
            


#            field_min = field.min()
#            if idx_f == 1: field_min = 0.001
#            field_max = field.max()
#            if idx_f in [2,3,4]: 
#                cmap = "coolwarm"
#                alpha = None
#            else: 
#                cmap = "rainbow"
#                alpha = 0.7
#                contours = ax[i,j].contour(grid_centers_x_, grid_centers_y_,
#                       field, no_contour_lines_, colors = 'black')
#                ax[i,j].clabel(contours, inline=True, fontsize=8)
#            CS = ax.pcolormesh(*grid.corners, field, cmap=cmap, alpha=alpha,
#                                    vmin=field_min, vmax=field_max,
#                                    edgecolor="face", zorder=1)
#            CS.cmap.set_under("white")
#            if idx_f == 1:
#                cbar = fig.colorbar(CS, ax=ax, extend = "min")
#            else: cbar = fig.colorbar(CS, ax=ax)
#            cbar.ax.tick_params(labelsize=TKFS)
#            ax.set_title(
#                field_names[idx_f] + ' (' + units[idx_f] + '), t = '
#                + str(int(save_times[idx_t]//60)) + " min", fontsize = TTFS )
#            ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                             tick_ranges[0,1],
#                                             no_ticks[0] ) )
#            ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                             tick_ranges[1,1],
#                                             no_ticks[1] ) )
#            ax.tick_params(axis='both', which='major', labelsize=TKFS)
#            if i == no_rows-1:
#                ax.set_xlabel(r'x (m)', fontsize = LFS)
#            if j == 0:
#                ax.set_ylabel(r'z (m)', fontsize = LFS)
#            ax.grid(color='gray', linestyle='dashed', zorder = 2)
          
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)



@njit()
# V0 = volume grid cell
def compute_moment_R_grid(n, R_p, xi, V0,
                          cells, active_ids, id_list, no_cells):
    moment = np.zeros( (no_cells[0], no_cells[1]), dtype = np.float64 )
    if n == 0:
        for ID in id_list[active_ids]:
            moment[cells[0,ID],cells[1,ID]] += xi[ID]
    else:
        for ID in id_list[active_ids]:
            moment[cells[0,ID],cells[1,ID]] += xi[ID] * R_p[ID]**n
    return moment / V0

from file_handling import load_grid_scalar_fields, load_particle_data_all

# possible field indices:
#0: r_v
#1: r_l
#2: Theta
#3: T
#4: p
#5: S
# possibe derived indices:
# 0: r_aero
# 1: r_cloud
# 2: r_rain     
# 3: n_aero
# 4: n_c
# 5: n_r 
# 6: R_avg
# 7: R_1/2 = 2nd moment / 1st moment
# 8: R_eff = 3rd moment/ 2nd moment of R-distribution
# NOTE that time_indices must be an array because of indexing below     
def generate_field_frame_data_avg(load_path_list,
                                  field_indices, time_indices,
                                  derived_indices,
                                  mass_dry_inv, grid_volume_cell,
                                  no_cells, solute_type):
    # output: fields_with_time = [ [time0: all fields[] ],
    #                              [time1],
    #                              [time2], .. ]
    # for collected fields:
    # unit_list
    # name_list
    # scales_list
    # save_times
    
    V0 = grid_volume_cell
    
    if solute_type == "AS":
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    
    bins_R_p_drop_classif = [0.5, 25.]
    
    field_names_orig = ["r_v", "r_l", "\Theta", "T", "p", "S"]
    scales_orig = [1000., 1000., 1, 1, 0.01, 1]
    units_orig = ["g/kg", "g/kg", "K", "K", "hPa", "-"]
    
    field_names_deri = ["r_\mathrm{aero}", "r_c", "r_r",
                       "n_\mathrm{aero}", "n_c", "n_r",
                       r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]
    units_deri = ["g/kg", "g/kg", "g/kg", "1/mg", "1/mg", "1/mg",
                  r"$\mathrm{\mu m}$", r"$\mathrm{\mu m}$", r"$\mathrm{\mu m}$"]
    scales_deri = [1000., 1000., 1000., 1E-6, 1E-6, 1E-6, 1., 1., 1.]    
    
    no_seeds = len(load_path_list)
    no_times = len(time_indices)
    no_fields_orig = len(field_indices)
    no_fields_derived = len(derived_indices)
    no_fields = no_fields_orig + no_fields_derived
    
#    print(no_fields_orig)
#    print(no_fields_derived)
#    print(no_fields)
    
    fields_with_time = np.zeros( (no_times, no_fields,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    fields_with_time_sq = np.zeros( (no_times, no_fields,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
    field_names_out = []
    units_out = []
    scales_out = []
    
    for cnt in range(no_fields_orig):
        idx_f = field_indices[cnt]
        field_names_out.append(field_names_orig[idx_f])
        units_out.append(units_orig[idx_f])
        scales_out.append(scales_orig[idx_f])
    
    for cnt in range(no_fields_derived):
        idx_f = derived_indices[cnt]
        field_names_out.append(field_names_deri[idx_f])
        units_out.append(units_deri[idx_f])
        scales_out.append(scales_deri[idx_f])         
    
    for time_n in range(no_times):
        idx_t = time_indices[time_n]
        save_times_out[time_n] = grid_save_times[idx_t]
    
    for seed_n, load_path in enumerate(load_path_list):
        
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        vec_data, cells_with_time, scal_data, xi_with_time, active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]
        
        for cnt in range(no_fields_orig):
            idx_f = field_indices[cnt]
            fields_with_time[:,cnt] += fields[time_indices,idx_f]
            fields_with_time_sq[:,cnt] += \
                fields[time_indices,idx_f]*fields[time_indices,idx_f]
        
        for time_n in range(no_times):
            idx_t = time_indices[time_n]
            
            no_SIPs = len(xi_with_time[idx_t])
            T_p = np.zeros(no_SIPs, dtype = np.float64)
            id_list = np.arange(no_SIPs)
            update_T_p(fields[idx_t, 3], cells_with_time[idx_t], T_p)
            R_p, w_s, rho_p = \
                compute_R_p_w_s_rho_p(m_w_with_time[idx_t],
                                      m_s_with_time[idx_t], T_p)
            idx_R_p = np.digitize(R_p, bins_R_p_drop_classif)
            idx_classification = np.arange(3).reshape((3,1))
            
            masks_R_p = idx_classification == idx_R_p
                   
            fields_derived = np.zeros((no_fields_derived, no_cells[0],
                                       no_cells[1]),
                                       dtype = np.float64)
            
            mom0 = compute_moment_R_grid(0, R_p, xi_with_time[idx_t], V0,
                                         cells_with_time[idx_t],
                                         active_ids_with_time[idx_t],
                                         id_list, no_cells)
            mom1 = compute_moment_R_grid(1, R_p, xi_with_time[idx_t], V0,
                                         cells_with_time[idx_t],
                                         active_ids_with_time[idx_t],
                                         id_list, no_cells)
            mom2 = compute_moment_R_grid(2, R_p, xi_with_time[idx_t], V0,
                                         cells_with_time[idx_t],
                                         active_ids_with_time[idx_t],
                                         id_list, no_cells)
            mom3 = compute_moment_R_grid(3, R_p, xi_with_time[idx_t], V0,
                                         cells_with_time[idx_t],
                                         active_ids_with_time[idx_t],
                                         id_list, no_cells)
            
            # calculate R_eff only from cloud range (as Arabas 2015)
            mom1_cloud = compute_moment_R_grid(
                             1,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)
            mom2_cloud = compute_moment_R_grid(
                             2,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)
            mom3_cloud = compute_moment_R_grid(
                             3,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)
#            mom2_cloud = compute_moment_R_grid(3,
#                                               R_p[masks_R_p[1]],
#                                               xi_with_time[idx_t][masks_R_p[1]],
#                                               cells_with_time[idx_t][:,[masks_R_p[1]]],
#                                               active_ids_with_time[idx_t][masks_R_p[1]],
#                                               id_list, no_cells)
#            mom3_cloud = compute_moment_R_grid(3, R_p, xi_with_time[idx_t],
#                                               cells_with_time[idx_t],
#                                               active_ids_with_time[idx_t],
#                                               id_list, no_cells)

            for cnt in range(no_fields_derived):
                idx_f = derived_indices[cnt]
                if idx_f < 6:
                    mask = np.logical_and(masks_R_p[idx_f%3],
                                          active_ids_with_time[idx_t])
                    if idx_f in range(3):
                        update_mixing_ratio(fields_derived[cnt],
                                            m_w_with_time[idx_t],
                                            xi_with_time[idx_t],
                                            cells_with_time[idx_t],
                                            mass_dry_inv, 
                                            id_list, mask)   
                    elif idx_f in range(3,6):
                        update_number_concentration_per_dry_mass(
                                fields_derived[cnt],
                                xi_with_time[idx_t],
                                cells_with_time[idx_t],
                                mass_dry_inv, 
                                id_list, mask)
                elif idx_f == 6:
                    # R_mean
                    fields_derived[cnt] = np.where(mom0 == 0.0, 0.0, mom1/mom0)
                elif idx_f == 7:
                    # R_2/1
                    fields_derived[cnt] = np.where(mom1_cloud == 0.0, 0.0,
                                                   mom2_cloud/mom1_cloud)
                elif idx_f == 8:
                    # R_eff
                    fields_derived[cnt] = np.where(mom2_cloud == 0.0, 0.0,
                                                   mom3_cloud/mom2_cloud)
                
                        
            
#            for mask_n in range(3):
#                mask = np.logical_and(masks_R_p[mask_n],
#                                      active_ids_with_time[idx_t])
#                update_mixing_ratio(fields_derived[mask_n],
#                                    m_w_with_time[idx_t],
#                                    xi_with_time[idx_t],
#                                    cells_with_time[idx_t],
#                                    mass_dry_inv, 
#                                    id_list, mask)   
#                update_number_concentration_per_dry_mass(
#                        fields_derived[mask_n+3],
#                        xi_with_time[idx_t],
#                        cells_with_time[idx_t],
#                        mass_dry_inv, 
#                        id_list, mask)
#            for cnt in range(no_fields_derived):
#                fields_with_time[idx_t,cnt+no_fields_orig] += \
#                    fields_derived[cnt]
                
#            print(fields_with_time.shape)    
#            print(fields_derived.shape)    
            fields_with_time[time_n,no_fields_orig:no_fields] += \
                fields_derived
            fields_with_time_sq[time_n,no_fields_orig:no_fields] += \
                fields_derived * fields_derived
    
    fields_with_time /= no_seeds
    
    fields_with_time_std = np.sqrt((fields_with_time_sq \
                                  - no_seeds*fields_with_time*fields_with_time) \
                           / (no_seeds * (no_seeds-1)) )
    
    return fields_with_time, fields_with_time_std, \
           save_times_out, field_names_out, units_out, \
           scales_out 

#%%

def generate_moments_avg_std(load_path_list,
                             no_moments, time_indices,
                             grid_volume_cell,
                             no_cells, solute_type):

    if solute_type == "AS":
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl

    no_seeds = len(load_path_list)
    no_times = len(time_indices)
    
    moments_vs_time_all_seeds = np.zeros( (no_seeds, no_times, no_moments,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
    for time_n in range(no_times):
        idx_t = time_indices[time_n]
        save_times_out[time_n] = grid_save_times[idx_t]
    
    V0 = grid_volume_cell
    
    for seed_n, load_path in enumerate(load_path_list):
        
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        vec_data, cells_with_time, scal_data, xi_with_time, active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]
        
        for time_n in range(no_times):
            idx_t = time_indices[time_n]
            
            no_SIPs = len(xi_with_time[idx_t])
            T_p = np.zeros(no_SIPs, dtype = np.float64)
            id_list = np.arange(no_SIPs)
            update_T_p(fields[idx_t, 3], cells_with_time[idx_t], T_p)
            R_p, w_s, rho_p = \
                compute_R_p_w_s_rho_p(m_w_with_time[idx_t],
                                      m_s_with_time[idx_t], T_p)
            
            for mom_n in range(no_moments):
            
                moments_vs_time_all_seeds[seed_n, time_n, mom_n] =\
                    compute_moment_R_grid(mom_n, R_p, xi_with_time[idx_t], V0,
                                             cells_with_time[idx_t],
                                             active_ids_with_time[idx_t],
                                             id_list, no_cells)
   
#    moments_vs_time_avg = np.average(moments_vs_time_all_seeds, axis=0)
#    moments_vs_time_std = np.std(moments_vs_time_all_seeds, axis=1, ddof=1)
    
    return moments_vs_time_all_seeds, save_times_out



#%% PLOT SCALAR FIELD FRAMES EXTENDED FIELD VARIATY

def plot_scalar_field_frames_extend_avg(grid, fields_with_time,
                                        save_times,
                                        field_names,
                                        units,
                                        scales,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path=None,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        ):
    
    tick_ranges = grid.ranges
    
    no_rows = len(save_times)
    no_cols = len(field_names)
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = (4.5*no_cols, 4*no_rows))
    
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            ax = axes[time_n,field_n]
            field = fields_with_time[time_n, field_n] * scales[field_n]
            ax_title = field_names[field_n]
            unit = units[field_n]
            if ax_title in ["T","p",r"\Theta"]:
                cmap = "coolwarm"
                alpha = 1.0
            else :
#                cmap = "rainbow"
                cmap = cmap_lcpp
#                alpha = 0.8
                
            field_max = field.max()
            field_min = field.min()
            
            norm_ = mpl.colors.Normalize 
            if ax_title in ["r_r", "n_r"] and field_max > 1E-2:
                norm_ = mpl.colors.LogNorm
                field_min = 0.01
                cmap = cmap_lcpp                
                if ax_title == "r_r":
                    field_max = 1.
                elif ax_title == "n_r":
                    field_max = 10.
            else: norm_ = mpl.colors.Normalize   
            
            if ax_title == "r_c":
                field_min = 0.0
                field_max = 1.3
            if ax_title == "n_c":
                field_min = 0.0
                field_max = 150.
            if ax_title == "n_\mathrm{aero}":
                field_min = 0.0
                field_max = 150.
            if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
#                field_min = 0.
                field_min = 1.2
#                field_min = 1.5
                field_max = 20.
#                cmap = cmap_new
                # Arabas 2015
                cmap = cmap_lcpp
                
                
            oom_max = oom = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
            if oom_max > 2 or oom_max < 0:
                my_format = True
                oom_factor = 10**(-oom)
                
                field_min *= oom_factor
                field_max *= oom_factor            
            
            if oom_max ==2: str_format = "%.1f"
            else: str_format = "%.2f"
            
            if field_min/field_max < 1E-4:
#                cmap = cmap_new
                # Arabas 2015                
                cmap = cmap_lcpp
#                alpha = 0.8
            
            # REMOVE FIX APLHA HERE
#            alpha = 1.0
            
            CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor="face", zorder=1,
                                norm = norm_(vmin=field_min, vmax=field_max)
                                )
            CS.cmap.set_under("white")
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )
            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_xlabel(r'x (m)', fontsize = LFS)
            ax.set_ylabel(r'z (m)', fontsize = LFS)
            ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
                         int(save_times[time_n]/60)),
                         fontsize = TTFS)
            cbar = plt.colorbar(CS, ax=ax,
                                format=mticker.FormatStrFormatter(str_format))
            # my_format dos not work with log scale here!!
            if my_format:
                cbar.ax.text(field_min - (field_max-field_min),
                             field_max + (field_max-field_min)*0.01,
                             r'$\times\,10^{{{}}}$'.format(oom_max),
                             va='bottom', ha='left', fontsize = TKFS)
            cbar.ax.tick_params(labelsize=TKFS)
            
            if show_target_cells:
                ### ad the target cells
                no_neigh_x = no_cells_x // 2
                no_neigh_z = no_cells_z // 2
                dx = grid.steps[0]
                dz = grid.steps[1]
                
                no_tg_cells = len(target_cell_list[0])
                LW_rect = 2.
                for tg_cell_n in range(no_tg_cells):
                    x = (target_cell_list[0, tg_cell_n] - no_neigh_x - 0.1) * dx
                    z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
                    
            #        dx *= no_cells_x
            #        dz *= no_cells_z
                    
                    rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
                                         fill=False,
                                         linewidth = LW_rect,
        #                                 linestyle = "dashed",
                                         edgecolor='k',
                                         zorder = 99)        
                    ax.add_patch(rect)

    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)    
    
# the 2D field plots get shifted "to the left" by shift_cells_x cells
def plot_scalar_field_frames_extend_avg_shift(grid, fields_with_time,
                                            save_times,
                                            field_names,
                                            units,
                                            scales,
                                            solute_type,
                                            simulation_mode, # for time in label
                                            fig_path=None,
                                            no_ticks=[6,6],
                                            alpha = 1.0,
                                            TTFS = 12, LFS = 10, TKFS = 10,
                                            cbar_precision = 2,
                                            show_target_cells = False,
                                            target_cell_list = None,
                                            no_cells_x = 0,
                                            no_cells_z = 0,
                                            shift_cells_x = 0
                                            ):
    
    tick_ranges = grid.ranges
    
    no_rows = len(save_times)
    no_cols = len(field_names)
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = (4.5*no_cols, 4*no_rows))
    
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            ax = axes[time_n,field_n]
            field = fields_with_time[time_n, field_n] * scales[field_n]
            field = np.concatenate( (field[shift_cells_x:,:],
                                     field[:shift_cells_x,:]),
                                   axis = 0)
            ax_title = field_names[field_n]
            unit = units[field_n]
            if ax_title in ["T","p",r"\Theta"]:
                cmap = "coolwarm"
                alpha = 1.0
            else :
#                cmap = "rainbow"
                cmap = cmap_lcpp
#                alpha = 0.8
                
            field_max = field.max()
            field_min = field.min()
            
            norm_ = mpl.colors.Normalize 
            if ax_title in ["r_r", "n_r"] and field_max > 1E-2:
                norm_ = mpl.colors.LogNorm
                field_min = 0.01
                cmap = cmap_lcpp                
                if ax_title == "r_r":
                    field_max = 1.
                elif ax_title == "n_r":
                    field_max = 10.
            else: norm_ = mpl.colors.Normalize   
            
            if ax_title == "r_c":
                field_min = 0.0
                field_max = 1.3
            if ax_title == "n_c":
                field_min = 0.0
                field_max = 150.
            if ax_title == "n_\mathrm{aero}":
                field_min = 0.0
                field_max = 150.
            if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
#                field_min = 0.
                field_min = 1.2
#                field_min = 1.5
                field_max = 20.
#                cmap = cmap_new
                # Arabas 2015
                cmap = cmap_lcpp
                
                
            oom_max = oom = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
            if oom_max > 2 or oom_max < 0:
                my_format = True
                oom_factor = 10**(-oom)
                
                field_min *= oom_factor
                field_max *= oom_factor            
            
            if oom_max ==2: str_format = "%.1f"
            else: str_format = "%.2f"
            
            if field_min/field_max < 1E-4:
#                cmap = cmap_new
                # Arabas 2015
                cmap = cmap_lcpp
#                alpha = 0.8
            
            # REMOVE FIX APLHA HERE
#            alpha = 1.0
            
            CS = ax.pcolormesh(*grid.corners,
                               field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor="face", zorder=1,
                                norm = norm_(vmin=field_min, vmax=field_max)
                                )
            CS.cmap.set_under("white")
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )
            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_xlabel(r'x (m)', fontsize = LFS)
            ax.set_ylabel(r'z (m)', fontsize = LFS)
            ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
                         int(save_times[time_n]/60)),
                         fontsize = TTFS)
            cbar = plt.colorbar(CS, ax=ax,
                                format=mticker.FormatStrFormatter(str_format))
            # my_format dos not work with log scale here!!
            if my_format:
                cbar.ax.text(field_min - (field_max-field_min),
                             field_max + (field_max-field_min)*0.01,
                             r'$\times\,10^{{{}}}$'.format(oom_max),
                             va='bottom', ha='left', fontsize = TKFS)
            cbar.ax.tick_params(labelsize=TKFS)
            
            if show_target_cells:
                ### ad the target cells
                no_neigh_x = no_cells_x // 2
                no_neigh_z = no_cells_z // 2
                dx = grid.steps[0]
                dz = grid.steps[1]
                
                no_tg_cells = len(target_cell_list[0])
                LW_rect = 2.
                for tg_cell_n in range(no_tg_cells):
                    x = ((target_cell_list[0, tg_cell_n]-shift_cells_x) % grid.no_cells[0]
                         - no_neigh_x - 0.1) * dx
                    z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
                    
            #        dx *= no_cells_x
            #        dz *= no_cells_z
                    
                    rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
                                         fill=False,
                                         linewidth = LW_rect,
        #                                 linestyle = "dashed",
                                         edgecolor='k',
                                         zorder = 99)        
                    ax.add_patch(rect)

    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)    
    
# for one seed only for now...
# load_path_list = [[load_path0]] 
# target_cell_list = [ [tgc1], [tgc2], ... ]; tgc1 = [i1, j1]
# ind_time = [it1, it2, ..] = ind. of save times belonging to tgc1, tgc2, ...
# -> to create one cycle cf with particle trajectories

def generate_size_spectra_R_Arabas(load_path_list,
                                   ind_time,
                                   grid_mass_dry_inv,
                                   grid_no_cells,
                                   solute_type,
                                   target_cell_list,
                                   no_cells_x, no_cells_z,
                                   no_bins_R_p, no_bins_R_s):                                   

#    if solute_type == "AS":
#        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
#    elif solute_type == "NaCl":
#        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    
    no_seeds = len(load_path_list)
    no_times = len(ind_time)
    no_tg_cells = len(target_cell_list[0])
    
#    print(no_fields_orig)
#    print(no_fields_derived)
#    print(no_fields)
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
#    i_list = target_cell_list[0]
#    j_list = target_cell_list[1]    
    
    
    R_p_list = []
    R_s_list = []
    weights_list = []
    R_min_list =[]    
    R_max_list =[]    
    
    grid_r_l_list = np.zeros( (no_times, grid_no_cells[0], grid_no_cells[1]),
                        dtype = np.float64)
    
#    for tg_cell_n in no_tg_cells:
    
    for tg_cell_n in range(no_tg_cells):
        R_p_list.append([])
        R_s_list.append([])
        weights_list.append([])
        save_times_out[tg_cell_n] = grid_save_times[ind_time[tg_cell_n]]
        
    for seed_n, load_path in enumerate(load_path_list):
#        fields = load_grid_scalar_fields(load_path, grid_save_times)
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        grid_temperature_with_time = fields[:,3] #??
        
        grid_r_l_with_time = fields[:,1]
        
        vec_data, cells_with_time, scal_data,\
        xi_with_time, active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]    
        
        for tg_cell_n in range(no_tg_cells):
            target_cell = target_cell_list[:,tg_cell_n]
            idx_t = ind_time[tg_cell_n]
            
            
            id_list = np.arange(len(xi_with_time[idx_t]))
            
#            print("seed")
#            print(seed_n)
#            print("no_tg_cells, tg_cell_n")
#            print(no_tg_cells, tg_cell_n)
#            print("target_cell")
#            print(target_cell)
            
            R_p_tg, R_s_tg, xi_tg, weights_tg, no_cells_eval = \
                sample_radii_per_m_dry(m_w_with_time[idx_t],
                                       m_s_with_time[idx_t],
                                       xi_with_time[idx_t],
                                       cells_with_time[idx_t],
                                       solute_type, id_list,
                                       grid_temperature_with_time[idx_t],
                                       grid_mass_dry_inv,
                                       target_cell, no_cells_x, no_cells_z)
#            print("R_p_tg")
#            print(R_p_tg)
                
            R_p_list[tg_cell_n].append(R_p_tg)
            R_s_list[tg_cell_n].append(R_s_tg)
            weights_list[tg_cell_n].append(weights_tg)
            
            grid_r_l_list[tg_cell_n] += grid_r_l_with_time[idx_t]
    
    grid_r_l_list /= no_seeds
        
    f_R_p_list = np.zeros( (no_tg_cells, no_seeds, no_bins_R_p),
                          dtype = np.float64 )
    f_R_s_list = np.zeros( (no_tg_cells, no_seeds, no_bins_R_s),
                          dtype = np.float64 )
    bins_R_p_list = np.zeros( (no_tg_cells, no_bins_R_p+1),
                             dtype = np.float64 )
    bins_R_s_list = np.zeros( (no_tg_cells, no_bins_R_s+1),
                             dtype = np.float64 )
    
    for tg_cell_n in range(no_tg_cells):
#        R_p_tg = np.concatenate(R_p_list[tg_cell_n])
#        R_s_tg = np.concatenate(R_s_list[tg_cell_n])
#        weights_tg = np.concatenate(weights_list[tg_cell_n])
            
        R_p_min = np.amin(np.concatenate(R_p_list[tg_cell_n]))
        R_p_max = np.amax(np.concatenate(R_p_list[tg_cell_n]))
        
        R_min_list.append(R_p_min)
        R_max_list.append(R_p_max)
        
#        print("tg_cell_n, R_p_min, R_p_max")
#        print(tg_cell_n, R_p_min, R_p_max)
        
        R_s_min = np.amin(np.concatenate(R_s_list[tg_cell_n]))
        R_s_max = np.amax(np.concatenate(R_s_list[tg_cell_n]))
        
#        R_p_min = np.amin(R_p_list[tg_cell_n])
#        R_p_max = np.amax(R_p_list[tg_cell_n])
#        
#        R_s_min = np.amin(R_s_list[tg_cell_n])
#        R_s_max = np.amax(R_s_list[tg_cell_n])
        
        R_min_factor = 0.5
        R_max_factor = 2.
        
        bins_R_p = np.logspace(np.log10(R_p_min*R_min_factor),
                               np.log10(R_p_max*R_max_factor), no_bins_R_p+1 )
        
        bins_R_p_list[tg_cell_n] = np.copy(bins_R_p )
        
        bins_width_R_p = bins_R_p[1:] - bins_R_p[:-1]

        bins_R_s = np.logspace(np.log10(R_s_min*R_min_factor),
                               np.log10(R_s_max*R_max_factor), no_bins_R_s+1 )
    
        bins_width_R_s = bins_R_s[1:] - bins_R_s[:-1]

        bins_R_s_list[tg_cell_n] = np.copy(bins_R_s)
        
        for seed_n in range(no_seeds):
            R_p_tg = R_p_list[tg_cell_n][seed_n]
            R_s_tg = R_s_list[tg_cell_n][seed_n]
            weights_tg = weights_list[tg_cell_n][seed_n]
    
            h_p, b_p = np.histogram(R_p_tg, bins_R_p, weights= weights_tg)

#            f_R_p = 1E-6 * h_p / bins_width_R_p / no_tg_cells
        #            f_R_p_min = f_R_p.min()
        #            f_R_p_max = f_R_p.max()            
            
            # convert from 1/(kg*micrometer) to unit 1/(milligram * micro_meter)
            f_R_p_list[tg_cell_n, seed_n] =\
                1E-6 * h_p / bins_width_R_p / no_cells_eval
        
            h_s, b_s = np.histogram(R_s_tg, bins_R_s, weights= weights_tg)
            
#            f_R_s = 1E-6 * h_s / bins_width_R_s / no_tg_cells
#            f_R_s_min = f_R_s.min()
#            f_R_s_max = f_R_s.max()
            
            # convert from 1/(kg*micrometer) to unit 1/(milligram * micro_meter)
            f_R_s_list[tg_cell_n, seed_n] =\
                1E-6 * h_s / bins_width_R_s / no_cells_eval
    
    return f_R_p_list, f_R_s_list, bins_R_p_list, bins_R_s_list, \
           save_times_out, grid_r_l_list, R_min_list, R_max_list

#%% PLOT SIZE SPECTRA AND TRACER TRAJECTORY AND EFFECTIVE RADIUS
           
def plot_size_spectra_R_Arabas(f_R_p_list, f_R_s_list,
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
                               TTFS=12, LFS=10, TKFS=10, LW = 4.0, MS = 3.0,
                               figsize_spectra = None,
                               figsize_trace_traj = None,
                               fig_path = None,
                               show_target_cells = False,
                               fig_path_tg_cells = None,
                               fig_path_R_eff = None,
                               trajectory = None
                               ):
    
#    f_R_p_list, f_R_s_list, bins_R_p_list, bins_R_s_list, save_times_out = \
#        generate_size_spectra_R_Arabas(load_path_list,
#                                       ind_time,
#                                       grid_mass_dry_inv,
#                                       solute_type,
#                                       target_cell_list,
#                                       no_cells_x, no_cells_z,
#                                       no_bins_R_p, no_bins_R_s)  
    grid_steps = grid.steps
    no_seeds = len(f_R_p_list[0])
    no_times = len(save_times_out)
    no_tg_cells = len(target_cell_list[0])    

    f_R_p_avg = np.average(f_R_p_list, axis=1)
    f_R_p_std = np.std(f_R_p_list, axis=1, ddof=1)
    f_R_s_avg = np.average(f_R_s_list, axis=1)
    f_R_s_std = np.std(f_R_s_list, axis=1, ddof=1)
    
    
    if figsize_spectra is not None:
        figsize = figsize_spectra
    else:        
        figsize = (no_cols*5, no_rows*4)
    
    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                             figsize = figsize )
    
    plot_n = -1
    for row_n in range(no_rows):
        for col_n in range(no_cols):
            plot_n += 1
            if no_cols == 1:
                if no_rows == 1:
                    ax = axes
                else:                    
                    ax = axes[row_n]
            else:
                ax = axes[row_n, col_n]        
            
            target_cell = target_cell_list[:,plot_n]
            
            f_R_p = f_R_p_avg[plot_n]
            f_R_s = f_R_s_avg[plot_n]
            f_R_p_err = f_R_p_std[plot_n]
            f_R_s_err = f_R_s_std[plot_n]
            
            bins_R_p = bins_R_p_list[plot_n]
            bins_R_s = bins_R_s_list[plot_n]
            
            f_R_p_min = f_R_p.min()
            f_R_p_max = f_R_p.max()
            f_R_s_min = f_R_s.min()
            f_R_s_max = f_R_s.max()            
            
            ax.plot(np.repeat(bins_R_p,2),
                    np.hstack( [[f_R_p_min*1E-1],
                                np.repeat(f_R_p,2),
                                [f_R_p_min*1E-1] ] ),
                    linewidth = LW, label = "wet")
            ax.plot(np.repeat(bins_R_s,2),
                    np.hstack( [[f_R_s_min*1E-1],
                                np.repeat(f_R_s,2),
                                [f_R_s_min*1E-1] ] ),
                    linewidth = LW, label = "dry")            
    
#            ax.vlines(0.5, f_R_s_min, f_R_s_max)
            ax.axvline(0.5, c ="k", linewidth=1.0)
            ax.axvline(25., c ="k", linewidth=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim( [2E-3, 1E2] )    
            ax.set_ylim( [8E-3, 4E3] )    
#            ax.set_xlim( [2E-3, 3E2] )    
#            ax.set_ylim( [1E-5, 4E3] )    
            
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 5)
            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                           length = 3)
            
#            height = int(grid.compute_location(*target_cell,0.0,0.0)[1])
            xx = int((target_cell[0])*grid_steps[0])
            height = int((target_cell[1])*grid_steps[1])
#            xx = int((target_cell[0]+0.5)*grid_steps[1])
#            height = int((target_cell[1]+0.5)*grid_steps[1])
            ax.set_xlabel(r"particle radius ($\mathrm{\mu m}$)",
                          fontsize = LFS)
            ax.set_ylabel(r"distribution (${\mathrm{mg}}^{-1}\, {\mathrm{\mu m}}^{-1}$)",
                          fontsize = LFS)
            # ax.set_xlabel(r"particle radius ($\si{\micro m}$)", fontsize = LFS)
            # ax.set_ylabel(r"concentration ($\si{\# / cm^{3}}$)", fontsize = LFS)
            ax.set_title( f'x = {xx}, h = {height} m ' +
                         f"cell ({target_cell[0]} {target_cell[1]}) "
                            + f"Nc ({no_cells_x}, {no_cells_z}) "
                            +f"t = {save_times_out[plot_n]//60} min",
                            fontsize = TTFS )            
            ax.grid()
            ax.legend(loc='upper right')

#            ax.annotate(f"({R_min_list[plot_n]:.2e}\n{R_max_list[plot_n]:.2e})",
            ax.annotate(f"$R_{{min/max}}$\n{R_min_list[plot_n]:.2e}\n{R_max_list[plot_n]:.2e}",            
                        (12.,40.))
            
            #ax.set_ylim( [f_R_min*0.5,4E3] )
#    fig.tight_layout()
    
    if fig_path is not None:
#        fig_name =\
#            fig_path \
#            + f"spectrum_cell_list_j_from_{j_low}_to_{j_high}_" \
#            + f"Ntgcells_{no_tg_cells}_no_cells_{no_cells_x}_{no_cells_z}_" \
#            + f"Nseeds_{no_seeds}.pdf"
        fig.savefig(fig_path)

    if figsize_spectra is not None:
        figsize = figsize_spectra
    else:        
        figsize = (no_cols*5, no_rows*4)

    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                             figsize = figsize )
    
    plot_n = -1
    for row_n in range(no_rows):
        for col_n in range(no_cols):
            plot_n += 1
            if no_cols == 1:
                if no_rows == 1:
                    ax = axes
                else:                    
                    ax = axes[row_n]
            else:
                ax = axes[row_n, col_n]        
            
            target_cell = target_cell_list[:,plot_n]
            
            f_R_p = f_R_p_avg[plot_n]
            f_R_s = f_R_s_avg[plot_n]
            f_R_p_err = f_R_p_std[plot_n]
            f_R_s_err = f_R_s_std[plot_n]
            
            bins_R_p = bins_R_p_list[plot_n]
            bins_R_s = bins_R_s_list[plot_n]
            
            f_R_p_min = f_R_p.min()
            f_R_p_max = f_R_p.max()
            f_R_s_min = f_R_s.min()
            f_R_s_max = f_R_s.max()            
            
            ax.plot(np.repeat(bins_R_p,2),
                    np.hstack( [[f_R_p_min*1E-1],
                                np.repeat(f_R_p,2),
                                [f_R_p_min*1E-1] ] ),
                    linewidth = LW, label = "wet")
            ax.plot(np.repeat(bins_R_s,2),
                    np.hstack( [[f_R_s_min*1E-1],
                                np.repeat(f_R_s,2),
                                [f_R_s_min*1E-1] ] ),
                    linewidth = LW, label = "dry")            
    
#            ax.vlines(0.5, f_R_s_min, f_R_s_max)
            ax.axvline(0.5, c ="k", linewidth=1.0)
            ax.axvline(25., c ="k", linewidth=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
#            ax.set_xlim( [2E-3, 1E2] )    
#            ax.set_ylim( [8E-3, 4E3] )    
            ax.set_xlim( [2E-3, 3E2] )    
            ax.set_ylim( [1E-5, 4E3] )    
            
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 5)
            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                           length = 3)
            
#            height = int(grid.compute_location(*target_cell,0.0,0.0)[1])
            xx = int((target_cell[0])*grid_steps[0])
            height = int((target_cell[1])*grid_steps[1])
#            xx = int((target_cell[0]+0.5)*grid_steps[1])
#            height = int((target_cell[1]+0.5)*grid_steps[1])
            ax.set_xlabel(r"particle radius ($\mathrm{\mu m}$)",
                          fontsize = LFS)
            ax.set_ylabel(r"distribution (${\mathrm{mg}}^{-1}\, {\mathrm{\mu m}}^{-1}$)",
                          fontsize = LFS)
            # ax.set_xlabel(r"particle radius ($\si{\micro m}$)", fontsize = LFS)
            # ax.set_ylabel(r"concentration ($\si{\# / cm^{3}}$)", fontsize = LFS)
            ax.set_title( f'x = {xx}, h = {height} m ' +
                         f"cell ({target_cell[0]} {target_cell[1]}) "
                            + f"Nc ({no_cells_x}, {no_cells_z}) "
                            +f"t = {save_times_out[plot_n]//60} min",
                            fontsize = TTFS )            
            ax.grid()
            ax.legend(loc='upper right')
            
            ax.annotate(f"$R_{{min/max}}$\n{R_min_list[plot_n]:.2e}\n{R_max_list[plot_n]:.2e}",
                        (25.,6.))
            
            #ax.set_ylim( [f_R_min*0.5,4E3] )
#    fig.tight_layout()
    
    if fig_path is not None:
#        fig_name =\
#            fig_path \
#            + f"spectrum_cell_list_j_from_{j_low}_to_{j_high}_" \
#            + f"Ntgcells_{no_tg_cells}_no_cells_{no_cells_x}_{no_cells_z}_" \
#            + f"Nseeds_{no_seeds}.pdf"
        fig.savefig(fig_path[:-4] + "_ext.pdf")
    
    ### CALC EFFECTIVE RADIUS = MOMENT3/MOMENT2 FROM ANALYSIS OF f_R
    R_eff_list = np.zeros((no_tg_cells, 3), dtype = np.float64)
    no_rows = 1
    no_cols = 1      
    fig3, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                 figsize = (no_cols*6, no_rows*5) )
    for tg_cell_n in range(no_tg_cells):
        bins_log = np.log(bins_R_p_list[tg_cell_n])
        bins_width = bins_R_p_list[tg_cell_n,1:] - bins_R_p_list[tg_cell_n,:-1]
        bins_width_log = bins_log[1:] - bins_log[:-1]
        bins_center_log = 0.5 * (bins_log[1:] + bins_log[:-1])
        bins_center_log_lin = np.exp(bins_center_log)
        
        mom0 = np.sum( f_R_p_list[tg_cell_n] * bins_width)
        mom1 = np.sum( f_R_p_list[tg_cell_n] * bins_width
                       * bins_center_log_lin)
        mom2 = np.sum( f_R_p_list[tg_cell_n] * bins_width
                       * bins_center_log_lin * bins_center_log_lin)
        mom3 = np.sum( f_R_p_list[tg_cell_n] * bins_width
                       * bins_center_log_lin**3)
#        R_eff0 = mom1/mom0
#        R_eff1 = mom2/mom1
#        R_eff2 = mom3/mom2
        
        R_eff_list[tg_cell_n,0] = mom1/mom0
        R_eff_list[tg_cell_n,1] = mom2/mom1
        R_eff_list[tg_cell_n,2] = mom3/mom2
    
    for mom_n in range(3):        
        axes.plot( np.arange(1,no_tg_cells+1), R_eff_list[:,mom_n], "o",
                  label = f"mom {mom_n}")
        if mom_n == 2:
            for tg_cell_n in range(no_tg_cells):
                axes.annotate(f"({target_cell_list[0,tg_cell_n]} "
                              + f" {target_cell_list[1,tg_cell_n]})",
                              (tg_cell_n+1, 3. + R_eff_list[tg_cell_n,mom_n]),
                              fontsize=8)
    axes.legend()        
    if fig_path is not None:
        fig3.savefig(fig_path_R_eff)            
    
    ########################################## PLOT TARGET CELLS IN EXTRA PLOT
    ########################################## PLOT TARGET CELLS IN EXTRA PLOT
    
    if show_target_cells:
        grid_r_l = grid_r_l_list[0]*1E3
        no_rows = 1
        no_cols = 1  
#        cmap = cmap_new
        # arabas 2015
        cmap = cmap_lcpp
#        cmap = "my_rainbow"
#        alpha = 0.7
        alpha = 1.0
        no_ticks = [6,6]
        str_format = "%.2f"
        
        tick_ranges = grid.ranges
        
        if figsize_trace_traj is not None:
            figsize = figsize_trace_traj
        else:
            figsize = (no_cols*9, no_rows*8)            
        
        fig2, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                         figsize = figsize )
        ax = axes
        
        field_min = 0.0
        field_max = grid_r_l.max()
        CS = ax.pcolormesh(*grid.corners, grid_r_l,
                           cmap=cmap, alpha=alpha,
                            edgecolor="face", zorder=1,
                            vmin=field_min, vmax=field_max,
                            antialiased=True, linewidth=0.0
#                            norm = norm_(vmin=field_min, vmax=field_max)
                            )
        CS.cmap.set_under("white")
        
#        ax.axis("equal", "box")

        ax.set_xticks( np.linspace( tick_ranges[0,0],
                                         tick_ranges[0,1],
                                         no_ticks[0] ) )
        ax.set_yticks( np.linspace( tick_ranges[1,0],
                                         tick_ranges[1,1],
                                         no_ticks[1] ) )
#        ax.set_xlim()

#        ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
#                     int(save_times[time_n]/60)),
#                     fontsize = TTFS)
#        cax = fig2.add_axes([ax.get_position().x1+0.01,
#                            ax.get_position().y0,0.02,
#                            ax.get_position().height])
        cbar = plt.colorbar(CS, ax=ax, fraction =.045,
                            format=mticker.FormatStrFormatter(str_format))    
        
        ### add vel. field
        ARROW_WIDTH= 0.004
        ARROW_SCALE = 20.0
        no_arrows_u=16
        no_arrows_v=16
        if no_arrows_u < grid.no_cells[0]:
            arrow_every_x = grid.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < grid.no_cells[1]:
            arrow_every_y = grid.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1
        centered_u_field = ( grid.velocity[0][0:-1,0:-1]\
                             + grid.velocity[0][1:,0:-1] ) * 0.5
        centered_w_field = ( grid.velocity[1][0:-1,0:-1]\
                             + grid.velocity[1][0:-1,1:] ) * 0.5
        ax.quiver(
            grid.centers[0][arrow_every_y//2::arrow_every_y,
                        arrow_every_x//2::arrow_every_x],
            grid.centers[1][arrow_every_y//2::arrow_every_y,
                        arrow_every_x//2::arrow_every_x],
            centered_u_field[::arrow_every_y,::arrow_every_x],
            centered_w_field[::arrow_every_y,::arrow_every_x],
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )                            
        
        ### ad the target cells
        no_neigh_x = no_cells_x // 2
        no_neigh_z = no_cells_z // 2
        dx = grid_steps[0]
        dz = grid_steps[1]
        
        LW_rect = 0.8
        
        for tg_cell_n in range(no_tg_cells):
            x = (target_cell_list[0, tg_cell_n] - no_neigh_x - 0.1) * dx
            z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
            
    #        dx *= no_cells_x
    #        dz *= no_cells_z
            
            rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z, fill=False,
                                 linewidth = LW_rect,
#                                 linestyle = "dashed",
                                 edgecolor='k',
                                 zorder = 99)        
            ax.add_patch(rect)
        
#        MS = 2
        if trajectory is not None:
#            print("should plot here")
            ax.plot(trajectory[:,0],trajectory[:,1],"o",
                    markersize = MS, c="k")
        
        ax.tick_params(axis='both', which='major', labelsize=TKFS)
        ax.grid(color='gray', linestyle='dashed', zorder = 2)
        ax.set_title(r"$r_l$ (g/kg), t = {} min".format(-120 + save_times_out[0]//60))
        ax.set_xlabel(r'x (m)', fontsize = LFS)
        ax.set_ylabel(r'z (m)', fontsize = LFS)  
#        ax.axis("equal")
#        ax.axis("scaled")
#        ax.set(xlim=grid.ranges[0], ylim=grid.ranges[1])          
        ax.set_xlim((0,1500))
        ax.set_ylim((0,1500))
        ax.set_aspect('equal')
#        ax.set_aspect('equal', 'box')
        
#        fig2.tight_layout(pad = 2.)
#        plt.subplots_adjust(top=1, bottom=1)
        if fig_path_tg_cells is not None:
            fig2.savefig(fig_path_tg_cells,
        #                    bbox_inches = 0,
                        bbox_inches = 'tight',
                        pad_inches = 0.04
                        )   
        
        
        
        