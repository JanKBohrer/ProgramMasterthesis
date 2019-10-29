#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:31:53 2019

@author: jdesk
"""

#%% IMPORTS

import math
import numpy as np
#from numba import njit
import matplotlib.pyplot as plt

#import kernel
#from kernel import compute_kernel_Long_Bott_m, compute_kernel_hydro, \
#                   compute_E_col_Long_Bott
from .kernel import update_velocity_Beard
from microphysics import compute_radius_from_mass_jit
from microphysics import compute_radius_from_mass_vec

from .AON import collision_step_Golovin
from .AON import collision_step_Long_Bott_m
from .AON import collision_step_Ecol_grid_R
#from .AON import collision_step_Long_Bott_Ecol_grid_R
from .AON import collision_step_Long_Bott_kernel_grid_m

#%% DEFINITIONS

### SIMULATION
def simulate_collisions(SIP_quantities,
                        kernel_quantities, kernel_name, kernel_method,
                        dV, dt, t_end, dt_save, no_cols, seed, save_dir):
    if kernel_name == "Long_Bott":
        if kernel_method == "Ecol_grid_R":
            collision_step = collision_step_Ecol_grid_R
            (xis, masses, radii, vel, mass_densities) = SIP_quantities
            (E_col_grid, no_kernel_bins, R_kernel_low_log, bin_factor_R_log) =\
                kernel_quantities
        elif kernel_method == "kernel_grid_m":
            collision_step = collision_step_Long_Bott_kernel_grid_m
            (xis, masses) = SIP_quantities
            (kernel_grid, no_kernel_bins, m_kernel_low_log, bin_factor_m_log)=\
                kernel_quantities
        elif kernel_method == "analytic":
            collision_step = collision_step_Long_Bott_m
            (xis, masses, mass_density) = SIP_quantities

    if kernel_name == "Hall_Bott":
        if kernel_method == "Ecol_grid_R":
            collision_step = collision_step_Ecol_grid_R
            (xis, masses, radii, vel, mass_densities) = SIP_quantities
            (E_col_grid, no_kernel_bins, R_kernel_low_log, bin_factor_R_log) =\
                kernel_quantities
                
    if kernel_name == "Golovin":
        collision_step = collision_step_Golovin
        (xis, masses) = SIP_quantities
        
    np.random.seed(seed)
    # save_path = save_dir + f"seed_{seed}/"
    # t = 0.0
    no_SIPs = xis.shape[0]
    no_steps = int(math.ceil(t_end/dt))
    # save data at t=0, every dt_save and at the end
    no_saves = int(t_end/dt_save - 0.001) + 2
    dn_save = int(math.ceil(dt_save/dt))
    
    dt_over_dV = dt/dV
    
    xis_vs_time = np.zeros((no_saves,no_SIPs), dtype=np.float64)
    masses_vs_time = np.zeros((no_saves,no_SIPs), dtype=np.float64)
    save_times = np.zeros(no_saves)
    # step_n = 0
    save_n = 0
    if kernel_method == "Ecol_grid_R":
        for step_n in range(no_steps):
            if step_n % dn_save == 0:
                t = step_n * dt
                xis_vs_time[save_n] = np.copy(xis)
                masses_vs_time[save_n] = np.copy(masses)
                save_times[save_n] = t
                save_n += 1
            # for box model: calc. velocity from terminal vel.
            # in general: vel given from dynamic simulation
            collision_step(xis, masses, radii, vel, mass_densities,
                           dt_over_dV, E_col_grid, no_kernel_bins,
                           R_kernel_low_log, bin_factor_R_log, no_cols)
            update_velocity_Beard(vel,radii)
    elif kernel_method == "kernel_grid_m":
        for step_n in range(no_steps):
            if step_n % dn_save == 0:
                t = step_n * dt
                xis_vs_time[save_n] = np.copy(xis)
                masses_vs_time[save_n] = np.copy(masses)
                save_times[save_n] = t
                save_n += 1
            collision_step(xis, masses, dt_over_dV,
                           kernel_grid, no_kernel_bins,
                           m_kernel_low_log, bin_factor_m_log, no_cols)
    elif kernel_method == "analytic":           
        if kernel_name == "Golovin":
            for step_n in range(no_steps):
                if step_n % dn_save == 0:
                    t = step_n * dt
                    xis_vs_time[save_n] = np.copy(xis)
                    masses_vs_time[save_n] = np.copy(masses)
                    save_times[save_n] = t
                    save_n += 1
                collision_step(xis, masses, dt_over_dV, no_cols)
        elif kernel_name == "Long_Bott":
            for step_n in range(no_steps):
                if step_n % dn_save == 0:
                    t = step_n * dt
                    xis_vs_time[save_n] = np.copy(xis)
                    masses_vs_time[save_n] = np.copy(masses)
                    save_times[save_n] = t
                    save_n += 1
                collision_step(xis, masses, mass_density, dt_over_dV, no_cols)
    
    t = no_steps * dt
#    t = (step_n+1) * dt
    xis_vs_time[save_n] = np.copy(xis)
    masses_vs_time[save_n] = np.copy(masses)
    save_times[save_n] = t
    np.save(save_dir + f"xis_vs_time_{seed}", xis_vs_time)
    np.save(save_dir + f"masses_vs_time_{seed}", masses_vs_time)
    np.save(save_dir + f"save_times_{seed}", save_times)
    
### ANALYSIS OF SIM DATA

# for given kappa:
# the simulation yields masses in unit 1E-18 kg
# to compare moments etc with Unterstrasser, masses are converted to kg
def analyze_sim_data(kappa, mass_density, dV, no_sims, start_seed, no_bins, load_dir):
    # f"/mnt/D/sim_data/col_box_mod/results/{kernel_name}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/"
    # f"/mnt/D/sim_data/col_box_mod/results/{kernel_name}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/perm/"
    
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    
    seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
    
    masses_vs_time = []
    xis_vs_time = []
    for seed in seed_list:
        # convert to kg
        masses_vs_time.append(1E-18*np.load(load_dir + f"masses_vs_time_{seed}.npy"))
#        masses_vs_time.append(np.load(load_dir + f"masses_vs_time_{seed}.npy"))
        xis_vs_time.append(np.load(load_dir + f"xis_vs_time_{seed}.npy"))
    
    masses_vs_time_T = []
    xis_vs_time_T = []
    
    no_times = len(save_times)
    
    for time_n in range(no_times):
        masses_ = []
        xis_ = []
        for i,m in enumerate(masses_vs_time):
            masses_.append(m[time_n])
            xis_.append(xis_vs_time[i][time_n])
        masses_vs_time_T.append(masses_)
        xis_vs_time_T.append(xis_)
    
    f_m_num_avg_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
    f_m_num_std_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
    g_m_num_avg_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
    g_m_num_std_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
    g_ln_r_num_avg_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
    g_ln_r_num_std_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
    
    bins_mass_vs_time = np.zeros((no_times,no_bins+1),dtype=np.float64)
    bins_mass_width_vs_time = np.zeros((no_times,no_bins),dtype=np.float64)
    bins_rad_width_log_vs_time = np.zeros((no_times,no_bins),dtype=np.float64)
    bins_mass_centers = []
    bins_rad_centers = []
    
    m_max_vs_time = np.zeros(no_times,dtype=np.float64)
    m_min_vs_time = np.zeros(no_times,dtype=np.float64)
    bin_factors_vs_time = np.zeros(no_times,dtype=np.float64)
    
    moments_vs_time = np.zeros((no_times,4,no_sims),dtype=np.float64)
    
    last_bin_factor = 1.0
    # last_bin_factor = 1.5
    first_bin_factor = 1.0
    # first_bin_factor = 0.8
    for time_n, masses in enumerate(masses_vs_time_T):
        xis = xis_vs_time_T[time_n]
        masses_sampled = np.concatenate(masses)
        xis_sampled = np.concatenate(xis)
        # print(time_n, xis_sampled.min(), xis_sampled.max())
    
        m_min = masses_sampled.min()
        m_max = masses_sampled.max()
        
        # convert to microns
        R_min = compute_radius_from_mass_jit(1E18*m_min, mass_density)
        R_max = compute_radius_from_mass_jit(1E18*m_max, mass_density)

        xi_min = xis_sampled.min()
        xi_max = xis_sampled.max()

        print(kappa, time_n, f"{xi_max/xi_min:.3e}",
              xis_sampled.shape[0]/no_sims, R_min, R_max)
    
        m_min_vs_time[time_n] = m_min
        m_max_vs_time[time_n] = m_max
    
        bin_factor = (m_max/m_min)**(1.0/no_bins)
        bin_factors_vs_time[time_n] = bin_factor
        # bin_log_dist = np.log(bin_factor)
        # bin_log_dist_half = 0.5 * bin_log_dist
        # add dummy bins for overflow
        # bins_mass = np.zeros(no_bins+3,dtype=np.float64)
        bins_mass = np.zeros(no_bins+1,dtype=np.float64)
        bins_mass[0] = m_min
        # bins_mass[0] = m_min / bin_factor
        for bin_n in range(1,no_bins+1):
            bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
        # the factor 1.01 is for numerical stability: to be sure
        # that m_max does not contribute to a bin larger than the
        # last bin
        # bins_mass[-1] *= 1.0001
        bins_mass[-1] *= last_bin_factor
        # the factor 0.99 is for numerical stability: to be sure
        # that m_min does not contribute to a bin smaller than the
        # 0-th bin
        # bins_mass[0] *= 0.9999
        bins_mass[0] *= first_bin_factor
        # m_0 = m_min / np.sqrt(bin_factor)
        # bins_mass_log = np.log(bins_mass)
    
        bins_mass_vs_time[time_n] = bins_mass
        # convert to microns
        bins_rad = compute_radius_from_mass_vec(1E18*bins_mass, mass_density)
        bins_mass_log = np.log(bins_mass)
        bins_rad_log = np.log(bins_rad)
    
        bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
        bins_rad_width = (bins_rad[1:]-bins_rad[:-1])
        bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])
        bins_mass_width_vs_time[time_n] = bins_mass_width
        bins_rad_width_log_vs_time[time_n] = bins_rad_width_log
    
        f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
    
        # define centers on lin scale
        bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
        bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
    
        # define centers on the logarithmic scale
        bins_mass_center_log = np.exp(0.5 * (bins_mass_log[:-1] + bins_mass_log[1:]))
        bins_rad_center_log = np.exp(0.5 * (bins_rad_log[:-1] + bins_rad_log[1:]))
    
        # bins_mass are not equally spaced on log scale because of scaling
        # of the first and last bin
        # bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
        # bins_rad_center_log = bins_rad[:-1] * np.sqrt(bin_factor)
    
        # bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
        # bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
    
        # define the center of mass for each bin and set it as the "bin center"
        # bins_mass_center_COM = g_m_num_sampled/f_m_num_sampled
        # bins_rad_center_COM =\
        #     compute_radius_from_mass(bins_mass_center_COM*1.0E18,
        #                              c.mass_density_water_liquid_NTP)
    
        # set the bin "mass centers" at the right spot such that
        # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
        m_avg = masses_sampled.sum() / xis_sampled.sum()
        bins_mass_center_exact = bins_mass[:-1] \
                                 + m_avg * np.log(bins_mass_width\
              / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
        # convert to microns
        bins_rad_center_exact =\
            compute_radius_from_mass_vec(1E18*bins_mass_center_exact,
                                         mass_density)
        bins_mass_centers.append( np.array((bins_mass_center_lin,
                                  bins_mass_center_log,
                                  bins_mass_center_exact)) )
    
        bins_rad_centers.append( np.array((bins_rad_center_lin,
                                 bins_rad_center_log,
                                 bins_rad_center_exact)) )
    
        ### STATISTICAL ANALYSIS OVER no_sim runs
        # get f(m_i) curve for each "run" with same bins for all ensembles
        f_m_num = []
        g_m_num = []
        g_ln_r_num = []
    
    
        for sim_n,mass in enumerate(masses):
            # convert to microns
            rad = compute_radius_from_mass_vec(1E18*mass, mass_density)
            f_m_num.append(np.histogram(mass, bins_mass, weights=xis[sim_n])[0] \
                           / (bins_mass_width * dV))
            g_m_num.append(np.histogram(mass, bins_mass, weights=xis[sim_n]*mass)[0] \
                           / (bins_mass_width * dV))
    
            # build g_ln_r = 3*m*g_m DIRECTLY from data
            g_ln_r_num.append( np.histogram(rad, bins_rad,
                                            weights=xis[sim_n]*mass)[0] \
                               / (bins_rad_width_log * dV) )
    
            moments_vs_time[time_n,0,sim_n] = xis[sim_n].sum() / dV
            for n in range(1,4):
                moments_vs_time[time_n,n,sim_n] = np.sum(xis[sim_n]*mass**n) / dV
    
        # f_m_num = np.array(f_m_num)
        # g_m_num = np.array(g_m_num)
        # g_ln_r_num = np.array(g_ln_r_num)
        
        f_m_num_avg_vs_time[time_n] = np.average(f_m_num, axis=0)
        f_m_num_std_vs_time[time_n] = \
            np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
        g_m_num_avg_vs_time[time_n] = np.average(g_m_num, axis=0)
        g_m_num_std_vs_time[time_n] = \
            np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
        g_ln_r_num_avg_vs_time[time_n] = np.average(g_ln_r_num, axis=0)
        g_ln_r_num_std_vs_time[time_n] = \
            np.std(g_ln_r_num, axis=0, ddof=1) / np.sqrt(no_sims)
    # convert to microns
    R_min_vs_time = compute_radius_from_mass_vec(1E18*m_min_vs_time,
                                                 mass_density)
    R_max_vs_time = compute_radius_from_mass_vec(1E18*m_max_vs_time,
                                                 mass_density)
    
    moments_vs_time_avg = np.average(moments_vs_time, axis=2)
    moments_vs_time_std = np.std(moments_vs_time, axis=2, ddof=1) \
                          / np.sqrt(no_sims)
    
    moments_vs_time_Unt = np.zeros_like(moments_vs_time_avg)
    
    # mom_fac = math.log(10)/(3*kappa)
    for time_n in range(no_times):
        for n in range(4):
            moments_vs_time_Unt[time_n,n] =\
                math.log(bin_factors_vs_time[time_n]) / 3.0 \
                * np.sum( g_ln_r_num_avg_vs_time[time_n]
                                    * (bins_mass_centers[time_n][1])**(n-1) )
                # np.sum( g_ln_r_num_avg_vs_time[time_n]
                #         * (bins_mass_centers[time_n][1])**(n-1)
                #         * bins_rad_width_log_vs_time[time_n] )
                # mom_fac * np.sum( g_m_num_avg_vs_time[time_n]
                #                    * (bins_mass_centers[time_n][1])**(n-1) )

    np.save(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy", moments_vs_time_avg)
    np.save(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy", moments_vs_time_std)
    np.save(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", f_m_num_avg_vs_time)
    np.save(load_dir + f"f_m_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", f_m_num_std_vs_time)
    np.save(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_m_num_avg_vs_time)
    np.save(load_dir + f"g_m_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_m_num_std_vs_time)
    np.save(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_ln_r_num_avg_vs_time)
    np.save(load_dir + f"g_ln_r_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_ln_r_num_std_vs_time)
    np.save(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy", bins_mass_centers)
    np.save(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy", bins_rad_centers)    


# PLOTTING
def plot_for_given_kappa(kappa, eta, dt, no_sims, start_seed, no_bins,
                         kernel_name, gen_method, bin_method, load_dir):
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    no_times = len(save_times)
    # bins_mass_centers =
    bins_mass_centers = np.load(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy")
    bins_rad_centers = np.load(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy")

    f_m_num_avg_vs_time = np.load(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_m_num_avg_vs_time = np.load(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_ln_r_num_avg_vs_time = np.load(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")

    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")

    fig_name = "fm_gm_glnr_vs_t"
    fig_name += f"_kappa_{kappa}_dt_{int(dt)}_no_sims_{no_sims}_no_bins_{no_bins}.png"
    no_rows = 3
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,8*no_rows))
    # ax.loglog(radii, xis, "x")
    # ax.loglog(bins_mid[:51], H, "x-")
    # ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
    ax = axes[0]
    for time_n in range(no_times):
        ax.plot(bins_mass_centers[time_n][0], f_m_num_avg_vs_time[time_n])
    # ax.plot(bins_mass_centers[0][0], f_m_num_avg_vs_time[0], "x")
    # ax.plot(m_, f_m_ana_)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mass (kg)")
    ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$")
    if kernel_name == "Golovin":
        ax.set_xticks( np.logspace(-15,-5,11) )
        ax.set_yticks( np.logspace(5,21,17) )
    ax.grid()
    
    ax = axes[1]
    for time_n in range(no_times):
        ax.plot(bins_mass_centers[time_n][0], g_m_num_avg_vs_time[time_n])
    # ax.plot(bins_mass_centers[0][0], g_m_num_avg_vs_time[0], "x")
    # ax.plot(m_, g_m_ana_)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mass (kg)")
    ax.set_ylabel(r"$g_m$ $\mathrm{(m^{-3})}$")
    if kernel_name == "Golovin":
        ax.set_xticks( np.logspace(-15,-5,11) )
        ax.set_yticks( np.logspace(-2,9,12) )
    ax.grid()
    
    ax = axes[2]
    for time_n in range(no_times):
        ax.plot(bins_rad_centers[time_n][0], g_ln_r_num_avg_vs_time[time_n]*1000.0)
    # ax.plot(bins_rad_centers[0][0], g_ln_r_num_avg_vs_time[0]*1000.0, "x")
    # ax.plot(R_, g_ln_r_ana_)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("radius $\mathrm{(\mu m)}$")
    ax.set_ylabel(r"$g_{\ln(r)}$ $\mathrm{(g \; m^{-3})}$")
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
    ax.grid(which="major")
    
    for ax in axes:
        ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    
    # ax = axes[4]
    # for n in range(4):
    #     ax.plot(n*np.ones_like(moments_sampled[n]),
    #             moments_sampled[n]/moments_an[n], "o")
    # ax.errorbar(np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
    #             fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0,
    #             capsize=10, elinewidth=5, markeredgewidth=2,
    #             zorder=99)
    # ax.plot(np.arange(4), np.ones_like(np.arange(4)))
    # ax.xaxis.set_ticks([0,1,2,3])
    # ax.set_xlabel("$k$")
    # # ax.set_ylabel(r"($k$-th moment of $f_m$)/(analytic value)")
    # ax.set_ylabel(r"$\lambda_k / \lambda_{k,analytic}$")

    fig.suptitle(
f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
gen_method={gen_method}, kernel={kernel_name}, bin_method={bin_method}")
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(load_dir + fig_name)
    
    ### PLOT MOMENTS VS TIME
    t_Unt = [0,10,20,30,35,40,50,55,60]
    lam0_Unt = [2.97E8, 2.92E8, 2.82E8, 2.67E8, 2.1E8, 1.4E8,  1.4E7, 4.0E6, 1.2E6]
    t_Unt2 = [0,10,20,30,40,50,60]
    lam2_Unt = [8.0E-15, 9.0E-15, 9.5E-15, 6E-13, 2E-10, 7E-9, 2.5E-8]
    
    fig_name = "moments_vs_time"
    fig_name += f"_kappa_{kappa}_dt_{int(dt)}_no_sims_{no_sims}_no_bins_{no_bins}.png"
    no_rows = 4
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
    for i,ax in enumerate(axes):
        ax.plot(save_times/60, moments_vs_time_avg[:,i],"x-")
        if i != 1:
            ax.set_yscale("log")
        ax.grid()
        ax.set_xticks(save_times/60)
        ax.set_xlim([save_times[0]/60, save_times[-1]/60])
        # ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
        #                  bottom=True, top=False, left=False, right=False)
        ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    if kernel_name == "Golovin":
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
        axes[2].set_yticks( np.logspace(-15,-9,7) )
        axes[3].set_yticks( np.logspace(-26,-15,12) )
    
    axes[0].plot(t_Unt,lam0_Unt, "o")
    axes[2].plot(t_Unt2,lam2_Unt, "o")
    fig.suptitle(
f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
gen_method={gen_method}, kernel={kernel_name}, bin_method={bin_method}")
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(load_dir + fig_name)
    plt.close("all")

#################################################################
### PLOT MOMENTS VS TIME for several kappa

# TTFS, LFS, TKFS: title, labels, ticks font size
#def plot_moments_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
#                           kernel_name, gen_method,
#                           dist, start_seed, ref_data_path, sim_data_path,
#                           result_path_add,
#                           fig_dir, TTFS, LFS, TKFS):
def plot_moments_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
                           kernel_name, gen_method,
                           dist, start_seed,
                           moments_ref, times_ref,
                           sim_data_path,
                           result_path_add,
                           fig_dir, TTFS, LFS, TKFS):
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
    
    fig_name = f"moments_vs_time_kappa_var_{no_kappas}"
    # fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.png"
#    fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.png"
    fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.pdf"
    no_rows = 4
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows), sharex=True)
    
#    mom0_last_time = np.zeros(len(kappa_list),dtype=np.float64)
    
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#        moments_vs_time_avg[:,1] *= 1.0E3
#        mom0_last_time[kappa_n] = moments_vs_time_avg[-1,0]
        
        if kappa_n < 10: fmt = "x-"
        else: fmt = "x--"            
        
        for i,ax in enumerate(axes):
            ax.plot(save_times/60, moments_vs_time_avg[:,i],fmt,label=f"{kappa}")

    for i,ax in enumerate(axes):
        if kernel_name == "Long_Bott" or kernel_name == "Hall_Bott":
            ax.plot(times_ref/60, moments_ref[i],
                    "o", c = "k",fillstyle='none', markersize = 8, mew=1.0, label="Wang")
        if i != 1:
            ax.set_yscale("log")
        ax.grid()
        # if i ==0: ax.legend()
        if i != 1:
            ax.legend(fontsize=TKFS)
        if i == 1:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.05),
                      fontsize=TKFS)
        ax.set_xticks(save_times/60)
        ax.set_xlim([save_times[0]/60, save_times[-1]/60])
        # ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
        #                  bottom=True, top=False, left=False, right=False)
        ax.tick_params(which="both", bottom=True, top=True,
                       left=True, right=True
                       )
        ax.tick_params(axis='both', which='major', labelsize=TKFS,
                       width=2, size=10)
        ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                       width=1, size=6)
    axes[-1].set_xlabel("time (min)",fontsize=LFS)
    axes[0].set_ylabel(r"$\lambda_0$ = DNC $(\mathrm{m^{-3}})$ ",
                       fontsize=LFS)
    axes[1].set_ylabel(r"$\lambda_1$ = LWC $(\mathrm{kg \, m^{-3}})$ ",
                       fontsize=LFS)
    axes[2].set_ylabel(r"$\lambda_2$ $(\mathrm{kg^2 \, m^{-3}})$ ",
                       fontsize=LFS)
    axes[3].set_ylabel(r"$\lambda_3$ $(\mathrm{kg^3 \, m^{-3}})$ ",
                       fontsize=LFS)
    if kernel_name == "Golovin":
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
        axes[2].set_yticks( np.logspace(-15,-9,7) )
        axes[3].set_yticks( np.logspace(-26,-15,12) )
    elif kernel_name == "Long_Bott":
        # axes[0].set_yticks([1.0E6,1.0E7,1.0E8,3.0E8,4.0E8])
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8])
        axes[0].set_ylim([1.0E6,4.0E8])
        # axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        axes[2].set_yticks( np.logspace(-15,-7,9) )
        axes[2].set_ylim([1.0E-15,1.0E-7])
        axes[3].set_yticks( np.logspace(-26,-11,16) )
        axes[3].set_ylim([1.0E-26,1.0E-11])
    # axes[0].plot(t_Unt,lam0_Unt, "o", c = "k")
    # axes[2].plot(t_Unt2,lam2_Unt, "o", c = "k")
        
#    for mom0_last in mom0_last_time:
#    print(mom0_last_time/mom0_last_time.min())
#    if len(mom0_last_time) >= 3:
#        print(mom0_last_time/mom0_last_time[-2])
#        print(mom0_last_time_Unt/mom0_last_time_Unt.min())
#        print()
    title=\
f"Moments of the distribution for various $\kappa$ (see legend)\n\
dt={dt:.1e}, eta={eta:.0e}, r_critmin=0.6, no_sims={no_sims}, \
gen_method={gen_method}, kernel={kernel_name}"
#     title=\
# f"Moments of the distribution for various $\kappa$ (see legend)\n\
# dt={dt}, eta={eta:.0e}, no_sims={no_sims}, \
# gen_method={gen_method}, kernel=LONG"
    fig.suptitle(title, fontsize=TTFS, y = 0.997)
    fig.tight_layout()
    # fig.subplots_adjust()
    plt.subplots_adjust(top=0.965)
    fig.savefig(fig_dir + fig_name)
    plt.close("all")