#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 07:26:11 2019

@author: jdesk
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color, LinearSegmentedColormap
import matplotlib.ticker as mticker

import os
import math
import numpy as np
# import os
# from datetime import datetime
# import timeit

from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from microphysics import compute_mass_from_radius_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

import constants as c

from plotting import cm2inch
#from plotting import generate_rcParams_dict
#from plotting import pgf_dict, pdf_dict
#plt.rcParams.update(pgf_dict)
#plt.rcParams.update(pdf_dict)

from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

from analysis import sample_masses, sample_radii
from analysis import sample_masses_per_m_dry , sample_radii_per_m_dry
from analysis import plot_size_spectra_R_Arabas, generate_size_spectra_R_Arabas

                          
                          
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
rgb_colors = [hex2color(cl) + tuple([1.0]) for cl in hex_colors]
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

#%% FUNCTION DEF POSITIONS AND VELOCITIES

def plot_pos_vel_pt_MA(pos, vel, grid,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARROWSCALE=2, ARROWWIDTH=0.02,
                    HEADW=1, HEADL=1.5,
                    fig_name=None):
    # u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
    # v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
    
    scale_x = 1E-3
    
    pos = pos*scale_x
    fig, ax = plt.subplots(figsize=figsize)
#    ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
#    ax.plot(pos[0],pos[1], "o", color="k", markersize=MS)
    ax.quiver(*pos, *vel, scale=ARROWSCALE, width=ARROWWIDTH,
              headwidth = HEADW, headlength = HEADL, pivot="mid",
#              rasterized=True
              )
    # ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
              # scale=ARRSCALE, pivot="mid", color="red")
    # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
    #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
    #           scale=0.5, pivot="mid", color="red")
    # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
    #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
    #           scale=0.5, pivot="mid", color="blue")
    
    x_min = grid.ranges[0,0] * scale_x
    x_max = grid.ranges[0,1] * scale_x
    y_min = grid.ranges[1,0] * scale_x
    y_max = grid.ranges[1,1] * scale_x
    ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
    ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
    # ax.set_xticks(grid.corners[0][:,0])
    # ax.set_yticks(grid.corners[1][0,:])
    ax.set_xticks(grid.corners[0][:,0]*scale_x, minor = True)
    ax.set_yticks(grid.corners[1][0,:]*scale_x, minor = True)
    ax.set_xlim((0,1.5))
    ax.set_ylim((0,1.5))
    ax.set_aspect('equal')  
    ax.set_xlabel(r'$x$ (km)',
#                  fontsize = LFS
                  )
    ax.set_ylabel(r'$z$ (km)',
#                  fontsize = LFS
                  )      
    # plt.minorticks_off()
    # plt.minorticks_on()
#    ax.grid()
#    fig.tight_layout()
    # ax.grid(which="minor")
    # plt.show()
    if fig_name is not None:
        fig.savefig(fig_name,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.05, dpi=300
                    )           
#        fig.savefig(fig_name)
    plt.close("all")
#%% FUNCTION DEF SIZE SPECTRA

def plot_size_spectra_R_Arabas_MA(f_R_p_list, f_R_s_list,
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
                               t_spin_up,
                               SIM_N = None,
                               TTFS=12, LFS=10, TKFS=10, LW = 4.0, MS = 3.0,
                               figsize_spectra = None,
                               figsize_trace_traj = None,
                               fig_path = None,
                               show_target_cells = False,
                               fig_path_tg_cells = None,
                               fig_path_R_eff = None,
                               trajectory = None,
                               show_textbox = True,
                               xshift_textbox = 0.5,
                               yshift_textbox = -0.35
                               ):
#    f_R_p_list, f_R_s_list, bins_R_p_list, bins_R_s_list, save_times_out = \
#        generate_size_spectra_R_Arabas(load_path_list,
#                                       ind_time,
#                                       grid_mass_dry_inv,
#                                       solute_type,
#                                       target_cell_list,
#                                       no_cells_x, no_cells_z,
#                                       no_bins_R_p, no_bins_R_s)  
    
    scale_x = 1E-3
    
    annotations = ["A", "B", "C", "D", "E", "F",
                   "G", "H", "I", "J", "K", "L",
                   "M", "N", "O", "P", "Q", "R"]
    
    if SIM_N == 1:
        annotations = np.array(annotations)
        annotations[0:6] = annotations[0:6][::-1]
        annotations = (np.reshape(annotations, (3,6)).T).flatten()
    
#    annotations_reordered = []
#    for cnt in range(no_rows):
#        annotations_reordered += annotations[no_rows-cnt-1::no_rows]
    
    if SIM_N == 7:
        annotations = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
#        i_list = target_cell_list[0]
#        j_list = target_cell_list[1]
        print ("target_cell_list")
        print (target_cell_list)
        target_cell_list = np.delete(target_cell_list, np.s_[2::3], 1 )
        target_cell_list = np.delete(target_cell_list, np.s_[8:10:], 1 )
        print (target_cell_list)
        
        idx_chosen = np.array([0,1,3,4,6,7,9,10,15,16])
        mask = np.zeros(18, dtype=bool)
        mask[idx_chosen] = True
        
        no_rows = 5
        no_cols = 2
    
    no_seeds = len(f_R_p_list[0])
    print("no_seeds =", no_seeds)
    grid_steps = grid.steps
#    no_seeds = len(f_R_p_list[0])
#    no_times = len(save_times_out)
    no_tg_cells = len(target_cell_list[0])    
    
    f_R_p_avg = np.average(f_R_p_list, axis=1)
    f_R_s_avg = np.average(f_R_s_list, axis=1)
    
    if f_R_p_list.shape[1] != 1:
        f_R_p_std = np.std(f_R_p_list, axis=1, ddof=1) / np.sqrt(no_seeds)
        f_R_s_std = np.std(f_R_s_list, axis=1, ddof=1) / np.sqrt(no_seeds)
    
    if SIM_N == 7:
        f_R_p_avg = f_R_p_avg[mask]
        f_R_s_avg = f_R_s_avg[mask]
        f_R_p_std = f_R_p_std[mask]
        f_R_s_std = f_R_s_std[mask]
        bins_R_p_list = bins_R_p_list[mask]
        bins_R_s_list = bins_R_s_list[mask]

#    print("len(f_R_p_list)")
#    print(f_R_p_list.shape)
#    print(f_R_p_avg)
#    print(f_R_p_std)

#%% SPECTRA EXTENDED
    if figsize_spectra is not None:
        figsize = figsize_spectra
    else:        
        figsize = (no_cols*5, no_rows*4)

    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                             figsize = figsize, sharex = True, sharey=True )
    
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
            
            if f_R_p_list.shape[1] != 1:
                f_R_p_err = f_R_p_std[plot_n]
                f_R_s_err = f_R_s_std[plot_n]
            
            bins_R_p = bins_R_p_list[plot_n]
            bins_R_s = bins_R_s_list[plot_n]
            
            bins_centers_R_p = 0.5 * (bins_R_p[1:] + bins_R_p[:-1])
            bins_centers_R_s = 0.5 * (bins_R_s[1:] + bins_R_s[:-1])            
            
            f_R_p_min = f_R_p.min()
            f_R_p_max = f_R_p.max()
            f_R_s_min = f_R_s.min()
            f_R_s_max = f_R_s.max()            
            
            LW_spectra = 1.
            
            ax.plot(np.repeat(bins_R_p,2),
                    np.hstack( [[f_R_p_min*1E-1],
                                np.repeat(f_R_p,2),
                                [f_R_p_min*1E-1] ] ),
                    linewidth = LW_spectra, label = "wet")
            ax.plot(np.repeat(bins_R_s,2),
                    np.hstack( [[f_R_s_min*1E-1],
                                np.repeat(f_R_s,2),
                                [f_R_s_min*1E-1] ] ),
                    linewidth = LW_spectra, label = "dry")  
            
            # added error-regions here
            if f_R_p_list.shape[1] != 1:
                ax.fill_between(np.repeat(bins_R_p,2)[1:-1],
                                np.repeat(f_R_p,2) - np.repeat(f_R_p_err,2),
                                np.repeat(f_R_p,2) + np.repeat(f_R_p_err,2),
                                alpha=0.5,
                                facecolor="lightblue",
                                edgecolor="blue", lw=0.5,
                                zorder=0)
                ax.fill_between(np.repeat(bins_R_s,2)[1:-1],
                                np.repeat(f_R_s,2) - np.repeat(f_R_s_err,2),
                                np.repeat(f_R_s,2) + np.repeat(f_R_s_err,2),
                                alpha=0.4,
                                facecolor="orange",
                                edgecolor="darkorange", lw=0.5,
                                zorder=0)
#            ax.fill_between(bins_centers_R_s,
#                            f_R_s - f_R_s_err,
#                            f_R_s + f_R_s_err,
#                            alpha=0.4,
#                            facecolor="orange",
#                            edgecolor="darkorange", lw=0.5,
#                            zorder=0)
#            ax.fill_between(bins_centers_R_p,
#                            f_R_p - f_R_p_err,
#                            f_R_p + f_R_p_err,
#                            alpha=0.4,
#                            facecolor="lightblue",
#                            edgecolor="blue", lw=0.5,
#                            zorder=0)
#            ax.fill_between(bins_centers_R_s,
#                            f_R_s - f_R_s_err,
#                            f_R_s + f_R_s_err,
#                            alpha=0.4,
#                            facecolor="orange",
#                            edgecolor="darkorange", lw=0.5,
#                            zorder=0)
            if SIM_N == 7:
                AN_pos = (2.5E-3, 2E3)
            else:
                AN_pos = (2.5E-3, 1E3)
            ax.annotate(f"\\textbf{{{annotations[plot_n]}}}",
#                    annotations[plot_n],
                    AN_pos
#                    (2.5E-3, 1E3)
#                    fontsize = ANS
                    )
#            ax.vlines(0.5, f_R_s_min, f_R_s_max)
            LW_vline = 1
            LS_vline = "--"
            ax.axvline(0.5, c ="k", linewidth=LW_vline,ls=LS_vline)
            ax.axvline(25., c ="k", linewidth=LW_vline, ls = LS_vline)            
#            ax.axvline(0.5, c ="k", linewidth=0.8)
#            ax.axvline(25., c ="k", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
#            ax.set_xlim( [2E-3, 1E2] )    
#            ax.set_ylim( [8E-3, 4E3] )    
            
            if SIM_N == 7:
                ax.set_yticks(np.logspace(-2,4,7))
#            else:
            
            ax.set_xticks(np.logspace(-2,2,5))
            ax.set_xticks(
                    np.concatenate((
                            np.linspace(2E-3,9E-3,8),
                            np.linspace(1E-2,9E-2,9),
                            np.linspace(1E-1,9E-1,9),
                            np.linspace(1,9,9),
                            np.linspace(10,90,9),
                            np.linspace(1E2,3E2,3),
                            )),
                    minor=True
                    )
                    
            ax.set_xlim( [2E-3, 3E2] )  
            
            if SIM_N == 7:
                ax.set_ylim( [5E-3, 1E4] )    
            else:
                ax.set_ylim( [1E-5, 1E4] )    
            
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 6, width=1)
            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                           length = 3)
            
#            height = int(grid.compute_location(*target_cell,0.0,0.0)[1])
            xx = ( (target_cell[0] + 0.5 )*grid_steps[0])
            height = ( (target_cell[1] + 0.5 )*grid_steps[1])
#            xx = int((target_cell[0])*grid_steps[0])
#            height = int((target_cell[1])*grid_steps[1])
#            xx = int((target_cell[0]+0.5)*grid_steps[1])
#            height = int((target_cell[1]+0.5)*grid_steps[1])
            if row_n==no_rows-1:
                ax.set_xlabel(r"$R_p$ (\si{\micro\meter})",
                              fontsize = LFS)
            if col_n==0:
#                ax.set_ylabel(r"$f_m$ (${\mathrm{mg}}^{-1}\, {\mathrm{\mu m}}^{-1}$)",
                
                ax.set_ylabel("$f_R$ $(\\si{\\micro\\meter^{-1}\\,mg^{-1}})$",
                              fontsize = LFS)
                
            # ax.set_xlabel(r"particle radius ($\si{\micro m}$)", fontsize = LFS)
            # ax.set_ylabel(r"concentration ($\si{\# / cm^{3}}$)", fontsize = LFS)
#            ax.set_title( f'$(x,z)=$ ({xx:.2}, {height:.2}) km, ' 
            if trajectory is None:            
                ax.set_title( f'$(x,z)=$ ({xx:.2}, {height:.2}) km, ', 
                              fontsize = TTFS )            
            else:
                ax.set_title( f'$(x,z)=$ ({xx:.2}, {height:.2}), ' 
                              + f"t = {save_times_out[plot_n]//60-t_spin_up//60} min",
                              fontsize = TTFS )            
#            ax.set_title( f'$(x,z)=$ ({xx:.2}, {height:.2}), ' 
##                         + f"cell ({target_cell[0]} {target_cell[1]}) "
##                            + f"Nc ({no_cells_x}, {no_cells_z}) "
#                          + f"t = {save_times_out[plot_n]//60-120} min",
##                          + f"t = {save_times_out[plot_n]//60-120} min",
#                          fontsize = TTFS )            
            ax.grid()
            ax.legend(
                       loc = "upper right",
#                       bbox_to_anchor=(),
                       ncol = 2,
                       handlelength=1, handletextpad=0.3,
                      columnspacing=0.8, borderpad=0.25)            
#            ax.legend(loc='upper right')
            
#            ax.annotate(f"$R_{{min/max}}$\n{R_min_list[plot_n]:.2e}\n{R_max_list[plot_n]:.2e}",
#                        (25.,6.))
            
            #ax.set_ylim( [f_R_min*0.5,4E3] )
#    fig.tight_layout()
    pad_ax_h = 0.35          
    if trajectory is None:
        pad_ax_v = 0.08
        fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
    else:        
        fig.subplots_adjust(hspace=pad_ax_h)
    if fig_path is not None:
#        fig_name =\
#            fig_path \
#            + f"spectrum_cell_list_j_from_{j_low}_to_{j_high}_" \
#            + f"Ntgcells_{no_tg_cells}_no_cells_{no_cells_x}_{no_cells_z}_" \
#            + f"Nseeds_{no_seeds}.pdf"
        fig.savefig(fig_path[:-4] + "_ext.pdf",
    #                bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.05
                    )        
    ### CALC EFFECTIVE RADIUS = MOMENT3/MOMENT2 FROM ANALYSIS OF f_R
    R_eff_list = np.zeros((no_tg_cells, 3), dtype = np.float64)
#    no_rows = 1
#    no_cols = 1      
    fig3, axes = plt.subplots(nrows = 1, ncols = 1,
                 figsize = (6, 5) )
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
    if fig_path_R_eff is not None:
        fig3.savefig(fig_path_R_eff)            

#%% PLOT TARGET CELLS IN EXTRA PLOT
    ########################################## PLOT TARGET CELLS IN EXTRA PLOT
    
    if show_target_cells:
        grid_r_l = grid_r_l_list[1]*1E3
#        no_rows = 1
#        no_cols = 1  
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
            figsize = (9, 8)            
        
        fig2, axes = plt.subplots(nrows = 1, ncols = 1,
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
        
        from matplotlib.text import TextPath
#        annotate_labels = []
#        annotate_markers = []
        textbox = []
        for tg_cell_n in range(no_tg_cells):
            x = (target_cell_list[0, tg_cell_n] - no_neigh_x - 0.1) * dx
            z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
            
            print("x,z tg cells")
            print(x,z)
            
    #        dx *= no_cells_x
    #        dz *= no_cells_z
            
            rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z, fill=False,
                                 linewidth = LW_rect,
#                                 linestyle = "dashed",
                                 edgecolor='k',
                                 zorder = 99)        
            ax.add_patch(rect)
#            x_ann_shift = 100E-3
#            ax.scatter(x-x_ann_shift,z-z_ann_shift,
#                       marker=f"\\textbf{{{annotations[tg_cell_n]}}}",
##                        xytext=(3, 1.5),
##                        fontsize=8,
#                       edgecolors = None,
#                       zorder=100,
#                       label=r"{}".format(-120 + save_times_out[tg_cell_n]//60))
#            annotate_handles.append(
            if trajectory is None:
#                print((tg_cell_n // no_cols))
#                print((tg_cell_n // no_cols) % 2)
                x_ann_shift = 80E-3
                z_ann_shift = 30E-3
                # gives col number
                if (tg_cell_n % no_cols) == 2:
                    x_ann_shift = -80E-3
                    
                # gives row number
                if (tg_cell_n // no_cols) == 1:
                    z_ann_shift = -40E-3
                if (tg_cell_n // no_cols) == 2:
                    z_ann_shift = -20E-3
                if (tg_cell_n // no_cols) == 3:
                    z_ann_shift = 40E-3
                if (tg_cell_n // no_cols) == 4:
                    z_ann_shift = 30E-3
                    
                if SIM_N == 7: ANS = 10
                else: ANS = 8
                ax.annotate(f"\\textbf{{{annotations[tg_cell_n]}}}",
                            (x-x_ann_shift,z-z_ann_shift),
                #                        xytext=(3, 1.5),
                            fontsize=ANS, zorder=100,
                #            label=r"{}".format(-120 + save_times_out[tg_cell_n]//60)
                            )
#                else:
#                    ax.annotate(f"\\textbf{{{annotations[tg_cell_n]}}}",
#                            (x+x_ann_shift,z-z_ann_shift),
#                #                        xytext=(3, 1.5),
#                            fontsize=8, zorder=100,
#                #            label=r"{}".format(-120 + save_times_out[tg_cell_n]//60)
#                            )
            else:
                x_ann_shift = 60E-3
                z_ann_shift = 60E-3
                ax.annotate(f"\\textbf{{{annotations[tg_cell_n]}}}",
                            (x-x_ann_shift,z-z_ann_shift),
                #                        xytext=(3, 1.5),
                            fontsize=8, zorder=100,
                #            label=r"{}".format(-120 + save_times_out[tg_cell_n]//60)
                            )
            textbox.append(f"\\textbf{{{annotations[tg_cell_n]}}}: "
                           + f"{save_times_out[tg_cell_n]//60 - 120}")
        #    textbox += f"\\textbf{{{annotations[i]}}}: {t[i]} "
#            annotate_markers.append(TextPath((0,0),
#                                             f"\\textbf{{{annotations[tg_cell_n]}}}"))
#            annotate_labels.append(r"{}".format(-120 + save_times_out[tg_cell_n]//60))
        if show_textbox:
#            ax.legend(annotate_markers, annotate_labels)
#            ax.legend(handles=annotate_handles)
            textbox = r"$t$ (min): \quad" + ", ".join(textbox)
            props = dict(boxstyle='round', facecolor="white", alpha=1)
            ax.text(xshift_textbox, yshift_textbox, textbox, transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=8,
                    bbox=props)            
            
#        MS = 2
        if trajectory is not None:
#            print("should plot here")
            ax.plot(scale_x * trajectory[:,0], scale_x * trajectory[:,1],"o",
                    markersize = MS, c="k")
        
        ax.tick_params(axis='both', which='major', labelsize=TKFS)
        ax.grid(color='gray', linestyle='dashed', zorder = 2)
        
        ax.set_title(r"$r_l$ (g/kg), $t$ = {} min".format(-t_spin_up//60 + save_times_out[1]//60))
        ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
        ax.set_ylabel(r'$z$ (km)', fontsize = LFS)  
#        ax.axis("equal")
#        ax.axis("scaled")
#        ax.set(xlim=grid.ranges[0], ylim=grid.ranges[1])          
        ax.set_xlim((0,1.5))
        ax.set_ylim((0,1.5))
        ax.set_aspect('equal')
#        ax.set_aspect('equal', 'box')
        
#        fig2.tight_layout(pad = 2.)
#        plt.subplots_adjust(top=1, bottom=1)
        if fig_path_tg_cells is not None:
            fig2.savefig(fig_path_tg_cells,
        #                    bbox_inches = 0,
                        bbox_inches = 'tight',
                        pad_inches = 0.05
                        )   
            
#%% FUNCTION DEF: PLOT SCALAR FIELDS
def plot_scalar_field_frames_extend_avg_MA(grid, fields_with_time,
                                        save_times,
                                        field_names,
                                        units,
                                        scales,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path,
                                        figsize,
                                        SIM_N=None,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        ):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    for i,fm in enumerate(field_names):
        print(i,fm)
    print(save_times)
    
    tick_ranges = grid.ranges
#    tick_ranges_label = [[0,1.2],[0,1.2]]

    no_rows = len(save_times)
    no_cols = len(field_names)
    
    if SIM_N == 7:
        print ("target_cell_list")
        print (target_cell_list)
        target_cell_list = np.delete(target_cell_list, np.s_[2::3], 1 )
        target_cell_list = np.delete(target_cell_list, np.s_[8:10:], 1 )
        print (target_cell_list)        
    
#    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                       figsize = (4.5*no_cols, 4*no_rows),
#                       sharex=True, sharey=True)    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
    
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
            
            xticks_major = None
            xticks_minor = None

            norm_ = mpl.colors.Normalize 
            if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
                norm_ = mpl.colors.LogNorm
                field_min = 0.01
                cmap = cmap_lcpp                
                if ax_title == "r_r":
                    field_max = 1.
                    xticks_major = [0.01,0.1,1.]
                    xticks_minor = np.concatenate((
                            np.linspace(2E-2,1E-1,9),
                            np.linspace(2E-1,1,9),
                            ))
                elif ax_title == "n_r":
                    field_max = 10.
                    xticks_major = [0.01,0.1,1.,10.]
                    xticks_minor = np.concatenate((
                            np.linspace(2E-2,1E-1,9),
                            np.linspace(2E-1,1,9),
                            np.linspace(2,10,9),
                            ))
            else: norm_ = mpl.colors.Normalize   
            
            if ax_title == r"\Theta":
#                if SIM_N == 7:
#                    field_min = 289.0
#                    field_max = 292.5
#                    xticks_major = [289,290,291,292]
#                else:
                field_min = 289.2
                field_max = 292.5
                xticks_major = [290,291,292]
            if ax_title == "r_v":
                field_min = 6.5
                field_max = 7.6
                xticks_minor = [6.75,7.25]
            if ax_title == "r_l":
                field_min = 0.0
                field_max = 1.3
                xticks_minor = [0.25,0.75,1.25]
                
            if ax_title == "r_c":
                field_min = 0.0
                field_max = 1.3
#                xticks_major = np.linspace(0,1.2,7)
#                xticks_major = np.linspace(0,1.25,6)
                xticks_major = [0,0.5,1]
                xticks_minor = [0.25,0.75,1.25]                
            if ax_title == "n_c":
                field_min = 0.0
                field_max = 150.
                xticks_major = [0,50,100,150]
                xticks_minor = [25,75,125]                
            if ax_title == "n_\mathrm{aero}":
                field_min = 0.0
                field_max = 200.
                xticks_major = [0,50,100,150,200]
#                xticks_minor = [25,75,125]
            if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
                xticks_major = [1,5,10,15,20]
#                field_min = 0.
                field_min = 1
#                field_min = 1.5
                field_max = 20.
#                cmap = cmap_new
                # Arabas 2015
                cmap = cmap_lcpp
                unit = r"\si{\micro\meter}"
                
            oom_max = oom = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
            if oom_max > 2 or oom_max < 0:
                my_format = True
                oom_factor = 10**(-oom)
                
                field_min *= oom_factor
                field_max *= oom_factor            
            
            if oom_max ==2: str_format = "%.0f"
#            if oom_max ==2: str_format = "%.1f"
            
            else: str_format = "%.2g"
#            else: str_format = "%.2f"
            
            if field_min/field_max < 1E-4:
#                cmap = cmap_new
                # Arabas 2015                
                cmap = cmap_lcpp
#                alpha = 0.8
            
            # REMOVE FIX APLHA HERE
#            alpha = 1.0
#        CS = ax.pcolormesh(*grid.corners, grid_r_l,
#                           cmap=cmap, alpha=alpha,
#                            edgecolor="face", zorder=1,
#                            vmin=field_min, vmax=field_max,
#                            antialiased=True, linewidth=0.0
##                            norm = norm_(vmin=field_min, vmax=field_max)
#                            )            
            CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor="face", zorder=1,
                                norm = norm_(vmin=field_min, vmax=field_max),
                                rasterized=True,
                                antialiased=True, linewidth=0.0
                                )
            if ax_title == r"\Theta":
                cmap_x = mpl.cm.get_cmap('coolwarm')
                print("cmap_x(0.0)")
                print(cmap_x(0.0))
                CS.cmap.set_under(cmap_x(0.0))
#                pass
            else:
#            if ax_title != r"\Theta":
                CS.cmap.set_under("white")
            
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )

#            ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                             tick_ranges[0,1],
#                                             no_ticks[0] ) )
#            ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                             tick_ranges[1,1],
#                                             no_ticks[1] ) )
#            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 3, width=1)
#            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
#                           length = 3)            
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_aspect('equal')
            if time_n == no_rows-1:
#                tlabels = ax.get_xticklabels()
#                print(tlabels)
#                tlabels[-1] = ""
#                print(tlabels)
#                ax.set_xticklabels(tlabels)    
                xticks1 = ax.xaxis.get_major_ticks()
                xticks1[-1].label1.set_visible(False)
                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
            if field_n == 0:            
                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
#            if time_n == 0:
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#            else:                
            ax.set_title( r"$t$ = {0} min".format(int(save_times[time_n]/60)),
                         fontsize = TTFS)
#            ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
#                         int(save_times[time_n]/60)),
#                         fontsize = TTFS)
            if time_n == 0:
                if no_cols == 4:
#            if time_n == no_rows - 1:
                    axins = inset_axes(ax,
                                       width="90%",  # width = 5% of parent_bbox width
                                       height="8%",  # height
                                       loc='lower center',
                                       bbox_to_anchor=(0.0, 1.35, 1, 1),
    #                                   , 1, 1),
                                       bbox_transform=ax.transAxes,
                                       borderpad=0,
                                       )      
                else:
                    axins = inset_axes(ax,
                                       width="90%",  # width = 5% of parent_bbox width
                                       height="8%",  # height
                                       loc='lower center',
                                       bbox_to_anchor=(0.0, 1.4, 1, 1),
    #                                   , 1, 1),
                                       bbox_transform=ax.transAxes,
                                       borderpad=0,
                                       )      
#                divider = make_axes_locatable(ax)
#                cax = divider.append_axes("top", size="6%", pad=0.3)
                
                cbar = plt.colorbar(CS, cax=axins,
#                                    fraction=0.046, pad=-0.1,
                                    format=mticker.FormatStrFormatter(str_format),
                                    orientation="horizontal"
                                    )
#                axins.xaxis.set_ticks_position("bottom")
                
                axins.xaxis.set_ticks_position("bottom")
                axins.tick_params(axis="x",direction="inout",which="both")
#                axins.tick_params(axis="x",direction="inout")
                axins.tick_params(axis='x', which='major', labelsize=TKFS,
                               length = 7, width=1)                
                axins.tick_params(axis='x', which='minor', labelsize=TKFS,
                               length = 5, width=0.5,bottom=True)                
                
                
                if xticks_major is not None:
                    axins.xaxis.set_ticks(xticks_major)
                if xticks_minor is not None:
                    axins.xaxis.set_ticks(xticks_minor, minor=True)
                if ax_title == "n_c":
                    xticks2 = axins.xaxis.get_major_ticks()
                    xticks2[-1].label1.set_visible(False)                
                axins.set_title(r"${0}$ ({1})".format(ax_title, unit))
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
                LW_rect = .5
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

    if no_cols == 4:
        pad_ax_h = 0.1     
        pad_ax_v = 0.05
    else:        
        pad_ax_h = -0.5 
        pad_ax_v = 0.08
    #    pad_ax_v = 0.005
    fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
#    fig.subplots_adjust(wspace=pad_ax_v)
             
#    fig.tight_layout()
    if fig_path is not None:
#        if 
#        DPI =
        fig.savefig(fig_path,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.05,
#                    dpi=300,
                    dpi=600
                    )   
        
#%% FUNCTION DEF: PLOT ABSOLUTE DEVS OF TWO SCALAR FIELDS
def plot_scalar_field_frames_abs_dev_MA(grid,
                                        fields_with_time1,
                                        fields_with_time_std1,
                                        fields_with_time2,
                                        fields_with_time_std2,
                                        save_times,
                                        field_names,
                                        units,
                                        scales,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        compare_type,
                                        fig_path,
                                        fig_path_abs_err,
                                        figsize,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        ):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    for i,fm in enumerate(field_names):
        print(i,fm)
    print(save_times)
    
    tick_ranges = grid.ranges
#    tick_ranges_label = [[0,1.2],[0,1.2]]

    no_rows = len(save_times)
    no_cols = len(field_names)
    
    abs_dev_with_time = fields_with_time2 - fields_with_time1
    
#    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                       figsize = (4.5*no_cols, 4*no_rows),
#                       sharex=True, sharey=True)    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
    
    vline_pos = (0.33, 1.17, 1.33)
    
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            ax = axes[time_n, field_n]
            
            if compare_type == "Nsip" and time_n == 2:
                for vline_pos_ in vline_pos:
                    ax.axvline(vline_pos_, alpha=0.5, c="k", zorder= 3, linewidth = 1.3)
            
            field = abs_dev_with_time[time_n, field_n] * scales[field_n]
            ax_title = field_names[field_n]

            print(time_n, ax_title, field.min(), field.max())

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
            
            xticks_major = None
            xticks_minor = None
            
            norm_ = mpl.colors.Normalize 

            if compare_type in ["Ncell", "solute", "Kernel"]:
                if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
    #                norm_ = mpl.colors.SymLogNorm
                    norm_ = mpl.colors.Normalize
    #                norm_ = mpl.colors.LogNorm
                    cmap = cmap_lcpp                
                    if ax_title == "r_r":
                        field_max = 0.1
                        field_min = -field_max
    #                    linthresh=(-0.01,0.01)
                        linthresh=0.01
    #                    xticks_major = [-0.1, 0, 0.1]
                        xticks_major = np.linspace(field_min,field_max,5)
#                        xticks_major = [field_min, -0.04, 0., 0.04, field_max]
    #                    xticks_major = [0, 0.01, 0.1]
                    elif ax_title == "n_r":
                        pass
                        field_max = 0.9
                        field_min = -field_max
#                        xticks_major = [0.01,0.1,1.,10.]
#                        xticks_minor = np.concatenate((
#                                np.linspace(2E-2,1E-1,9),
#                                np.linspace(2E-1,1,9),
#                                np.linspace(2,10,9),
#                                ))
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r"\Theta":
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == "r_v":
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == "r_l":
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == "r_c":
                    field_max = 0.3
                    field_min = -field_max                
    #                field_min = -0.2
    #                field_max = 0.2
#                    xticks_major = [field_min, -0.05, 0., 0.05, field_max]
                    xticks_major = np.linspace(field_min,field_max,5)
    #                xticks_major = np.linspace(0,1.2,7)
                if ax_title == "n_c":
#                    pass
                    field_max = 40.
                    field_min = -field_max
                if ax_title == "n_\mathrm{aero}":
                    field_max = 40.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
#                    xticks_major = [field_min, -4, 0., 4, field_max]
    #                xticks_minor = [25,75,125]
                if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
    #                xticks_major = [1,5,10,15,20]
    #                field_min = 0.
    #                field_min = 1.5
                    field_max = 8.
#                    field_max = 5.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
    #                cmap = cmap_new
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r"\si{\micro\meter}"
                    
            elif compare_type == "Nsip":
#            elif compare_type == "dt_col":
                if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
    #                norm_ = mpl.colors.SymLogNorm
                    norm_ = mpl.colors.Normalize
    #                norm_ = mpl.colors.LogNorm
                    cmap = cmap_lcpp                
                    if ax_title == "r_r":
                        field_max = 0.08
                        field_min = -field_max
    #                    linthresh=(-0.01,0.01)
                        linthresh=0.01
    #                    xticks_major = [-0.1, 0, 0.1]
                        xticks_major = [field_min, -0.04, 0., 0.04, field_max]
    #                    xticks_major = [0, 0.01, 0.1]
                    elif ax_title == "n_r":
                        field_max = 10.
                        xticks_major = [0.01,0.1,1.,10.]
                        xticks_minor = np.concatenate((
                                np.linspace(2E-2,1E-1,9),
                                np.linspace(2E-1,1,9),
                                np.linspace(2,10,9),
                                ))
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r"\Theta":
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == "r_v":
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == "r_l":
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == "r_c":
                    field_max = 0.12
                    field_min = -field_max                
    #                field_min = -0.2
    #                field_max = 0.2
#                    xticks_major = np.linspace(field_min,field_max,5)
                    xticks_major = [-0.1, -0.05, 0., 0.05, 0.1]
    #                xticks_major = np.linspace(0,1.2,7)
                if ax_title == "n_c":
                    field_min = 0.0
                    field_max = 150.
                if ax_title == "n_\mathrm{aero}":
                    field_max = 20.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
#                    xticks_major = [field_min, -4, 0., 4, field_max]
    #                xticks_minor = [25,75,125]
                if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
    #                xticks_major = [1,5,10,15,20]
    #                field_min = 0.
    #                field_min = 1.5
#                    field_max = 8.
                    field_max = 5.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
    #                cmap = cmap_new
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r"\si{\micro\meter}"
            else:
#            elif compare_type == "dt_col":
                if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
    #                norm_ = mpl.colors.SymLogNorm
                    norm_ = mpl.colors.Normalize
    #                norm_ = mpl.colors.LogNorm
                    cmap = cmap_lcpp                
                    if ax_title == "r_r":
                        field_max = 0.08
                        field_min = -field_max
    #                    linthresh=(-0.01,0.01)
                        linthresh=0.01
    #                    xticks_major = [-0.1, 0, 0.1]
                        xticks_major = [field_min, -0.04, 0., 0.04, field_max]
    #                    xticks_major = [0, 0.01, 0.1]
                    elif ax_title == "n_r":
                        field_max = 10.
                        xticks_major = [0.01,0.1,1.,10.]
                        xticks_minor = np.concatenate((
                                np.linspace(2E-2,1E-1,9),
                                np.linspace(2E-1,1,9),
                                np.linspace(2,10,9),
                                ))
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r"\Theta":
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == "r_v":
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == "r_l":
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == "r_c":
                    field_max = 0.1
                    field_min = -field_max                
    #                field_min = -0.2
    #                field_max = 0.2
                    xticks_major = [field_min, -0.05, 0., 0.05, field_max]
    #                xticks_major = np.linspace(0,1.2,7)
                if ax_title == "n_c":
                    field_min = 0.0
                    field_max = 150.
                if ax_title == "n_\mathrm{aero}":
                    field_max = 8.
                    field_min = -field_max
                    xticks_major = [field_min, -4, 0., 4, field_max]
    #                xticks_minor = [25,75,125]
                if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
    #                xticks_major = [1,5,10,15,20]
    #                field_min = 0.
    #                field_min = 1.5
                    field_max = 3.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,7)
    #                cmap = cmap_new
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r"\si{\micro\meter}"
                    
            oom_max = oom = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
#            if oom_max > 2 or oom_max < 0:
#                my_format = True
#                oom_factor = 10**(-oom)
#                
#                field_min *= oom_factor
#                field_max *= oom_factor            
            
            if oom_max ==2: str_format = "%.0f"
#            if oom_max ==2: str_format = "%.1f"
            
            else: str_format = "%.2g"
#            else: str_format = "%.2f"
            
            cmap = "bwr"
            
            # REMOVE FIX APLHA HERE
#            alpha = 1.0
#        CS = ax.pcolormesh(*grid.corners, grid_r_l,
#                           cmap=cmap, alpha=alpha,
#                            edgecolor="face", zorder=1,
#                            vmin=field_min, vmax=field_max,
#                            antialiased=True, linewidth=0.0
##                            norm = norm_(vmin=field_min, vmax=field_max)
#                            )          
            if False:
#            if ax_title in ["r_r", "n_r"]:
                norm = norm_(linthresh = linthresh, vmin=field_min, vmax=field_max)
            else:
                norm = norm_(vmin=field_min, vmax=field_max)
            CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor="face", zorder=1,
                                norm = norm,
                                rasterized=True,
                                antialiased=True, linewidth=0.0
                                )
            CS.cmap.set_under("blue")
            CS.cmap.set_over("red")
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )

#            ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                             tick_ranges[0,1],
#                                             no_ticks[0] ) )
#            ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                             tick_ranges[1,1],
#                                             no_ticks[1] ) )
#            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 3, width=1)
#            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
#                           length = 3)            
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_aspect('equal')
            if time_n == no_rows-1:
#                tlabels = ax.get_xticklabels()
#                print(tlabels)
#                tlabels[-1] = ""
#                print(tlabels)
#                ax.set_xticklabels(tlabels)    
                xticks1 = ax.xaxis.get_major_ticks()
                xticks1[-1].label1.set_visible(False)
                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
            if field_n == 0:            
                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
#            if time_n == 0:
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#            else:                
            ax.set_title( r"$t$ = {0} min".format(int(save_times[time_n]/60)),
                         fontsize = TTFS)
#            ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
#                         int(save_times[time_n]/60)),
#                         fontsize = TTFS)
            if time_n == 0:
#            if time_n == no_rows - 1:
                axins = inset_axes(ax,
                                   width="90%",  # width = 5% of parent_bbox width
                                   height="8%",  # height
                                   loc='lower center',
                                   bbox_to_anchor=(0.0, 1.35, 1, 1),
#                                   , 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )      
#                divider = make_axes_locatable(ax)
#                cax = divider.append_axes("top", size="6%", pad=0.3)
                
                cbar = plt.colorbar(CS, cax=axins,
#                                    fraction=0.046, pad=-0.1,
                                    format=mticker.FormatStrFormatter(str_format),
                                    orientation="horizontal"
                                    )
#                axins.xaxis.set_ticks_position("bottom")
                
                axins.xaxis.set_ticks_position("bottom")
                axins.tick_params(axis="x",direction="inout",which="both")
#                axins.tick_params(axis="x",direction="inout")
                axins.tick_params(axis='x', which='major', labelsize=TKFS,
                               length = 7, width=1)                
                axins.tick_params(axis='x', which='minor', labelsize=TKFS,
                               length = 5, width=0.5,bottom=True)                
                
                if xticks_major is not None:
                    axins.xaxis.set_ticks(xticks_major)
                if xticks_minor is not None:
                    axins.xaxis.set_ticks(xticks_minor, minor=True)
                axins.set_title(r"$\Delta {0}$ ({1})".format(ax_title, unit))
#                axins.set_title(r"${0}$ ({1})".format(ax_title, unit))
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
                LW_rect = .5
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


    pad_ax_h = 0.1     
    pad_ax_v = 0.05
#    pad_ax_v = 0.005
    fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
#    fig.subplots_adjust(wspace=pad_ax_v)
             
#    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.05,
                    dpi=600
                    )     
    plt.close("all")
    #######################################################################
    ### PLOT ABS ERROR OF THE DIFFERENCE OF TWO FIELDS:
#    # Var = Var_1 + Var_2 (assuming no correlations)
    # 
#    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
#    for i,fm in enumerate(field_names):
#        print(i,fm)
#    print(save_times)
    
    print("plotting ABS ERROR of field1 - field2")
    
    std1 = np.nan_to_num( fields_with_time_std1 )
    std2 = np.nan_to_num( fields_with_time_std2 )
    std = np.sqrt( std1**2 + std2**2 )
#    rel_std = fields_with_time_std
#    rel_std = np.zeros_like(std)
    
#    rel_std = np.where(fields_with_time == 0.,
#                       0.,
#                       fields_with_time_std / fields_with_time)
    
    tick_ranges = grid.ranges
#    tick_ranges_label = [[0,1.2],[0,1.2]]
    
    no_rows = len(save_times)
    no_cols = len(field_names)

#    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                       figsize = (4.5*no_cols, 4*no_rows),
#                       sharex=True, sharey=True)    
    
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            std[time_n, field_n] *= scales[field_n]
#            fields_with_time[time_n, field_n] *= scales[field_n]
#            rel_std[time_n, field_n] = \
#                np.where(fields_with_time[time_n, field_n] <= 0.,
#                         0.,
#                         std[time_n, field_n] / fields_with_time[time_n, field_n])

    field_max_all = np.amax(std, axis=(0,2,3))
    field_min_all = np.amin(std, axis=(0,2,3))
#    field_max_all_std = np.amax(rel_std, axis=(0,2,3))
#    field_min_all_std = np.amin(rel_std, axis=(0,2,3))

    pad_ax_h = 0.1     
    pad_ax_v = 0.03

    ### PLOT ABS ERROR  
#    if plot_abs:
    fig2, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
            
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            ax = axes[time_n,field_n]
#            field = rel_std[time_n, field_n] * scales[field_n]
            field = std[time_n, field_n]
            ax_title = field_names[field_n]
            unit = units[field_n]
            
#            cmap = "coolwarm"
#            cmap = "Greys"
#            cmap = "Greens"
#            cmap = "Reds"
            cmap = cmap_lcpp
            
            field_min = 0.
            field_max = field_max_all[field_n]
#            field_min = field_min_all[field_n]

            xticks_major = None
            xticks_minor = None
            
            if compare_type == "Nsip":
                if ax_title == "n_\mathrm{aero}":
#                    field_max = 5.
#                    xticks_minor = np.linspace(1.25,5,3)
                    xticks_minor = np.array((1.25,3.75))
                if ax_title == "r_c":
                    field_max = 0.06
                    xticks_minor = np.linspace(0.01,0.05,3)
                if ax_title == "r_r":
#                    field_max = 0.04
                    xticks_minor = np.linspace(0.01,0.03,2)
#                        xticks_minor = np.linspace(1,5,3)
                #, r"R_{2/1}", r"R_\mathrm{eff}"]:                    
                if ax_title in [r"R_\mathrm{eff}"]:
                    xticks_major = np.linspace(0.,1.5,4)
            
            if compare_type == "dt_col":
                if ax_title == "n_\mathrm{aero}":
#                    field_max = 5.
#                    xticks_minor = np.linspace(1.25,5,3)
                    xticks_minor = np.array((1.25, 3.75, 6.25))
                if ax_title == "r_c":
#                    field_max = 0.0
#                    xticks_minor = np.linspace(0.01,0.05,3)
                    xticks_major = np.array((0.,0.02,0.04,0.06))
                if ax_title == "r_r":
#                    field_max = 0.04
                    xticks_minor = np.linspace(0.01,0.05,3)
#                        xticks_minor = np.linspace(1,5,3)
                #, r"R_{2/1}", r"R_\mathrm{eff}"]:                    
                if ax_title in [r"R_\mathrm{eff}"]:
                    xticks_major = np.linspace(0.,1.5,4)
            
#            if SIM_N == 2:
#                if ax_title == "r_r":
#                    field_max = 0.03
##                        xticks_minor = np.linspace(0.01,0.03,2)
##                        xticks_minor = np.linspace(1,5,3)
#                if ax_title == "r_c":
##                        field_max = 0.05
#                    xticks_minor = np.linspace(0.01,0.03,2)
#                if ax_title == "n_\mathrm{aero}":
##                        pass
##                        field_max = 5.
#                    xticks_minor = np.linspace(1,3,2)
#            if SIM_N == 10:
#                if ax_title == "r_r":
##                        field_max = 0.04
#                    xticks_minor = np.linspace(0.01,0.03,2)
##                        xticks_minor = np.linspace(1,5,3)
#                if ax_title == "r_c":
#                    field_max = 0.05
#                    xticks_minor = np.linspace(0.01,0.05,3)
#                if ax_title == "n_\mathrm{aero}":
#                    field_max = 5.
#                    xticks_minor = np.linspace(1,5,3)
            
#            field_max = field.max()
#            field_min = field.min()
            
            norm_ = mpl.colors.Normalize 
                
            str_format = "%.2g"
            
            my_format = False
            oom_factor = 1.0

            oom_max = oom = 1

            CS = ax.pcolormesh(*grid.corners,
                               field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor="face", zorder=1,
#                                norm = norm_(vmin=-minmax, vmax=minmax),
                                norm = norm_(vmin=field_min, vmax=field_max),
#                                norm = norm_(vmin=field_min, vmax=field_max),
                                rasterized=True,
                                antialiased=True, linewidth=0.0
                                )
            CS.cmap.set_under("white")
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )

            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 3, width=1)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_aspect('equal')
            if time_n == no_rows-1:
                xticks1 = ax.xaxis.get_major_ticks()
                xticks1[-1].label1.set_visible(False)
                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
            if field_n == 0:            
                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
            ax.set_title( r"$t$ = {0} min".format(int(save_times[time_n]/60)),
                         fontsize = TTFS)
            if time_n == 0:
                axins = inset_axes(ax,
                                   width="90%",  # width = 5% of parent_bbox width
                                   height="8%",  # height
                                   loc='lower center',
                                   bbox_to_anchor=(0.0, 1.35, 1, 1),
#                                   , 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )      
                cbar = plt.colorbar(CS, cax=axins,
#                                    fraction=0.046, pad=-0.1,
                                    format=mticker.FormatStrFormatter(str_format),
                                    orientation="horizontal"
                                    )
                axins.xaxis.set_ticks_position("bottom")
                axins.tick_params(axis="x",direction="inout",which="both")
                axins.tick_params(axis='x', which='major', labelsize=TKFS,
                               length = 7, width=1)                
                axins.tick_params(axis='x', which='minor', labelsize=TKFS,
                               length = 5, width=0.5,bottom=True)                
                
                if xticks_major is not None:
                    axins.xaxis.set_ticks(xticks_major)
                if xticks_minor is not None:
                    axins.xaxis.set_ticks(xticks_minor, minor=True)
                if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
                    unit = r"\si{\micro\meter}"                                            
                axins.set_title(r"${0}$ abs. error ({1})".format(ax_title, unit))
            # my_format dos not work with log scale here!!
                
                if my_format:
                    cbar.ax.text(1.0,1.0,
                                 r'$\times\,10^{{{}}}$'.format(oom_max),
                                 va='bottom', ha='right', fontsize = TKFS,
                                 transform=ax.transAxes)
#                    cbar.ax.text(field_min - (field_max-field_min),
#                                 field_max + (field_max-field_min)*0.01,
#                                 r'$\times\,10^{{{}}}$'.format(oom_max),
#                                 va='bottom', ha='left', fontsize = TKFS,
#                                 transform=ax.transAxes)
                cbar.ax.tick_params(labelsize=TKFS)

            if show_target_cells:
                ### ad the target cells
                no_neigh_x = no_cells_x // 2
                no_neigh_z = no_cells_z // 2
                dx = grid.steps[0]
                dz = grid.steps[1]
                
                no_tg_cells = len(target_cell_list[0])
                LW_rect = .5
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
    
    
    
    #    pad_ax_v = 0.005
    fig2.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
#    fig.subplots_adjust(wspace=pad_ax_v)
             
#    fig.tight_layout()
    if fig_path_abs_err is not None:
        fig2.savefig(fig_path_abs_err,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.05,
                    dpi=600
                    )       

        
#%% FUNCTION DEF: PLOT INITIAL SCALAR FIELDS INITIAL
def plot_scalar_field_frames_init_avg_MA(grid, fields_with_time,
                                        save_times,
                                        field_names,
                                        units,
                                        scales,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path,
                                        figsize,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        ):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    for i,fm in enumerate(field_names):
        print(i,fm)
    print(save_times)
    
    
    tick_ranges = grid.ranges
#    tick_ranges_label = [[0,1.2],[0,1.2]]
    
#    no_rows = len(save_times)
#    no_cols = len(field_names)

    field_names_orig = ["r_v", "r_l", "\Theta", "T", "p", "S"]
    scales_orig = [1000., 1000., 1, 1, 0.01, 1]    
    units_orig = ["g/kg", "g/kg", "K", "K", "hPa", "-"]    

#    field_names_deri = ["r_\mathrm{aero}", "r_c", "r_r",
#                       "n_\mathrm{aero}", "n_c", "n_r",
#                       r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]
#    units_deri = ["g/kg", "g/kg", "g/kg", "1/mg", "1/mg", "1/mg",
#                  r"$\mathrm{\mu m}$", r"$\mathrm{\mu m}$", r"$\mathrm{\mu m}$"]
#    scales_deri = [1000., 1000., 1000., 1E-6, 1E-6, 1E-6, 1., 1., 1.]    
    
    ii = (3,4,0,1)
    
    field_names = []
    units = []
    scales = []
    
    for i in ii:
        field_names.append(field_names_orig[i])
        units.append(units_orig[i])
        scales.append(scales_orig[i])

    T = grid.temperature
    p = grid.pressure
    S = grid.saturation
    
    r_v = grid.mixing_ratio_water_vapor
    r_l = grid.mixing_ratio_water_liquid

    fields = np.array([T,p,r_v,r_l])

    no_rows = 2
    no_cols = 2

    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
    
#    for time_n in range(no_rows):
#        for field_n in range(no_cols):

    for field_n, ax in enumerate(axes.flatten()):
        field = fields[field_n]*scales[field_n]
        ax_title = field_names[field_n]
        unit = units[field_n]        
        if ax_title in ["T","p",r"\Theta"]:
            cmap = "coolwarm"
        else :
            cmap = cmap_lcpp
        alpha = 1.0    
            
        field_max = field.max()
        field_min = field.min()
        
        xticks_major = None
        xticks_minor = None

        norm_ = mpl.colors.Normalize 
        if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
            norm_ = mpl.colors.LogNorm
            field_min = 0.01
            cmap = cmap_lcpp                
            if ax_title == "r_r":
                field_max = 1.
                xticks_major = [0.01,0.1,1.]
            elif ax_title == "n_r":
                field_max = 10.
                xticks_major = [0.01,0.1,1.,10.]
                xticks_minor = np.concatenate((
                        np.linspace(2E-2,1E-1,9),
                        np.linspace(2E-1,1,9),
                        np.linspace(2,10,9),
                        ))
        else: norm_ = mpl.colors.Normalize   
        
        
        if ax_title == r"\Theta":
            field_min = 289.2
            field_max = 292.5
            xticks_major = [290,291,292]
        if ax_title == "r_v":
            field_min = 6.5
            field_max = 7.6
            xticks_minor = [6.75,7.25]
        if ax_title == "r_l":
            field_min = 0.0
            field_max = 1.3
            xticks_minor = [0.25,0.75,1.25]
            
        if ax_title == "r_c":
            field_min = 0.0
            field_max = 1.3
            xticks_major = np.linspace(0,1.2,7)
        if ax_title == "n_c":
            field_min = 0.0
            field_max = 150.
        if ax_title == "n_\mathrm{aero}":
            field_min = 0.0
            field_max = 150.
            xticks_minor = [25,75,125]
        if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
            xticks_major = [1,5,10,15,20]
#                field_min = 0.
            field_min = 1
#                field_min = 1.5
            field_max = 20.
#                cmap = cmap_new
            # Arabas 2015
            cmap = cmap_lcpp
            unit = r"\si{\micro\meter}"
            
            
        oom_max = oom = int(math.log10(field_max))
        
        my_format = False
        oom_factor = 1.0
        
        if oom_max > 3 or oom_max < 0:
            my_format = True
            oom_factor = 10**(-oom)
            
            field_min *= oom_factor
            field_max *= oom_factor            
        
        if oom_max in (2,3): str_format = "%.0f"
#            if oom_max ==2: str_format = "%.1f"
        
        else: str_format = "%.2g"
#            else: str_format = "%.2f"
        
        if field_min/field_max < 1E-4:
#                cmap = cmap_new
            # Arabas 2015                
            cmap = cmap_lcpp
#                alpha = 0.8
        
        # REMOVE FIX APLHA HERE
#            alpha = 1.0
#        CS = ax.pcolormesh(*grid.corners, grid_r_l,
#                           cmap=cmap, alpha=alpha,
#                            edgecolor="face", zorder=1,
#                            vmin=field_min, vmax=field_max,
#                            antialiased=True, linewidth=0.0
##                            norm = norm_(vmin=field_min, vmax=field_max)
#                            )            
        CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                           cmap=cmap, alpha=alpha,
                            edgecolor="face", zorder=1,
                            norm = norm_(vmin=field_min, vmax=field_max),
                            rasterized=True,
                            antialiased=True, linewidth=0.0
                            )
        CS.cmap.set_under("white")
        
        ax.set_xticks( np.linspace( tick_ranges[0,0],
                                         tick_ranges[0,1],
                                         no_ticks[0] ) )
        ax.set_yticks( np.linspace( tick_ranges[1,0],
                                         tick_ranges[1,1],
                                         no_ticks[1] ) )

#            ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                             tick_ranges[0,1],
#                                             no_ticks[0] ) )
#            ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                             tick_ranges[1,1],
#                                             no_ticks[1] ) )
#            ax.tick_params(axis='both', which='major', labelsize=TKFS)
        ax.tick_params(axis='both', which='major', labelsize=TKFS,
                       length = 3, width=1)
#            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
#                           length = 3)            
        ax.grid(color='gray', linestyle='dashed', zorder = 2)
        ax.set_aspect('equal')
#        if time_n == no_rows-1:
#                tlabels = ax.get_xticklabels()
#                print(tlabels)
#                tlabels[-1] = ""
#                print(tlabels)
#                ax.set_xticklabels(tlabels)    
        xticks1 = ax.xaxis.get_major_ticks()
        xticks1[-1].label1.set_visible(False)
        if field_n >= 2:
            ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
#        if field_n == 0:            
        if field_n%2 == 0:
            ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
#            if time_n == 0:
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#            else:                
        ax.set_title( r"${0}$ ({1})".format(ax_title, unit),
                     fontsize = TTFS)
#            ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
#                         int(save_times[time_n]/60)),
#                         fontsize = TTFS)
#        if time_n == 0:
#            if time_n == no_rows - 1:
#        axins = inset_axes(ax,
#                           width="90%",  # width = 5% of parent_bbox width
#                           height="8%",  # height
#                           loc='lower center',
#                           bbox_to_anchor=(0.0, 1.35, 1, 1),
##                                   , 1, 1),
#                           bbox_transform=ax.transAxes,
#                           borderpad=0,
#                           )      
#                divider = make_axes_locatable(ax)
#                cax = divider.append_axes("top", size="6%", pad=0.3)
        
        cbar = plt.colorbar(CS, ax=ax,
                                    fraction=0.0467, pad=0.02,
                            format=mticker.FormatStrFormatter(str_format),
                            orientation="vertical"
                            )
#                axins.xaxis.set_ticks_position("bottom")
        
#        axins = cbar.axis
#        axins.xaxis.set_ticks_position("bottom")
#        axins.tick_params(axis="x",direction="inout",which="both")
##                axins.tick_params(axis="x",direction="inout")
#        axins.tick_params(axis='x', which='major', labelsize=TKFS,
#                       length = 7, width=1)                
#        axins.tick_params(axis='x', which='minor', labelsize=TKFS,
#                       length = 5, width=0.5,bottom=True)                
#        
#        if xticks_major is not None:
#            axins.xaxis.set_ticks(xticks_major)
#        if xticks_minor is not None:
#            axins.xaxis.set_ticks(xticks_minor, minor=True)
        
##        axins.set_title(r"${0}$ ({1})".format(ax_title, unit))
    # my_format dos not work with log scale here!!

        if my_format:
            cbar.ax.text(field_min - (field_max-field_min),
                         field_max + (field_max-field_min)*0.01,
                         r'$\times\,10^{{{}}}$'.format(oom_max),
                         va='bottom', ha='left', fontsize = TKFS)
        cbar.ax.tick_params(labelsize=TKFS)

#        if show_target_cells:
#            ### ad the target cells
#            no_neigh_x = no_cells_x // 2
#            no_neigh_z = no_cells_z // 2
#            dx = grid.steps[0]
#            dz = grid.steps[1]
#            
#            no_tg_cells = len(target_cell_list[0])
#            LW_rect = .5
#            for tg_cell_n in range(no_tg_cells):
#                x = (target_cell_list[0, tg_cell_n] - no_neigh_x - 0.1) * dx
#                z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
#                
#        #        dx *= no_cells_x
#        #        dz *= no_cells_z
#                
#                rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
#                                     fill=False,
#                                     linewidth = LW_rect,
#    #                                 linestyle = "dashed",
#                                     edgecolor='k',
#                                     zorder = 99)        
#                ax.add_patch(rect)


    pad_ax_h = 0.02     
    pad_ax_v = 0.35
#    pad_ax_v = 0.005
    fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
#    fig.subplots_adjust(wspace=pad_ax_v)
             
#    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.05,
                    dpi=600
                    )           
        
#%% FUNCTION DEF: PLOT FIELDS STD ERRORS
def plot_scalar_field_frames_std_MA(grid, fields_with_time,
                                           fields_with_time_std,
                                        save_times,
                                        field_names,
                                        units,
                                        scales,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path_abs,
                                        fig_path_rel,
                                        figsize,
                                        SIM_N,
                                        plot_abs=True,
                                        plot_rel=True,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        ):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    for i,fm in enumerate(field_names):
        print(i,fm)
    print(save_times)
    
    std = np.nan_to_num( fields_with_time_std )
#    rel_std = fields_with_time_std
    rel_std = np.zeros_like(std)
    
#    rel_std = np.where(fields_with_time == 0.,
#                       0.,
#                       fields_with_time_std / fields_with_time)
    
    tick_ranges = grid.ranges
#    tick_ranges_label = [[0,1.2],[0,1.2]]
    
    no_rows = len(save_times)
    no_cols = len(field_names)

#    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                       figsize = (4.5*no_cols, 4*no_rows),
#                       sharex=True, sharey=True)    
    
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            std[time_n, field_n] *= scales[field_n]
            fields_with_time[time_n, field_n] *= scales[field_n]
            rel_std[time_n, field_n] = \
                np.where(fields_with_time[time_n, field_n] <= 0.,
                         0.,
                         std[time_n, field_n] / fields_with_time[time_n, field_n])

    field_max_all = np.amax(std, axis=(0,2,3))
    field_min_all = np.amin(std, axis=(0,2,3))
    field_max_all_std = np.amax(rel_std, axis=(0,2,3))
    field_min_all_std = np.amin(rel_std, axis=(0,2,3))

    pad_ax_h = 0.1     
    pad_ax_v = 0.03

    ### PLOT ABS ERROR  
    if plot_abs:
        fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize = figsize,
                           sharex=True, sharey=True)
                
        for time_n in range(no_rows):
            for field_n in range(no_cols):
                ax = axes[time_n,field_n]
    #            field = rel_std[time_n, field_n] * scales[field_n]
                field = std[time_n, field_n]
                ax_title = field_names[field_n]
                unit = units[field_n]
                
    #            cmap = "coolwarm"
    #            cmap = "Greys"
    #            cmap = "Greens"
    #            cmap = "Reds"
                cmap = cmap_lcpp
                
                field_min = 0.
                field_max = field_max_all[field_n]
    #            field_min = field_min_all[field_n]

                xticks_major = None
                xticks_minor = None
                
                if SIM_N == 1:
                    if ax_title == "r_r":
                        field_max = 0.04
                        xticks_minor = np.linspace(0.01,0.03,2)
#                        xticks_minor = np.linspace(1,5,3)
                    if ax_title == "r_c":
#                        field_max = 0.05
                        xticks_minor = np.linspace(0.01,0.05,3)
                    if ax_title == "n_\mathrm{aero}":
                        field_max = 5.
                        xticks_minor = np.linspace(1,5,3)
                
                if SIM_N == 2:
                    if ax_title == "r_r":
                        field_max = 0.03
#                        xticks_minor = np.linspace(0.01,0.03,2)
#                        xticks_minor = np.linspace(1,5,3)
                    if ax_title == "r_c":
#                        field_max = 0.05
                        xticks_minor = np.linspace(0.01,0.03,2)
                    if ax_title == "n_\mathrm{aero}":
#                        pass
#                        field_max = 5.
                        xticks_minor = np.linspace(1,3,2)
                if SIM_N == 10:
                    if ax_title == "r_r":
#                        field_max = 0.04
                        xticks_minor = np.linspace(0.01,0.03,2)
#                        xticks_minor = np.linspace(1,5,3)
                    if ax_title == "r_c":
                        field_max = 0.05
                        xticks_minor = np.linspace(0.01,0.05,3)
                    if ax_title == "n_\mathrm{aero}":
                        field_max = 5.
                        xticks_minor = np.linspace(1,5,3)
                
    #            field_max = field.max()
    #            field_min = field.min()
                
                norm_ = mpl.colors.Normalize 
                    
                str_format = "%.2g"
                
                my_format = False
                oom_factor = 1.0
    
                oom_max = oom = 1
    
                CS = ax.pcolormesh(*grid.corners,
                                   field*oom_factor,
                                   cmap=cmap, alpha=alpha,
                                    edgecolor="face", zorder=1,
    #                                norm = norm_(vmin=-minmax, vmax=minmax),
                                    norm = norm_(vmin=field_min, vmax=field_max),
    #                                norm = norm_(vmin=field_min, vmax=field_max),
                                    rasterized=True,
                                    antialiased=True, linewidth=0.0
                                    )
                CS.cmap.set_under("white")
                
                ax.set_xticks( np.linspace( tick_ranges[0,0],
                                                 tick_ranges[0,1],
                                                 no_ticks[0] ) )
                ax.set_yticks( np.linspace( tick_ranges[1,0],
                                                 tick_ranges[1,1],
                                                 no_ticks[1] ) )
    
                ax.tick_params(axis='both', which='major', labelsize=TKFS,
                               length = 3, width=1)
                ax.grid(color='gray', linestyle='dashed', zorder = 2)
                ax.set_aspect('equal')
                if time_n == no_rows-1:
                    xticks1 = ax.xaxis.get_major_ticks()
                    xticks1[-1].label1.set_visible(False)
                    ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
                if field_n == 0:            
                    ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
                ax.set_title( r"$t$ = {0} min".format(int(save_times[time_n]/60)),
                             fontsize = TTFS)
                if time_n == 0:
                    axins = inset_axes(ax,
                                       width="90%",  # width = 5% of parent_bbox width
                                       height="8%",  # height
                                       loc='lower center',
                                       bbox_to_anchor=(0.0, 1.35, 1, 1),
    #                                   , 1, 1),
                                       bbox_transform=ax.transAxes,
                                       borderpad=0,
                                       )      
                    cbar = plt.colorbar(CS, cax=axins,
    #                                    fraction=0.046, pad=-0.1,
                                        format=mticker.FormatStrFormatter(str_format),
                                        orientation="horizontal"
                                        )
                    axins.xaxis.set_ticks_position("bottom")
                    axins.tick_params(axis="x",direction="inout",which="both")
                    axins.tick_params(axis='x', which='major', labelsize=TKFS,
                                   length = 7, width=1)                
                    axins.tick_params(axis='x', which='minor', labelsize=TKFS,
                                   length = 5, width=0.5,bottom=True)                
                    
                    if xticks_major is not None:
                        axins.xaxis.set_ticks(xticks_major)
                    if xticks_minor is not None:
                        axins.xaxis.set_ticks(xticks_minor, minor=True)
                    if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
                        unit = r"\si{\micro\meter}"                                            
                    axins.set_title(r"${0}$ abs. error ({1})".format(ax_title, unit))
                # my_format dos not work with log scale here!!
                    
                    if my_format:
                        cbar.ax.text(1.0,1.0,
                                     r'$\times\,10^{{{}}}$'.format(oom_max),
                                     va='bottom', ha='right', fontsize = TKFS,
                                     transform=ax.transAxes)
    #                    cbar.ax.text(field_min - (field_max-field_min),
    #                                 field_max + (field_max-field_min)*0.01,
    #                                 r'$\times\,10^{{{}}}$'.format(oom_max),
    #                                 va='bottom', ha='left', fontsize = TKFS,
    #                                 transform=ax.transAxes)
                    cbar.ax.tick_params(labelsize=TKFS)
    
                if show_target_cells:
                    ### ad the target cells
                    no_neigh_x = no_cells_x // 2
                    no_neigh_z = no_cells_z // 2
                    dx = grid.steps[0]
                    dz = grid.steps[1]
                    
                    no_tg_cells = len(target_cell_list[0])
                    LW_rect = .5
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
    
    
    
    #    pad_ax_v = 0.005
        fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
    #    fig.subplots_adjust(wspace=pad_ax_v)
                 
    #    fig.tight_layout()
        if fig_path_abs is not None:
            fig.savefig(fig_path_abs,
        #                    bbox_inches = 0,
                        bbox_inches = 'tight',
                        pad_inches = 0.05,
                        dpi=600
                        )           
    
    ### PLOT REL ERROR  
    if plot_rel:
        fig2, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize = figsize,
                           sharex=True, sharey=True)
                
        for time_n in range(no_rows):
            for field_n in range(no_cols):
                ax = axes[time_n,field_n]
                field = rel_std[time_n, field_n]
    #            field = rel_std[time_n, field_n] * scales[field_n]
    #            field = std[time_n, field_n]
                ax_title = field_names[field_n]
                unit = units[field_n]
                
    #            cmap = "coolwarm"
    #            cmap = "Greys"
#                cmap = "Greens"
                cmap = "Reds"
#                cmap = "YlOrRd"
    #            cmap = cmap_lcpp
                
                field_min = 0.
                field_max = field_max_all_std[field_n]
    #            field_min = field_min_all_std[field_n]
                
    #            field_max = field.max()
    #            field_min = field.min()
                
                xticks_major = None
                xticks_minor = None
    
                norm_ = mpl.colors.Normalize 
                    
                str_format = "%.2g"
                
                my_format = False
                oom_factor = 1.0
    
                oom_max = oom = 1
    
                CS = ax.pcolormesh(*grid.corners,
                                   field*oom_factor,
                                   cmap=cmap, alpha=alpha,
                                    edgecolor="face", zorder=1,
    #                                norm = norm_(vmin=-minmax, vmax=minmax),
                                    norm = norm_(vmin=field_min, vmax=field_max),
    #                                norm = norm_(vmin=field_min, vmax=field_max),
                                    rasterized=True,
                                    antialiased=True, linewidth=0.0
                                    )
                CS.cmap.set_under("white")
                
                ax.set_xticks( np.linspace( tick_ranges[0,0],
                                                 tick_ranges[0,1],
                                                 no_ticks[0] ) )
                ax.set_yticks( np.linspace( tick_ranges[1,0],
                                                 tick_ranges[1,1],
                                                 no_ticks[1] ) )
    
                ax.tick_params(axis='both', which='major', labelsize=TKFS,
                               length = 3, width=1)
                ax.grid(color='gray', linestyle='dashed', zorder = 2)
                ax.set_aspect('equal')
                if time_n == no_rows-1:
                    xticks1 = ax.xaxis.get_major_ticks()
                    xticks1[-1].label1.set_visible(False)
                    ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
                if field_n == 0:            
                    ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
                ax.set_title( r"$t$ = {0} min".format(int(save_times[time_n]/60)),
                             fontsize = TTFS)
                if time_n == 0:
                    axins = inset_axes(ax,
                                       width="90%",  # width = 5% of parent_bbox width
                                       height="8%",  # height
                                       loc='lower center',
                                       bbox_to_anchor=(0.0, 1.35, 1, 1),
    #                                   , 1, 1),
                                       bbox_transform=ax.transAxes,
                                       borderpad=0,
                                       )      
                    cbar = plt.colorbar(CS, cax=axins,
    #                                    fraction=0.046, pad=-0.1,
                                        format=mticker.FormatStrFormatter(str_format),
                                        orientation="horizontal"
                                        )
                    axins.xaxis.set_ticks_position("bottom")
                    axins.tick_params(axis="x",direction="inout",which="both")
                    axins.tick_params(axis='x', which='major', labelsize=TKFS,
                                   length = 7, width=1)                
                    axins.tick_params(axis='x', which='minor', labelsize=TKFS,
                                   length = 5, width=0.5,bottom=True)                
                    
                    if xticks_major is not None:
                        axins.xaxis.set_ticks(xticks_major)
                    if xticks_minor is not None:
                        axins.xaxis.set_ticks(xticks_minor, minor=True)
                    axins.set_title(r"${0}$ rel. error".format(ax_title))
                # my_format dos not work with log scale here!!
                    
                    if my_format:
                        cbar.ax.text(1.0,1.0,
                                     r'$\times\,10^{{{}}}$'.format(oom_max),
                                     va='bottom', ha='right', fontsize = TKFS,
                                     transform=ax.transAxes)
    #                    cbar.ax.text(field_min - (field_max-field_min),
    #                                 field_max + (field_max-field_min)*0.01,
    #                                 r'$\times\,10^{{{}}}$'.format(oom_max),
    #                                 va='bottom', ha='left', fontsize = TKFS,
    #                                 transform=ax.transAxes)
                    cbar.ax.tick_params(labelsize=TKFS)
    
                if show_target_cells:
                    ### ad the target cells
                    no_neigh_x = no_cells_x // 2
                    no_neigh_z = no_cells_z // 2
                    dx = grid.steps[0]
                    dz = grid.steps[1]
                    
                    no_tg_cells = len(target_cell_list[0])
                    LW_rect = .5
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
    
    
    #    pad_ax_h = 0.1     
    #    pad_ax_v = 0.05
    #    pad_ax_v = 0.005
        fig2.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
    #    fig.subplots_adjust(wspace=pad_ax_v)
                 
    #    fig.tight_layout()
        if fig_path_rel is not None:
            fig2.savefig(fig_path_rel,
        #                    bbox_inches = 0,
                        bbox_inches = 'tight',
                        pad_inches = 0.05,
                        dpi=600
                        )           

#%% FUNCTION DEF: PARTICLES TRACKING

# traj = [ pos0, pos1, pos2, .. ]
# where pos0 = [pos_x_0, pos_z_0] is pos at time0
# where pos_x_0 = [x0, x1, x2, ...]
# selection = [n1, n2, ...] --> take only these indic. from the list of traj !!!
def plot_particle_trajectories_MA(traj, grid, selection=None,
                               no_ticks=[6,6], figsize=(8,8),
                               MS=1.0, LW=0.6, arrow_every=5,
                               ARROW_SCALE=12,ARROW_WIDTH=0.005, 
                               TTFS = 10, LFS=10,TKFS=10,fig_name=None,
                               t_start=0, t_end=3600, label_y=True):
    scale_x = 1E-3    
    traj *= scale_x
    
    print (traj.shape)
    
    traj[:,0] = (0.75 - traj[:,0]) % 1.5
    
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
                            arrow_every//2::arrow_every] * scale_x
    pos_z = grid.centers[1][arrow_every//2::arrow_every,
                            arrow_every//2::arrow_every] * scale_x
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
    
    tick_ranges = grid.ranges * scale_x
    
    fig, ax = plt.subplots(figsize=figsize)
    
    marker_list = ("o","x", "+", "D", "s", "^", "P"  )
    
    if traj[0,0].size == 1:
#        ax.plot(traj[:,0], traj[:,1] ,"o", markersize = MS)
        ax.plot(traj[:,0], traj[:,1] ,"-", markersize = MS, linewidth=LW)
    # print(selection)
    else:        
        if selection is None: selection=range(len(traj[0,0]))
        for ID_n, ID in enumerate(selection):
            x = traj[:,0,ID]
            z = traj[:,1,ID]
            ax.plot(x,z, markersize = MS, marker=marker_list[0],
                    linestyle="None", mew=0.2,
                    zorder=99, label=f"{ID_n+1}")
#            ax.plot(x,z,"-", markersize = MS)
#            ax.annotate( f"({ID})", (x[0],z[0]),zorder=100)
    ax.quiver(pos_x, pos_z, vel_x, vel_z,
              pivot = 'mid',
              width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=0 )
    ax.set_xticks( np.linspace( tick_ranges[0,0], tick_ranges[0,1],
                                no_ticks[0] ) )
    ax.set_yticks( np.linspace( tick_ranges[1,0], tick_ranges[1,1],
                                no_ticks[1] ) )
    ax.set_xlim( tick_ranges[0,0], tick_ranges[0,1] )
    ax.set_ylim( tick_ranges[1,0], tick_ranges[1,1] )
    
    ax.set_aspect("equal")
    
    ax.tick_params(axis='both', which='major', labelsize=TKFS)
    ax.set_xlabel('$x$ (km)', fontsize = LFS)
    if label_y:
        ax.set_ylabel('$z$ (km)', fontsize = LFS)
    else:
        ax.tick_params(labelleft=False)
    
#    ax.set_title(
#    'Air velocity field and arbitrary particle trajectories\nfrom $t = $'\
#    + str(t_start) + " s to " + str(t_end) + " s",
#        fontsize = TTFS, y = 1.04)
    ax.legend(
               loc = "lower center",
               bbox_to_anchor=(0.5,1.00),
               ncol = len(selection),
               handlelength=1, handletextpad=0.08,
               columnspacing=0.4, borderpad=0.2,
               markerscale=3)      
#    ax.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncols=6)
    ax.grid(color='gray', linestyle='dashed', zorder = 0)    
    # ax.grid()
#    fig.tight_layout()
    if fig_name is not None:
#        fig.savefig(fig_name)
        fig.savefig(fig_name,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.03,
                    dpi=600
                    )           
        

            