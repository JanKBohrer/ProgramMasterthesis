#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:03:34 2019

@author: jdesk
"""

import numpy as np

import matplotlib.pyplot as plt

from microphysics import compute_water_activity_AS
from microphysics import compute_water_activity_NaCl
from microphysics import compute_mass_from_radius_jit
from microphysics import compute_radius_from_mass_jit
from microphysics import compute_molality_from_mass_fraction
from microphysics import compute_mass_fraction_from_molality
from microphysics import compute_density_AS_solution
from microphysics import compute_density_NaCl_solution
from microphysics import compute_equilibrium_saturation_AS_mf
from microphysics import compute_equilibrium_saturation_NaCl_mf
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl

import constants as c


#%%

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

#%%

#compute_water_activity_AS(w_s)

#compute_water_activity_NaCl(m_w, m_s, w_s):

#compute_molality_from_mass_fraction(mass_fraction_, molecular_weight_)

#compute_equilibrium_saturation_AS(w_s, R_p, T_p, rho_p, sigma_w)
#compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s)

figdir = "/home/jdesk/Masterthesis/Figures/03Particles/"
figname = "Seq_vs_Rp.pdf"

rho_AS = c.mass_density_AS_dry
rho_SC = c.mass_density_NaCl_dry


T_p = 285
R_s_list = [5E-3,10E-3,20E-3]

fig2, axes = plt.subplots(ncols=3, figsize=(6,2.5), sharey = True)
#fig2, axes = plt.subplots(ncols=3, figsize=(6,3))

#TTFS = 10
#LFS = 10
#TKFS = 8

ax_titles = [
        "$R_\mathrm{dry} =$ 5 nm",
        "$R_\mathrm{dry} =$ 10 nm",
        "$R_\mathrm{dry} =$ 20 nm"
        ]

for cnt, R_s in enumerate(R_s_list):
    ax = axes[cnt]

    m_AS = compute_mass_from_radius_jit(R_s, rho_AS)
    m_SC = compute_mass_from_radius_jit(R_s, rho_SC)
    
    w_s_max_AS = 0.5
    
    w_s = np.logspace(-6,np.log10(w_s_max_AS),100)
    
    #m_w = np.logspace( np.log10(m_AS) )
    
    m_w_AS = m_AS * (1/w_s - 1)
    m_w_SC = m_SC * (1/w_s - 1)
    
    aw_AS = compute_water_activity_AS(w_s)
    aw_SC = compute_water_activity_NaCl(m_w_SC, m_SC, w_s)
    
    #fig, axes = plt.subplots(figsize=(6,6))
    #
    #ax = axes
    #ax.plot(m_w_AS, aw_AS)
    #ax.plot(m_w_SC, aw_SC)
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    
    ###
    
    S_AS = compute_equilibrium_saturation_AS_mf(w_s, T_p, m_AS)
    S_SC = compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_SC)
    
    R_AS,_,_ = compute_R_p_w_s_rho_p_AS(m_w_AS, m_AS, T_p)
    R_SC,_,_ = compute_R_p_w_s_rho_p_NaCl(m_w_SC, m_SC, T_p)
    
    ax.plot(R_AS, S_AS, label = "AS")
    ax.plot(R_SC, S_SC, label = "SC")
#    ax.set_xlim(R_AS[-1],8E-1)
    if cnt == 0:
        ax.set_xlim(7E-3,5E-1)
    else:
        ax.set_xlim(1E-2,5E-1)
#    ax.set_xticks((7E-3,1E-2,1E-1,5E-1))
    ax.set_xscale("log")
#    ax.set_yscale("log")
    ax.grid()
    ax.legend(loc="lower right")
    ax.set_xlabel(r"$R_p$ ($\si{\micro m}$)")
    ax.set_title(ax_titles[cnt])
axes[0].set_ylabel(r"$S_\mathrm{eq}$ (-)")

pad_ax_h = 0.1
pad_ax_v = 0.1
fig2.subplots_adjust(hspace=pad_ax_h) #, wspace=pad_ax_v)                    
fig2.subplots_adjust(wspace=pad_ax_v)    
fig2.savefig(figdir + figname,
                    bbox_inches = 'tight',
                    pad_inches = 0.04,
                    dpi=600
                    )     
#fig2, axes = plt.subplots(figsize=(6,12))
#
#mol_SC = compute_molality_from_mass_fraction(w_s, c.molar_mass_NaCl)
#
#ax = axes
##ax.plot(m_w_AS, aw_AS)
#ax.plot(mol_SC, aw_SC)
#ax.set_xscale("log")
#ax.set_yticks(np.arange(0.18,1.02,0.02))
#ax.grid(which="both")
##ax.set_yscale("log")
