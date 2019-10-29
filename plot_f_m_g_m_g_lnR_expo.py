import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
mpl.rcParams.update(plt.rcParamsDefault)
#mpl.use("pdf")
mpl.use("pgf")

import numpy as np

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
#import numpy as np

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
LW = 1.5
MS = 2

# raster resolution for e.g. .png
DPI = 600

mpl.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))

#%%

def compute_f_m0(m, DNC, LWC):
    k = DNC/LWC
    return DNC * k * np.exp(-k * m)

def plot_f_m_g_m_g_ln_R(DNC, LWC, m, R, xlim_m, xlim_R, figsize, figname):
    
    f_m0 = compute_f_m0(m, DNC, LWC)
    g_m0 = f_m0 * m
    g_ln_R = 3. * f_m0 * m * m
    
    f_m_max = f_m0.max()
    g_m_max = g_m0.max()
    g_R_max = g_ln_R.max()
    
    scale_f = 1E-9
    scale_gR = 1E9
    
    fig, ax = plt.subplots(figsize=( cm2inch(figsize) ), tight_layout=True)
    
    
    
#    ax.plot(m, f_m0 * scale_f)    
#    ax.plot(m, f_m0 / DNC, c = "k")    
#    ax.plot(m, g_m0 / LWC, "--", c = "0.3")    
    ax.plot(m, f_m0 / f_m_max, c = "k", label = "$f_m$")    
    ax.plot(m, g_m0 / g_m_max, "--", c = "0.3", label = "$g_m$")    
#    ax.plot(m, g_ln_R)    
#    ax.set_xticks(np.arange(-10,41,10))
    ax.set_xlim(xlim_m)
#    ax.set_ylim((1E-6,1E12))
#    ax.set_ylim((3,1E2))
#    ax.set_xlabel("$T$ (K)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Droplet mass (kg)")
#    ax.set_ylabel("$f_m$ ($\mathrm{kg^{-1}\,m^{-3}}$)")
    ax.set_ylabel("Distributions (relative units)")
    ax_ = ax.twiny()
#    ax2 = ax_.twiny()
#    ax_.plot(R, g_ln_R / LWC, ":", c = "0.3")
    ax_.plot(R, g_ln_R / g_R_max, ":", c = "0.3", label = "$g_{\ln R}$")
#    ax_.plot(R, g_ln_R*scale_gR)
    ax_.set_xscale("log")
    ax_.set_xlim(xlim_R)
    ax_.set_xlabel(r"Droplet radius ($\si{\micro\meter}$)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower center")
#    ax.legend()
#    ax_ = ax.twinx()
#    ax2 = ax_.twiny()
#    ax2.plot(R, g_ln_R*scale_gR)
#    ax2.set_xscale("log")
#    ax2.set_yscale("log")
#    ax.grid(which = "both", linestyle="--", linewidth=0.5, c = "0.5", zorder = 0)
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.065
                )

from generate_SIP_ensemble_dst import dst_lognormal

def compute_f_R_lognormal_multi_modal(x, mu_log, sigma_log, DNC):
    res = 0.
    for n in range(len(mu_log)):
        res += DNC[n] * dst_lognormal(x, mu_log[n], sigma_log[n])
    return res  

def plot_f_R_g_R_g_ln_R_lognorm(mu_log, sigma_log, DNC, LWC_tot, R, m, xlim_m,
                                xlim_R, figsize, figname):
    
#    f_m0 = compute_f_m0(m, DNC, LWC)
    f_R0 = compute_f_R_lognormal_multi_modal(R, mu_log, sigma_log, DNC)
#    f_m0 = 
    g_R0 = f_R0 * m
    g_ln_R = g_R0 * R
#    
#    DNC_tot = DNC.sum()
    
    f_R_max = f_R0.max()
    g_R_max = g_R0.max()
    g_lnR_max = g_ln_R.max()
    
#    scale_f = 1E-9
#    scale_gR = 1E9
    
    fig, ax = plt.subplots(figsize=( cm2inch(figsize) ), tight_layout=True)
    
#    ax.plot(m, f_m0 * scale_f)    
#    ax.plot(R, f_R0 / DNC_tot, c = "k")    
#    ax.plot(R, g_R0 / LWC_tot, "--", c = "0.3")    
    ax.plot(R, f_R0 / f_R_max, c = "k", label = "$f_R$")    
    ax.plot(R, g_R0 / g_R_max, "--", c = "0.3", label = "$g_R$")    
#    ax.plot(m, g_ln_R)    
#    ax.set_xticks(np.arange(-10,41,10))
    ax.set_xlim(xlim_m)
#    ax.set_ylim((1E-6,1E12))
#    ax.set_ylim((3,1E2))
#    ax.set_xlabel("$T$ (K)")
#    ax.set_xscale("log")
#    ax.set_yscale("log")
#    ax.set_xlabel("Droplet mass (kg)")
#    ax.set_ylabel("$f_m$ ($\mathrm{kg^{-1}\,m^{-3}}$)")
    ax.set_ylabel("Distributions (relative units)")
    
#    ax_ = ax.twiny()
    ax_ = ax
#    ax2 = ax_.twiny()
#    ax_.plot(R, g_ln_R / LWC_tot, ":", c = "0.3")
    ax_.plot(R, g_ln_R / g_lnR_max, ":", c = "0.3", label = "$g_{\ln R}$")
#    ax_.plot(R, g_ln_R*scale_gR)
    ax_.set_xscale("log")
    ax_.set_yscale("log")
    ax_.set_xlim(xlim_R)
    ax_.set_xlabel(r"Droplet radius ($\si{\micro\meter}$)")
    ax_.legend()
#    ax_ = ax.twinx()
#    ax2 = ax_.twiny()
#    ax2.plot(R, g_ln_R*scale_gR)
#    ax2.set_xscale("log")
#    ax2.set_yscale("log")
#    ax.grid(which = "both", linestyle="--", linewidth=0.5, c = "0.5", zorder = 0)
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.065
                )
  
#%%
import math
LWC0 = 1.0E-3 # kg/m^3
R_mean = 9.3 # in mu
mass_density = 1E3 # approx for water
#mass_density = c.mass_density_water_liquid_NTP
#mass_density = c.mass_density_NaCl_dry
c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
c_mass_to_radius = 1.0 / c_radius_to_mass
m_mean = c_radius_to_mass * R_mean**3 # in kg
DNC0 = LWC0 / m_mean # in 1/m^3

figname = "/home/jdesk/Masterthesis/Figures/02Theory/initDistExpo.pdf"
figsize = (7.6,6)

min_log = -1
max_log = np.log10(30)
#min_log = -15
#max_log = -10
R = np.logspace(min_log,max_log,1000)
#R = (m*c_mass_to_radius)**(1./3.)
m = c_radius_to_mass * R**3
#m = np.logspace(min_log,max_log)
#R = (m*c_mass_to_radius)**(1./3.)

xlim_m = (m[0], m[-1])
xlim_R = (R[0], R[-1])
plot_f_m_g_m_g_ln_R(DNC0, LWC0, m, R, xlim_m, xlim_R, figsize, figname)

#%% lognormal
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

mu_R_log = np.log( mu_R )
sigma_R_log = np.log( sigma_R )    
# derive parameters of lognormal distribution of mass f_m(m)
# assuming mu_R in mu and density in kg/m^3
# mu_m in 1E-18 kg
mu_m_log = np.log(compute_mass_from_radius_vec(mu_R,
                                               c.mass_density_NaCl_dry))
sigma_m_log = 3.0 * sigma_R_log
dst_par = (mu_m_log, sigma_m_log)

def compute_mean_mass_lognormal(mu_m_log, sigma_m_log):
    return np.exp(mu_m_log) * np.exp(0.5 * (sigma_m_log)**2)

mean_mass = compute_mean_mass_lognormal(mu_m_log, sigma_m_log)

LWC0_tot = DNC0[0] * mean_mass[0] + DNC0[1] * mean_mass[1]

mass_density = c.mass_density_NaCl_dry
c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
c_mass_to_radius = 1.0 / c_radius_to_mass

#min_log = -2
min_log = np.log10(4E-3)
max_log = 0
#max_log = np.log10(2)
#min_log = -15
#max_log = -10
R = np.logspace(min_log,max_log,1000)
#R = (m*c_mass_to_radius)**(1./3.)
m = c_radius_to_mass * R**3

figname = "/home/jdesk/Masterthesis/Figures/02Theory/initDistLognorm.pdf"
figsize = (7.6,6)

xlim_m = (m[0], m[-1])
xlim_R = (R[0], R[-1])
plot_f_R_g_R_g_ln_R_lognorm(mu_R_log, sigma_R_log,
                            DNC0, LWC0_tot, R, m, xlim_m,
                            xlim_R, figsize, figname)
