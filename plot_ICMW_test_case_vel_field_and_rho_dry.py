#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:05:53 2019

@author: bohrer
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
mpl.rcParams.update(plt.rcParamsDefault)
#mpl.use("pdf")
mpl.use("pgf")

import numpy as np

from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict, pdf_dict
plt.rcParams.update(pgf_dict)
#plt.rcParams.update(pdf_dict)

from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

#%%
#from os.path import expanduser
# for Mac OS:
#fontpath = expanduser("~/Library/Fonts/LinLibertine_R.otf")
# for LinuxDesk
#fontpath = "/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf"
#fontprop = font_manager.FontProperties(fname=fontpath)

#print(fontpath)
#for font in font_manager.findSystemFonts():
#    print(font)

#font_manager._rebuild()

#%%

#pgf_with_lualatex = {
#    "text.usetex": True,
#    "pgf.rcfonts": False,   # Do not set up fonts from rc parameters.
#    "pgf.texsystem": "lualatex",
##    "pgf.texsystem": "pdflatex",
##    "pgf.texsystem": "xelatex",
#    "pgf.preamble": [
#            r'\PassOptionsToPackage{no-math}{fontspec}',
#            r'\usepackage[ttscale=.9]{libertine}',
##            r'\usepackage[T1]{fontenc}',
#            r'\usepackage[libertine]{newtxmath}',
##            r'\usepackage{unicode-math}',
##            r'\usepackage[]{mathspec}',
#            r'\setmainfont{LinLibertine_R}',
#            r'\setromanfont[]{LinLibertine_R}',
#            r'\setsansfont[]{LinLibertine_R}',
#            r'\DeclareSymbolFont{digits}{TU}{\sfdefault}{m}{n}',
#            r'\DeclareMathSymbol{0}{\mathalpha}{digits}{`0}',
#            r'\DeclareMathSymbol{1}{\mathalpha}{digits}{`1}',
#            r'\DeclareMathSymbol{2}{\mathalpha}{digits}{`2}',
#            r'\DeclareMathSymbol{3}{\mathalpha}{digits}{`3}',
#            r'\DeclareMathSymbol{4}{\mathalpha}{digits}{`4}',
#            r'\DeclareMathSymbol{5}{\mathalpha}{digits}{`5}',
#            r'\DeclareMathSymbol{6}{\mathalpha}{digits}{`6}',
#            r'\DeclareMathSymbol{7}{\mathalpha}{digits}{`7}',
#            r'\DeclareMathSymbol{8}{\mathalpha}{digits}{`8}',
#            r'\DeclareMathSymbol{9}{\mathalpha}{digits}{`9}'           
##            r'\setromanfont[Mapping=tex-text]{LinLibertine_R}',
##            r'\setsansfont[Mapping=tex-text]{LinLibertine_R}',
##            r'\setmathfont{LinLibertine_R}'
##            r'\setmathfont[range={"0031-"0040}]{LinLibertine_R}'
##            r'\setmathsfont(Digits){LinLibertine_R}'
##        r'\usepackage{libertine}',
##        r'\usepackage[libertine]{newtxmath}',
##        r'\usepackage[]{fontspec}',
##        r'\usepackage[no-math]{fontspec}',
#        ]
#}
#mpl.rcParams.update(pgf_with_lualatex)

#pgf_dict = {
##    "backend" : "pgf",    
#    "text.usetex": True,
##    "pgf.rcfonts": False,   # Do not set up fonts from rc parameters.
##    "pgf.texsystem": "lualatex",
#    "pgf.texsystem": "pdflatex",
#    "pgf.preamble": [
#        r'\usepackage{libertine}',
#        r'\usepackage[libertine]{newtxmath}',
#        r'\usepackage[T1]{fontenc}',
##        r'\usepackage[no-math]{fontspec}',
#        ],
#    "font.family": "serif"
#}
#plt.rcParams.update(pgf_dict)
#plt.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))
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

## alignemnt of ticks
#plt.rcParams['xtick.alignment'] = "center"
#plt.rcParams['ytick.alignment'] = "center"

#%%
# data for kinematic velocity field

# input:
# velocity (u,w) at the grid surfaces
# corners (x_c, z_c)
# centers (x,z)
# shifts = (in_x, int_z): shift the drawn arrows right and up
# no_arrows in (x,z)
# no_ticks in (x,z)
# figsize in cm (x,y)
# figname = full path incl ".pdf"
# ARROW_SCALE, ARROW_WIDTH
def plot_velocity_field_centered(velocity, corners, centers, shifts,
                                 no_cells,
                                 no_arrows, no_ticks, figsize, figname,
                                 ARROW_SCALE, ARROW_WIDTH):
    centered_u_field = ( velocity[0][0:-1,0:-1]\
                         + velocity[0][1:,0:-1] ) * 0.5
    centered_w_field = ( velocity[1][0:-1,0:-1]\
                         + velocity[1][0:-1,1:] ) * 0.5

#    no_cells = np.array((len(centered_u_field), len(centered_w_field)))
    
    if no_ticks[0] < no_cells[0]:
        # take no_major_xticks - 1 to get the right spacing
        # in dimension of full cells widths
        tick_every_x = no_cells[0] // (no_ticks[0] - 1)
    else:
        tick_every_x = 1

    if no_ticks[1] < no_cells[1]:
        tick_every_y = no_cells[1] // (no_ticks[1] - 1)
    else:
        tick_every_y = 1

    if no_arrows[0] < no_cells[0]:
        arrow_every_x = no_cells[0] // (no_arrows[0] - 1)
    else:
        arrow_every_x = 1

    if no_arrows[1] < no_cells[1]:
        arrow_every_y = no_cells[1] // (no_arrows[1] - 1)
    else:
        arrow_every_y = 1    
    
    fig, ax = plt.subplots(figsize=( cm2inch(figsize) ), tight_layout=True)
    
    sx = shifts[0]
    sy = shifts[1]
    
    ax.quiver(
        centers[0][sx::arrow_every_x,sy::arrow_every_y],
        centers[1][sx::arrow_every_x,sy::arrow_every_y],
        centered_u_field[sx::arrow_every_x,sy::arrow_every_y],
        centered_w_field[sx::arrow_every_x,sy::arrow_every_y],
              pivot = 'mid',
              width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )    
#    ax.axis('equal',"box")
    ax.set_aspect('equal')
#    ax.set_aspect('equal', 'box')
    ax.set_xticks(np.linspace(corners[0][0,0], corners[0][-1,0], no_ticks[0]) )
    ax.set_yticks(np.linspace(corners[1][0,0], corners[1][0,-1], no_ticks[1]) )
#    ax.set_xticks(corners[0][::tick_every_x,0])
#    ax.set_yticks(corners[1][0,::tick_every_y])
#    ax.set_xticks(corners[0][:,0], minor = True)
#    ax.set_yticks(corners[1][0,:], minor = True)
    ax.set_xlim(corners[0][0,0], corners[0][-1,0])
    ax.set_ylim(corners[1][0,0], corners[1][0,-1])
    
    ax.grid(linestyle="dashed")

    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$z$ (m)')
#    ax2 = ax.twiny()
#    ax2.set_xlabel("$S$ (-) and $r_l/r_{l,\mathrm{max}}$ (-)", color="white")
#    fig.tight_layout()
#    plt.gca().set_axis_off()
#    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#                hspace = 0, wspace = 0)
#    plt.margins(0,0)    
#    plt.gca().xaxis.set_major_locator(plt.NullLocator())
#    plt.gca().yaxis.set_major_locator(plt.NullLocator())    
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.065
                )


#{'center', 'top', 'bottom', 'baseline', 'center_baseline'}
#center_baseline seems to be def, center is OK, little bit too far down but OK
#for label in ax.yaxis.get_ticklabels():
##    label.set_verticalalignment('baseline')
#    label.set_verticalalignment('center')
#grid.mixing_ratio_water_liquid
#grid.mixing_ratio_water_vapor
def plot_rho_T_p_init(centers, corners, no_cells,
                      density, temperature, pressure,
                      potential_temperature, saturation,
                      mixing_ratio_water_liquid,
                      mixing_ratio_water_vapor,
                      no_ticks, figsize, figname):
    
    rho0 = density[0,0]
    T0 = temperature[0,0]
    p0 = pressure[0,0]
    Theta0 = potential_temperature[0,0]
    r_v0 = np.average( mixing_ratio_water_vapor[:,0] )
    r_l0 = np.average( mixing_ratio_water_liquid[:,-1] )
#    S0 = np.average( saturation[:,0] )
    S0 = 1.


#    r_l0 = r_l[0,0]
#    S0 = r_l[0,0]
    
    rho = density[0,:]
    T = temperature[0,:]
    p = pressure[0,:]
    Theta = potential_temperature[0,:]
    r_v = np.average( mixing_ratio_water_vapor, axis = 0 )
    r_l = np.average( mixing_ratio_water_liquid, axis = 0 )
    S = np.average( saturation, axis = 0 )
    
    
    data0 = np.array((rho0,r_v0,T0,p0,Theta0))
    data = np.array((rho, r_v, T ,p ,Theta))
    
    names = [r"$\rho_\mathrm{dry}$", r"$r_v$", r"$T$",
             r"$p$", r"$\Theta$"]
    ls = ["-","--",":","-.",(0, (3, 1, 1, 1, 1, 1))]
    
    z = centers[1][0,:]
    
    fig, ax = plt.subplots(figsize=( cm2inch(figsize) ), tight_layout=True)    
#    ax.plot(rho,z, c = "0.0")
#    ax.plot(p,z, c = "0.2", linestyle = "dashed")
#    ax.plot(T,z, c = "0.4", linestyle = "dashdot")

    for n in range(len(data0)):
#        ax.plot(data[n]/data0[n], z, label = "$r$")
        ax.plot(data[n]/data0[n], z, label = names[n], c = "k",
                linestyle=ls[n])
    
#    ax.set_xlabel("Hallo")

    ax2 = ax.twiny()
    ax2.plot(r_l/r_l0, z, label = "$r_l$", c = "0.6")
    ax2.plot(S,z, label = "$S$", c = "0.6", linestyle="dashdot")
#    ax.set_xticks(np.linspace(corners[0][0,0], corners[0][-1,0], no_ticks[0]) )
    ax.set_yticks(np.linspace(corners[1][0,0], corners[1][0,-1], no_ticks[1]) )
#    ax.set_xticks(corners[0][::tick_every_x,0])
#    ax.set_yticks(corners[1][0,::tick_every_y])
#    ax.set_xticks(corners[0][:,0], minor = True)
#    ax.set_yticks(corners[1][0,:], minor = True)
    
    ax.set_xlim(0.82,1.03)
    ax.set_ylim(corners[1][0,0], corners[1][0,-1])    

    ax.set_xlabel(r'$\rho_\mathrm{dry}$, $r_v$, $T$, $p$, $\Theta$'
                  + ' (relative values)')
#    ax.set_xlabel('Atmospheric state variables (-)')
#    ax.set_xlabel("$r$ (-)")
#    ax.set_xlabel('$S$ (m)')
#    ax.set_ylabel('Hallo (m)')
    ax.set_ylabel('$z$ (m)')
    ax.grid(linestyle="dashed", c = "0.6")
    ax.legend(loc='lower left', bbox_to_anchor=(0.06, -0.01))

    ax2.set_xlabel("$S$ (-) and $r_l/r_{l,\mathrm{max}}$ (-)")    
    ax2.tick_params(axis='x', colors='0.5')
    ax2.legend(loc='lower left', bbox_to_anchor=(0.34, -0.01))
#    ax2.legend()
    
    for n,d_ in enumerate(data0):
        print(names[n], f"{d_:.4e}")
    print(f"{r_l0*1000:.4}")
    
#    fig.tight_layout()
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.05
                )    

#%% load data

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
no_spcm = np.array([16, 24])
#no_spcm = np.array([20, 30])
#no_spcm = np.array([26, 38])

# seed of the SIP generation -> needed for the right grid folder
# 3711, 3713, 3715, 3717
# 3719, 3721, 3723, 3725
seed_SIP_gen = 3711
#seed_SIP_gen_list = [3711, 3713]
#seed_SIP_gen_list = [3711, 3713, 3715, 3717]
no_sims = 50
seed_SIP_gen_list = np.arange(seed_SIP_gen, seed_SIP_gen + no_sims * 2, 2)

# for collisons
seed_sim = 4711
#seed_sim = 6711
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

spin_up_finished = True
#spin_up_finished = False

# path = simdata_path + folder_load_base
t_grid = 0
#t_grid = 7200
#t_grid = 10800
#t_grid = 14400

t_start = 0
#t_start = 7200

#t_end = 60
#t_end = 3600
t_end = 7200
#t_end = 10800
#t_end = 14400
    
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
    
    
#%% plot velocity field
#figname = home_path + "/Masterthesis/Figures/02Theory/kinVelfield.pdf"
#
#
#figsize = (7,7.4)
#
#no_arrows = (16,16)
#shifts=(2,2)
##no_ticks = (11,11)
#no_ticks = (6,6)
#ARROW_SCALE = 17
#ARROW_WIDTH = 0.004
#
#plt.close("all")
#
#plot_velocity_field_centered(grid.velocity, grid.corners, grid.centers,
#                             shifts,
#                             grid.no_cells,
#                                 no_arrows, no_ticks, figsize, figname,
#                                 ARROW_SCALE, ARROW_WIDTH)

#%% plot density, p, T init

#figname = home_path + "/Masterthesis/Figures/02Theory/rhoTpInit.pdf"
#figsize = (8.5,7.4)
#
#no_ticks = (6,6)
#
#plot_rho_T_p_init(grid.centers, grid.corners, grid.no_cells,
#                      grid.mass_density_air_dry, grid.temperature, grid.pressure,
#                      grid.potential_temperature, grid.saturation,
#                      grid.mixing_ratio_water_liquid,
#                      grid.mixing_ratio_water_vapor,
#                      no_ticks, figsize, figname)


#%% DERIVATION OF THE TRANSPORT EQUATION FOR THETA_DRY -> check 
    # if r_v terms and R_m term can be neglected
    
import constants as c
rho_v = grid.mixing_ratio_water_vapor * grid.mass_density_air_dry

e = rho_v * c.specific_gas_constant_water_vapor * grid.temperature

p_d = grid.pressure - e

p_r = grid.p_ref

print("p_r/p_d")
print(p_r/p_d)
print("1 + e/p_d")
print(1 + e/p_d)

kappa_air_dry = c.specific_gas_constant_air_dry\
                / c.specific_heat_capacity_air_dry_NTP

def compute_kappa_air_moist(mixing_ratio_vapor_):
    return kappa_air_dry * ( 1 - 0.289 * mixing_ratio_vapor_ )

# J/(kg K)
def compute_specific_gas_constant_air_moist(specific_humidity_):
    return c.specific_gas_constant_air_dry * (1 + 0.608 * specific_humidity_ )

def compute_heat_of_vaporization(temperature_):
    return 1.0E3 * ( 2500.82 - 2.358 * (temperature_ - 273.0) ) 

kappa_m = compute_kappa_air_moist(grid.mixing_ratio_water_vapor)

print(kappa_air_dry)
print(kappa_m)

Th_d_over_Th = (p_r/p_d)**(kappa_air_dry*0.289 * grid.mixing_ratio_water_vapor) \
               * (1 + e/p_d)**(kappa_m) 

print(Th_d_over_Th)
print(Th_d_over_Th*289.)

c_d = c.specific_heat_capacity_air_dry_NTP
R_d = c.specific_gas_constant_air_dry
c_v = c.specific_heat_capacity_water_vapor_20C
R_v = c.specific_gas_constant_water_vapor

print("1-kappa_d")
print(1 - kappa_air_dry)
print("c_d-R_d")
print(c.specific_heat_capacity_air_dry_NTP - c.specific_gas_constant_air_dry)
print("c_v-R_v")
print(c.specific_heat_capacity_water_vapor_20C - c.specific_gas_constant_water_vapor)

R_m = compute_specific_gas_constant_air_moist(7.5E-3)
print("R_m")
print(R_m)

T = 270
L_v = compute_heat_of_vaporization(270)

print("L_v/T")
print(L_v/T)
print("(L_v/T)/R_m")
print(1/((L_v/T)/R_m))

print("(c_v-R_v) / (c_d - R_d)")
print((c_v-R_v) / (c_d - R_d))

####
p0 = 1015
ps = 1000
rt = 7.5E-3
pd0 = p0 / (1 + R_v/R_d * rt)
T0 = 290.23
print(T0 * (ps/pd0)**kappa_air_dry)

#%%
def compute_viscosity_air(T_):
    return 1.458E-6 * T_**(1.5) / (T_ + 110.4)

print(compute_viscosity_air(290))



