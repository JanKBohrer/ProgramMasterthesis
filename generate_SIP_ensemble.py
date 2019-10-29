#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:50:53 2019

@author: jdesk
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import constants as c

from init import dst_expo
from init import compute_quantiles
from init import generate_SIP_ensemble_expo_SingleSIP_weak_threshold
from init import generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint
from init import generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint2
from init import generate_SIP_ensemble_expo_my_xi_rnd
from microphysics import compute_radius_from_mass

#%% GENERATE SIP ENSEMBLE
### SET PARAMETERS
#myOS = "Linux"
myOS = "MacOS"

dV = 1.0
dt = 1.0
# dt = 10.0
# dt = 20.0
t_end = 3600.0
dt_store = 600.0

algorithm = "Shima"
# algorithm = "AON_Unt"

# kernel = "Bott"
# kernel = "Hall"
# kernel = "Long"
kernel = "Long_Bott"
# kernel = "Golovin"

init = "SingleSIP"
# init = "my_xi_random"

no_sims = 5
start_seed = 4711

## for SingleSIP random:
# bin exponential scaling factor
kappa = 10
# kappa = 2000

## for my xi random initialization:
# INTENDED number of SIP:
no_spc = 160
# bin linear spreading parameter
eps = 200
# area of cumulative PDF that is covered, also determines the bin width
p_min = 0
p_max = 1.0 - 1.0E-6

# droplet concentration
#n = 100 # cm^(-3)
n0 = 297.0 # cm^(-3)
# liquid water content (per volume)
LWC0 = 1.0E-6 # g/cm^3
# total number of droplets
no_rpc = int(n0 * dV * 1.0E6)
print("no_rpc=", no_rpc)

### DERIVED
# Unterstrasser 2017 uses monomodal exponential distribution:
# f = 1/mu exp(m/mu)
# mean droplet mass
mu = 1.0E15*LWC0 / n0
print("mu_m=", mu)
# mean radius
# mu_R = 9.3 # mu
mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
print("mu_R=", mu_R)
total_mass_in_cell = dV*LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg

# numerical integration parameters for my xi random init
dm = mu*1.0E-5
m0 = 0.0
m1 = 100*mu

if init == "SingleSIP":
    init_pars = [kappa]
elif init == "my_xi_random":
    init_pars = [no_spc, eps]
# simdata_path, path =\
#     generate_folder_collision(myOS, dV, dt, algorithm, kernel,
#                               init, init_pars, no_sims, gen = True)
store_every = int(math.ceil(dt_store/dt))

seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

#simdata_path = "/home/jdesk/OneDrive/python/sim_data/"
simdata_path = "/Users/bohrer/OneDrive - bwedu/python/sim_data/"
folder = f"collision_box_model/generate_SIPs_nonint2/kappa_{kappa}/"
path = simdata_path + folder

import os
if not os.path.exists(path):
    os.makedirs(path)

# seed = seed_list[4]
for seed in seed_list:
    if init == "SingleSIP":
        masses, xis, m_low, bins =\
            generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint2(
                                  1.0/mu, no_rpc, m_high_by_m_low=1.0E6,
                                  kappa=kappa, seed = seed)    
    elif init == "my_xi_random": 
        masses, xis, m_low, m_high =\
            generate_SIP_ensemble_expo_my_xi_rnd(1.0/mu, no_spc, no_rpc,
                                  total_mass_in_cell,
                                  p_min, p_max, eps,
                                  m0, m1, dm, seed, setseed = True)
    
    np.save(path + f"masses_dV_{dV}_kappa_{kappa}_seed_{seed}.npy", masses)
    np.save(path + f"xis_dV_{dV}_kappa_{kappa}_seed_{seed}.npy", xis)


#%%
# from init import num_int_expo
# print( no_rpc * num_int_expo(0.0, m_low, 1.0/mu, steps=1.0E5) )

#%%

m_high = bins[-1]

no_rpc_dev = []
mass_dev = []
R_maxs = []
R_mins = []
xi0s = []
no_SIPs = []
for seed in seed_list:
    masses = np.load(path + f"masses_dV_{dV}_kappa_{kappa}_seed_{seed}.npy")
    radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
    bins_R = compute_radius_from_mass(bins, c.mass_density_water_liquid_NTP)
    xis = np.load(path + f"xis_dV_{dV}_kappa_{kappa}_seed_{seed}.npy")
    # masses = np.load(path + f"masses_dV_{dV}_no_spc_aim_{no_spc}.npy")
    # xis = np.load(path + f"xis_dV_{dV}_no_spc_aim_{no_spc}.npy")
    
    xi_min = xis.min()
    xi_max = xis.max()
    
    no_SIPs.append(len(xis))
    
    R_min = radii.min()
    R_max = radii.max()
    
    R_maxs.append(R_max)
    R_mins.append(R_min)
    
    xi0s.append(xis[0])
    
    
    m_max = np.amax(masses)
    m_ = np.linspace(0.0,m_max, 10000)
    # m_max = masses[-2]
    m_min = np.amin(masses)
    # no_spc = len(masses)
    no_rpc_is = xis.sum()
    
    total_mass_is = np.sum(masses*xis)
    
    # m_ges = np.sum(masses*xis)
    # bin_size = m_max/no_spc
    # bin_size = m_max/(no_spc-1)
    
    bin_size = (m_high - m_low)/(no_rpc_is)
    
    # print()
    # print(bins_R)
    # print()
    no_rpc_dev.append((no_rpc-no_rpc_is)/no_rpc)
    mass_dev.append((total_mass_in_cell-total_mass_is)/total_mass_in_cell)
    print("seed =", seed)
    print("(no_rpc-no_rpc_is)/no_rpc =", (no_rpc-no_rpc_is)/no_rpc)
    print("(total_mass_in_cell-total_mass_is)/total_mass_in_cell =",
          (total_mass_in_cell-total_mass_is)/total_mass_in_cell)
    
    # print()
    # print()

#%%
    fig_name = f"SIP_ensemble_dV_{dV}_kappa_{kappa}_seed_{seed}.png"
    fig, ax = plt.subplots(figsize=(12,8))
    # ax.plot(masses, xis, "o")
    ax.plot(radii, xis, "o")
    # ax.plot(m_, no_rpc*np.exp(-m_/mu)\
    #             *(np.exp(0.5*bin_size/mu)-np.exp(-0.5*bin_size/mu)))
    # ax.plot(m_, no_rpc*bin_size*dst_expo(m_,1.0/mu))
    ax.vlines(bins_R,xi_min,xi_max, linewidths = 0.5)
    # ax.set_xlim([R_min*0.9, R_max*1.1])
    ax.set_xlim([R_min*0.9, 40.0])
    ax.set_ylim([0.01, 1.0E8])
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    fig.savefig(path + fig_name)
    
    plt.show()
    plt.close()

#%%

# masses1 = np.load(path + "masses.npy")
# xis1 = np.load(path + "xis.npy")
# masses2 = np.load(path + "masses2.npy")
# xis2 = np.load(path + "xis2.npy")
# masses3 = np.load(path + "masses3.npy")
# xis3 = np.load(path + "xis3.npy")

#%% compare with PDF

# import matplotlib.pyplot as plt

# masses = np.load(path + "masses0.npy")
# xis = np.load(path + "xis0.npy")

# m_max = np.amax(masses)
# # xi = np.ones(no_spct, dtype = np.int64)
# # print("sim_n", sim_n, "; masses.shape=", masses.shape, "; m_max=", m_max)

# no_rpc_should = int(n0 * dV * 1.0E6)
# print(xis.sum()-no_rpc_should)

# total_mass_in_cell = dV * LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg
# print(np.sum(masses*xis)-total_mass_in_cell)


# #%%
# from init import compute_quantiles

# dV = 1.0E-6 # m^3
# dt = 1.0
# no_spc = 20

# # dt = 1.0
# store_every = 600
# t_end = 3600.0

# # droplet concentration
# #n = 100 # cm^(-3)
# n0 = 297.0 # cm^(-3)
# # liquid water content (per volume)
# LWC0 = 1.0E-6 # g/cm^3
# # total number of droplets
# no_rpc = int(n0 * dV * 1.0E6)
# print("no_rpc=", no_rpc)
# print("no_spc=", no_spc)
# total_mass_in_cell = dV * LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg
# # we start with a monomodal exponential distribution
# # mean droplet mass
# mu = 1.0E15*LWC0 / n0
# print("mu_m=", mu)
# from microphysics import compute_radius_from_mass
# import constants as c
# mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
# print("mu_R=", mu_R)

# dst = dst_expo
# par = 1/mu
# p_min = 0
# p_max = 0.9999
# dm = mu*1.0E-5
# m0 = 0.0
# m1 = 100*mu
# seed = 4711

# (m_low, m_high), Ps = compute_quantiles(dst_expo, par, m0, m1, dm,
#                                  [p_min, p_max], None)

# print(m_low, f"{m_high:.2e}")

# #%%

# fig_name = f"SIP_ensemble.png"
# fig, ax = plt.subplots(figsize=(8,8))
# ax.plot(masses, xis, "o")
# m_max = np.amax(masses)
# m_ = np.linspace(0.0,m_max, 10000)
# # m_max = masses[-2]
# m_min = np.amin(masses)
# no_spc = len(masses)
# no_rpc = xis.sum()
# m_ges = np.sum(masses*xis)
# # bin_size = m_max/no_spc
# # bin_size = m_max/(no_spc-1)
# bin_size = (m_high - m_low)/(20)
# # ax.plot(m_, no_rpc*np.exp(-m_/mu)\
# #             *(np.exp(0.5*bin_size/mu)-np.exp(-0.5*bin_size/mu)))
# ax.plot(m_, no_rpc*bin_size*dst_expo(m_,1.0/mu))

# print(bin_size)
# # no_bins = 50
# # ax.hist(masses, density=True, bins=50)
# # bins = ax.hist(masses, weights=xis, bins=8)[1]
# # no_bins = len(bins - 1)
# # ax.plot(m_, (bins[-1]-bins[0])/no_bins*masses.shape[0]*dst_expo(m_, 1.0/mu))
# # ax.plot(m_, (bins[-1]-bins[0])/no_bins*no_rpc*dst_expo(m_, 1.0/mu))

