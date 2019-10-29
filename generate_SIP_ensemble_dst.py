#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:08:57 2019

@author: bohrer
"""

import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import constants as c
# from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass_vec
from microphysics import compute_radius_from_mass_jit
from microphysics import compute_mass_from_radius_vec
from microphysics import compute_mass_from_radius_jit


# exponential distr. such that int ( f(x) dx ) = 1
# x_mean_inv = 1.0/x_avg, where x_mean is the mean (and the STD)
@njit()
def dst_expo(x, x_mean_inv):
    return x_mean_inv * np.exp(- x * x_mean_inv)

# log-normal distr. (monomodal) such that int ( f(x) dx ) = 1
# mu_log is the ln() of the geometric mean "mu" (also "mode") of the dst
# sigma_log is the ln() of the geometric STD "sigma" of the dst
# it is x_mean := arithm. expect(x) = mu * exp( 0.5 * sigma_log**2)
# and x_std = SD(x) = x_mean * sqrt( exp( sigma_log**2 ) - 1 )
two_pi_sqrt = np.sqrt(2.0*np.pi)
@njit()
def dst_lognormal(x, mu_log, sigma_log):
    return np.exp( -0.5*( ( np.log( x ) - mu_log ) / sigma_log )**2 ) \
           / (x * two_pi_sqrt * sigma_log)


# generate SIP ensemble from pure PDFs
# return masses and weights 
# -> In a second step multiply weights by 
# total number of real particles in cell "nrpc" to get multiplicities
# note that the multiplicities are not integers but floats
# exponential distribution:
# f = 1/m_avg * exp(-m/m_avg)
def gen_mass_ensemble_weights_SinSIP_expo_np(
        m_mean, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low=1.0E6,
        seed=3711, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = compute_mass_from_radius_jit(r_critmin, mass_density) # in 1E-18 kg
    m_mean_inv = 1.0 / m_mean
    
    bin_factor = 10**(1.0/kappa)
    m_high = m_low * m_high_over_m_low
    m_left = m_low

    l_max = int(kappa * np.log10(m_high_over_m_low))
    rnd = np.random.rand( l_max )
    
    if weak_threshold:
        rnd2 = np.random.rand( l_max )

    weights = np.zeros(l_max, dtype = np.float64)
    masses = np.zeros(l_max, dtype = np.float64)
    bins = np.zeros(l_max+1, dtype = np.float64)
    bins[0] = m_left

    bin_n = 0
    while m_left < m_high:
        m_right = m_left * bin_factor
        bin_width = m_right - m_left
        mu = m_left + rnd[bin_n] * bin_width
        weights[bin_n] = dst_expo(mu, m_mean_inv) * bin_width
        masses[bin_n] = mu
        m_left = m_right

        bin_n += 1
        bins[bin_n] = m_left

    weight_max = weights.max()
    weight_critmin = weight_max * eta

#    valid_ids = np.ones(l_max, dtype = np.bool8)
    valid_ids = np.full(l_max, True)
    for bin_n in range(l_max):
        if weights[bin_n] < weight_critmin:
            if weak_threshold:
                if rnd2[bin_n] < weights[bin_n] / weight_critmin:
                    weights[bin_n] = weight_critmin
                else: valid_ids[bin_n] = False
            else: valid_ids[bin_n] = False
    weights = weights[valid_ids]
    masses = masses[valid_ids]
    
    return masses, weights, m_low, bins
gen_mass_ensemble_weights_SinSIP_expo =\
    njit()(gen_mass_ensemble_weights_SinSIP_expo_np)   

# generate SIP ensemble from pure PDFs
# return masses and weights 
# -> In a second step multiply weights by 
# total number of real particles in cell "nrpc" to get multiplicities
# note that the multiplicities are not integers but floats
# exponential distribution:
# f = 1/m_avg * exp(-m/m_avg)
def gen_mass_ensemble_weights_SinSIP_lognormal_np(
        mu_m_log, sigma_m_log,
        mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low,
        seed, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = compute_mass_from_radius_jit(r_critmin, mass_density) # in 1E-18 kg
    
    bin_factor = 10**(1.0/kappa)
    m_high = m_low * m_high_over_m_low
    m_left = m_low

    l_max = int( math.ceil( kappa * np.log10(m_high_over_m_low)))
    rnd = np.random.rand( l_max )
    
    if weak_threshold:
        rnd2 = np.random.rand( l_max )

    weights = np.zeros(l_max, dtype = np.float64)
    masses = np.zeros(l_max, dtype = np.float64)
    bins = np.zeros(l_max+1, dtype = np.float64)
    bins[0] = m_left

    bin_n = 0
    
    for bin_n in range(l_max):
        m_right = m_left * bin_factor
        bin_width = m_right - m_left
        mu = m_left + rnd[bin_n] * bin_width
        weights[bin_n] = dst_lognormal(mu, mu_m_log, sigma_m_log) * bin_width
        masses[bin_n] = mu
        bins[bin_n+1] = m_right
        m_left = m_right
        
#        print(m_high - m_left)

#    while m_left < m_high:
#        m_right = m_left * bin_factor
#        bin_width = m_right - m_left
#        mu = m_left + rnd[bin_n] * bin_width
#        weights[bin_n] = dst_lognormal(mu, mu_m_log, sigma_m_log) * bin_width
#        masses[bin_n] = mu
#        m_left = m_right
#        
#        print(m_high - m_left)
#
#        bin_n += 1
#        bins[bin_n] = m_left

    weight_max = weights.max()
    weight_critmin = weight_max * eta

#    valid_ids = np.ones(l_max, dtype = np.bool8)
    valid_ids = np.full(l_max, True)
    for bin_n in range(l_max):
        if weights[bin_n] < weight_critmin:
            if weak_threshold:
                if rnd2[bin_n] < weights[bin_n] / weight_critmin:
                    weights[bin_n] = weight_critmin
                else: valid_ids[bin_n] = False
            else: valid_ids[bin_n] = False
    weights = weights[valid_ids]
    masses = masses[valid_ids]
    
    return masses, weights, m_low, bins
gen_mass_ensemble_weights_SinSIP_lognormal =\
    njit()(gen_mass_ensemble_weights_SinSIP_lognormal_np)   


#%% CREATE ENSEMBLE MASSES AND WEIGHTS IN EACH CELL of a given z-level (j-level)
# at the z-level, the mass density and thereby the number of real particles per
# cell is known
# create SIPs for each cell of the level, then
# lump all in one array and assign cell - x values for indexing
# if no_modes = 1: monomodal, mu_m_log, sigma_m_log = scalars
# if no_modes > 1: multimodal, mu_m_log = [mu0, mu1, ...] same for sigm & kappa
# mass density is only for conversion r_critmin -> m_critmin
# for lognorm weak threshold: kappa = 3.5 -> no_SIPs_avg = 20.2
def gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl(no_modes,
        mu_m_log, sigma_m_log, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, seed, no_cells_x, no_rpcm, setseed = True):
    # NOTE that array not possible since nr of SIPs is not equal in each cell
    
    if setseed:
        np.random.seed(seed)
    
    masses_lvl = []
    xis_lvl = []
    cells_x_lvl = []
    
    no_spc_lvl = np.zeros(no_cells_x, dtype = np.int64)
    
    if no_modes > 1:
        modes_lvl = []
    
    for i in range(no_cells_x):
        if no_modes == 1:
            masses, weights, m_low, bins = \
                gen_mass_ensemble_weights_SinSIP_lognormal(
                    mu_m_log, sigma_m_log, mass_density,
                    dV, kappa, eta, weak_threshold, r_critmin,
                    m_high_over_m_low, seed, setseed=False)
            no_sp_cell = len(masses)
            no_spc_lvl[i] += no_sp_cell
            masses_lvl.append(masses)
            xis_lvl.append(weights*no_rpcm)
            cells_x_lvl.append( i*np.ones(no_sp_cell, dtype=np.int64) )
        elif no_modes > 1:
            for mode_n in range(no_modes):
                masses, weights, m_low, bins = \
                    gen_mass_ensemble_weights_SinSIP_lognormal(
                        mu_m_log[mode_n], sigma_m_log[mode_n], mass_density,
                        dV, kappa[mode_n], eta, weak_threshold,
                        r_critmin[mode_n],
                        m_high_over_m_low, seed, setseed=False)
                no_sp_cell = len(masses)
                no_spc_lvl[i] += no_sp_cell
                masses_lvl.append(masses)
                xis_lvl.append(weights * no_rpcm[mode_n])
                cells_x_lvl.append( i*np.ones(no_sp_cell, dtype=np.int64) )
                modes_lvl.append(mode_n*np.ones(no_sp_cell, dtype=np.int64))
    
    masses_lvl = np.concatenate(masses_lvl)
    xis_lvl = np.concatenate(xis_lvl)
    cells_x_lvl = np.concatenate(cells_x_lvl)
    if no_modes > 1:
        modes_lvl = np.concatenate(modes_lvl)
    else: modes_lvl = np.zeros_like(cells_x_lvl)
                    
    return masses_lvl, xis_lvl, cells_x_lvl, modes_lvl, no_spc_lvl

# see above, now for expo dist
# at the z-level, the mass density and thereby the number of real particles per
# cell is known
# create SIPs for each cell of the level, then
# lump all in one array and assign cell - x values for indexing
# if no_modes = 1: monomodal, mu_m_log, sigma_m_log = scalars
# if no_modes > 1: multimodal, mu_m_log = [mu0, mu1, ...] same for sigm & kappa
# mass density is only for conversion r_critmin -> m_critmin
# for lognorm weak threshold: kappa = 3.5 -> no_SIPs_avg = 20.2
# expo is monomodal ONLY!
# IN WORK: UNFINISHED!
def gen_mass_ensemble_weights_SinSIP_expo_z_lvl(
        m_mean, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, seed, no_cells_x, no_rpcm, setseed = True):
    # NOTE that array not possible since nr of SIPs is not equal in each cell
    
    if setseed:
        np.random.seed(seed)
    
    masses_lvl = []
    xis_lvl = []
    cells_x_lvl = []
    
    no_spc_lvl = np.zeros(no_cells_x, dtype = np.int64)
    
#    modes_lvl = []
    
    for i in range(no_cells_x):
        # UNFINISHED!
        masses, weights, m_low, bins = \
            gen_mass_ensemble_weights_SinSIP_expo(
                    m_mean, mass_density,
                    dV, kappa, eta, weak_threshold, r_critmin,
                    m_high_over_m_low,
                    seed, setseed=False)        
        no_sp_cell = len(masses)
        no_spc_lvl[i] += no_sp_cell
        masses_lvl.append(masses)
        xis_lvl.append(weights*no_rpcm)
        cells_x_lvl.append( i*np.ones(no_sp_cell, dtype=np.int64) )
    
    masses_lvl = np.concatenate(masses_lvl)
    xis_lvl = np.concatenate(xis_lvl)
    cells_x_lvl = np.concatenate(cells_x_lvl)
#    modes_lvl = np.concatenate(modes_lvl)
                    
    return masses_lvl, xis_lvl, cells_x_lvl, no_spc_lvl
    
    
#%% CREATE ENSEMBLE MASSES AND WEIGHTS IN EACH CELL

# no_cells = [no_c_x, no_c_z]
def gen_mass_ensemble_weights_SinSIP_expo_grid_np(
        m_mean, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, seed, no_cells, no_rpcm):
    
    np.random.seed(seed)
    
    masses = []
    cells_x = []
    cells_z = []
    xis = []
    
    for j in range(no_cells[1]):
        masses_lvl, xis_lvl, cells_x_lvl, no_spc_lvl = \
            gen_mass_ensemble_weights_SinSIP_expo_z_lvl(
                m_mean, mass_density,
                dV, kappa, eta, weak_threshold, r_critmin,
                m_high_over_m_low, seed, no_cells[0], no_rpcm, setseed = False)
        masses.append(masses_lvl)
        cells_x.append(cells_x_lvl)
        cells_z.append(np.ones_like(cells_x_lvl) * j)
        xis.append(xis_lvl)
    
    masses = np.concatenate(masses)
    xis = np.concatenate(xis)
    cells = np.array( (np.concatenate(cells_x), np.concatenate(cells_z)) )
    
    return masses, xis, cells
    
    
# njit not possible because of lists
def gen_mass_ensemble_weights_SinSIP_lognormal_grid(
        mu_m_log, sigma_m_log, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, seed, no_cells):
    # NOTE that array not possible since the nr of SIPs is not equal in each
    # cell
#    mass_grid = np.zeros( no_cells, dtype = np.float64 )
#    weights_grid = np.zeros( no_cells, dtype = np.float64 )
    
    mass_grid_ji = []
    weights_grid_ji = []
    
    np.random.seed(seed)
    
    no_sp_placed_ji = np.zeros( no_cells, dtype = np.int64 )
    
    for j in range(no_cells[0]):
        mg_ = []
        wg_ = []
        for i in range(no_cells[1]):
            masses, weights, m_low, bins = \
                gen_mass_ensemble_weights_SinSIP_lognormal_np(
                    mu_m_log, sigma_m_log, mass_density,
                    dV, kappa, eta, weak_threshold, r_critmin,
                    m_high_over_m_low, seed, setseed=False)
            mg_.append(masses)
            wg_.append(weights)
            no_sp_placed_ji[j,i] = len(masses)
        mass_grid_ji.append(mg_)
        weights_grid_ji.append(wg_)
    return mass_grid_ji, weights_grid_ji, no_sp_placed_ji
#gen_mass_ensemble_weights_SinSIP_lognormal_grid = \
#    njit()(gen_mass_ensemble_weights_SinSIP_lognormal_grid_np)

#def gen_mass_ensemble_weights_SinSIP_expo_grid_np(
#        m_mean, mass_density,
#        dV, kappa, eta, weak_threshold, r_critmin,
#        m_high_over_m_low,
#        seed, no_cells):


#def moments_analytical_expo(n, DNC, DNC_over_LWC):
#    if n == 0:
#        return DNC
#    else:
#        LWC_over_DNC = 1.0 / DNC_over_LWC
#        return math.factorial(n) * DNC * LWC_over_DNC**n
   
#%%

#mass_density = 1E3
#
#R_mean = 9.3 # mu
#m_mean = compute_mass_from_radius_jit(R_mean, mass_density) # in 1E-18 kg
#
#dV = 1.0
#kappa = 10
#eta = 1E-9
#
##weak_threshold = False
#weak_threshold = True
#
#r_critmin = 0.6
#m_high_over_m_low = 1E6
#
#seed = 3711
#
#no_sims = 100
#
#dist = "lognormal"
#
#no_cells = np.array((10,10))
#
#if dist == "lognormal":
#    r_critmin = 0.001 # mu
#    m_high_over_m_low = 1.0E8
#    
#    # set DNC0 manually only for lognormal distr. NOT for expo
#    DNC0 = 60.0E6 # 1/m^3
#    #DNC0 = 2.97E8 # 1/m^3
#    #DNC0 = 3.0E8 # 1/m^3
#    mu_R = 0.02 # in mu
#    sigma_R = 1.4 #  no unit
#    
#    mu_R_log = np.log(mu_R)
#    sigma_R_log = np.log(sigma_R)
#    
#    # derive parameters of lognormal distribution of mass f_m(m)
#    # assuming mu_R in mu and density in kg/m^3
#    # mu_m in 1E-18 kg
#    mu_m_log = 3.0 * mu_R_log + np.log(c.four_pi_over_three * mass_density)
#    sigma_m_log = 3.0 * sigma_R_log
#    print("dist = lognormal", f"DNC0 = {DNC0:.3e}",
#          "mu_R =", mu_R, "sigma_R = ", sigma_R)
#    dist_par = (DNC0, mu_m_log, sigma_m_log)    
#
#mass_grid, weights_grid = \
#    gen_mass_ensemble_weights_SinSIP_lognormal_grid_np(
#        mu_m_log, sigma_m_log, mass_density,
#        dV, kappa, eta, weak_threshold, r_critmin,
#        m_high_over_m_low, seed, no_cells)
#
    
#%%    
##%%
#
#masses, weights, m_low, bins = \
#    gen_mass_ensemble_weights_SinSIP_expo(
#        m_mean, mass_density, dV, kappa, eta, weak_threshold, r_critmin,
#        m_high_over_m_low, seed, setseed=True)
#
#LWC0 = 1E-3
#DNC0 = 1E18 * LWC0 / m_mean
#nrpc = dV * DNC0
#
#xis = weights * nrpc
#
#radii = compute_radius_from_mass_vec(masses, mass_density)
#
#no_rows = 1
#fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows))
#ax=axes
#ax.plot(radii, xis, "x")
#
#ax.set_xscale("log")
#ax.set_yscale("log")
#if dist == "expo":
#    ax.set_yticks(np.logspace(-4,8,13))
#    ax.set_ylim((1.0E-4,1.0E8))
#ax.grid()
#
#ax.set_xlabel(r"radius ($\mathrm{\mu m}$)")
#ax.set_ylabel(r"mean multiplicity per SIP")
#
#fig.tight_layout()
#
#
##fig, ax = plt.subplots()
##
##ax.plot(radii, xis, "o")
##
##ax.set_xscale("log")
##ax.set_yscale("log")
#
##%%
#
#nrpc_is = np.sum(xis)
#
#print(nrpc, nrpc_is, (nrpc - nrpc_is)/nrpc)
#
#seed_list = np.arange(seed, seed + no_sims*2, 2)
#
#masses_list = []
#xis_list = []
#moments = []
#
#
#for sim_n, seed in enumerate(seed_list):
#    masses, weights, m_low, bins = \
#    gen_mass_ensemble_weights_SinSIP_expo(
#        m_mean, mass_density, dV, kappa, eta, weak_threshold, r_critmin,
#        m_high_over_m_low, seed, setseed=True)
#    masses_list.append(masses)
#    xis_list.append(weights * nrpc)
#    lam0 = xis_list[sim_n].sum()/dV
#    lam1 = np.sum(xis_list[sim_n] * masses * 1E-18)
#    moments.append( (lam0,lam1) )
#
#moments = np.array(moments).T    
#
#moments_an = []
#
#for i in range(2):
#    moments_an.append(moments_analytical_expo(i, DNC0, DNC0/LWC0))
#
#no_rows = 1
#fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows))
#ax=axes
#
#for n in range(2):
#    ax.plot(n*np.ones_like(moments[n]) , moments[n]/moments_an[n],  "o")
#
##ax.set_xscale("log")
##ax.set_yscale("log")
##if dist == "expo":
##    ax.set_yticks(np.logspace(-4,8,13))
##    ax.set_ylim((1.0E-4,1.0E8))
##ax.grid()
#
##ax.set_xlabel(r"radius ($\mathrm{\mu m}$)")
##ax.set_ylabel(r"mean multiplicity per SIP")
#
#fig.tight_layout()

    
