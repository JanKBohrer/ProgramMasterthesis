#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:03:03 2019

@author: jdesk

module contains functions for modelling the collision of warm cloud droplets
"""

#import os
import numpy as np
from numba import njit, vectorize
import math

import constants as c
from microphysics import compute_radius_from_mass_jit
#from microphysics import compute_mass_from_radius_jit
from grid import bilinear_weight

#%% PERMUTATION
# returns a permutation of the list of integers [0,1,2,..,N-1] by Shima method
@njit()
def generate_permutation(N):
    permutation = np.zeros(N, dtype=np.int64)
    for n_next in range(1, N):
        q = np.random.randint(0,n_next+1)
        if q==n_next:
            permutation[n_next] = n_next
        else:
            permutation[n_next] = permutation[q]
            permutation[q] = n_next
    return permutation

#%% TERMINAL VELOCITY AS FUNCTION OF RADIUS

# par[0] belongs to the largest exponential x^(n-1) for par[i], i = 0, .., n 
@njit()
def compute_polynom(par,x):
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res

# terminal velocity of falling cloud droplets
# From Beard 1976, used also in Bott 1998 and Unterstrasser 2017
# For radii larger the cutoff R_max, we set R = R_max for w_sed
# i.e. the term. velocity is set const. at a max. value for R >= R_max
# these parameters for the polys are checked with Beard 1976
# note, that they dont have units, because Y = polynom(pi,X)
# and Y and X are dimensionless   
# the material constants are taken from Bott
# int Bott's Code, there is a another version of the algorithm
# implemented, which is, however, 1 to 1 equivalent
# with the parameter factor 1.255 instead of 1.257
# this function gets close to the provided values of Bott (vBeard.dat)
# 2E6 is the max. rel. error (coming from rounding errors?)
p1_Beard = (-0.318657e1, 0.992696, -0.153193e-2,
          -0.987059e-3,-0.578878e-3,0.855176e-4,-0.327815e-5)[::-1]
p2_Beard = (-0.500015e1,0.523778e1,-0.204914e1,0.475294,-0.542819e-1,
           0.238449e-2)[::-1]
one_sixth = 1.0/6.0

v_max_Beard = 9.04929248
#v_max_Beard2 = 9.04927508845
@njit()
def compute_terminal_velocity_Beard(R):
    rho_w = 1.0E3
    rho_a = 1.225
    viscosity_air_NTP = 1.818E-5
    sigma_w_NTP = 73.0E-3
    # R in mu = 1E-6 m
    R_0 = 10.0
    R_1 = 535.0
    R_max = 3500.0
    drho = rho_w-rho_a
    # drho = c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP
    if R < R_0:
        l0 = 6.62E-2 # mu
        # this is converted for radius instead of diameter
        # i.e. my_C1 = 4*C1_Beard
        C1 = drho*c.earth_gravity / (4.5*viscosity_air_NTP)
        # Bott uses factor 1.257 in data from Unterstrasser, but in paper 
        # factor is 1.255 ...
        C_sc = 1.0 + 1.257 * l0 / R
#        C_sc = 1.0 + 1.255 * l0 / R
        v = C1 * C_sc * R * R * 1.0E-12
    elif R < R_1:
        N_Da = 32.0E-18 * R*R*R * rho_a * drho \
               * c.earth_gravity / (3.0 * viscosity_air_NTP*viscosity_air_NTP)
        Y = np.log(N_Da)
        Y = compute_polynom(p1_Beard,Y)
        l0 = 6.62E-2 # mu
        # Bott uses factor 1.257 in data from Unterstrasser, but in paper 
        # factor is 1.255 ...
        C_sc = 1.0 + 1.257 * l0 / R
#        C_sc = 1.0 + 1.255 * l0 / R
        v = viscosity_air_NTP * C_sc * np.exp(Y)\
            / (rho_a * R * 2.0E-6)
    elif R < R_max:
        N_Bo = 16.0E-12 * R*R * drho * c.earth_gravity / (3.0 * sigma_w_NTP)
        N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
                 * rho_a * rho_a 
                 / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
        Y = np.log(N_Bo * N_P16)
        Y = compute_polynom(p2_Beard,Y)
        v = viscosity_air_NTP * N_P16 * np.exp(Y)\
            / (rho_a * R * 2.0E-6)
    else: v = v_max_Beard
    # else:
    #     # IN WORK: precalc v_max form R_max
    #     N_Bo = 16.0E-12 * drho * c.earth_gravity\
    #             / (3.0 * sigma_w_NTP) * R_max * R_max
    #     N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
    #               * rho_a * rho_a 
    #               / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
    #     Y = np.log(N_Bo * N_P16)
    #     Y = compute_polynom(p2_Beard,Y)
    #     v = viscosity_air_NTP * N_P16 * np.exp(Y)\
    #         / (rho_a * R_max * 2.0E-6)
    return v

@vectorize("float64(float64)")
def compute_terminal_velocity_Beard_vec(R):
    return compute_terminal_velocity_Beard(R)

@njit()
def update_velocity_Beard(vel, R):
    for i in range(vel.shape[0]):
        vel[i] = compute_terminal_velocity_Beard(R[i])
    
# terminal velocity of falling cloud droplets
# From Beard 1976, used also in Bott 1998 and Unterstrasser 2017
# For radii larger the cutoff R_max, we set R = R_max for w_sed
# i.e. the term. velocity is set const. at a max. value for R >= R_max
# these parameters for the polys are checked with Beard 1976
# note, that they dont have units, because Y = polynom(pi,X)
# and Y and X are dimensionless
# in Bott Code: sigma_w = 73.0, rho_w = 1.0, rho_a = 1.225E-3, g = 980.665
# the function was tested vs the tabulated values from Unterstrasser/Beard
# the rel devs. are (my_v_sed > v_sed_Beard) and 
# (rel dev <0.6% for small R and <0.8% for large R)
# and might come from different material
# constants: sigma_w, g, rho_a, rho_w ...
from atmosphere import compute_viscosity_air
from atmosphere import compute_surface_tension_water
viscosity_air_NTP = compute_viscosity_air(293.15)
sigma_w_NTP = compute_surface_tension_water(293.15)

#p1_Beard = (-0.318657e1, 0.992696, -0.153193e-2,
#          -0.987059e-3,-0.578878e-3,0.855176e-4,-0.327815e-5)[::-1]
#p2_Beard = (-0.500015e1,0.523778e1,-0.204914e1,0.475294,-0.542819e-1,
#           0.238449e-2)[::-1]
#one_sixth = 1.0/6.0
v_max_Beard_my_mat_const = 9.11498
@njit()
def compute_terminal_velocity_Beard_my_mat_const(R):
    # R in mu = 1E-6 m
    R_0 = 9.5
    R_1 = 535.0
    R_max = 3500.0
    drho = c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP
    if R < R_0:
        l0 = 6.62E-2 # mu
        # this is converted for radius instead of diameter
        # i.e. my_C1 = 4*C1_Beard
        C1 = drho*c.earth_gravity / (4.5*viscosity_air_NTP)
        C_sc = 1.0 + 1.255 * l0 / R
        v = C1 * C_sc * R * R * 1.0E-12
    elif R < R_1:
        N_Da = 32.0E-18 * R*R*R * c.mass_density_air_dry_NTP * drho \
               * c.earth_gravity / (3.0 * viscosity_air_NTP*viscosity_air_NTP)
        Y = np.log(N_Da)
        Y = compute_polynom(p1_Beard,Y)
        l0 = 6.62E-2 # mu
        C_sc = 1.0 + 1.255 * l0 / R
        v = viscosity_air_NTP * C_sc * np.exp(Y)\
            / (c.mass_density_air_dry_NTP * R * 2.0E-6)
    elif R < R_max:
        N_Bo = 16.0E-12 * R*R * drho * c.earth_gravity / (3.0 * sigma_w_NTP)
        N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
                 * c.mass_density_air_dry_NTP * c.mass_density_air_dry_NTP 
                 / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
        Y = np.log(N_Bo * N_P16)
        Y = compute_polynom(p2_Beard,Y)
        v = viscosity_air_NTP * N_P16 * np.exp(Y)\
            / (c.mass_density_air_dry_NTP * R * 2.0E-6)
    else: v = v_max_Beard_my_mat_const
    # else:
    #     # IN WORK: precalc v_max form R_max
    #     N_Bo = 16.0E-12 * drho * c.earth_gravity\
    #            / (3.0 * sigma_w_NTP) * R_max * R_max
    #     N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
    #              * c.mass_density_air_dry_NTP * c.mass_density_air_dry_NTP 
    #              / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
    #     Y = np.log(N_Bo * N_P16)
    #     Y = compute_polynom(p2_Beard,Y)
    #     v = viscosity_air_NTP * N_P16 * np.exp(Y)\
    #         / (c.mass_density_air_dry_NTP * R_max * 2.0E-6)
    return v

#%% COLLECTION, COLLISION, COALESCENCES EFFICIENCIES
# E_collection = E_collision * E_coalescence

### Hall 1980: Collision Efficiency table -> assume E_coalescence = 1.0 f.a. R

# in Hall_E_col:
# row =  collector drop radius (radius of larger drop)
# column = ratio of R_small/R_large "collector ratio"  
Hall_E_col = np.load("collision/kernel_data/Hall/Hall_collision_efficiency.npy")
Hall_R_col = np.load("collision/kernel_data/Hall/Hall_collector_radius.npy")
Hall_R_col_ratio = np.load("collision/kernel_data/Hall/Hall_radius_ratio.npy")
@njit()
def linear_weight(i, weight, f):
    return f[i+1]*weight + f[i]*(1.0-weight)
@njit()
def compute_E_col_Hall(R_i, R_j):
    if R_i <= 0.0 or R_j <= 0.0:
        return 0.0
    if R_i < R_j:
        R_col = R_j
        R_ratio = R_i/R_j
    else:
        R_col = R_i
        R_ratio = R_j/R_i
    if R_col > 300.0:
        return 1.0
    else:
        # NOTE that ind_R is for index of R_collection,
        # which indicates the row of Hall_E_col
        ind_R = int(R_col/10.0)
        ind_ratio = int(R_ratio/0.05)
        if ind_R == Hall_R_col.shape[0]-1:
            if ind_ratio == Hall_R_col_ratio.shape[0]-1:
                return 1.0
            else:
                weight = (R_ratio - ind_ratio * 0.05) / 0.05
                return linear_weight(ind_ratio, weight, Hall_E_col[ind_R])
        elif ind_ratio == Hall_R_col_ratio.shape[0]-1:
            weight = (R_col - ind_R * 10.0) / 10.0
            return linear_weight(ind_R, weight, Hall_E_col[:,ind_ratio])
        else:
            weight_1 = (R_col - ind_R * 10.0) / 10.0
            weight_2 = (R_ratio - ind_ratio * 0.05) / 0.05
#        print(R_col, R_ratio)
#        print(ind_R, ind_ratio, weight_1, weight_2)
            return bilinear_weight(ind_R, ind_ratio,
                                   weight_1, weight_2, Hall_E_col)
        # E_col = bilinear_weight(ind_1, ind_2, weight_1, weight_2, Hall_E_col)

# Col. Eff. "Long" as used in Unterstrasser 2017, which he got from Bott (1998?)
# IN: rad in mu
@njit()
def compute_E_col_Long_Bott(R_i, R_j):
    if R_j >= R_i:
        R_max = R_j
        R_min = R_i
    else:
        R_max = R_i
        R_min = R_j
        
    if R_max <= 50:
        return 4.5E-4 * R_max * R_max \
                * ( 1.0 - 3.0 / ( max(3.0, R_min) + 1.0E-2) )
    else: return 1.0    
    
### Kernel "Hall" used by Unterstrasser 2017, he got it from Bott bin-code
# NOTE that there are differences in the Effi table:        
# For R_collec <= 30 mu: major differences: Botts Effis are LARGER
# Bott gives values for 6,8,15,25 mu, which are not incl in Hall orig.
# Bott: if Effi  is Effi_crit > 1.0 for R = R_crit and some R_ratio, then all
# Effi for R > R_crit are Effi = Effi_crit
# for R_ratio >= 0.2, the values are quite close, even for small R, BUT
# at some points, there are deviations...
# in Hall_E_col:
# row =  collector drop radius (radius of larger drop)
# column = ratio of R_small/R_large "collector ratio"  
Hall_Bott_E_col = np.load("collision/kernel_data/Hall/Hall_Bott_collision_efficiency.npy")
Hall_Bott_R_col = np.load("collision/kernel_data/Hall/Hall_Bott_collector_radius.npy")
Hall_Bott_R_col_ratio = np.load("collision/kernel_data/Hall/Hall_Bott_radius_ratio.npy")        
@njit()
def compute_E_col_Hall_Bott(R_i, R_j):
    dR_ipol = 2.0
    dratio = 0.05
    # if R_i <= 0.0 or R_j <= 0.0:
    #     return 0.0
    if R_i < R_j:
        R_col = R_j
        if R_col <= 0.0:
            return Hall_Bott_E_col[0,0]
        else:
            R_ratio = R_i/R_j
        # R_ratio = R_i/R_j
    else:
        R_col = R_i
        if R_col <= 0.0:
            return Hall_Bott_E_col[0,0]
        else:
            R_ratio = R_j/R_i
        # R_ratio = R_j/R_i
    ind_ratio = min(int(R_ratio/dratio), Hall_Bott_R_col_ratio.shape[0]-1)
    # ind_R is index of R_collection,
    # which indicates the row of Hall_Bott_E_col 
    ind_R = min(int(R_col/dR_ipol), Hall_Bott_R_col.shape[0]-1)
    
    if ind_R == Hall_Bott_R_col.shape[0]-1:
        if ind_ratio == Hall_Bott_R_col_ratio.shape[0]-1:
            return 4.0
        else:
            weight = (R_ratio - ind_ratio * dratio) / dratio
            return linear_weight(ind_ratio, weight, Hall_Bott_E_col[ind_R])
    elif ind_ratio == Hall_Bott_R_col_ratio.shape[0]-1:
        weight = (R_col - ind_R * dR_ipol) / dR_ipol
        return linear_weight(ind_R, weight, Hall_Bott_E_col[:,ind_ratio])
    else:
        weight_1 = (R_col - ind_R * dR_ipol) / dR_ipol
        weight_2 = (R_ratio - ind_ratio * dratio) / dratio
#        print(R_col, R_ratio)
#        print(ind_R, ind_ratio, weight_1, weight_2)
        return bilinear_weight(ind_R, ind_ratio,
                               weight_1, weight_2, Hall_Bott_E_col)

#%% COLLECTION KERNELS
## IN:
# R1, R2 (in mu)
# collection effic. E_col (-)
# absolute difference in terminal velocity (>0) in m/s
## OUT: Hydrodynamic collection kernel in m^3/s
@njit()
def compute_kernel_hydro(R_1, R_2, E_col, dv):
    return math.pi * (R_1 + R_2) * (R_1 + R_2) * E_col * dv * 1.0E-12

# Kernel "Long" as used in Unterstrasser 2017, which he got from Bott (1998?)
# NOTE that Unterstrasser uses a discretized version of the kernel
# this function does a direct computation for given (R_i, R_j)
@njit()
def compute_kernel_Long_Bott_R(R_i, R_j):
    E_col = compute_E_col_Long_Bott(R_i, R_j)
    dv = abs(compute_terminal_velocity_Beard(R_i)\
             - compute_terminal_velocity_Beard(R_j))
    return compute_kernel_hydro(R_i, R_j, E_col, dv)

# Kernel "Long" direct compute for given masses 
# masses in 1E-18 kg
@njit()
def compute_kernel_Long_Bott_m(m_i, m_j, mass_density):
    R_i = compute_radius_from_mass_jit(m_i, mass_density)
    R_j = compute_radius_from_mass_jit(m_j, mass_density)
    return compute_kernel_Long_Bott_R(R_i, R_j)

# Kernel "Golovin" 
# masses in 1E-18 kg
# parameter from Unterstrasser 2017 b = 1.5 m^3/(kg s)
@njit()
def compute_kernel_Golovin(m_i, m_j):
    return (m_i + m_j) * 1.5E-18
    
#%% GENERATION (DISCRETIZATION) OF KERNEL AND EFFICIENCY GRIDS
    
# R_low, R_high in mu = 1E-6 m
# no_bins_10: number of bins per radius decade
# mass_density in kg/m^3
# R_low and R_high are both included in the radius_grid range interval
# but R_high itself might NOT be a value of the radius_grid
# (which is def by no_bins_10 and R_low)
def generate_kernel_grid_Long_Bott_np(R_low, R_high, no_bins_10,
                                      mass_density):
    
    bin_factor = 10**(1.0/no_bins_10)
    
    no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
    
    radius_grid = np.zeros( no_bins, dtype = np.float64 )
    radius_grid[0] = R_low
    for bin_n in range(1,no_bins):
        radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
    
    # generate velocity grid first at pos. of radius_grid
    
    vel_grid = compute_terminal_velocity_Beard_vec(radius_grid)
    
#    vel_grid = np.zeros( no_bins, dtype = np.float64 )
#    for i in range(no_bins):
#        vel_grid[i] = compute_terminal_velocity_Beard(radius_grid[i])
    
    kernel_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    for j in range(1,no_bins):
        R_j = radius_grid[j]
        v_j = vel_grid[j]
        for i in range(j):
            R_i = radius_grid[i]
            ### Kernel "LONG" provided by Bott (via Unterstrasser)
            if R_j <= 50.0:
                E_col = 4.5E-4 * R_j * R_j \
                        * ( 1.0 - 3.0 / ( max(3.0, R_i) + 1.0E-2 ) )
            else: E_col = 1.0
            
            kernel_grid[j,i] = 1.0E-12 * math.pi * (R_i + R_j) * (R_i + R_j) \
                               * E_col * abs(v_j - vel_grid[i]) 
            kernel_grid[i,j] = kernel_grid[j,i]

    c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
#    c_mass_to_radius = 1.0 / c_radius_to_mass
    mass_grid = c_radius_to_mass * radius_grid**3

    return kernel_grid, vel_grid, mass_grid, radius_grid
generate_kernel_grid_Long_Bott = njit()(generate_kernel_grid_Long_Bott_np)

@njit()
def interpol_bilin(i,j,p,q,f):
    return (1.-p)*(1.-q)*f[i,j] + (1.-p)*q*f[i,j+1]\
           + p*(1.-q)*f[i+1,j] + p*q*f[i+1,j+1]

Hall_Bott_E_col_raw_corr = np.load("collision/kernel_data/Hall/Hall_Bott_E_col_table_raw_corr.npy")
#Hall_Bott_E_col_raw_corr = np.load("collision/kernel_data/Hall/Hall_Bott_E_col_table_raw.npy")
Hall_Bott_R_col_raw = np.load("collision/kernel_data/Hall/Hall_Bott_R_col_table_raw.npy")
Hall_Bott_R_ratio_raw = np.load("collision/kernel_data/Hall/Hall_Bott_R_ratio_table_raw.npy") 
# R_low must be > 0 !
def generate_E_col_grid_R_Hall_Bott_corr_np(R_low, R_high, no_bins_10,
                                            radius_grid_in=None):
    
    no_R_table = Hall_Bott_R_col_raw.shape[0]
    no_rat_table = Hall_Bott_R_ratio_raw.shape[0]
    
    if radius_grid_in is None:
        bin_factor = 10**(1.0/no_bins_10)
        no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
        radius_grid = np.zeros( no_bins, dtype = np.float64 )
        
        radius_grid[0] = R_low
        for bin_n in range(1,no_bins):
            radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor   
    else:
        radius_grid = radius_grid_in
        no_bins = radius_grid_in.shape[0]
    
    kernel_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    # R = larger radius
    # r = smaller radius
    for ind_R_out, R_col in enumerate(radius_grid):
        # get index of coll. radius lower boundary (floor)
        if R_col <= Hall_Bott_R_col_raw[0]:
            ind_R_table = -1
        elif R_col > Hall_Bott_R_col_raw[-1]:
            ind_R_table = no_R_table - 1 # = 14
        else:
            for ind_R_table_ in range(0,no_R_table-1):
                # we want drops larger 300 mu to have Ecol=1
                # thus drops with R = 300 mu (exact) go to another category
                if (Hall_Bott_R_col_raw[ind_R_table_] < R_col) \
                   and (R_col <= Hall_Bott_R_col_raw[ind_R_table_+1]):
                    ind_R_table = ind_R_table_
                    break
        for ind_r_out in range(ind_R_out+1):
            ratio = radius_grid[ind_r_out] / R_col
            # get index of radius ratio lower boundary (floor)
            # index of ratio should be at least one smaller
            # than the max. possible index. The case ratio = 1 is also covered
            # by the bilinear interpolation
            if ratio >= 1.0: ind_ratio = no_rat_table-2
            else:
                for ind_ratio_ in range(no_rat_table-1):
                    if (Hall_Bott_R_ratio_raw[ind_ratio_] <= ratio) \
                       and (ratio < Hall_Bott_R_ratio_raw[ind_ratio_+1]):
                        ind_ratio = ind_ratio_
                        break
            # bilinear interpolation:
            # linear interpol in ratio, if R >= R_max
            if ind_R_table == no_R_table - 1:
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )
                E_col = (1.-q) * Hall_Bott_E_col_raw_corr[ind_R_table,ind_ratio]\
                        + q * Hall_Bott_E_col_raw_corr[ind_R_table,ind_ratio+1]
                if E_col <= 1.0:
                    kernel_grid[ind_R_out, ind_r_out] = E_col
                else:    
                    kernel_grid[ind_R_out, ind_r_out] = 1.0
#                kernel_grid[ind_R_out, ind_r_out] = min(E_col, 1.0)
            elif ind_R_table == -1:
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )
                kernel_grid[ind_R_out, ind_r_out] =\
                        (1.-q) * Hall_Bott_E_col_raw_corr[0,ind_ratio]\
                        + q * Hall_Bott_E_col_raw_corr[0,ind_ratio+1] 
            else:
                p = ( R_col - Hall_Bott_R_col_raw[ind_R_table] ) \
                    / ( Hall_Bott_R_col_raw[ind_R_table+1] 
                        - Hall_Bott_R_col_raw[ind_R_table] )                
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )                
                kernel_grid[ind_R_out, ind_r_out] = \
                    interpol_bilin(ind_R_table, ind_ratio, p, q,
                                   Hall_Bott_E_col_raw_corr)
            kernel_grid[ind_r_out, ind_R_out] =\
                kernel_grid[ind_R_out, ind_r_out]
    return kernel_grid, radius_grid
generate_E_col_grid_R_Hall_Bott_corr = \
    njit()(generate_E_col_grid_R_Hall_Bott_corr_np)

# R_low, R_high in mu = 1E-6 m
# no_bins_10: number of bins per radius decade
# mass_density in kg/m^3
# R_low and R_high are both included in the radius_grid range interval
# but R_high itself might NOT be a value of the radius_grid
# (which is def by no_bins_10 and R_low)
def generate_and_save_kernel_grid_Long_Bott(R_low, R_high, no_bins_10,
                                               mass_density, save_dir):
    kernel_grid, vel_grid, mass_grid, radius_grid = \
        generate_kernel_grid_Long_Bott(R_low, R_high, no_bins_10,
                                       mass_density)
    
    np.save(save_dir + "radius_grid_out.npy", radius_grid)
    np.save(save_dir + "mass_grid_out.npy", mass_grid)
    np.save(save_dir + "kernel_grid.npy", kernel_grid)
    np.save(save_dir + "velocity_grid.npy", vel_grid)        
    
    return kernel_grid, vel_grid, mass_grid, radius_grid

def generate_E_col_grid_R_np(R_low, R_high, no_bins_10, kernel_name):
    
    if kernel_name == "Hall_Bott":
        E_col_grid, radius_grid = \
            generate_E_col_grid_R_Hall_Bott_corr(R_low, R_high,
                                                 no_bins_10, None)
    elif kernel_name == "Long_Bott":
        bin_factor = 10**(1.0/no_bins_10)
        no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
        radius_grid = np.zeros( no_bins, dtype = np.float64 )
        
        radius_grid[0] = R_low
        for bin_n in range(1,no_bins):
            radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
        
        # vel grid ONLY for testing...
    #    vel_grid = np.zeros( no_bins, dtype = np.float64 )
    #    for i in range(no_bins):
    #        vel_grid[i] = kernel.compute_terminal_velocity_Beard(radius_grid[i])
        
        E_col_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
        
        for j in range(0,no_bins):
            R_j = radius_grid[j]
            for i in range(j+1):
                E_col_grid[j,i] = compute_E_col_Long_Bott(radius_grid[i], R_j)
                E_col_grid[i,j] = E_col_grid[j,i]
    
    return E_col_grid, radius_grid
#    return E_col_grid, radius_grid, vel_grid
generate_E_col_grid_R = \
    njit()(generate_E_col_grid_R_np)
    
#def generate_E_col_grid_R_np(R_low, R_high, no_bins_10, kernel_name):
#
#    bin_factor = 10**(1.0/no_bins_10)
#    no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
#    radius_grid = np.zeros( no_bins, dtype = np.float64 )
#    
#    radius_grid[0] = R_low
#    for bin_n in range(1,no_bins):
#        radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
#    
#    # vel grid ONLY for testing...
##    vel_grid = np.zeros( no_bins, dtype = np.float64 )
##    for i in range(no_bins):
##        vel_grid[i] = kernel.compute_terminal_velocity_Beard(radius_grid[i])
#    
#    E_col_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
#    
#    if kernel_name == "Long_Bott":
#        for j in range(0,no_bins):
#            R_j = radius_grid[j]
#            for i in range(j+1):
#                E_col_grid[j,i] = compute_E_col_Long_Bott(radius_grid[i], R_j)
#                E_col_grid[i,j] = E_col_grid[j,i]
#    elif kernel_name == "Hall_Bott":
#        for j in range(0,no_bins):
#            R_j = radius_grid[j]
#            for i in range(j+1):
#                E_col_grid[j,i] = compute_E_col_Hall_Bott(radius_grid[i], R_j)
#                E_col_grid[i,j] = E_col_grid[j,i]
#    
#    
#    return E_col_grid, radius_grid
##    return E_col_grid, radius_grid, vel_grid
#generate_E_col_grid_R = \
#    njit()(generate_E_col_grid_R_np)

def generate_E_col_grid_R_from_R_grid_np(radius_grid, kernel_name):

#    bin_factor = 10**(1.0/no_bins_10)
#    no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
#    radius_grid = np.zeros( no_bins, dtype = np.float64 )
#    
#    radius_grid[0] = R_low
#    for bin_n in range(1,no_bins):
#        radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
    
    # vel grid ONLY for testing...
#    vel_grid = np.zeros( no_bins, dtype = np.float64 )
#    for i in range(no_bins):
#        vel_grid[i] = kernel.compute_terminal_velocity_Beard(radius_grid[i])
    no_bins = len(radius_grid)
    
    E_col_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    if kernel_name == "Long_Bott":
        for j in range(0,no_bins):
            R_j = radius_grid[j]
            for i in range(j+1):
                E_col_grid[j,i] = compute_E_col_Long_Bott(radius_grid[i], R_j)
                E_col_grid[i,j] = E_col_grid[j,i]
    elif kernel_name == "Hall_Bott":
        for j in range(0,no_bins):
            R_j = radius_grid[j]
            for i in range(j+1):
                E_col_grid[j,i] = compute_E_col_Hall_Bott(radius_grid[i], R_j)
                E_col_grid[i,j] = E_col_grid[j,i]
    
    
    return E_col_grid, radius_grid
#    return E_col_grid, radius_grid, vel_grid
generate_E_col_grid_R_from_R_grid = \
    njit()(generate_E_col_grid_R_from_R_grid_np)
    
### OLD WORKING FOR LONG BOTT ONLY    
#def generate_E_col_grid_Long_Bott_np(R_low, R_high, no_bins_10):
#
#    bin_factor = 10**(1.0/no_bins_10)
#    no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
#    radius_grid = np.zeros( no_bins, dtype = np.float64 )
#    
#    radius_grid[0] = R_low
#    for bin_n in range(1,no_bins):
#        radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
#    
#    # vel grid ONLY for testing...
##    vel_grid = np.zeros( no_bins, dtype = np.float64 )
##    for i in range(no_bins):
##        vel_grid[i] = kernel.compute_terminal_velocity_Beard(radius_grid[i])
#    
#    E_col_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
#    
#    for j in range(0,no_bins):
#        R_j = radius_grid[j]
#        for i in range(j+1):
#            E_col_grid[j,i] = compute_E_col_Long_Bott(radius_grid[i], R_j)
#            E_col_grid[i,j] = E_col_grid[j,i]
#    
#    return E_col_grid, radius_grid
##    return E_col_grid, radius_grid, vel_grid
#generate_E_col_grid_Long_Bott = \
#    njit()(generate_E_col_grid_Long_Bott_np)

def generate_and_save_E_col_grid_R(R_low, R_high,
                                   no_bins_10, kernel_name,
                                   save_dir):
    E_col_grid, radius_grid = \
        generate_E_col_grid_R(R_low, R_high, no_bins_10, kernel_name)
    
    np.save(save_dir + "radius_grid_out.npy", radius_grid)
    np.save(save_dir + "E_col_grid.npy", E_col_grid)
    
    return E_col_grid, radius_grid

# WORKING VERSION FOR LONG BOTT ONLY
#def generate_and_save_E_col_grid_Long_Bott(R_low, R_high,
#                                           no_bins_10, save_dir):
#    E_col_grid, radius_grid = \
#        generate_E_col_grid_Long_Bott(R_low, R_high, no_bins_10)
#    
#    np.save(save_dir + "radius_grid_out.npy", radius_grid)
#    np.save(save_dir + "E_col_grid.npy", E_col_grid)
#    
#    return E_col_grid, radius_grid
