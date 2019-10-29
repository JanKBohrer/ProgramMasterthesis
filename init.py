#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:31:49 2019

@author: jdesk
"""
#%%
import numpy as np
import math
# import matplotlib.pyplot as plt
import sys
from numba import njit

import constants as c
from grid import Grid
from grid import interpolate_velocity_from_cell_bilinear
from microphysics import compute_mass_from_radius_jit
from microphysics import compute_initial_mass_fraction_solute_m_s_NaCl
from microphysics import compute_initial_mass_fraction_solute_m_s_AS, \
                         compute_dml_and_gamma_impl_Newton_full_NaCl,\
                         compute_dml_and_gamma_impl_Newton_full_AS,\
                         compute_R_p_w_s_rho_p_NaCl, \
                         compute_R_p_w_s_rho_p_AS, \
                         compute_surface_tension_NaCl, \
                         compute_surface_tension_AS
#                         compute_mass_from_radius_vec,\
#                         compute_radius_from_mass_vec,\
#                         compute_density_particle,\
#                         compute_initial_mass_fraction_solute_NaCl,\
                         
from atmosphere import compute_kappa_air_moist,\
                       compute_diffusion_constant,\
                       compute_thermal_conductivity_air,\
                       compute_heat_of_vaporization,\
                       compute_saturation_pressure_vapor_liquid,\
                       compute_pressure_vapor,\
                       epsilon_gc, compute_surface_tension_water,\
                       kappa_air_dry,\
                       compute_beta_without_liquid
from file_handling import save_grid_and_particles_full

#from generate_SIP_ensemble_dst import gen_mass_ensemble_weights_SinSIP_lognormal
#from generate_SIP_ensemble_dst import gen_mass_ensemble_weights_SinSIP_lognormal_grid
from generate_SIP_ensemble_dst import \
    gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl    


#%%
# IN WORK: what do you need from grid.py?
# from grid import *
# from physical_relations_and_constants import *

# stream function
pi_inv = 1.0/np.pi
def compute_stream_function_Arabas(x_, z_, j_max_, x_domain_, z_domain_):
    return -j_max_ * x_domain_ * pi_inv * np.sin(np.pi * z_ / z_domain_)\
                                        * np.cos( 2 * np.pi * x_ / x_domain_)

# dry mass flux  j_d = rho_d * vel
# Z = domain height z
# X = domain width x
# X_over_Z = X/Z
# k_z = pi/Z
# k_x = 2 * pi / X
# j_max = Amplitude
def compute_mass_flux_air_dry_Arabas( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
    j_x = j_max_ * X_over_Z_ * np.cos(k_x_ * x_) * np.cos(k_z_ * z_ )
    j_z = 2 * j_max_ * np.sin(k_x_ * x_) * np.sin(k_z_ * z_)
    return j_x, j_z

#def compute_mass_flux_air_dry_x( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
#    return j_max_ * X_over_Z_ * np.cos(k_x_ * x_) * np.cos(k_z_ * z_ )
#def compute_mass_flux_air_dry_z( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
#    return 2 * j_max_ * np.sin(k_x_ * x_) * np.sin(k_z_ * z_)
#def mass_flux_air_dry(x_, z_):
#    return compute_mass_flux_air_dry_Arabas(x_, z_, j_max, k_x, k_z, X_over_Z)
#
#def mass_flux_air_dry_x(x_, z_):
#    return compute_mass_flux_air_dry_x(x_, z_, j_max, k_x, k_z, X_over_Z)
#def mass_flux_air_dry_z(x_, z_):
#    return compute_mass_flux_air_dry_z(x_, z_, j_max, k_x, k_z, X_over_Z)

def compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1( grid_,
                                                                    j_max_ ):
    X = grid_.sizes[0]
    Z = grid_.sizes[1]
    k_x = 2.0 * np.pi / X
    k_z = np.pi / Z
    X_over_Z = X / Z
#    j_max = 0.6 # (m/s)*(m^3/kg)
    # grid only has corners as positions...
    vel_pos_u = [grid_.corners[0], grid_.corners[1] + 0.5 * grid_.steps[1]]
    vel_pos_w = [grid_.corners[0] + 0.5 * grid_.steps[0], grid_.corners[1]]
    j_x = compute_mass_flux_air_dry_Arabas( *vel_pos_u, j_max_,
                                            k_x, k_z, X_over_Z )[0]
    j_z = compute_mass_flux_air_dry_Arabas( *vel_pos_w, j_max_,
                                            k_x, k_z, X_over_Z )[1]
    return j_x, j_z

# j_max_base = the appropriate value for a grid of 1500 x 1500 m^2
def compute_j_max(j_max_base, grid):
    return np.sqrt( grid.sizes[0] * grid.sizes[0]
                  + grid.sizes[1] * grid.sizes[1] ) \
           / np.sqrt(2.0 * 1500.0 * 1500.0)

####
           
#%% DISTRIBUTIONS
           
# par[0] = mu
# par[1] = sigma
two_pi_sqrt = math.sqrt(2.0 * math.pi)
def dst_normal(x, par):
    return np.exp( -0.5 * ( ( x - par[0] ) / par[1] )**2 ) \
           / (two_pi_sqrt * par[1])

# par[0] = mu^* = geometric mean of the log-normal dist
# par[1] = ln(sigma^*) = lognat of geometric std dev of log-normal dist
def dst_log_normal(x, par):
    # sig = math.log(par[1])
    f = np.exp( -0.5 * ( np.log( x / par[0] ) / par[1] )**2 ) \
        / ( x * math.sqrt(2 * math.pi) * par[1] )
    return f


def dst_expo_np(x,k):
    return np.exp(-x*k) * k
dst_expo = njit()(dst_expo_np)

def num_int_np(func, x0, x1, steps=1E5):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    # cnt = 0
    while (x < x1):
        intl += dx * func(x)
        x += dx
        # cnt += 1
    return intl
# num_int = njit()(num_int_np)

def num_int_expo_np(x0, x1, k, steps=1E5):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_expo(x,k)
    # cnt = 0
    while (x < x1):
        f2 = dst_expo(x + 0.5*dx, k)
        f3 = dst_expo(x + dx, k)
        # intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4 * f2 + f3)
        x += dx
        f1 = f3
        # cnt += 1        
        # intl += dx * x * dst_expo(x,k)
        # x += dx
        # cnt += 1
    return intl
num_int_expo = njit()(num_int_expo_np)

# def num_int_expo_np(x0, x1, k, steps=1.0E5):
#     dx = (x1 - x0) / steps
#     x = x0
#     intl = 0.0
#     # cnt = 0
#     while (x < x1):
#         intl += dx * dst_expo(x,k)
#         x += dx
#         # cnt += 1
#     return intl
# num_int_expo = njit()(num_int_expo_np)

def num_int_expo_mean_np(x0, x1, k, steps=1E5):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_expo(x,k) * x
    # cnt = 0
    while (x < x1):
        f2 = dst_expo(x + 0.5*dx, k) * (x + 0.5*dx)
        f3 = dst_expo(x + dx, k) * (x + dx)
        # intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4 * f2 + f3)
        x += dx
        f1 = f3
        # cnt += 1        
        # intl += dx * x * dst_expo(x,k)
        # x += dx
        # cnt += 1
    return intl
num_int_expo_mean = njit()(num_int_expo_mean_np)

def num_int_impl_right_border(func, x0, intl_value, dx, cnt_lim=1E7):
    dx0 = dx
    x = x0
    intl = 0.0
    cnt = 0
    f1 = func(x)
    print("dx0 =", dx0)
    while (intl < intl_value and cnt < cnt_lim):
        # if cnt % 100 == 0:
        #     dx = dx0
        #     while (np.abs((func(x0+dx)-func(x0))/func(x0)) < 1.0E-6) :
        #         dx *= 2.0
        #     while (np.abs((func(x0+dx)-func(x0))/func(x0)) > 1.0E-6) :
        #         dx *= 0.5
        # dp_lim = 1.0E-5
        # if cnt % 100 == 0:
        #     dx = dx0
        #     if f1*dx > dp_lim:
        #         dx = dp_lim/f1
        #         print("cnt=",cnt,"dx=",dx)
        f2 = func(x + 0.5*dx)
        f3 = func(x + dx)
        intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4 * f2 + f3)
        x += dx
        f1 = f3
        cnt += 1
    # print(f"{cnt:.2e}")
    # if x > dx: 
    #     x = x - dx
    # elif x > 0.5 * dx:
    #     x = x - 0.5 * dx
    # else: x = 0.0        
    # return x + 0.5 *dx    
    return x - dx + dx * (intl_value - intl_bef)/(intl - intl_bef)
    
def num_int_expo_impl_right_border_np(x0, intl_value, dx, k, cnt_lim=1E7):
    # dx0 = dx
    x = x0
    intl = 0.0
    cnt = 0
    f1 = dst_expo(x,k)
    # print("dx0 =", dx0)
    while (intl < intl_value and cnt < cnt_lim):
        f2 = dst_expo(x + 0.5*dx, k)
        f3 = dst_expo(x + dx, k)
        intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4 * f2 + f3)
        x += dx
        f1 = f3
        cnt += 1
    return x - dx + dx * (intl_value - intl_bef)/(intl - intl_bef)
num_int_expo_impl_right_border = njit()(num_int_expo_impl_right_border_np)

def prob_exp(x,k):
    return 1.0 - np.exp(-k*x)

def compute_right_border_impl_exp(x0, intl_value, k):
    return -1.0/k * np.log( np.exp(-k * x0) - intl_value )

#%% STOCHASTICS

# input: prob = [p1, p2, p3, ...] = quantile probab. -> need to set no_q = None
# OR (if prob = None)
# input: no_q = number of quantiles (including prob = 1.0)
# returns N-1 quantiles q with given (prob)
# or equal probability distances (no_q):
# N = 4 -> P(X < q[0]) = 1/4, P(X < q[1]) = 2/4, P(X < q[2]) = 3/4
# this function was checked with the std normal distribution
def compute_quantiles(func, par, x0, x1, dx, prob, no_q=None):
    # probabilities of the quantiles
    if par is not None:
        f = lambda x : func(x, par)
    else: f = func
    
    if prob is None:
        prob = np.linspace(1.0/no_q,1.0,no_q)[:-1]
    print ("quantile probabilities = ", prob)
    
    intl = 0.0
    x = x0
    q = []
    
    cnt = 0
    for i,p in enumerate(prob):
        while (intl < p and p < 1.0 and x <= x1 and cnt < 1E8):
            intl += dx * f(x)
            x += dx
            cnt += 1
        # the quantile value is somewhere between x - dx and x
        # q.append(x)    
        # q.append(x - 0.5 * dx)    
        q.append(max(x - dx,x0))    
    
    print ("quantile values = ", q)
    return q, prob
          
# for a given grid with grid center positions grid_centers[i,j]:
# create random radii rad[i,j] = array of shape (Nx, Ny, no_spc)
# and the corresp. values of the distribution p[i,j]
# input:
# p_min, p_max: quantile probs to cut off e.g. (0.001,0.999)
# no_spc: number of super-droplets per cell (scalar!)
# dst: distribution to use
# par: params of the distribution par -> "None" possible if dst = dst(x)
# r0, r1, dr: parameters for the numerical integration to find the cutoffs
def generate_random_radii_monomodal(grid, dst, par, no_spc, p_min, p_max,
                                    r0, r1, dr, seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    if par is not None:
        func = lambda x : dst(x, par)
    else: func = dst
    
    qs, Ps = compute_quantiles(func, None, r0, r1, dr, [p_min, p_max], None)
    
    r_min = qs[0]
    r_max = qs[1]
    
    bins = np.linspace(r_min, r_max, no_spc+1)
    
    rnd = []
    for i,b in enumerate(bins[0:-1]):
        # we need 1 radius value for each bin for each cell
        rnd.append( np.random.uniform(b, bins[i+1], grid.no_cells_tot) )
    
    rnd = np.array(rnd)
    
    rnd = np.transpose(rnd)    
    shape = np.hstack( [np.array(np.shape(grid.centers)[1:]), [no_spc]] )
    rnd = np.reshape(rnd, shape)
    
    weights = func(rnd)
    
    for p_row in weights:
        for p_cell in p_row:
            p_cell /= np.sum(p_cell)
    
    return rnd, weights #, bins

# no_spc = super particles per cell
# creates no_spc random radii per cell and the no_spc weights per cell
# where sum(weight_i) = 1.0
# the weights are drawn from a normal distribution with mu = 1.0, sigma = 0.2
# and rejected if weight < 0. The weights are then normalized such that sum = 1
# par = (mu*, ln(sigma*)), where mu* and sigma* are the
# GEOMETRIC expectation value and standard dev. of the lognormal distr. resp.
def generate_random_radii_monomodal_lognorm(grid, par, no_spc,
                                            seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    no_spt = grid.no_cells_tot * no_spc
    
    # draw random numbers from log normal distr. by this procedure
    Rs = np.random.normal(0.0, par[1], no_spt)
    Rs = np.exp(Rs)
    Rs *= par[0]
    # draw random weights
    weights = np.abs(np.random.normal(1.0, 0.2, no_spt))
    
    # bring to shape Rs[i,j,k], where Rs[i,j] = arr[k] k = 0...no_spc-1
    shape = np.hstack( [np.array(np.shape(grid.centers)[1:]), [no_spc]] )
    Rs = np.reshape(Rs, shape)
    weights = np.reshape(weights, shape)
    
    for p_row in weights:
        for p_cell in p_row:
            p_cell /= np.sum(p_cell)
    
    return Rs, weights #, bins

# no_spcm MUST be a list/array with [no_1, no_2, .., no_N], N = number of modes
# no_spcm[k] = 0 is possible for some mode k. Then no particles are generated
# for this mode...
# par MUST be a list with [par1, par2, .., parN]
# where par1 = [p11, p12, ...] , par2 = [...] etc
# everything else CAN be a list or a scalar single float/int
# if given as scalar, all modes will use the same value
# the seed is set at least once
# reseed = True will reset the seed everytime for every new mode with the value
# given in the seed list, it can also be used with seed being a single scalar
# about the seeds:
# using np.random.seed(seed) in a function sets the seed globaly
# after leaving the function, the seed remains the set seed
def generate_random_radii_multimodal(grid, dst, par, no_spcm, p_min, p_max, 
                                     r0, r1, dr, seed, reseed = False):
    no_modes = len(no_spcm)
    if not isinstance(p_min, (list, tuple, np.ndarray)):
        p_min = [p_min] * no_modes
    if not isinstance(p_max, (list, tuple, np.ndarray)):
        p_max = [p_max] * no_modes
    if not isinstance(dst, (list, tuple, np.ndarray)):
        dst = [dst] * no_modes
    if not isinstance(r0, (list, tuple, np.ndarray)):
        r0 = [r0] * no_modes
    if not isinstance(r1, (list, tuple, np.ndarray)):
        r1 = [r1] * no_modes
    if not isinstance(dr, (list, tuple, np.ndarray)):
        dr = [dr] * no_modes
    if not isinstance(seed, (list, tuple, np.ndarray)):
        seed = [seed] * no_modes        

    rad = []
    weights = []
    print(p_min)
    # set seed always once
    setseed = True
    for k in range(no_modes):
        if no_spcm[k] > 0:
            r, w = generate_random_radii_monomodal(
                       grid, dst[k], par[k], no_spcm[k], p_min[k], p_max[k],
                       r0[k], r1[k], dr[k], seed[k], setseed)
            rad.append( r )
            weights.append( w )
            setseed = reseed
    # we need the different modes separately, because they 
    # are weighted with different total concentrations
    # if len(rad)>1:
    #     rad = np.concatenate(rad, axis = 2)
    #     weights = np.concatenate(weights, axis = 2)
    
    return np.array(rad), np.array(weights)

# 
def generate_random_radii_multimodal_lognorm(grid, par, no_spcm,
                                             seed, reseed = False):
    no_modes = len(no_spcm)
    if not isinstance(seed, (list, tuple, np.ndarray)):
        seed = [seed] * no_modes       
        
    rad = []
    weights = []
    # set seed always once
    setseed = True
    for k in range(no_modes):
        if no_spcm[k] > 0:
            r, w = generate_random_radii_monomodal_lognorm(
                       grid, par[k], no_spcm[k], seed[k], setseed)
            rad.append( r )
            weights.append( w )
            setseed = reseed
    # we need the different modes separately, because they 
    # are weighted with different total concentrations
    # if len(rad)>1:
    #     rad = np.concatenate(rad, axis = 2)
    #     weights = np.concatenate(weights, axis = 2)
    
    return np.array(rad), np.array(weights)

# no_spcm = super-part/cell in modes [N_mode1_per_cell, N_mode2_per_cell, ...]
# no_spc = # super-part/cell
# no_spt = # SP total in full domain
# returns positions of particles of shape (2, no_spt)
def generate_random_positions(grid, no_spc, seed, set_seed = False):
    if isinstance(no_spc, (list, tuple, np.ndarray)):
        no_spt = np.sum(no_spc)
    else:
        no_spt = grid.no_cells_tot * no_spc
        no_spc = np.ones(grid.no_cells, dtype = np.int64) * no_spc
    if set_seed:
        np.random.seed(seed)        
    rnd_x = np.random.rand(no_spt)
    rnd_z = np.random.rand(no_spt)
    dx = grid.steps[0]
    dz = grid.steps[1]
    x = []
    z = []
    cells = [[],[]]
    n = 0
    for j in range(grid.no_cells[1]):
        z0 = grid.corners[1][0,j]
        for i in range(grid.no_cells[0]):
            x0 = grid.corners[0][i,j]
            for k in range(no_spc[i,j]):
                x.append(x0 + dx * rnd_x[n])
                z.append(z0 + dz * rnd_z[n])
                cells[0].append(i)
                cells[1].append(j)
                n += 1
    pos = np.array([x, z])
    rel_pos = np.array([rnd_x, rnd_z])
    return pos, rel_pos, np.array(cells)

## no_spcm = super-part/cell in modes [N_mode1_per_cell, N_mode2_per_cell, ...]
## no_spc = # super-part/cell
## no_spt = # SP total in full domain
## returns positions of particles of shape (2, no_spt)
#def generate_random_positions_cells(grid, cells, seed, set_seed = False):
#    if isinstance(no_spc, (list, tuple, np.ndarray)):
#        no_spt = np.sum(no_spc)
#    else:
#        no_spt = grid.no_cells_tot * no_spc
#        no_spc = np.ones(grid.no_cells, dtype = np.int64) * no_spc
#    if set_seed:
#        np.random.seed(seed)        
#    rnd_x = np.random.rand(no_spt)
#    rnd_z = np.random.rand(no_spt)
#    dx = grid.steps[0]
#    dz = grid.steps[1]
#    x = []
#    z = []
#    cells = [[],[]]
#    n = 0
#    for j in range(grid.no_cells[1]):
#        z0 = grid.corners[1][0,j]
#        for i in range(grid.no_cells[0]):
#            x0 = grid.corners[0][i,j]
#            for k in range(no_spc[i,j]):
#                x.append(x0 + dx * rnd_x[n])
#                z.append(z0 + dz * rnd_z[n])
#                cells[0].append(i)
#                cells[1].append(j)
#                n += 1
#    pos = np.array([x, z])
#    rel_pos = np.array([rnd_x, rnd_z])
#    return pos, rel_pos, np.array(cells)

#%%

#grid_ranges = [[0.,1500.], [0.,1500.]]
#grid_steps = [150.,150.]
#dy = 1.0
#
#grid = Grid(grid_ranges, grid_steps, dy)
#
#no_spc = np.random.randint(5,10,grid.no_cells)
#
#seed = 3711
#    
#pos, rel_pos, cells = generate_random_positions(grid, no_spc, seed, True)


#%% GENERATE SIP ENSEMBLES

### NEW APPROACH AFTER UNTERSTRASSER CODE 
    



#############################################################################
### OLD APPROACH

# par = "rate" parameter "k" of the expo distr: k*exp(-k*m) (in 10^18 kg)
# no_rpc = number of real particles in cell
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold(
        par, no_rpc, r_critmin, m_high_by_m_low, kappa,
        eta, seed, setseed):
    
    if setseed: np.random.seed(seed)
    m_low = compute_mass_from_radius_jit(r_critmin,
                                     c.mass_density_water_liquid_NTP)
    # m_high = num_int_expo_impl_right_border(0.0, p_max, 1.0/par*0E-6, par,
    #                                            cnt_lim=1E8)
    
    m_high = m_low * m_high_by_m_low
    # since we consider only particles with m > m_low, the total number of
    # placed particles and the total placed mass will be underestimated
    # to fix this, we could adjust the PDF
    # by e.g. one of the two possibilities:
    # 1. decrease the total number of particles and thereby the total pt conc.
    # 2. keep the total number of particles, and thus increase the PDF
    # in the interval [m_low, m_high]
    # # For 1.: decrease the total number of particles by multiplication 
    # # with factor num_int_expo_np(m_low, m_high, par, steps=1.0E6)
    # print(no_rpc)
    # no_rpc *= num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    # for 2.:
    # increase the total number of
    # real particle "no_rpc" by 1.0 / int_(m_low)^(m_high)
    # no_rpc *= 1.0/num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    kappa_inv = 1.0/kappa
    bin_fac = 10**kappa_inv
    
    masses = []
    xis = []
    bins = [m_low]
    no_rp_set = 0
    no_sp_set = 0
    bin_n = 0
    
    m = np.random.uniform(m_low/2, m_low)
    xi = dst_expo(m,par) * m_low * no_rpc
    masses.append(m)
    xis.append(xi)
    no_rp_set += xi
    no_sp_set += 1    
    
    m_left = m_low
    # m = 0.0
    
    # print("no_rpc =", no_rpc)
    # while(no_rp_set < no_rpc and m < m_high):
    while(m < m_high):
        # m_right = m_left * 10**kappa_inv
        m_right = m_left * bin_fac
        # print("missing particles =", no_rpc - no_rp_set)
        # print("m_left, m_right, m_high")
        # print(bin_n, m_left, m_right, m_high)
        m = np.random.uniform(m_left, m_right)
        # print("m =", m)
        # we do not round to integer here, because of the weak threshold below
        # the rounding is done afterwards
        xi = dst_expo(m,par) * (m_right - m_left) * no_rpc
        # xi = int((dst_expo(m,par) * (m_right - m_left) * no_rpc))
        # if xi < 1 : xi = 1
        # if no_rp_set + xi > no_rpc:
        #     xi = no_rpc - no_rp_set
            # print("no_rpc reached")
        # print("xi =", xi)
        masses.append(m)
        xis.append(xi)
        no_rp_set += xi
        no_sp_set += 1
        m_left = m_right
        bins.append(m_right)
        bin_n += 1
    
    xis = np.array(xis)
    masses = np.array(masses)
    
    xi_max = xis.max()
    # print("xi_max =", f"{xi_max:.2e}")
    
    xi_critmin = int(xi_max * eta)
    if xi_critmin < 1: xi_critmin = 1
    # print("xi_critmin =", xi_critmin)
    
    for bin_n,xi in enumerate(xis):
        if xi < xi_critmin:
            # print("")
            p = xi / xi_critmin
            if np.random.rand() >= p:
                xis[bin_n] = 0
            else: xis[bin_n] = xi_critmin
    
    ind = np.nonzero(xis)
    xis = xis[ind].astype(np.int64)
    masses = masses[ind]
    
    # now, some bins are empty, thus the total number sum(xis) is not right
    # moreover, the total mass sum(masses*xis) is not right ...
    # FIRST: test, how badly the total number and total mass of the cell
    # is violated, if this is bad:
    # -> create a number of particles, such that the total number is right
    # and assign the average mass to it
    # then reweight all of the masses such that the total mass is right.
    
    return masses, xis, m_low, m_high, bins



# par = "rate" parameter "k" of the expo distr: k*exp(-k*m) (in 10^18 kg)
# no_rpc = number of real particles in cell
# NON INTERGER WEIGHTS ARE POSSIBLE A SIN UNTERSTRASSER 2017
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint(
        par, no_rpc, r_critmin=0.6, m_high_by_m_low=1.0E6, kappa=40,
        eta=1.0E-9, seed=4711, setseed = True):
    
    if setseed: np.random.seed(seed)
    m_low = compute_mass_from_radius_jit(r_critmin,
                                     c.mass_density_water_liquid_NTP)
    # m_high = num_int_expo_impl_right_border(0.0, p_max, 1.0/par*0E-6, par,
    #                                            cnt_lim=1E8)
    
    m_high = m_low * m_high_by_m_low
    # since we consider only particles with m > m_low, the total number of
    # placed particles and the total placed mass will be underestimated
    # to fix this, we could adjust the PDF
    # by e.g. one of the two possibilities:
    # 1. decrease the total number of particles and thereby the total pt conc.
    # 2. keep the total number of particles, and thus increase the PDF
    # in the interval [m_low, m_high]
    # # For 1.: decrease the total number of particles by multiplication 
    # # with factor num_int_expo_np(m_low, m_high, par, steps=1.0E6)
    # print(no_rpc)
    # no_rpc *= num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    # for 2.:
    # increase the total number of
    # real particle "no_rpc" by 1.0 / int_(m_low)^(m_high)
    # no_rpc *= 1.0/num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    kappa_inv = 1.0/kappa
    bin_fac = 10**kappa_inv
    
    masses = []
    xis = []
    bins = [m_low]
    no_rp_set = 0
    no_sp_set = 0
    bin_n = 0
    
    m = np.random.uniform(m_low/2, m_low)
    xi = dst_expo(m,par) * m_low * no_rpc
    masses.append(m)
    xis.append(xi)
    no_rp_set += xi
    no_sp_set += 1    
    
    m_left = m_low
    # m = 0.0
    
    # print("no_rpc =", no_rpc)
    # while(no_rp_set < no_rpc and m < m_high):
    while(m < m_high):
        # m_right = m_left * 10**kappa_inv
        m_right = m_left * bin_fac
        # print("missing particles =", no_rpc - no_rp_set)
        # print("m_left, m_right, m_high")
        # print(bin_n, m_left, m_right, m_high)
        m = np.random.uniform(m_left, m_right)
        # print("m =", m)
        # we do not round to integer here, because of the weak threshold below
        # the rounding is done afterwards
        xi = dst_expo(m,par) * (m_right - m_left) * no_rpc
        # xi = int((dst_expo(m,par) * (m_right - m_left) * no_rpc))
        # if xi < 1 : xi = 1
        # if no_rp_set + xi > no_rpc:
        #     xi = no_rpc - no_rp_set
            # print("no_rpc reached")
        # print("xi =", xi)
        masses.append(m)
        xis.append(xi)
        no_rp_set += xi
        no_sp_set += 1
        m_left = m_right
        bins.append(m_right)
        bin_n += 1
    
    xis = np.array(xis)
    masses = np.array(masses)
    
    xi_max = xis.max()
    # print("xi_max =", f"{xi_max:.2e}")
    
    # xi_critmin = int(xi_max * eta)
    xi_critmin = xi_max * eta
    # if xi_critmin < 1: xi_critmin = 1
    # print("xi_critmin =", xi_critmin)
    
    valid_ind = []
    
    for bin_n,xi in enumerate(xis):
        if xi < xi_critmin:
            # print("")
            p = xi / xi_critmin
            if np.random.rand() >= p:
                xis[bin_n] = 0
            else:
                xis[bin_n] = xi_critmin
                valid_ind.append(bin_n)
        else: valid_ind.append(bin_n)
    
    # ind = np.nonzero(xis)
    # xis = xis[ind].astype(np.int64)
    valid_ind = np.array(valid_ind)
    xis = xis[valid_ind]
    masses = masses[valid_ind]
    
    # now, some bins are empty, thus the total number sum(xis) is not right
    # moreover, the total mass sum(masses*xis) is not right ...
    # FIRST: test, how badly the total number and total mass of the cell
    # is violated, if this is bad:
    # -> create a number of particles, such that the total number is right
    # and assign the average mass to it
    # then reweight all of the masses such that the total mass is right.
    
    return masses, xis, m_low, m_high, bins

@njit()
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint2(
        par, no_rpc, r_critmin=0.6, m_high_by_m_low=1.0E6, kappa=40,
        eta=1.0E-9, seed=4711, setseed = True):
    bin_factor = 10**(1.0/kappa)
    m_low = compute_mass_from_radius_jit(r_critmin,c.mass_density_water_liquid_NTP)
    m_left = m_low
    # l_max = kappa * log_10(m_high/m_low)
    l_max = int(kappa * np.log10(m_high_by_m_low)) + 1
    rnd = np.random.rand( l_max )
#    cnt = 0
    
    masses = np.zeros(l_max, dtype = np.float64)
    xis = np.zeros(l_max, dtype = np.float64)
    bins = np.zeros(l_max+1, dtype = np.float64)
    bins[0] = m_left
    
    for l in range(l_max):
        m_right = m_left * bin_factor
        bins[l+1] = m_right
        dm = m_right - m_left
        
        m = m_left + dm * rnd[l]
        masses[l] = m
        xis[l] = no_rpc * dm * dst_expo(m, par)
        
        m_left = m_right
        
#        cnt += 1
    
    xi_max = xis.max()
    xi_critmin = xi_max * eta
    
    switch = np.ones(l_max, dtype=np.int64)
    
    for l in range(l_max):
        if xis[l] < xi_critmin:
            if np.random.rand() < xis[l] / xi_critmin:
                xis[l] = xi_critmin
            else: switch[l] = 0
    
    ind = np.nonzero(switch)[0]
    
    xis = xis[ind]
    masses = masses[ind]
    
    
    return masses, xis, m_low, bins

# no_spc is the intended number of super particles per cell,
# this will right on average, but will vary due to the random assigning 
# process of the xi_i
# m0, m1, dm: parameters for the numerical integration to find the cutoffs
# and for the numerical integration of integrals
# dV = volume size of the cell (in m^3)
# n0: initial particle number distribution (in 1/m^3)
# IN WORK: make the bin size smaller for low masses to get
# finer resolution here...
# -> "steer against" the expo PDF for low m -> something in between
# we need higher resolution in very small radii (Unterstrasser has
# R_min = 0.8 mu
# at least need to go down to 1 mu (resolution there...)
# eps = parameter for the bin linear spreading:
# bin_size(m = m_high) = eps * bin_size(m = m_low)
# @njit()
def generate_SIP_ensemble_expo_my_xi_rnd(par, no_spc, no_rpc,
                                         total_mass_in_cell,
                                         p_min, p_max, eps,
                                         m0, m1, dm, seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    # if par is not None:
    #     func = lambda x : dst(x, par)
    # else: func = dst
    
    # define m_low, m_high by probability threshold p_min
    # m_thresh = [m_low, m_high] 
    
    # (m_low, m_high), Ps = compute_quantiles(func, None, m0, m1, dm,
    #                                  [p_min, p_max], None)
    m_low = 0.0
    m_high = num_int_expo_impl_right_border(m_low,p_max, dm, par, 1.0E8)
    
    # if p_min == 0:
    #     m_low = 0.0
    # estimate value for dm, which is the bin size of the hypothetical
    # bin assignment, which is approximated by the random process:
    
    # the bin mean size must be adjusted by ~np.log10(eps)
    # to get to a SIP number in the cell of about the intended number
    # if the reweighting is not done, the SIP number will be much larger
    # due to small bins for low masses
    bin_size_mean = (m_high - m_low) / no_spc * np.log10(eps)
    
    # eps = 10
    a = bin_size_mean * (eps - 1) / (m_high * 0.5 * (eps + 1))
    b = bin_size_mean / (0.5 * (eps + 1))
    
    # no of real particle cell
    # no_rpc = dV*n0
    
    # the current position of the left bin border
    m_left = m_low
    
    # generate list of masses and multiplicities
    masses = []
    xis = []
    
    # no of particles placed:
    no_pt = 0
    
    # let f(m) be the PDF!! (m can be the mass or the radius, depending,
    # which distr. is given), NOTE that f(m) is not the number concentration..
    
    pt_n = 0
    while no_pt < no_rpc:
        ### i) determine the next xi_mean by
        # integrating xi_mean = no_rpc * int_(m_left)^(m_left+dm) dm f(m)
        # print(pt_n, "m_left=", m_left)
        # m = m_left
        bin_size = a * m_left + b
        m_right = m_left + bin_size
        
        intl = num_int_expo(m_left, m_right, par, steps=1.0E5)
        
        # intl = 0.0
        # cnt = 0
        # while (m < m_right and cnt < 1E6):
        #     intl += dm * func(m)
        #     m += dm
        #     cnt += 1
        #     if cnt == 1E6:
        #         print(pt_n, "cnt=1E6")  
                
        xi_mean = no_rpc * intl
        # print(pt_n, "xi_mean=", xi_mean)
        ### ii) draw xi from distribution: 
        # try here normal distribution if xi_mean > 10
        # else Poisson
        # with parms: mu = xi_mean, sigma = sqrt(xi_mean)
        if xi_mean > 10:
            # xi = np.random.normal(xi_mean, np.sqrt(xi_mean))
            xi = np.random.normal(xi_mean, 0.2*xi_mean)
        else:
            xi = np.random.poisson(xi_mean)
        xi = int(math.ceil(xi))
        if xi <= 0: xi = 1
        if no_pt + xi >= no_rpc:
            xi = no_rpc - no_pt
            # last = True
            M_sys = np.sum( np.array(xis)*np.array(masses) )
            M_should = no_rpc * num_int_expo_mean(0.0,m_left,par,1.0E7)
            masses = [m * M_should / M_sys for m in masses]
            # M = p_max * total_mass_in_cell - M
            M_diff = total_mass_in_cell - M_should
            if M_diff <= 0.0:
                M_diff = 1.0/par                
            # if M_diff <= 0.0:
            #     M_diff = total_mass_in_cell - M_should
                # xis = np.array(xis, dtype=np.int64)
                # masses = np.array(masses)
                # masses *= p_max*total_mass_in_cell/M_sys
                # masses = [m*p_max*total_mass_in_cell/M_sys for m in masses]
                # M_sys = p_max*total_mass_in_cell
                # M_diff = total_mass_in_cell * (1 - p_max)
            # else:
            # mu = max(p_max*M/xi,m_left)
            mu = M_diff/xi
            if mu <= 1.02*masses[pt_n-1]:
                xi_sum = xi + xis[pt_n-1]
                m_sum = xi*mu + xis[pt_n-1] * masses[pt_n-1]
                xis[pt_n-1] = xi_sum
                masses[pt_n-1] = m_sum / xi_sum
                no_pt += xi
                # print(pt_n, xi, mu, no_pt)
            else:                
                masses.append(mu)    
                xis.append(xi)
                no_pt += xi
                # print(pt_n, xi, mu, no_pt)
                pt_n += 1            
            # masses.append(mu)    
            # xis.append(xi)
            # no_pt += xi
            # print(pt_n, xi, mu, no_pt)
            # pt_n += 1
            
            # xi = no_rpc - no_pt
            # # last = True
            # M_sys = np.sum( np.array(xis)*np.array(masses) )
            # # M = p_max * total_mass_in_cell - M
            # M_diff = total_mass_in_cell - M_sys
            # if M_diff <= 0.0:
            #     # xis = np.array(xis, dtype=np.int64)
            #     masses = [m*p_max*total_mass_in_cell/M_sys for m in masses]
            #     # masses = np.array(masses)
            #     # masses *= p_max*total_mass_in_cell/M_sys
            # else:
            #     # mu = max(p_max*M/xi,m_left)
            #     mu = M_diff/xi
            #     masses.append(mu)    
            #     xis.append(xi)
            #     no_pt += xi
            #     print(pt_n, xi, mu, no_pt)
            #     pt_n += 1
            # print("no_pt + xi=", no_pt + xi, "no_rpc =", no_rpc )
            # xi = no_rpc - no_pt
            # last = True
            # M = np.sum( np.array(xis)*np.array(masses) )
            # M = p_max * total_mass_in_cell - M
            # M = total_mass_in_cell - M
            # mu = max(p_max*M/xi,m_left)
            # mu = M/xi
            # masses.append(mu)
        else:            
            ### iii) set the right right bin border
            # by no_rpc * int_(m_left)^m_right dm f(m) = xi
            
            m_right = num_int_expo_impl_right_border(m_left, xi/no_rpc,
                                                        dm, par)
            
            # m = m_left
            # intl = 0.0
            # cnt = 0
            # while (intl < xi/no_rpc and cnt < 1E7):
            #     intl += dm * func(m)
            #     m += dm
            #     cnt += 1
            #     if cnt == 1E7:
            #         print(pt_n, "cnt=", cnt)
            # m_right = m
            # if m_right >= m_high:
                
            # print(pt_n, "new m_right=", m_right)
            # iv) set SIP mass mu by mu=M/xi (M = total mass in the bin)
            
            intl = num_int_expo_mean(m_left, m_right, par)
            
            # intl = 0.0
            # m = m_left
            # cnt = 0
            # while (m < m_right and cnt < 1E7):
            #     intl += dm * func(m) * m
            #     m += dm
            #     cnt += 1         
            #     if cnt == 1E7:
            #         print(pt_n, "cnt=", cnt)                
            
            mu = intl * no_rpc / xi
            masses.append(mu)
            xis.append(xi)
            no_pt += xi
            # print(pt_n, xi, mu, no_pt, f"{m_left:.2e}", f"{m_right:.2e}")
            pt_n += 1
            m_left = m_right
        
        if m_left >= m_high and no_pt < no_rpc:

            xi = no_rpc - no_pt
            # last = True
            M_sys = np.sum( np.array(xis)*np.array(masses) )
            M_should = no_rpc * num_int_expo_mean(0.0,m_left,par,1.0E7)
            masses = [m * M_should / M_sys for m in masses]
            # M = p_max * total_mass_in_cell - M
            M_diff = total_mass_in_cell - M_should
            if M_diff <= 0.0:
                M_diff = 1.0/par
            
            # print("m_left =", m_left, "m_left - m_high =", m_left-m_high)
            # print("no_pt + xi=", no_pt + xi, "no_rpc =", no_rpc )
            # xi = no_rpc - no_pt
            # last = True
            # M_sys = np.sum( np.array(xis)*np.array(masses) )
            # M = p_max * total_mass_in_cell - M
            # M_diff = total_mass_in_cell - M_sys

            # if M_diff <= 0.0:
                # M_should = no_rpc * num_int_expo_mean_np(0.0,m_left,par,1.0E7)
                # masses = [m * M_should / M_sys for m in masses]
                # M_diff = total_mass_in_cell - M_should
                # xis = np.array(xis, dtype=np.int64)
                # masses = np.array(masses)
                # masses *= p_max*total_mass_in_cell/M_sys
                # masses = [m*p_max*total_mass_in_cell/M_sys for m in masses]
                # # M_sys = p_max*total_mass_in_cell
                # M_diff = total_mass_in_cell * (1 - p_max)
            # else:
            # mu = max(p_max*M/xi,m_left)
            mu = M_diff/xi
            if mu <= 1.02*masses[pt_n-1]:
                xi_sum = xi + xis[pt_n-1]
                m_sum = xi*mu + xis[pt_n-1] * masses[pt_n-1]
                xis[pt_n-1] = xi_sum
                masses[pt_n-1] = m_sum / xi_sum
                no_pt += xi
                # print(pt_n, xi, mu, no_pt)
            else:                
                masses.append(mu)    
                xis.append(xi)
                no_pt += xi
                # print(pt_n, xi, mu, no_pt)
                pt_n += 1
    
    return np.array(masses), np.array(xis, dtype=np.int64), m_low, m_high

# # c "per cell", t "tot"
# # rad and weights are (Nx,Ny,no_spc) array
# # pos is (no_spt,) array
# def generate_particle_positions_and_dry_radii(grid, p_min, p_max, no_spc,
#                                               dst, par, r0, r1, dr, seed):
#     rad, weights = generate_random_radii(grid, p_min, p_max, no_spc,
#                                          dst, par, r0, r1, dr, seed)
    
#     return pos, rad, weights

#%% COMPUTE VERTICAL PROFILES WITHOUT LIQUID
def compute_profiles_T_p_rhod_S_without_liquid(
        z_, z_0_, p_0_, p_ref_, Theta_l_, r_tot_, SCALE_FACTOR_ = 1.0 ):

#    dz = grid_.steps[1] * SCALE_FACTOR_
#    z_0 = grid_.ranges[1,0]
    z_0 = z_0_
#    z_inversion_height = grid_.ranges[1,1]
#    zg = grid_.corners[1][0]
#    Nz = grid_.no_cells[1] + 1

    # constants:
    p_ref = p_ref_ # ref pressure for potential temperature in Pa
    p_0 = p_0_ # surface pressure in Pa
    p_0_over_p_ref = p_0 / p_ref

    r_tot = r_tot_ # kg water / kg dry air
    Theta_l = Theta_l_ # K

    kappa_tot = compute_kappa_air_moist(r_tot) # [-]
    kappa_tot_inv = 1.0 / kappa_tot # [-]
    p_0_over_p_ref_to_kappa_tot = p_0_over_p_ref**(kappa_tot) # [-]
    beta_tot = compute_beta_without_liquid(r_tot, Theta_l) # 1/m

    # in K
    # T_0 = compute_temperature_from_potential_temperature_moist(
    #           Theta_l, p_0, p_ref, r_tot)

    #################################################

    # analytically integrated profiles
    # for hydrostatic system with constant water vapor mixing ratio
    # r_v = r_tot and r_l = 0
    T_over_Theta_l = p_0_over_p_ref_to_kappa_tot - beta_tot * (z_ - z_0)
    T = T_over_Theta_l * Theta_l
    p_over_p_ref = T_over_Theta_l**(kappa_tot_inv)
    p = p_over_p_ref * p_ref
    rho_dry = p * beta_tot \
            / (T_over_Theta_l * c.earth_gravity * kappa_tot )
    S = compute_pressure_vapor( rho_dry * r_tot, T )\
        / compute_saturation_pressure_vapor_liquid(T)
            
#    def compute_T_over_Theta_l_init( z_ ):
#        return p_0_over_p_ref_to_kappa_tot - beta_tot * (z_ - z_0)
#    def compute_p_over_p_ref_init( z_ ):
#        return compute_T_over_Theta_l_init(z_)**(kappa_tot_inv)
#    def compute_density_dry_init( z_ ):
#        return compute_p_over_p_ref_init(z_) * p_ref * beta_tot \
#                / (compute_T_over_Theta_l_init( z_ )*earth_gravity*kappa_tot)
#    def compute_saturation_init( z_ ):
#        T = compute_T_over_Theta_l_init( z_ ) * Theta_l
#        return \
#            compute_pressure_vapor(compute_density_dry_init( z_ ) * r_tot, T) \
#            / compute_saturation_pressure_vapor_liquid(T)
            
    return T, p, rho_dry, S
###############################################################################
#%% INITIALIZE: GENERATE INIT GRID AND SUPER-PARTICLES
# p_0 = 101500 # surface pressure in Pa
# p_ref = 1.0E5 # ref pressure for potential temperature in Pa
# r_tot_0 = 7.5E-3 # kg water / kg dry air, r_tot = r_v + r_l (spatially const)
# Theta_l = 289.0 # K, Liquid potential temperature (spatially constant)
# n_p: list of number density of particles in the modes [n1, n2, ..]
# e.g. n_p = np.array([60.0E6, 40.0E6]) # m^3
# no_spcm = list of number of super particles per cell: [no_spc_1, no_spc_2, ..]
# where no_spc_k is the number of SP per cell in mode k
# dst: function or list of functions of the distribution to use,
# e.g. [dst1, dst2, ..]
# dst_par: parameters of the distributions, dst_par = [par1, par2, ...], 
# where par1 = [par11, par12, ...] are pars of dst1
# P_min = cutoff probability for the generation of random radii
# P_max = cutoff probability for the generation of random radii
# r0, r1, dr: parameters for the integration to set the quantiles during
# generation of random radii
# reseed: renew the random seed after the generation of the first mode
## for initialization phase
# S_init_max = 1.05
# dt_init = 0.1 # s
# Newton iterations for the implicit mass growth during initialization phase
# maximal allowed iter counts in initial particle water take up to equilibrium
# iter_cnt_limit = 500

# grid_file_list = ["grid_basics.txt", "arr_file1.npy", "arr_file2.npy"]
# grid_file_list = [path + s for s in grid_file_list]

# particle_file = "stored_particles.txt"
# particle_file = path + particle_file

def initialize_grid_and_particles_SinSIP(
        x_min, x_max, z_min, z_max, dx, dy, dz,
        p_0, p_ref, r_tot_0, Theta_l,
        solute_type,
        DNC0, no_spcm, no_modes, dist, dst_par,
        eta, eta_threshold, r_critmin, m_high_over_m_low,
        rnd_seed, reseed,
        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, save_path):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # VERSION WITH PARTICLE PROPERTIES IN ARRAYS, not particle class
    ##############################################
    ### 1. set base grid
    ##############################################
    # grid dimensions ("ranges")
    # note that the step size is fix.
    # The grid dimensions will be adjusted such that they are
    # AT LEAST x_max - x_min, etc... but may also be larger,
    # if the sizes are no integer multiples of the step sizes
    log_file = save_path + f"log_grid.txt"
#    if logfile is not None:
#        log_file = save_path + "log_grid.txt"
#        log_handle = open(log_file, "w")
#        sys.stdout = log_handle
        
    grid_ranges = [ [x_min, x_max],
    #                 [y_min, y_max], 
                    [z_min, z_max] ]
    grid_steps = [dx, dz]

    grid = Grid( grid_ranges, grid_steps, dy )
    grid.print_info()
    
    with open(log_file, "w+") as f:
        f.write("grid basic parameters:\n")
        f.write(f"grid ranges [x_min, x_max] [z_min, z_max]:\n")
        f.write(f"{x_min} {x_max} {z_min} {z_max}\n")
        f.write("number of cells: ")
        f.write(f"{grid.no_cells[0]}, {grid.no_cells[1]} \n")
        f.write("grid steps: ")
        f.write(f"{dx}, {dy}, {dz}\n\n")
    ###

    ###
    
    ##############################################
    ### 2. Set initial profiles without liquid water
    ##############################################
    
    # INITIAL PROFILES
    
    levels_z = grid.corners[1][0]
    centers_z = grid.centers[1][0]
    
    # for testing on a smaller grid: r_tot as array in centers
    r_tot_centers = np.ones_like( centers_z ) * r_tot_0
    # r_tot_centers = np.linspace(1.15, 1.3, np.shape(centers_z)[0]) * r_tot_0
    
    p_env_init_bottom = np.zeros_like(levels_z)
    p_env_init_bottom[0] = p_0
    
    p_env_init_center = np.zeros_like(centers_z)
    T_env_init_center = np.zeros_like(centers_z)
    rho_dry_env_init_center = np.zeros_like(centers_z)
    r_v_env_init_center = np.ones_like(centers_z) * r_tot_centers
    r_l_env_init_center = np.zeros_like(centers_z)
    S_env_init_center = np.zeros_like(centers_z)
    
    # p_env_init_center = np.zeros_like(levels_z)
    # T_env_init_center = np.zeros_like(levels_z)
    # rho_dry_env_init = np.zeros_like(levels_z)
    # r_v_env_init = np.ones_like(levels_z) * r_tot
    # r_l_env_init = np.zeros_like(r_v_env_init)
    # S_env_init = np.zeros_like(levels_z)
    # T_env_init = np.zeros_like(levels_z)
    # rho_dry_env_init = np.zeros_like(levels_z)
    # r_v_env_init = np.ones_like(levels_z) * r_tot
    # r_l_env_init = np.zeros_like(r_v_env_init)
    # S_env_init = np.zeros_like(levels_z)
    
    # T_env_init[0], p_env_init[0], rho_dry_env_init[0], S_env_init[0] =\
    #     compute_profiles_T_p_rhod_S_without_liquid(
    #         z_0, z_0, p_0, p_ref, Theta_l, r_tot )
    
    ##############################################
    ### 3. Go through levels from the ground and place particles 
    ##############################################
    print(
      "\n### particle placement and saturation adjustment for each z-level ###")
    print('timestep for sat. adj.: dt_init = ', dt_init)
    with open(log_file, "a") as f:
        f.write(
    "### particle placement and saturation adjustment for each z-level ###\n")
        f.write(f'solute material = {solute_type}\n')
        f.write(f'nr of modes in dry distribution = {no_modes}\n')
    ### DERIVED PARAMETERS SIP INIT
    if dist == "lognormal":
        mu_m_log = dst_par[0]
        sigma_m_log = dst_par[1]
        
        # derive scaling parameter kappa from no_spcm
        if no_modes == 1:
            kappa_dst = np.ceil( no_spcm / 20 * 28) * 0.1
            kappa_dst = np.maximum(kappa_dst, 0.1)
        elif no_modes == 2:        
            kappa_dst = np.ceil( no_spcm / 20 * np.array([33,25])) * 0.1
            kappa_dst = np.maximum(kappa_dst, 0.1)
        else:        
            kappa_dst = np.ceil( no_spcm / 20 * 28) * 0.1
            kappa_dst = np.maximum(kappa_dst, 0.1)
        print("kappa =", kappa_dst)
        with open(log_file, "a") as f:
            f.write("intended SIPs per mode and cell = ")
            for k_ in no_spcm:
                f.write(f"{k_:.1f} ")
            f.write("\n")
        with open(log_file, "a") as f:
            f.write("kappa = ")
            for k_ in kappa_dst:
                f.write(f"{k_:.1f} ")
            f.write("\n")

    with open(log_file, "a") as f:
        f.write(f'timestep for sat. adj.: dt_init = {dt_init}\n')
    
#    if dist == "expo":
#        DNC0 = dst_par[0]
#        DNC0_over_LWC0 = dst_par[1]
#
#        # derive scaling parameter kappa from no_spcm
#        kappa = np.rint( no_spcm * 2) * 0.1
#        kappa = np.maximum(kappa, 0.1)
#        print("kappa =", kappa)
    
    if eta_threshold == "weak":
        weak_threshold = True
    else: weak_threshold = False
    
    # start at level 0 (from surface!)
    
    # produce random numbers for relative location:
    np.random.seed( rnd_seed )
    
    # IN WORK: remove?
    V0 = grid.volume_cell
    
    # total number of real particles per mode in one grid cell:
    # total number of real particles in mode 'k' (= 1,2) in cell [i,j]
    # reference value at p_ref (marked by 0)
    no_rpcm_0 =  (np.ceil( V0*DNC0 )).astype(int)
    
    ### REMOVE 
#    print("{no_rpcm_0[0]} {no_rpcm_0[1]}" )
#    print(f"{no_rpcm_0[0]} {no_rpcm_0[1]}" )
    print("no_rpcm_0 = ", no_rpcm_0)
    with open(log_file, "a") as f:
        f.write("no_rpcm_0 = ")
        for no_rpcm_ in no_rpcm_0:
            f.write(f"{no_rpcm_:.2e} ")
        f.write("\n")
        f.write("\n")
    no_rpct_0 = np.sum(no_rpcm_0)
    
    # empty cell list
    ids_in_cell = []
    for i in range(grid.no_cells[0]):
        row_list = []
        for j in range(grid.no_cells[1]):
            row_list.append( [] )
        ids_in_cell.append(row_list)
    
    ids_in_level = []    
    for j in range(grid.no_cells[1]):
        ids_in_level.append( [] )
    
    # particle_list_by_id = []
    
    # cell numbers is sorted as [ [0,0], [1,0], [2,0] ]  
    # i.e. keeping z- level fixed and run through x-range
    # cell_numbers = []
    
    # running particle ID
#    ID = 0
    
    ##########################################################################
    # WORKING VERSION
    # number of super particles
#    no_spcm = np.array(no_spcm) # input: nr of super part. per cell and mode
#    no_spct = np.sum(no_spcm) # no super part. per cell (total)
#    # no_spt = no_spct * grid.no_cells_tot # no super part total in full domain
#    
#    ### generate particle radii and weights for the whole grid
#    R_s, weights_R_s = generate_random_radii_multimodal_lognorm(
#                               grid, dst_par, no_spcm, rnd_seed, reseed)
#    # convert to dry masses
##    print()
##    print("type(R_s)")
##    print(type(R_s))
##    print(R_s)
##    print()
#                   
#    m_s = compute_mass_from_radius_vec(R_s, c.mass_density_NaCl_dry)
#    w_s = np.zeros_like(m_s) # init weight fraction
#    m_w = np.zeros_like(m_s) # init particle water masses to zero
#    m_p = np.zeros_like(m_s) # init particle full mass
#    R_p = np.zeros_like(m_s) # init particle radius
#    rho_p = np.zeros_like(m_s) # init particle density
#    xi = np.zeros_like(m_s).astype(int)
    ##########################################################################
    
    mass_water_liquid_levels = []
    mass_water_vapor_levels = []
    # start with a cell [i,j]
    # we start with fixed 'z' value to assign levels as well
    iter_cnt_max = 0
    iter_cnt_max_level = 0
    # maximal allowed iter counts
    # iter_cnt_limit = 500
    rho_dry_0 = p_0 / (c.specific_gas_constant_air_dry * 293.0)
    no_rpt_should = np.zeros_like(no_rpcm_0)
    
    np.random.seed(rnd_seed)
    
    m_w = []
    m_s = []
    xi = []
    cells_x = []
    cells_z = []
    modes = []
    
    no_spc = np.zeros(grid.no_cells, dtype = np.int64)
    
    no_rpcm_scale_factors_lvl_wise = np.zeros(grid.no_cells[1])
    
    ### go through z-levels from the ground, j is the cell index resp. z
    for j in range(grid.no_cells[1]):
    #     print('next j')
    #     we are now in column 'j', fixed z-level

        n_top = j + 1
        n_bot = j
        
        z_bot = levels_z[n_bot]
        z_top = levels_z[n_top]
        
        r_tot = r_tot_centers[j]
        
        kappa_tot = compute_kappa_air_moist(r_tot) # [-]
        kappa_tot_inv = 1.0 / kappa_tot # [-]
        
    #     T_prev = T_env_init[n_prev]
        # p at bottom of level
        p_bot = p_env_init_bottom[n_bot]
    #     rho_dry_prev = rho_dry_env_init[n_prev]
    #     S_prev = S_env_init[n_prev]
    #     r_v_prev = r_v_env_init[n_prev]
    
        # initial guess at borders of new level from analytical
        # integration without liquid: r_v = r_tot
        # with boundary condition P(z_bot) = p_bot is set
        # calc values for bottom of level
        T_bot, p_bot, rho_dry_bot, S_bot = \
            compute_profiles_T_p_rhod_S_without_liquid(
                z_bot, z_bot, p_bot, p_ref, Theta_l, r_tot)
        # calc values for top of level
        T_top, p_top, rho_dry_top, S_top = \
            compute_profiles_T_p_rhod_S_without_liquid(
                z_top, z_bot, p_bot, p_ref, Theta_l, r_tot)
        
        p_avg = (p_bot + p_top) * 0.5
        T_avg = (T_bot + T_top) * 0.5
        rho_dry_avg = (rho_dry_bot + rho_dry_top  ) * 0.5
        S_avg = (S_bot + S_top) * 0.5
        r_v_avg = r_tot
        
        # ambient properties for this level
        # diffusion_constant = compute_diffusion_constant( T_avg, p_avg )
        # thermal_conductivity_air = compute_thermal_conductivity_air(T_avg)
        # specific_heat_capacity_air =\
        # compute_specific_heat_capacity_air_moist(r_v_avg)
        # adiabatic_index = 1.4
        # accomodation_coefficient = 1.0
        # condensation_coefficient = 0.0415
        
        # calc mass dry from 
        # int dV (rho_dry) = dx * dy * int dz rho_dry
        # rho_dry = dp/dz / [g*( 1+r_tot )]
        # => m_s = dx dy / (g*(1+r_tot)) * (p(z_1)-p(z_2))
        mass_air_dry_level = grid.sizes[0] * dy * (p_bot - p_top )\
                             / ( c.earth_gravity * (1 + r_tot) )
        mass_water_vapor_level = r_v_avg * mass_air_dry_level # in kg
        mass_water_liquid_level = 0.0
        mass_particles_level = 0.0
        dm_l_level = 0.0
        dm_p_level = 0.0
        
        print('\n### level', j, "###")
        print('S_env_init0 = ', S_avg)
        with open(log_file, "a") as f:
            f.write(f"### level {j} ###\n")
            f.write(f"S_env_init0 = {S_avg:.5f}\n")
    ########################################################
    ### 3a. (first initialization setting S_eq = S_amb if possible)
    ########################################################
        
        # nr of real particle per cell and mode is now given for rho_dry_avg
        no_rpcm_scale_factor = rho_dry_avg / rho_dry_0
        no_rpcm = np.rint( no_rpcm_0 * no_rpcm_scale_factor ).astype(int)
        ### REMOVE
#        print("{no_rpcm[0]} {no_rpcm[1]}" )
#        print(f"{no_rpcm[0]} {no_rpcm[1]}" )
        print("no_rpcm = ", no_rpcm)
        with open(log_file, "a") as f:
            f.write("no_rpcm = ")
            for no_rpcm_ in no_rpcm:
                f.write(f"{no_rpcm_:.2e} ")
            f.write("\n")
        
        no_rpcm_scale_factors_lvl_wise[j] = no_rpcm_scale_factor
        no_rpt_should += no_rpcm * grid.no_cells[0]
        
    ### create SIP ensemble for this level -> list of 
    
        if solute_type == "NaCl":
            mass_density_dry = c.mass_density_NaCl_dry
        elif solute_type == "AS":
            mass_density_dry = c.mass_density_AS_dry
        # m_s_lvl = [ m_s[0,j], m_s[1,j], ... ]
        # xi_lvl = ...
        if dist == "lognormal":
            m_s_lvl, xi_lvl, cells_x_lvl, modes_lvl, no_spc_lvl = \
                gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl(no_modes,
                        mu_m_log, sigma_m_log, mass_density_dry,
                        grid.volume_cell, kappa_dst, eta, weak_threshold,
                        r_critmin,
                        m_high_over_m_low, rnd_seed, grid.no_cells[0], no_rpcm,
                        setseed=False)
#        elif dist == "expo":
            
        no_spc[:,j] = no_spc_lvl
        if solute_type == "NaCl":
            w_s_lvl = compute_initial_mass_fraction_solute_m_s_NaCl(
                              m_s_lvl, S_avg, T_avg)
        elif solute_type == "AS":            
            w_s_lvl = compute_initial_mass_fraction_solute_m_s_AS(
                              m_s_lvl, S_avg, T_avg)
        m_p_lvl = m_s_lvl / w_s_lvl
#        rho_p_lvl = compute_density_particle(w_s_lvl, T_avg)
#        R_p_lvl = compute_radius_from_mass_vec(m_p_lvl, rho_p_lvl)
        m_w_lvl = m_p_lvl - m_s_lvl
        dm_l_level += np.sum(m_w_lvl * xi_lvl)
        dm_p_level += np.sum(m_p_lvl * xi_lvl)
        for mode_n in range(no_modes):
            no_pt_mode_ = len(xi_lvl[modes_lvl==mode_n])
            print("placed", no_pt_mode_,
                  f"SIPs in mode {mode_n}")
            with open(log_file, "a") as f:
                f.write(f"placed {no_pt_mode_} ")
                f.write(f"SIPs in mode {mode_n}, ")
                f.write(f"-> {no_pt_mode_/grid.no_cells[0]:.2f} ")
                f.write("SIPs per cell\n")
                
    #####################################################################
    # OLD WORKING                                 
#        for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
#            # print("weights_R_s[l][:,j]")
#            # print("no_rpcm[l]")
#            # print(weights_R_s[l][:,j])
#            # print(no_rpcm[l])
#            # the number concentration is not constant but depends on the
#            # mass density of the vicinity
#            xi[l][:,j] = np.rint(weights_R_s[l][:,j]
#                                 * no_rpcm[np.nonzero(no_spcm)][l]).astype(int)
#            # print("xi[l][:,j]")
#            # print(xi[l][:,j])
#            # initial weight fraction of this level dependent on S_amb
#            # -> try to place the particles with m_w such that S = S_eq
##            w_s[l][:,j] = compute_initial_mass_fraction_solute_m_s_NaCl(
##                          m_s[l][:,j], S_avg, T_avg)
#            w_s[l][:,j] = compute_initial_mass_fraction_solute_NaCl(
#                          R_s[l][:,j], S_avg, T_avg)
#            m_p[l][:,j] = m_s[l][:,j] / w_s[l][:,j]
#            rho_p[l][:,j] = compute_density_particle(w_s[l][:,j], T_avg)
#            R_p[l][:,j] = compute_radius_from_mass_vec(m_p[l][:,j],
#                                                       rho_p[l][:,j])
#            m_w[l][:,j] = m_p[l][:,j] - m_s[l][:,j]
#            dm_l_level += np.sum(m_w[l][:,j] * xi[l][:,j])
#            dm_p_level += np.sum(m_p[l][:,j] * xi[l][:,j])
#            print("placed", len(xi[l][:,j].flatten()), "particles in a mode")
    # OLD WORKING END                                 
    #####################################################################
    
        # print('level = ', j)
    
        # convert from 10^-18 kg to kg
        dm_l_level *= 1.0E-18 
        dm_p_level *= 1.0E-18 
        mass_water_liquid_level += dm_l_level
        mass_particles_level += dm_p_level
        
        # now we distributed the particles between levels 0 and 1
        # thereby sampling liquid water,
        # i.e. the mass of water vapor has dropped
        mass_water_vapor_level -= dm_l_level
        
        # and the ambient fluid is heated by Q = m_l * L_v = C_p dT
        # C_p = m_dry_end*c_p_dry + m_v*c_p_v + m_p_end*c_p_p
        heat_capacity_level =\
            mass_air_dry_level * c.specific_heat_capacity_air_dry_NTP\
            + mass_water_vapor_level * c.specific_heat_capacity_water_vapor_20C\
            + mass_particles_level * c.specific_heat_capacity_water_NTP
        heat_of_vaporization = compute_heat_of_vaporization(T_avg)
        
        dT_avg = dm_l_level * heat_of_vaporization / heat_capacity_level
        
        # assume homogeneous heating: bottom, mid and top are heated equally
        # i.e. the slope (lapse rate) of the temperature in the level
        # remains constant,
        # but the whole (linear) T-curve is shifted by dT_avg
        T_avg += dT_avg
        T_bot += dT_avg
        T_top += dT_avg
        
        r_v_avg = mass_water_vapor_level / mass_air_dry_level
        
    #     R_m_avg = compute_specific_gas_constant_air_moist(r_v_avg)
        
        # EVEN BETTER:
        # assume linear T-profile
        # BETTER: exponential for const T and r_v:
        # p_2 = p_1 np.exp(-a/b * dz/T)
        # also poss.:
        # implicit formula for p
        # p_n+1 = p_n (1 - a/(2b) * dz/T) / (1 + a/(2b) * dz/T)
        # a = (1 + r_t) g, b = R_d(1 + r_v/eps)
        # chi = ( 1 + r_tot / (1 + r_v_avg ) )
        # this is equal to expo above up to order 2 in taylor!
        
        # IN WORK: check if kappa_tot_inv is right or it should be kappa(r_v)

        p_top = p_bot * (T_top/T_bot)**( ( (1 + r_tot) * c.earth_gravity * dz )\
                / ( (1 + r_v_avg / epsilon_gc ) * c.specific_gas_constant_air_dry * (T_bot - T_top) ) )

        # for assumed Theta_moist = const:
        # then (p/p1) = (T/T1) ^ [(eps + r_t)/ (kappa_t * (eps + r_v))]
#        p_top = p_bot * (T_top/T_bot)**( kappa_tot_inv * ( epsilon_gc + r_tot )\
#                / (epsilon_gc + r_v_avg) )
        
        p_avg = 0.5 * (p_bot + p_top)
        
        # from integration of dp/dz = - (1 + r_tot) g rho_dry
        rho_dry_avg = (p_bot - p_top) / ( dz * (1 + r_tot) * c.earth_gravity )
        mass_air_dry_level = grid.sizes[0] * dy * dz * rho_dry_avg
        
        r_l_avg = mass_water_liquid_level / mass_air_dry_level
        r_v_avg = r_tot - r_l_avg
        mass_water_vapor_level = r_v_avg * mass_air_dry_level
        
        e_s_avg = compute_saturation_pressure_vapor_liquid(T_avg)
        
        S_avg = compute_pressure_vapor( rho_dry_avg * r_v_avg, T_avg ) \
                / e_s_avg
        
    #     rho_m_avg = p_avg / (R_m_avg * T_avg)
    #     rho_tot_avg = rho_m_avg * ((1 + r_tot)/(1 + r_v_avg))
    #     p = p_prev  - earth_gravity * rho_tot_avg * dz
        
    ########################################################
    ### 3b. saturation adjustment in level, CELL WISE
    # this was the initial placement of particles
    # now comes the saturation adjustment incl. condensation/vaporization
    # due to supersaturation/subsaturation
    # note that there will also be subsaturation if S is smaller than S_act,
    # because water vapor was taken from the atm. in the intitial
    # particle placement step
    ########################################################
    
        # initialize for saturation adjustment loop
        # loop until the change in dm_l_level is sufficiently small:
        grid.mixing_ratio_water_vapor[:,j] = r_v_avg
        grid.mixing_ratio_water_liquid[:,j] = r_l_avg
        grid.saturation_pressure[:,j] = e_s_avg
        grid.saturation[:,j] = S_avg
    
        dm_l_level = mass_water_liquid_level
        iter_cnt = 0
        
        print('S_env_init after placement =', S_avg)
        print("sat. adj. start")
        with open(log_file, "a") as f:
            f.write(f"S_env_init after placement = {S_avg:.5f}\n")
            f.write("sat. adj. start\n")
        # need to define criterium -> use relative change in liquid water
        while ( np.abs(dm_l_level/mass_water_liquid_level) > 1e-5
                and iter_cnt < iter_cnt_limit ):
            ## loop over particles in level:
            dm_l_level = 0.0
            dm_p_level = 0.0
            
            D_v = compute_diffusion_constant( T_avg, p_avg )
            K = compute_thermal_conductivity_air(T_avg)
            # c_p = compute_specific_heat_capacity_air_moist(r_v_avg)
            L_v = compute_heat_of_vaporization(T_avg)
    
            for i in range(grid.no_cells[0]):
                cell = (i,j)
                e_s_avg = grid.saturation_pressure[cell]
                # set arbitrary maximum saturation during spin up to
                # avoid overshoot
                # at high saturations, since S > 1.05 can happen initially
                S_avg = grid.saturation[cell]
                S_avg2 = np.min([S_avg, S_init_max ])
                
                ind_x = cells_x_lvl == i
                
                m_w_cell = m_w_lvl[ind_x]
                m_s_cell = m_s_lvl[ind_x]
                
                if solute_type == "NaCl":
                    R_p_cell, w_s_cell, rho_p_cell =\
                        compute_R_p_w_s_rho_p_NaCl(m_w_cell, m_s_cell, T_avg)
                    sigma_p_cell = compute_surface_tension_water(T_avg)
                    dm_l, gamma_ =\
                    compute_dml_and_gamma_impl_Newton_full_NaCl(
                        dt_init, Newton_iterations, m_w_cell,
                        m_s_cell, w_s_cell, R_p_cell, T_avg,
                        rho_p_cell,
                        T_avg, p_avg, S_avg2, e_s_avg,
                        L_v, K, D_v, sigma_p_cell)
                elif solute_type == "AS":
                    R_p_cell, w_s_cell, rho_p_cell =\
                        compute_R_p_w_s_rho_p_AS(m_w_cell, m_s_cell, T_avg)
                    sigma_p_cell = compute_surface_tension_AS(w_s_cell, T_avg)
                    dm_l, gamma_ =\
                    compute_dml_and_gamma_impl_Newton_full_AS(
                        dt_init, Newton_iterations, m_w_cell,
                        m_s_cell, w_s_cell, R_p_cell, T_avg,
                        rho_p_cell,
                        T_avg, p_avg, S_avg2, e_s_avg,
                        L_v, K, D_v, sigma_p_cell)
                
                
                m_w_lvl[ind_x] += dm_l
#                m_w[l][cell] += dm_l
                
                dm_l_level += np.sum(dm_l * xi_lvl[ind_x])
                dm_p_level += np.sum(dm_l * xi_lvl[ind_x])
                
            ######################################################
            # OLD WORKING                
#                for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
#                    m_w_cell = m_w[l][cell]
#                    m_s_cell = m_s[l][cell]
#                    
#                    R_p_cell, w_s_cell, rho_p_cell =\
#                        compute_R_p_w_s_rho_p(m_w_cell, m_s_cell, T_avg)
#                    
#                    dm_l, gamma_ =\
#                    compute_dml_and_gamma_impl_Newton_full_np(
#                        dt_init, Newton_iterations, m_w_cell,
#                        m_s_cell, w_s_cell, R_p_cell, T_avg,
#                        rho_p_cell,
#                        T_avg, p_avg, S_avg2, e_s_avg, L_v, K, D_v, sigma_w)
#                    
#                    m_w[l][cell] += dm_l
#                    
#                    dm_l_level += np.sum(dm_l * xi[l][cell])
#                    dm_p_level += np.sum(dm_l * xi[l][cell])
            # OLD WORKING                
            ######################################################
                
            # IN WORK: check if the temperature, pressure etc is right
            # BEFORE the level starts and check the addition to these values!
            # after 'sampling' water from all particles: heat the cell volume:
            # convert from 10^-18 kg to kg
            dm_l_level *= 1.0E-18 
            dm_p_level *= 1.0E-18 
            mass_water_liquid_level += dm_l_level
            mass_particles_level += dm_p_level
    
            # now we distributed the particles between levels 0 and 1
            # thereby sampling liquid water,
            # i.e. the mass of water vapor has dropped
            mass_water_vapor_level -= dm_l_level
    
            # and the ambient fluid is heated by Q = m_l * L_v = C_p dT
            # C_p = m_dry_end*c_p_dry + m_v*c_p_v + m_p_end*c_p_p
            heat_capacity_level =\
                mass_air_dry_level * c.specific_heat_capacity_air_dry_NTP \
                + mass_water_vapor_level\
                  * c.specific_heat_capacity_water_vapor_20C \
                + mass_particles_level * c.specific_heat_capacity_water_NTP
            heat_of_vaporization = compute_heat_of_vaporization(T_avg)
    
            dT_avg = dm_l_level * heat_of_vaporization / heat_capacity_level
    
            # assume homogeneous heating: bottom, mid and top are heated equally
            # i.e. the slope (lapse rate) of the temperature in the level
            # remains constant,
            # but the whole (linear) T-curve is shifted by dT_avg
            T_avg += dT_avg
            T_bot += dT_avg
            T_top += dT_avg
    
            r_v_avg = mass_water_vapor_level / mass_air_dry_level
    
            # EVEN BETTER:
            # assume linear T-profile
            # then (p/p1) = (T/T1) ^ [(eps + r_t)/ (kappa_t * (eps + r_v))]
            # BETTER: exponential for const T and r_v:
            # p_2 = p_1 np.exp(-a/b * dz/T)
            # also poss.:
            # implicit formula for p
            # p_n+1 = p_n (1 - a/(2b) * dz/T) / (1 + a/(2b) * dz/T)
            # a = (1 + r_t) g, b = R_d(1 + r_v/eps)
            # chi = ( 1 + r_tot / (1 + r_v_avg ) )
            # this is equal to expo above up to order 2 in taylor!
    
            # IN WORK: check if kappa_tot_inv is right or should be kappa(r_v)
            # -> should be right,
            # because the lapse rate dT/dt ~ beta_tot is not altered!
            # the T curve is shifted upwards in total
            p_top = p_bot * (T_top/T_bot)**( kappa_tot_inv
                                             * ( epsilon_gc + r_tot )
                                             / ( epsilon_gc + r_v_avg ) )
            p_avg = 0.5 * (p_bot + p_top)
    
            # from integration of dp/dz = - (1 + r_tot) * g * rho_dry
            rho_dry_avg = (p_bot - p_top)\
                          / ( dz * (1 + r_tot) * c.earth_gravity )
            mass_air_dry_level = grid.sizes[0] * dy * dz * rho_dry_avg
            mass_air_dry_cell = dx * dy * dz * rho_dry_avg
            
            r_l_avg = mass_water_liquid_level / mass_air_dry_level
            r_v_avg = r_tot - r_l_avg
            
            grid.mixing_ratio_water_liquid[:,j].fill(0.0)
            for i in range(grid.no_cells[0]):
                cell = (i,j)
                ind_x = cells_x_lvl == i
                grid.mixing_ratio_water_liquid[cell] +=\
                    np.sum(m_w_lvl[ind_x] * xi_lvl[ind_x])
                
#                for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
#                    grid.mixing_ratio_water_liquid[cell] +=\
#                        np.sum(m_w[l][cell] * xi[l][cell])
    
            grid.mixing_ratio_water_liquid[:,j] *= 1.0E-18 / mass_air_dry_cell
            grid.mixing_ratio_water_vapor[:,j] =\
                r_tot - grid.mixing_ratio_water_liquid[:,j]
            
            grid.saturation_pressure[:,j] =\
                compute_saturation_pressure_vapor_liquid(T_avg)
            grid.saturation[:,j] =\
                compute_pressure_vapor(
                    rho_dry_avg * grid.mixing_ratio_water_vapor[:,j], T_avg )\
                / grid.saturation_pressure[:,j]
            
            mass_water_vapor_level = r_v_avg * mass_air_dry_level
            
            e_s_avg = compute_saturation_pressure_vapor_liquid(T_avg)
    
            S_avg = compute_pressure_vapor( rho_dry_avg * r_v_avg, T_avg ) \
                    / e_s_avg
            
            iter_cnt += 1
        
#        print(grid.mixing_ratio_water_vapor[:,j])
        
        if iter_cnt_max < iter_cnt: 
            iter_cnt_max = iter_cnt
            iter_cnt_max_level = j
    #     print('iter_cnt_max: ', iter_cnt_max)
        print('sat. adj. end: iter_cnt = ', iter_cnt,
              ', S_avg_end = ', S_avg,
              '\ndm_l_level = ', dm_l_level,
              ', m_l_level = ', mass_water_liquid_level,
              '\ndm_l_level/mass_water_liquid_level = ',
              dm_l_level/mass_water_liquid_level)
        with open(log_file, "a") as f:
            f.write(f'sat. adj. end: iter_cnt = {iter_cnt}, ')
            f.write(f"S_avg_end = {S_avg:.5f}\n")
            f.write(f"dm_l_level = {dm_l_level}\n")
            f.write(f"m_l_level = {mass_water_liquid_level}\n")
            f.write(f"dm_l_level/mass_water_liquid_level = ")
            f.write(f"{dm_l_level/mass_water_liquid_level}\n\n")
        mass_water_liquid_levels.append(mass_water_liquid_level)
        mass_water_vapor_levels.append( mass_water_vapor_level )
    
        p_env_init_bottom[n_top] = p_top
        p_env_init_center[j] = p_avg
        T_env_init_center[j] = T_avg
        r_v_env_init_center[j] = r_v_avg
        r_l_env_init_center[j] = r_l_avg
        rho_dry_env_init_center[j] = rho_dry_avg
        S_env_init_center[j] = S_avg
        
        m_w.append(m_w_lvl)
        m_s.append(m_s_lvl)
        xi.append(xi_lvl)
        
        cells_x.append(cells_x_lvl)
        cells_z.append(np.ones_like(cells_x_lvl) * j)        
        
        modes.append(modes_lvl)        
        
    m_w = np.concatenate(m_w)        
    m_s = np.concatenate(m_s)        
    xi = np.concatenate(xi)        
    cells_x = np.concatenate(cells_x)        
    cells_z = np.concatenate(cells_z)   
    cells_comb = np.array( (cells_x, cells_z) )
    modes = np.concatenate(modes)   
    
    print('')
    print("### Saturation adjustment ended for all lvls ###")
    print('iter count max = ', iter_cnt_max, ' level = ', iter_cnt_max_level)
    # total number of particles in grid
    no_particles_tot = np.size(m_w)
    print('last particle ID = ', len(m_w) - 1 )
    print ('no_super_particles_tot placed = ', no_particles_tot)
    # print('no_super_particles_tot should',
    #       no_super_particles_tot  )
    print('no_cells x N_p_cell_tot =', np.sum(no_rpct_0)*grid.no_cells[0]\
                                                        *grid.no_cells[1] )
    
    with open(log_file, "a") as f:
        f.write("\n")
        f.write("### Saturation adjustment ended for all lvls ###\n")
        f.write(f'iter cnt max = {iter_cnt_max}, at lvl {iter_cnt_max_level}')
        f.write("\n")
        f.write(f'last particle ID = {len(m_w) - 1}\n' )
        f.write(f'no_super_particles_tot placed = {no_particles_tot}\n')
        f.write('no_cells x N_p_cell_tot = ')
        f.write(
        f"{np.sum(no_rpct_0) * grid.no_cells[0] * grid.no_cells[1]:.3e}\n")
    
    for i in range(grid.no_cells[0]):
        grid.pressure[i] = p_env_init_center
        grid.temperature[i] = T_env_init_center
        grid.mass_density_air_dry[i] = rho_dry_env_init_center
        
    p_dry = grid.mass_density_air_dry * c.specific_gas_constant_air_dry\
            * grid.temperature
    
    grid.potential_temperature = grid.temperature\
                                 * ( 1.0E5 / p_dry )**kappa_air_dry
                                 
    grid.saturation_pressure =\
        compute_saturation_pressure_vapor_liquid(grid.temperature)
       
    rho_dry_env_init_bottom = 0.5 * ( rho_dry_env_init_center[0:-1]
                                      + rho_dry_env_init_center[1:])
    rho_dry_env_init_bottom =\
        np.insert( rho_dry_env_init_bottom, 0, 
                   0.5 * ( 3.0 * rho_dry_env_init_center[0]
                           - rho_dry_env_init_center[1] ))
    rho_dry_env_init_bottom =\
        np.append( rho_dry_env_init_bottom,
                   0.5 * ( 3.0 * rho_dry_env_init_center[-1]
                           - rho_dry_env_init_center[-2] ) )
    
    print()
    print("placed ", len(m_w.flatten()), "super particles" )
    print("representing ", np.sum(xi.flatten()), "real particles:" )
    print("mode real_part_placed, real_part_should, " + 
          "rel_dev:")
    
    with open(log_file, "a") as f:
        f.write("\n")
        f.write(f"placed {len(m_w.flatten())} super particles\n")
        f.write(f"representing {np.sum(xi.flatten()):.3e} real particles:\n")
        f.write("mode real_part_placed, real_part_should, ")
        f.write("rel dev:\n")
    
    if no_modes == 1:
#        print(0, np.sum(xi), no_rpt_should, np.sum(xi) - no_rpt_should)
        print(0, f"{np.sum(xi):.3e}",
              f"{no_rpt_should:.3e}",
              f"{np.sum(xi) - no_rpt_should:.3e}")
        with open(log_file, "a") as f:
            f.write(f"0 {np.sum(xi):.3e} {no_rpt_should:.3e} ")
            f.write(f"{(np.sum(xi) - no_rpt_should)/no_rpt_should:.3e}\n")
    else:
        for mode_n in range(no_modes):
            ind_mode = modes == mode_n
            rel_dev_ = (np.sum(xi[ind_mode]) - no_rpt_should[mode_n])\
                      /no_rpt_should[mode_n]
            print(mode_n, f"{np.sum(xi[ind_mode]):.3e}",
                  f"{no_rpt_should[mode_n]:.3e}",
                  f"{rel_dev_:.3e}")
            with open(log_file, "a") as f:
                f.write(f"{mode_n} {np.sum(xi[ind_mode]):.3e} ")
                f.write(f"{no_rpt_should[mode_n]:.3e} ")
                f.write(f"{rel_dev_:.3e}\n")
    
#    l_ = 0
#    for l, N_l in enumerate( no_spcm ):
#        if l in np.nonzero(no_spcm)[0]:
#            print( l, np.sum(xi[l_]), no_rpt_should[l],
#                   np.sum(xi[l_]) - no_rpt_should[l])
#            l_ += 1
#        else:
#            print( l, 0, no_rpt_should[l], -no_rpt_should[l])
    
    r_tot_err =\
        np.sum(np.abs(grid.mixing_ratio_water_liquid
                      + grid.mixing_ratio_water_vapor
                      - r_tot_0))
    print()
    print(f"accumulated abs. error"
          + "|(r_l + r_v) - r_tot_should| over all cells = "
          + f"{r_tot_err:.4e}")
    with open(log_file, "a") as f:
        f.write("\n")
        f.write(f"accumulated abs. error |(r_l + r_v) - r_tot_should| ")
        f.write(f"over all cells ")
        f.write(f"= {r_tot_err:.4e}\n")
    ########################################################
    ### 4. set mass flux and velocity grid
    ######################################################## 
    
    j_max = 0.6 * np.sqrt(grid.sizes[0] * grid.sizes[0] +
                          grid.sizes[1] * grid.sizes[1])\
                / np.sqrt(2.0 * 1500.0*1500.0)
    print()
    print('j_max')
    print(j_max)
    with open(log_file, "a") as f:
        f.write(f"\nj_max = {j_max}")
    grid.mass_flux_air_dry =\
        compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1( grid,
                                                                        j_max )
    
    # grid.mass_flux_air_dry[1]:
    # j_z - positions taken at the z-bottom positions
    # every element mass_flux_air_dry[1][i] is a z-profile at fix x
    # and gets divided by the dry density profile at the bottoms
    # note that this is checked to provide the right division by numpy:
    # since mass_flux_air_dry[1] is an 2D array of shape (Nx, Nz)
    # and rho_dry_env_init_bottom is a 1D array of shape (Nz,)
    # each element of mass_flux_air_dry[1] (which is itself an array of dim Nz)
    # is divided BY THE FULL ARRAY 'rho_dry_env_init_bottom'
    # grid.mass_flux_air_dry[0]:
    # j_x - positions taken at the z-center positions
    # every element mass_flux_air_dry[0][i] is a z-profile at fix x_i
    # and gets divided by the dry density profile at the centers
    # note that this is checked to provide the right division by numpy:
    # since mass_flux_air_dry[0] is an 2D array of shape (Nx, Nz)
    # and rho_dry_env_init_bottom is a 1D array of shape (Nz,)
    # each element of mass_flux_air_dry[0] (which is itself an array of dim Nz)
    # is divided BY THE FULL ARRAY 'rho_dry_env_init_bottom' BUT
    # we need to add a dummy density for the center in a cell above
    # the highest cell, because velocity has
    # the same dimensions as grid.corners and not grid.centers
    grid.velocity[0] = grid.mass_flux_air_dry[0] / rho_dry_env_init_bottom
    grid.velocity[1] =\
        grid.mass_flux_air_dry[1]\
        / np.append(rho_dry_env_init_center, rho_dry_env_init_center[-1])
    
    grid.update_material_properties()
    V0_inv = 1.0 / grid.volume_cell
    grid.rho_dry_inv = np.ones_like(grid.mass_density_air_dry)\
                       / grid.mass_density_air_dry
    grid.mass_dry_inv = V0_inv * grid.rho_dry_inv
    
    # assign random positions to particles
    
    pos, rel_pos, cells = generate_random_positions(grid, no_spc, rnd_seed,
                                                    set_seed=False)
    # init velocities
    #vel = np.zeros_like(pos)
    
    # generate cell array [ [i1,j1], [i2,j2], ... ] and flatten particle masses
#    ID = 0
#    m_w_flat = np.zeros(len(pos[0]))
#    m_s_flat = np.zeros(len(pos[0]))
#    xi_flat = np.zeros(len(pos[0]), dtype = np.int64)
#    cell_list = np.zeros( (2, len(pos[0])), dtype = int )
    
    # @njit()
    # def flatten_m_w_m_s(m_w,m_s,m_w_flat,m_s_flat, no_spcm, grid_no_cells):
    #     for j in range(grid_no_cells[1]):
    #         for i in range(grid_no_cells[0]):
    #             for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
    #                 for n, m_w_ in m_w[l][i,j]:
    #                     cell_list[0,ID] = i
    #                     cell_list[1,ID] = j
    #                     m_w_flat[ID] = m_w[l][i,j][n]
    #                     m_s_flat[ID] = m_s[l][i,j][n]
                        
#    for j in range(grid.no_cells[1]):
#        for i in range(grid.no_cells[0]):
#            for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
#                if isinstance(m_w[l][i,j], (list, tuple, np.ndarray)):
#                    for n, m_w_ in enumerate(m_w[l][i,j]):
#                        cell_list[0,ID] = i
#                        cell_list[1,ID] = j
#                        m_w_flat[ID] = m_w[l][i,j][n]
#                        m_s_flat[ID] = m_s[l][i,j][n]
#                        xi_flat[ID] = xi[l][i,j][n]
#                        ID += 1
#                else:
#                    cell_list[0,ID] = i
#                    cell_list[1,ID] = j
#                    m_w_flat[ID] = m_w[l][i,j]
#                    m_s_flat[ID] = m_s[l][i,j]
#                    xi_flat[ID] = xi[l][i,j]
#                    ID += 1
### IN WORK    
    vel = interpolate_velocity_from_cell_bilinear(cells, rel_pos,
                                                      grid.velocity,
                                                      grid.no_cells)
    
#    active_ids = list(range(len(m_s_flat)))
#    active_ids = list(range(len(m_s)))
#    active_ids = list(range(len(m_s)))
#    removed_ids = []
    
    active_ids = np.full( len(m_s), True )
#    removed_ids = np.full( len(m_s), False )
    
    t = 0
    save_grid_and_particles_full(t, grid, pos, cells, vel,
                                 m_w, m_s, xi,
                                 active_ids, save_path)
    
    np.save(save_path + "modes_0", modes)
    np.save(save_path + "no_rpcm_scale_factors_lvl_wise",
            no_rpcm_scale_factors_lvl_wise)
    
#    save_grid_and_particles_full(t, grid, pos, cell_list, vel,
#                                 m_w_flat, m_s_flat, xi_flat,
#                                 active_ids, removed_ids, save_path)
    
    paras = [x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, r_tot_0,
             Theta_l, DNC0, no_spcm, dist,
             dst_par, kappa_dst, eta, eta_threshold,
             r_critmin, m_high_over_m_low,
             eta, rnd_seed, S_init_max, dt_init,
             Newton_iterations, iter_cnt_limit]
    para_names = "x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, " \
    + "r_tot_0, Theta_l, DNC0, no_super_particles_cell_mode, dist, " \
    + "mu_m_log, sigma_m_log, kappa_dst, eta, eta_threshold, " \
    + "r_critmin, m_high_over_m_low, rnd_seed, S_init_max, dt_init, " \
    + "Newton_iterations_init, iter_cnt_limit"
    
    grid_para_file = save_path + "grid_paras.txt"
    with open(grid_para_file, "w") as f:
        f.write( para_names + '\n' )
        for item in paras:
            type_ = type(item)
            if type_ is list or type_ is np.ndarray or type_ is tuple:
                for el in item:
                    tp_e = type(el)
                    if tp_e is list or tp_e is np.ndarray or tp_e is tuple:
                        for el_ in el:
                            f.write( f'{el_} ' )
                    else:
                        f.write( f'{el} ' )
            else: f.write( f'{item} ' )        
    
#    if logfile:
#        sys.stdout = sys.__stdout__
#        log_handle.close()
    
    return grid, pos, cells, cells_comb, vel, m_w, m_s, xi,\
           active_ids 
#    return grid, pos, cells, vel, m_w_flat, m_s_flat, xi_flat,\
#           active_ids, removed_ids


#%% testing (commented)
# particles: pos, vel, masses, multi,

# base grid without THD or flow values

# from grid import Grid
# from grid import interpolate_velocity_from_cell_bilinear_jit,\
#                  interpolate_velocity_from_position_bilinear_jit,\
#                  compute_cell_and_relative_position_jit
#                  # compute_cell_and_relative_position,\

# # domain size
# x_min = 0.0
# x_max = 120.0
# z_min = 0.0
# z_max = 200.0

# # grid steps
# dx = 20.0
# dy = 1.0
# dz = 20.0

# grid_ranges = [ [x_min, x_max], [z_min, z_max] ]
# grid_steps = [dx, dz]
# grid = Grid( grid_ranges, grid_steps, dy )

# # set mass flux, note that the velocity requires the dry density field first
# j_max = 0.6
# j_max = compute_j_max(j_max, grid)
# grid.mass_flux_air_dry = np.array(
#   compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1(grid, j_max))
# grid.velocity = grid.mass_flux_air_dry / c.mass_density_air_dry_NTP


# print("grid.velocity[0]")
# print(np.shape(grid.velocity))
# print(grid.velocity[0])

# # gen. positions of no_spc Sp per cell
# no_spc = 20
# seed = 4713
# pos, rel_pos = generate_random_positions(grid, no_spc, seed)
# cell_list = np.zeros_like(pos).astype(int)

# n = 0
# for j in range(grid.no_cells[1]):
#     for i in range(grid.no_cells[0]):
#         for sp_N in range(no_spc):
#             cell_list[0,n] = i
#             cell_list[1,n] = j
#             n += 1

# # cells = compute_cell(*pos,
# #                                            grid.steps[0],
# #                                            grid.steps[1])

# # i, j, x_rel, y_rel = compute_cell_and_relative_position_jit(*pos,
# cells, rel_pos = compute_cell_and_relative_position_jit(pos,
#                                                         grid.ranges,
#                                                         grid.steps)
#                                              # grid.ranges[0,0],
#                                              # grid.ranges[1,0],
#                                              # grid.steps[0],
#                                              # grid.steps[1])

# print()
# print("cells")
# print(cells)


# # print(i)
# # print(j)
# # print(x_rel)
# # print(y_rel)


# # cell_list = np.array((i, j))
# # rel_post = np.array((x_rel, y_rel))

# # vel = interpolate_velocity_from_position_bilinear_jit(
# #           grid.velocity, grid.no_cells, grid.ranges, grid.steps, pos)
# print()
# print(type(grid.velocity))
# print(type(grid.no_cells))
# print(type(cells))
# print(type(rel_pos))
# print()

# print("cells[0,1]")
# print(cells[0,1])


# vel3 = interpolate_velocity_from_cell_bilinear_jit(grid.velocity,
#                                                   grid.no_cells,
#                                                   cells, rel_pos)
# np.set_printoptions(precision=4)
# print()
# print("vel")
# print(vel)

# vel = interpolate_velocity_from_position_bilinear_jit(pos, grid.velocity,
#                                                        grid.no_cells,
#                                                        grid.ranges,
#                                                        grid.steps,
#                                                        )



# vel2 = np.zeros_like(vel)
# for n,x in enumerate(pos[0]):
#     u,v = grid.interpolate_velocity_from_location_bilinear(x, pos[1,n])
#     vel2[0,n] = u
#     vel2[1,n] = v

# print()
# print("vel2 - vel1")
# print(vel2 - vel)
# max_dev = 0.0
# max_n = 0
# for n,v in enumerate(vel[0]):
#     if abs(vel2[0,n] - vel[0,n]) > max_dev:
#         max_dev = vel2[0,n] - vel[0,n]
#         max_n = n
#     if abs( vel2[1,n] - vel[1,n]) > max_dev:
#         max_dev = vel2[0,n] - vel[0,n]
#         max_n = n
# print(max_n, max_dev)
# print()

# #%%

# # visualize particles in grid
# # %matplotlib inline
# def plot_pos_pt(pos, grid, figsize=(6,6), MS = 1.0):
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
#     ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
#     ax.set_xticks(grid.corners[0][:,0])
#     ax.set_yticks(grid.corners[1][0,:])
#     # ax.set_xticks(grid.corners[0][:,0], minor = True)
#     # ax.set_yticks(grid.corners[1][0,:], minor = True)
#     plt.minorticks_off()
#     # plt.minorticks_on()
#     ax.grid()
#     # ax.grid(which="minor")
#     plt.show()
# def plot_pos_vel_pt(pos, vel, grid, figsize=(8,8), MS = 1.0):
#     u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
#     v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
#     ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
#     ax.quiver(*pos, *vel, scale=2, pivot="mid")
#     ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
#               scale=2, pivot="mid", color="red")
#     # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
#     #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
#     #           scale=0.5, pivot="mid", color="red")
#     # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
#     #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
#     #           scale=0.5, pivot="mid", color="blue")
#     ax.set_xticks(grid.corners[0][:,0])
#     ax.set_yticks(grid.corners[1][0,:])
#     # ax.set_xticks(grid.corners[0][:,0], minor = True)
#     # ax.set_yticks(grid.corners[1][0,:], minor = True)
#     plt.minorticks_off()
#     # plt.minorticks_on()
#     ax.grid()
#     # ax.grid(which="minor")
#     plt.show()
    
# plot_pos_vel_pt(pos, vel, grid)

# #%%



# print(grid.corners[0][:,0])

# print("radii distribution")

# p_min = [0.001] * 2
# p_max = [0.999] * 2
# no_spcm = [4, 3]

# dst = [dst_log_normal, dst_log_normal]
# no_modes = 2
# mus = [0.02, 0.075]
# sigmas = [1.4, 1.6]
# pars = []
# for i,mu in enumerate(mus):
#     pars.append( [mu, math.log(sigmas[i])] )
# print("pars = ", pars)
# dr = [1E-4] * 2
# r0 = dr
# r1 = [ pars[0][1]*10, pars[1][1]*10 ]
# seed = [4711] * 2

# rad1,weights1 = generate_random_radii_monomodal(grid, p_min[0], p_max[0],
#                                                 no_spcm[0],
#                                 dst[0], pars[0], r0[0], r1[0], dr[0], seed[0])

# rad2,weights2 = generate_random_radii_monomodal(grid, p_min[1], p_max[1],
#                                                 no_spcm[1],
#                                 dst[1], pars[1], r0[1], r1[1], dr[0], seed[1])

# rad_merged = np.concatenate( (rad1, rad2),axis=2 )

# np.set_printoptions(precision=5)

# print(np.shape(rad1))
# print(np.shape(rad2))
# print(np.shape(rad_merged))
# print()
# # for i in range(grid.no_cells[0]):
# #     for j in range(grid.no_cells[1]):
# #         print(rad1[i,j])
# # print()
# # for i in range(grid.no_cells[0]):
# #     for j in range(grid.no_cells[1]):
# #         print(rad2[i,j])
# # print()
# # for i in range(grid.no_cells[0]):
# #     for j in range(grid.no_cells[1]):
# #         print(rad_merged[i,j])
# # print()

# p_min = 0.001
# p_max = 0.999

# no_modes = 2
# no_spcm = [4, 3]
# dst = dst_log_normal
# mus = [0.02, 0.075]
# sigmas = [1.4, 1.6]
# dist_pars = []
# for i,mu in enumerate(mus):
#     dist_pars.append( [mu, math.log(sigmas[i])] )
# print("pars = ", pars)
# dr = 1E-4
# r0 = dr
# r1 = [ 10 * p_[1] for p_ in pars ]
# print(r1)
# seed = 4711

# rad, weight = generate_random_radii_multimodal(grid, p_min, p_max, no_spcm,
#                                      dst, dist_pars, r0, r1, dr, seed)

# print(np.shape(rad))
# print(np.shape(weight))

# for i in range(grid.no_cells[0]):
#     for j in range(grid.no_cells[1]):
#         print(rad_merged[i,j])
# print()
# for i in range(grid.no_cells[0]):
#     for j in range(grid.no_cells[1]):
#         print(rad[i,j])
# print()
# for i in range(grid.no_cells[0]):
#     for j in range(grid.no_cells[1]):
#         print(rad[i,j] - rad_merged[i,j])

# # generate_random_radii_multimodal(grid, p_min, p_max, no_spcm,
# #                                      dst, pars, r0, r1, dr, seed, no_modes)
    
#     # for var in [p_min, p_max, no_spcm, dst, par, r0, r1, dr, seed]:
#     #     if not isinstance(var, (list, tuple, np.ndarray)):
#     #         var = [var] * no_modes
    
#     # print([p_min, p_max, no_spcm, dst, par, r0, r1, dr, seed])
# # for i in range(grid.no_cells[0]):
# #     for j in range(grid.no_cells[1]):
# #         print(weights1[i,j], np.sum(weights1[i,j]))


# # plot_pos_pt(pos[:,0:8*8], grid, MS = 2.0)
    
# #%%
# # print(1, np.random.get_state()[1][0])
# # print()

# # for seeds in np.arange(1,1000000,123456):
# #     np.random.seed(seeds)
# #     print(seeds, np.random.get_state()[1][0])


# # def test1(seed):
# #     print(np.random.get_state()[1][0])
# #     np.random.seed(seed)
# #     print(np.random.get_state()[1][0])
# # def test2(seed,reseed):
# #     print(np.random.get_state()[1][0])
# #     if reseed:
# #         test1(seed)
# #     print(np.random.get_state()[1][0])

# # print(np.random.get_state()[1][0])
# # print()
# # test1(4713)
# # print()
# # print(np.random.get_state()[1][0])
# # print()
# # test2(4715, True)
# # print()
# # print(np.random.get_state()[1][0])
