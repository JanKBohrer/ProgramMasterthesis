#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:26:45 2019

@author: jdesk
"""

import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import constants as c
# from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass
from microphysics import compute_mass_from_radius

two_pi_sqrt = np.sqrt(2.0*np.pi)
four_pi_over_three = 4.0*np.pi/3.0
four_pi_over_three_log = math.log(4.0*np.pi/3.0)
def dst_lognormal_R_np(R, mu_R_log, sigma_R_log):
    return np.exp( -0.5*( ( np.log( R ) - mu_R_log ) / sigma_R_log )**2 ) \
           / (R * two_pi_sqrt * sigma_R_log)
dst_lognormal_R = njit()(dst_lognormal_R_np)

# mass density in kg
def dst_lognormal_m_np(m, mu_R_log, sigma_R_log, mass_density): # = f_m(m)
#    print("m =",m)
    c = 1.0E-18 * four_pi_over_three * mass_density
    return np.exp( -0.5*( ( np.log(m/c) - 3.0*mu_R_log ) 
                          / (3.0*sigma_R_log) )**2 ) \
           / (two_pi_sqrt * 3.0 * sigma_R_log * m)
#    R = compute_radius_from_mass(1.0E18*m, c.mass_density_water_liquid_NTP)
#    return DNC * np.exp( -0.5*( ( np.log( R ) - mu_R_log ) / sigma_R_log )**2 ) \
#           / (R * two_pi_sqrt * sigma_R_log)
dst_lognormal_m = njit()(dst_lognormal_m_np)

# this function needs x0, x1 in radius values
def num_int_lognormal_R_np(x0, x1, par, steps=1E6):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_lognormal_R(x, par[0], par[1])
    # cnt = 0
    while (x < x1):
        f2 = dst_lognormal_R(x + 0.5*dx, par[0],par[1])
        f3 = dst_lognormal_R(x + dx, par[0],par[1])
        # intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
        # cnt += 1        
        # intl += dx * x * dst_expo(x,k)
        # x += dx
        # cnt += 1
    return intl
num_int_lognormal_R = njit()(num_int_lognormal_R_np)

def num_int_lognormal_m_np(x0, x1, par, steps=1E6):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_lognormal_m(x, par[0], par[1], par[2])
    # cnt = 0
    while (x < x1):
        f2 = dst_lognormal_m(x + 0.5*dx, par[0],par[1], par[2])
        f3 = dst_lognormal_m(x + dx, par[0],par[1], par[2])
        # intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
        # cnt += 1        
        # intl += dx * x * dst_expo(x,k)
        # x += dx
        # cnt += 1
    return intl
num_int_lognormal_m = njit()(num_int_lognormal_m_np)

def num_int_lognormal_moments_R_np(n, x0, x1, par, steps=1E6):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_lognormal_R(x, par[0], par[1]) * x**n
    # cnt = 0
    while (x < x1):
        f2 = dst_lognormal_R(x + 0.5*dx, par[0],par[1]) * (x+0.5*dx)**n
        f3 = dst_lognormal_R(x + dx, par[0],par[1]) * (x+dx)**n
        # intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
        # cnt += 1        
        # intl += dx * x * dst_expo(x,k)
        # x += dx
        # cnt += 1
    return intl
num_int_lognormal_moments_R = njit()(num_int_lognormal_moments_R_np)

def num_int_lognormal_mean_mass_R_np(n, x0, x1, par, steps=1E6):
    c = 1.0E-18 * four_pi_over_three * par[2]
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_lognormal_R(x, par[0], par[1]) * c * x**3
    # cnt = 0
    while (x < x1):
        f2 = dst_lognormal_R(x + 0.5*dx, par[0],par[1]) * c * (x+0.5*dx)**3
        f3 = dst_lognormal_R(x + dx, par[0],par[1]) * c * (x+dx)**3
        # intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
        # cnt += 1        
        # intl += dx * x * dst_expo(x,k)
        # x += dx
        # cnt += 1
    return intl
num_int_lognormal_mean_mass_R = njit()(num_int_lognormal_mean_mass_R_np)

def num_int_lognormal_moments_m_np(n, x0, x1, par, steps=1E6):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_lognormal_m(x, par[0], par[1], par[2]) * x**n
    # cnt = 0
    while (x < x1):
        f2 = dst_lognormal_m(x + 0.5*dx, par[0],par[1], par[2]) * (x+0.5*dx)**n
        f3 = dst_lognormal_m(x + dx, par[0],par[1], par[2]) * (x+dx)**n
        # intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
        # cnt += 1        
        # intl += dx * x * dst_expo(x,k)
        # x += dx
        # cnt += 1
    return intl
num_int_lognormal_moments_m = njit()(num_int_lognormal_moments_m_np)

def moments_analytical_lognormal_R(n, mu_R_log, sigma_R_log):
    if n == 0:
        return 1.0
    else:
        return np.exp(n * mu_R_log + 0.5 * n*n * sigma_R_log*sigma_R_log)

def moments_analytical_lognormal_m(n, mu_R_log, sigma_R_log, mass_density):
    if n == 0:
        return 1.0
    else:
        c = np.log(1.0E-18*four_pi_over_three*mass_density)
        return np.exp(n * (3.0*mu_R_log + c) \
                      + 4.5 * n*n * sigma_R_log*sigma_R_log)


#%% TEST LOGNORMAL DIST

### for lognormal dist
mu_R = 0.02 # in mu
sigma_R = 1.4 #  no unit

mu_R_log = np.log(mu_R)
sigma_R_log = np.log(sigma_R)

mass_density = c.mass_density_NaCl_dry # in kg/m^3           

dist_par_R = (mu_R_log, sigma_R_log)
dist_par_m = (mu_R_log, sigma_R_log, mass_density)

R_ = np.logspace(-3,0,1000) 
f_R_ana = dst_lognormal_R_np(R_, *dist_par_R)           

m_ = 1.0E-18 * compute_mass_from_radius(R_, mass_density)
f_m_ana = dst_lognormal_m_np(m_, *dist_par_m)


moments_ana_R = np.zeros(4)
moments_num_R = np.zeros(4)
moments_ana_m = np.zeros(4)
moments_num_m = np.zeros(4)

moments_num_R[0] = num_int_lognormal_R(R_[0], R_[-1], dist_par_R)
moments_ana_R[0] = 1.0

moments_num_m[0] = num_int_lognormal_m(m_[0], m_[-1], dist_par_m, steps=1E7)
moments_ana_m[0] = 1.0

m_mean_R = num_int_lognormal_mean_mass_R_np(n, R_[0], R_[-1], dist_par_m)

for n in range(1,4):
    moments_ana_R[n] = moments_analytical_lognormal_R(n, *dist_par_R)
    moments_num_R[n] = num_int_lognormal_moments_R(
                           n, R_[0], R_[-1], dist_par_R)
    moments_ana_m[n] = moments_analytical_lognormal_m(n, *dist_par_m)
    moments_num_m[n] = num_int_lognormal_moments_m(
                           n, m_[0], m_[-1], dist_par_m, steps=1E7)

for n in range(4):
    print(n, moments_ana_R[n], moments_num_R[n], moments_num_R[n]/moments_ana_R[n])
print()
for n in range(4):
    print(n, moments_ana_m[n], moments_num_m[n], moments_num_m[n]/moments_ana_m[n])

print()
print("m_mean_R/moments_ana_m[1]")
print(m_mean_R/moments_ana_m[1])

#%%
           
fig,axes = plt.subplots(3)
ax = axes[0]
ax.plot(R_, f_R_ana)
ax.plot(R_, f_m_ana * 3.0*m_/R_)
ax.set_xscale("log")

ax = axes[1]
#ax.plot(R_, f_m_ana)
ax.plot(m_, f_m_ana)
ax.plot(m_, f_R_ana * R_ / (3.0 * m_))
ax.set_xscale("log")

ax = axes[2]
ax.plot(R_, (f_R_ana - f_m_ana * 3.0*m_/R_) /f_R_ana )

fig.tight_layout()           