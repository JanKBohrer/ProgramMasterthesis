#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:10:17 2019

@author: bohrer
"""

from scipy import special
#import scipy.special
import numpy as np

import mpmath as mpm
mpm.mp.dps = 25; mpm.mp.pretty = True

import matplotlib.pyplot as plt

x = np.linspace(0.,1E-10,50)
#x = np.logspace(-15,-10,50)

y = special.iv(1,x)

# init distr is n(x,0) = n0/x0 * exp(-x/x0)
# thus int n(x,0) dx = n0
def dist_vs_time_golo_exp2(x,t,x0,n0,b):
    t = np.where(t<1E-12,1E-12,t)
    x_x0 = x/x0
    x = np.where(x_x0 < 1E-8, x0 * 1E-8, x)
    x_x0 = x/x0
    T = 1. - np.exp(-b*n0*x0*t)
    T_sqrt = np.sqrt(T)
    Bessel_term = np.where(special.iv(1, 2. * x_x0 * T_sqrt) > 1E240,
                           1E240,special.iv(1, 2. * x_x0 * T_sqrt))
    return n0 / x0 * Bessel_term \
           * (1. - T) * np.exp(-1. * x_x0 * (1. + T)) / (x_x0 * T_sqrt)

def dist_vs_time_golo_exp(x,t,x0,n0,b):
    res = np.zeros(len(x), dtype=np.float128)
    for n in range( len(x)):
        t = np.where(t<1E-12,1E-12,t)
        x_x0 = x[n]/x0
        x = np.where(x_x0 < 1E-8, x0 * 1E-8, x)
        x_x0 = x[n]/x0
        T = 1. - np.exp(-b*n0*x0*t)
        T_sqrt = np.sqrt(T)
        z = 2. * x_x0 * T_sqrt
        res[n] = n0 / x0 * (1. - T) * 2.\
                 * mpm.exp( -0.5 * (1. + T) * z / T_sqrt )\
                 * mpm.besseli(1,z) / z
#    Bessel_term = np.where(special.iv(1, 2. * x_x0 * T_sqrt) > 1E240,
#                           1E240,special.iv(1, 2. * x_x0 * T_sqrt))
    return res
#    return n0 / x0 * Bessel_term \
#           * (1. - T) * np.exp(-1. * x_x0 * (1. + T)) / (x_x0 * T_sqrt)
#    return np.where(x_x0 < 1E-6, n0/x0*(1.-T), 
#               n0/x0*special.iv(1, 2. * x_x0 * T_sqrt)
#               * (1. - T) * np.exp(-1. * x_x0 * (1. + T)) / (x_x0 * T_sqrt))




def dst_m_expo(x,x0,n0):
    return n0 / x0 * np.exp(-x/x0)

#%%

#%%
#x0 = 3.37E-12 # kg
#n0 = 300E6 # 1/m^3
#b = 1.5 # m^3 / (kg s)
#t0 = 3600
#x = np.loadtxt("masses_golo_test.txt").flatten()[:-2]
#
#
#T = 1. - np.exp(-b*n0*x0*t0)
#T_sqrt = np.sqrt(T)
#x_x0 = x/x0
#
#print(T)
#print(T_sqrt)
#print(x_x0)
#
#print("special.iv(1, 2. * x_x0 * T_sqrt)")
#print(special.iv(1, 2. * x_x0 * T_sqrt))
#print("(1. - T)")
#print((1. - T))
#print("np.exp(-1. * x_x0 * (1. + T))")
#print(np.exp(-1. * x_x0 * (1. + T)))
#print("(x_x0 * T_sqrt)")
#print((x_x0 * T_sqrt))
#print(dist_vs_time_golo_exp(x,t0,x0,n0,b))
#print(dist_vs_time_golo_exp2(x,t0,x0,n0,b))

#%%


#print(b*x0*n0*t0)
#print(np.exp(-b*x0*n0*t0))
#%%

#fig, ax = plt.subplots(figsize=(6,6))
##ax.plot(x,y)
##ax.loglog(x, dist_vs_time_golo_exp(x,1E-5,x0,n0,b)-dst_m_expo(x,x0,n0) )
##ax.loglog(x, dist_vs_time_golo_exp(x,1E-5,x0,n0,b))
##ax.plot(x,dst_m_expo(x,x0,n0))
#ax.plot(x, np.abs(dist_vs_time_golo_exp(x,t0,x0,n0,b)-dst_m_expo(x,x0,n0))/(1E-6+dst_m_expo(x,x0,n0)))
#ax.set_yscale("log")
#ax.grid()