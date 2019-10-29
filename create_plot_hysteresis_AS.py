#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:16:57 2019

@author: jdesk
"""

import matplotlib as mpl
mpl.use('svg')
import matplotlib.pyplot as plt
import numpy as np

from plotting import cm2inch
from plotting import generate_rcParams_dict

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

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

# Values idealized from Biskos2006 and Haemeri
# -> here also formula for rho_aq, a_w, sigma_s
# for dry size D = 10 nm

from scipy.optimize import curve_fit

def fit_exp1(x,a,b,c):
    return a*np.exp(b*x) + c


data1 = [
        [0,1],
        [35,1]
        ]
data2 = [
        [80,1.3],
        [90,1.4],
        [100,1.55]
        ]

data_crys1 = [
            [35,1],
            [80,1],
            ]

data_crys2 = [
            [80,1],
            [80,1.3],
            ]

data_deli = [
            [35,1.0],
            [35,1.08],
            [42,1.11],
            [50,1.13],
            [62,1.2],
            [70,1.25],
            [80,1.3],
            [80,1.3],
            ]

data_exp = [
           [35,1.08],
           [60,1.17],
           [80,1.3],
           [100,1.55]
           ]

data1 = np.array(data1).T
data2 = np.array(data2).T
data_exp = np.array(data_exp).T
data_crys1 = np.array(data_crys1).T
data_crys2 = np.array(data_crys2).T
data_deli = np.array(data_deli).T

p1,p2 = curve_fit(fit_exp1, *data_exp, p0 = (1.0,0.01,0.1))


x1 = np.linspace(35,80,100)
#fit1 = np.hstack( ( np.array( (1.0) ) , fit_exp1(x1,*p1) ) )
fit1 = fit_exp1(x1,*p1)
#x1 = np.hstack( ( np.array( (35) ), x1 ) )

x1b = np.array((35,35))
fit1b = np.array((1.,1.08))

x2 = np.linspace(80,100,10)
fit2 = fit_exp1(x2,*p1)
#fit2 = np.hstack( ( np.array( () ) , fit_exp1(x2,*p1) ) )

#%%

#S = np.arange()

# figsize in cm (x,y)
fig_size = (7.5,6.0)
fig, ax = plt.subplots(figsize=cm2inch(fig_size))

ax.plot(data1[0],data1[1],c="k")
ax.plot(x1, fit1,":",c="k")
ax.plot(x1b, fit1b,":",c="k")
ax.plot(x2, fit2,c="k")
#ax.plot(x1, fit1,":",c="0.6")
#ax.plot(x1, fit1,"..",c="k")
#ax.plot(data2[0],data2[1],c="k")
#ax.plot(data_deli[0],data_deli[1],c="0.6")
ax.plot(data_crys1[0],data_crys1[1],"--",c="k")
ax.plot(data_crys2[0],data_crys2[1],"--",c="k")
#ax.plot(data_crys[0],data_crys[1],c="0.4")

ax.set_xlim((0.0,100.0))
ax.set_ylim((0.91,1.56))

ax.set_xlabel("Saturation S")
ax.set_ylabel("Radius growth factor")

fig_name = "HysteresisPlt.svg"
fig.savefig("/home/jdesk/Masterthesis/Figures/02Theory/" + fig_name,
            bbox_inches = 'tight',
            pad_inches = 0,
            format = 'svg')



