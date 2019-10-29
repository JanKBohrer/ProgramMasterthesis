#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:18:48 2019

@author: jdesk
"""

import numpy as np

# row =  collector drop radius:

#%% CREATE TABLE "Hall_collision_efficiency.npy":
# two dimensional array: rows = collector radius, cols = ratio R_small/R_col

#R_col_ratio = np.arange(0.0,1.05,0.05)
#R_coll_drop = np.array([300,200,150,100,70,60,50,40,30,20,10])
#R_coll_drop = R_coll_drop[::-1]
#
#raw_data = np.loadtxt("Hall_1980_Collision_eff.csv")
#
#raw_data = raw_data[::-1]
#
#E_even_R_coll_drop = []
#
#R_coll_drop_interpol = []
#
#for i in range(len(R_coll_drop)-1):
#    R_col = R_coll_drop[i]
#    E_even_R_coll_drop.append(raw_data[i])
#    R_coll_drop_interpol.append(R_col)
#    R_interpol = R_col + 10
#    dR2 = R_coll_drop[i+1] - R_interpol
#    if dR2 > 0: dR = R_coll_drop[i+1] - R_col
#    while (dR2 > 0):
#        dR1 = R_interpol - R_col
#        E_even_R_coll_drop.append((raw_data[i+1]*dR1 + raw_data[i]*dR2)/dR)
#        R_coll_drop_interpol.append(R_interpol)
#        R_interpol += 10
#        dR2 = R_coll_drop[i+1] - R_interpol
#        
#R_coll_drop_interpol.append(300)        
#E_even_R_coll_drop.append(raw_data[-1])
#
#R_coll_drop_interpol = np.array(R_coll_drop_interpol)
#E_even_R_coll_drop = np.array(E_even_R_coll_drop)
#
#E_even_R_coll_drop = np.hstack((np.zeros_like(R_coll_drop_interpol, dtype=np.float64)[:,None], E_even_R_coll_drop))
#
#E_even_R_coll_drop = np.concatenate((np.zeros_like(R_col_ratio, dtype=np.float64)[None,:], E_even_R_coll_drop))
#
#R_coll_drop_interpol = np.insert(R_coll_drop_interpol, 0, 0).astype(float)
#
##R_col_ratio = np.arange(0.05,1.05,0.05)
#
#filename = "Hall_collision_efficiency.npy"
#np.save(filename, E_even_R_coll_drop)
#filename = "Hall_collector_radius.npy"
#np.save(filename, R_coll_drop_interpol)
#filename = "Hall_radius_ratio.npy"
#np.save(filename, R_col_ratio)
    
    
#%%

#R_col_ratio = np.arange(0.05,1.05,0.05)
    




#%%
import math

Hall_E_col = np.load("Hall_collision_efficiency.npy")
Hall_R_col = np.load("Hall_collector_radius.npy")
Hall_R_col_ratio = np.load("Hall_radius_ratio.npy")

from grid import bilinear_weight
        
#i1 = 29
#j1 = 20
#R1 = Hall_R_col[i1]
#R2 = Hall_R_col_ratio[j1] * R1
#print()
#print("R1,R2, Hall_R_col_ratio[j1]")
#print(R1,R2, Hall_R_col_ratio[j1])
#print("Hall_E_col[i1,j1]")
#print(Hall_E_col[i1,j1])
#print(compute_E_col_Hall(R1,R2))

#%% COMPARE EFFICIENCIES FROM BOTT AND ME
fn = "Hall_1980_Collision_eff_Bott2.txt"
Bott_E_Hall = np.loadtxt(fn, delimiter=",")
Bott_E_Hall = np.reshape(Bott_E_Hall, (21,15)).transpose()[:,:]
E = Bott_E_Hall
ind1 = np.arange(6,15)
ind1 = np.hstack(((2,4),ind1))
E_view = E[ind1][::-1]
# Bott_E_Hall = np.reshape(Bott_E_Hall, (21,15))[::-1,:]
my_E_Hall = np.loadtxt("Hall_1980_Collision_eff.csv")        

R0 = np.array([300,200,150,100,70,60,50,40,30,25,20,15,10,8,6])[::-1]
R_view = R0[ind1][::-1]
R0_even = np.arange(0,302,2)

R_ratio = np.arange(0.0,1.05,0.05)

R0_ipol = [0,2,4]
E_ipol = [E[0],E[0],E[0]]


# interpolation to data with distance dR_ipol = 2
for i in range(len(R0)-1):
    R_col = R0[i]
    # print(R_col)
    if R_col%2 == 0:
        R_next = R_col + 2
        R0_ipol.append(R_col)
        E_ipol.append(E[i])
    else: R_next = R_col + 1
    dR2 = R0[i+1] - R_next
    if dR2 > 0: dR = R0[i+1] - R_col
    while (dR2 > 0):
        dR1 = R_next - R_col
        E_ipol.append((E[i+1]*dR1 + E[i]*dR2)/dR)
        R0_ipol.append(R_next)
        R_next += 2
        dR2 = R0[i+1] - R_next

R0_ipol.append(300)        
E_ipol.append(E[-1])

R0_ipol = np.array(R0_ipol)
E_ipol = np.array(E_ipol)


filename = "Hall_Bott_collision_efficiency.npy"
np.save(filename, E_ipol)
filename = "Hall_Bott_collector_radius.npy"
np.save(filename, R0_ipol)
filename = "Hall_Bott_radius_ratio.npy"
np.save(filename, R_ratio)

#%% TEST: PRINTING FOR EXACT RADII FROM TABLE
# for i1 in range(1,151):
#     diff_Eff = []    
#     strng = ""
#     for j1 in range(21):
#         R1 = Hall_Bott_R_col[i1]
#         R2 = Hall_Bott_R_col_ratio[j1] * R1        
#         dE = Hall_Bott_E_col[i1,j1]-compute_E_col_Hall_Bott(R1,R2)
#         diff_Eff.append(dE)
#         strng += f"{dE:.2} "
#         if abs(dE) > 1.0E-14 or math.isnan(dE):
#             print(i1,j1, R1, R2, Hall_Bott_R_col_ratio[j1], dE)
#     # print(i1,strng)
    