#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:04:00 2019

@author: jdesk
"""
import math
import numpy as np
from numba import njit, vectorize
import matplotlib.pyplot as plt


import microphysics as mp
import constants as c
import atmosphere as atm    
#from atmosphere import compute_surface_tension_water,\
#                       compute_specific_heat_capacity_air_moist,\
#                       compute_diffusion_constant,\
#                       compute_thermal_conductivity_air

#%% TESTING MASS RATE

T = 298.    
T_amb = T
T_p = T

S_amb = 1.005
p_amb = 8E4
e_s_amb = atm.compute_saturation_pressure_vapor_liquid(T_amb)
L_v = atm.compute_heat_of_vaporization(T_amb)
K = atm.compute_thermal_conductivity_air(T_amb)
D_v = atm.compute_diffusion_constant(T_amb, p_amb)

#D_s = 6. # mu = 10 nm
D_s = 10. # mu = 10 nm
#D_s = 20. # mu = 10 nm
#D_s = 30. # mu = 10 nm
#D_s = 100. # mu = 10 nm

w_s = np.logspace(-5., np.log10(0.78), 10000)

D_s *= 1E-3
R_s = 0.5 * D_s

m_s_AS = mp.compute_mass_from_radius_jit(R_s, c.mass_density_AS_dry)
rho_AS = mp.compute_density_AS_solution(w_s, T_p) 
m_p_AS = m_s_AS/w_s
m_w_AS = m_p_AS - m_s_AS
R_p_AS = mp.compute_radius_from_mass_jit(m_p_AS, rho_AS)

m_s_SC = mp.compute_mass_from_radius_jit(R_s, c.mass_density_NaCl_dry)
rho_SC = mp.compute_density_NaCl_solution(w_s, T_p) 
m_p_SC = m_s_SC/w_s
m_w_SC = m_p_SC - m_s_SC
R_p_SC = mp.compute_radius_from_mass_jit(m_p_SC, rho_SC)

sigma_AS = mp.compute_surface_tension_AS(w_s, T_p)    
sigma_SC = atm.compute_surface_tension_water(T_p)  
  

S_eq_AS = mp.compute_equilibrium_saturation_AS(w_s, R_p_AS, T_p, rho_AS, sigma_AS)
S_eq_SC = mp.compute_equilibrium_saturation_NaCl(
              m_w_SC, m_s_SC, w_s, R_p_SC, T_p, rho_SC, sigma_SC)

gamma_AS = mp.compute_mass_rate_AS(w_s, R_p_AS, T_p, rho_AS,
                         T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_AS)

gamma_SC = mp.compute_mass_rate_NaCl(
               m_w_SC, m_s_SC, w_s, R_p_SC, T_p, rho_SC,
               T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_SC)
#gamma_NaCl = mp.compute_mass_rate_NaCl(m_w, m_s, w_s, R_p, T_p, rho_p,
#                      T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_NaCl)


#gamma_AS2, dgamma_dm_AS, dR_p_dm_over_R_p, dg1_dm, dSeq_dm = \
gamma_AS2, dgamma_dm_AS = \
    mp.compute_mass_rate_and_derivative_AS(m_w_AS, m_s_AS, w_s, R_p_AS, T_p,
                                           rho_AS,
                                           T_amb, p_amb, S_amb, e_s_amb,
                                           L_v, K, D_v, sigma_AS)

machine_epsilon_sqrt = 1.5E-8
def compute_mass_rate_derivative_AS_num(w_s, R_p, T_p, rho_p,
                         T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p):
    m_p = mp.compute_mass_from_radius_jit(R_p, rho_p)
    m_s = w_s * m_p
    
    h = m_p * machine_epsilon_sqrt
    
    m_p1 = m_p + h
    w_s1 = m_s / m_p1
    rho_p1 = mp.compute_density_AS_solution(w_s1, T_p)
    R_p1 = mp.compute_radius_from_mass_jit(m_p1, rho_p1)
    sigma_p1 = mp.compute_surface_tension_AS(w_s1, T)
    
    gamma = mp.compute_mass_rate_AS(w_s1, R_p1, T_p, rho_p1,
                         T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p1)
    m_p2 = m_p - h
    w_s2 = m_s / m_p2
    rho_p2 = mp.compute_density_AS_solution(w_s2, T_p)
    R_p2 = mp.compute_radius_from_mass_jit(m_p2, rho_p2)
    sigma_p2 = mp.compute_surface_tension_AS(w_s2, T)
    
    gamma -= mp.compute_mass_rate_AS(w_s2, R_p2, T_p, rho_p2,
                         T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p2)
    
    return gamma / (m_p1 - m_p2)
    
dgamma_dm_AS_num = compute_mass_rate_derivative_AS_num(
                    w_s, R_p_AS, T_p,
                    rho_AS, T_amb, p_amb, S_amb, e_s_amb,
                    L_v, K, D_v, sigma_AS)


###

dt_sub = 0.1
no_iter = 3

delta_ml_AS_lin, gamma_AS3 = \
    mp.compute_dml_and_gamma_impl_Newton_lin_AS(
            dt_sub, no_iter, m_w_AS, m_s_AS, w_s, R_p_AS, T_p, rho_AS,
            T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_AS)

delta_ml_AS, gamma_AS3 = \
    mp.compute_dml_and_gamma_impl_Newton_full_AS(
            dt_sub, no_iter, m_w_AS, m_s_AS, w_s, R_p_AS, T_p, rho_AS,
            T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_AS)

delta_ml_SC_lin, gamma_SC3 = \
    mp.compute_dml_and_gamma_impl_Newton_lin_NaCl(
            dt_sub, no_iter, m_w_SC, m_s_SC, w_s, R_p_SC, T_p, rho_SC,
            T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_SC)

delta_ml_SC, gamma_SC3 = \
    mp.compute_dml_and_gamma_impl_Newton_full_NaCl(
            dt_sub, no_iter, m_w_SC, m_s_SC, w_s, R_p_SC, T_p, rho_SC,
            T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_SC)

w_s_init_AS = \
mp.compute_initial_mass_fraction_solute_m_s_AS(m_s_AS, S_amb, T_amb)

w_s_init_SC = \
mp.compute_initial_mass_fraction_solute_m_s_NaCl(m_s_SC, S_amb, T_amb)


#%% PLOTTING

no_rows = 6
fig, axes = plt.subplots(no_rows, figsize=(10,6*no_rows))

ax = axes[0]
ax.plot(R_p_AS, S_eq_AS, label = "AS")
ax.plot(R_p_SC, S_eq_SC, label = "SC")
ax.axhline(S_amb)
ax.legend()

ax = axes[1]
ax.plot(m_p_AS, gamma_AS, label = "AS base")
ax.plot(m_p_AS, gamma_AS2, label = "AS fct 2")
ax.plot(m_p_SC, gamma_SC, label = "SC")
#ax.set_yscale("symlog")

ax2 = ax.twinx()
ax2.plot(m_p_AS, dgamma_dm_AS_num, label ="num")
ax2.set_yscale("symlog")
ax.legend()
ax.grid()
ax2.grid()

ax = axes[2]
ax.plot(m_p_AS, (gamma_AS2 - gamma_AS) / gamma_AS,
        label = "rel dev AS base and AS fct 2")
ax.plot(m_p_AS, (gamma_AS3 - gamma_AS) / gamma_AS,
        label = "rel dev AS base and AS fct 3")

ind_min = 0
#ind_min = 600
ax = axes[3]
ax.plot(m_p_AS[ind_min:], dgamma_dm_AS[ind_min:])
#ax.plot(R_p_AS, gamma_AS, label = "AS base")
ax.plot(m_p_AS[ind_min:], dgamma_dm_AS_num[ind_min:], label ="num")
ax.set_yscale("symlog")
ax.grid()

ind_min = 0
#ind_min = 600
ax = axes[4]
ax.plot(m_p_AS[ind_min:],
        (dgamma_dm_AS[ind_min:] - dgamma_dm_AS_num[ind_min:]) \
        / dgamma_dm_AS_num[ind_min:])
#ax.plot(R_p_AS, gamma_AS, label = "AS base")
#ax.plot(m_p_AS[ind_min:], , label ="num")
#ax.set_yscale("symlog")
ax.grid()

ax = axes[5]
ax.plot(m_p_AS, delta_ml_AS, label = "AS")
ax.plot(m_p_AS, delta_ml_AS_lin,label = "AS lin ")
ax.plot(m_p_SC, delta_ml_SC, label = "SC")
ax.plot(m_p_SC, delta_ml_SC_lin, label = "SC lin")
ax.grid()
ax.legend()

ax2 = ax.twinx()
ax2.plot(m_p_AS, gamma_AS3)
ax2.plot(m_p_SC, gamma_SC3)

for ax in axes[1:]:
    ax.set_xticks(np.arange(0,101,10))

#ind_max = 400
#ax = axes[4]
#ax.plot(m_p_AS[-ind_max:], dR_p_dm_over_R_p[-ind_max:])
#ax.plot(m_p_AS[-ind_max:], dg1_dm[-ind_max:])
#ax.plot(m_p_AS[-ind_max:], dSeq_dm[-ind_max:])
#ax.set_yscale("symlog")
#
#ind_max = 0
#ax = axes[5]
#ax.plot(m_p_AS[-ind_max:], dR_p_dm_over_R_p[-ind_max:])
#ax.plot(m_p_AS[-ind_max:], dg1_dm[-ind_max:])
#ax.plot(m_p_AS[-ind_max:], dSeq_dm[-ind_max:])
#ax.set_yscale("symlog")

#%%



