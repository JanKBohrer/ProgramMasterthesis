#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:43:19 2019

@author: jdesk
"""

# import numba
import numpy as np

# grid_scalar_fields[0] = grid.temperature
# grid_scalar_fields[1] = grid.pressure
# grid_scalar_fields[2] = grid.potential_temperature
# grid_scalar_fields[3] = grid.mass_density_air_dry
# grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
# grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
# grid_scalar_fields[6] = grid.saturation
# grid_scalar_fields[7] = grid.saturation_pressure
# grid_scalar_fields[8] = grid.mass_dry_inv
# grid_scalar_fields[9] = grid.rho_dry_inv

# grid_mat_prop[0] = grid.thermal_conductivity
# grid_mat_prop[1] = grid.diffusion_constant
# grid_mat_prop[2] = grid.heat_of_vaporization
# grid_mat_prop[3] = grid.surface_tension
# grid_mat_prop[4] = grid.specific_heat_capacity
# grid_mat_prop[5] = grid.viscosity
# grid_mat_prop[6] = grid.mass_density_fluid

index_dtype = np.dtype({"names" : ["T", "p", "Th", "rhod",
                                    "rv", "rl", "S", "es",
                                    "K", "Dv", "L", "sigmaw",
                                    "cpf", "muf", "rhof"],
                        "formats" : [np.int32, np.int32, np.int32, np.int32,
                                      np.int32, np.int32, np.int32, np.int32,
                                      np.int32, np.int32, np.int32, np.int32,
                                      np.int32, np.int32, np.int32] },
                        align=True)

ind = np.array( (0,1,2,3,4,5,6,7,0,1,2,3,4,5,6), index_dtype ) 
# for s in ["T", "p", "Th", "rhod", "rv", "rl", "S", "es",
#                                    "K", "Dv", "L", "sigmaw",
#                                    "cpf", "muf", "rhof"]:
#     print(s, ind[s])

#%%

# from collections import namedtuple

# Indices = namedtuple("Indices",
#             "T, p, Th, rhod, rv, rl, S, es, K, Dv, L, sigmaw, cpf, muf, rhof")

# ind = Indices(
# T = 0,
# p = 1,
# Th = 2,
# rhod = 3,
# rv = 4,
# rl = 5,
# S = 6,
# es = 7,
# K = 0,
# Dv = 1,
# L = 2,
# sigmaw = 3,
# cpf = 4,
# muf = 5,
# rhof = 6)

# print(ind)
# print(ind.T)
# print(ind.p)

# print(type(ind.T))



#%%
# import numba
# import numpy as np
# from grid import Grid
### grid_scalar_fields =
# 0 [T,
# 1 p,
# 2 Theta,
# 3 rho_dry,
# 4 r_v,
# 5 r_l,
# 6 S,
# 7 e_s]
# grid_scalar_fields = np.array( ( grid.temperature,
#                                  grid.pressure,
#                                  grid.potential_temperature,
#                                  grid.mass_density_air_dry,
#                                  grid.mixing_ratio_water_vapor,
#                                  grid.mixing_ratio_water_liquid,
#                                  grid.saturation,
#                                  grid.saturation_pressure) )
T = 0
# T = numba.i4(0)
p = 1
Th = 2
rhod = 3
rv = 4
rl = 5
S = 6
es = 7

# idx_T = 0

# print("T")
# print(T)
# print(type(T))

### grid_mat_prop =
# 0 [K,
# 1  Dv,
# 2  L,
# 3  sigmaw,
# 4  cpf,
# 5  muf,
# 6  rhof]
# grid_mat_prop = np.array( ( grid.thermal_conductivity,
#                             grid.diffusion_constant,
#                             grid.heat_of_vaporization,
#                             grid.surface_tension,
#                             grid.specific_heat_capacity,
#                             grid.viscosity,
#                             grid.mass_density_fluid ) )
K = 0
Dv = 1
L = 2
sigmaw = 3
cpf = 4
muf = 5
rhof = 6
