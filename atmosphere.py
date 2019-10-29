#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:33:07 2019

@author: jdesk
"""

import numpy as np
import constants as c
from numba import vectorize, njit

# J/(kg K)
def compute_specific_gas_constant_air_moist(specific_humidity_):
    return c.specific_gas_constant_air_dry * (1 + 0.608 * specific_humidity_ )

# J/(kg K)
@njit()
def compute_specific_heat_capacity_air_moist(mixing_ratio_vapor_):
    return c.specific_heat_capacity_air_dry_NTP * \
            ( 1 + 0.897 * mixing_ratio_vapor_ )

##############################################################################
# atmospheric environmental profile
kappa_air_dry = c.specific_gas_constant_air_dry\
                / c.specific_heat_capacity_air_dry_NTP

def compute_kappa_air_moist(mixing_ratio_vapor_):
    return kappa_air_dry * ( 1 - 0.289 * mixing_ratio_vapor_ )

epsilon_gc = c.specific_gas_constant_air_dry\
             / c.specific_gas_constant_water_vapor

epsilon_gc_prime = 1.0 / epsilon_gc - 1

def compute_beta_without_liquid(mixing_ratio_total_,
                                liquid_potential_temperature_):
    return c.earth_gravity * compute_kappa_air_moist(mixing_ratio_total_)\
           * (1 + mixing_ratio_total_) \
            / ( c.specific_gas_constant_air_dry * liquid_potential_temperature_
                * (1 + mixing_ratio_total_/epsilon_gc) )            

# general formula for any Theta_l, r_tot
# for z0 = 0 (surface)
def compute_T_over_Theta_l_without_liquid( z_, p_0_over_p_ref_to_kappa_tot_,
                                          beta_tot_ ):
    return p_0_over_p_ref_to_kappa_tot_ - beta_tot_ * z_

def compute_potential_temperature_moist( temperature_, pressure_,
                                        pressure_reference_,
                                        mixing_ratio_vapor_ ):
    return temperature_ \
            * ( pressure_reference_ / pressure_ )\
            **( compute_kappa_air_moist(mixing_ratio_vapor_) )

def compute_potential_temperature_dry( temperature_, pressure_,
                                      pressure_reference_ ):
    return temperature_ * ( pressure_reference_ / pressure_ )\
                        **( kappa_air_dry )

def compute_temperature_from_potential_temperature_moist(potential_temperature_, 
                                                         pressure_, 
                                                         pressure_reference_, 
                                                         mixing_ratio_vapor_ ):
    return potential_temperature_ * \
           ( pressure_ / pressure_reference_ )\
           **( compute_kappa_air_moist(mixing_ratio_vapor_) )
@njit()
def compute_temperature_from_potential_temperature_dry( potential_temperature_,
                                                       pressure_,
                                                       pressure_reference_ ):
    return potential_temperature_ * \
            ( pressure_ / pressure_reference_ )**( kappa_air_dry )

@vectorize("float64(float64, float64, float64)") 
def compute_pressure_ideal_gas( mass_density_, temperature_, 
                                specific_gas_constant_ ):
    return mass_density_ * temperature_ * specific_gas_constant_ 

@vectorize("float64(float64, float64)") 
def compute_pressure_vapor( density_vapor_, temperature_ ):
    return compute_pressure_ideal_gas( density_vapor_,
                                      temperature_,
                                      c.specific_gas_constant_water_vapor )
@vectorize("float64(float64, float64)") 
def compute_density_air_dry(temperature_, pressure_):
    return pressure_ / ( c.specific_gas_constant_air_dry * temperature_ )
@vectorize("float64(float64, float64)", target="parallel") 
def compute_density_air_dry_par(temperature_, pressure_):
    return pressure_ / ( c.specific_gas_constant_air_dry * temperature_ )


# IN WORK:
# thermal conductivity in air dependent on ambient temperature in Kelvin 
# empirical formula from Beard and Pruppacher 1971 (in Lohmann, p. 191)
# K_air in W/(m K)
@vectorize("float64(float64)")
def compute_thermal_conductivity_air(temperature_):
    return 4.1868E-3 * ( 5.69 + 0.017 * ( temperature_ - 273.15 ) )

# Formula from Pruppacher 1997
# m^2 / s
@vectorize("float64(float64, float64)")
def compute_diffusion_constant(ambient_temperature_ = 293.15,
                               ambient_pressure_ = 101325 ):
    return 4.01218E-5 * ambient_temperature_**1.94 / ambient_pressure_ # m^2/s
@vectorize("float64(float64, float64)", target="parallel") 
def compute_diffusion_constant_par(ambient_temperature_,
                               ambient_pressure_):
    return 4.01218E-5 * ambient_temperature_**1.94 / ambient_pressure_ # m^2/s

# dynamic viscosity "\mu" in Pa * s
## Fit to Data From Kadoya 1985,
#use linear approx of data in range 250 .. 350 K
#def compute_viscosity_air_approx(T_):
#    return 1.0E-6 * (18.56 + 0.0484 * (T_ - 300))
# ISO ISA 1975: "formula based on kinetic theory..."
# "with Sutherland's empirical coeff.
@vectorize("float64(float64)")
def compute_viscosity_air(T_):
    return 1.458E-6 * T_**(1.5) / (T_ + 110.4)
@vectorize("float64(float64)", target="parallel") 
def compute_viscosity_air_par(T_):
    return 1.458E-6 * T_**(1.5) / (T_ + 110.4)

# IN WORK: shift to microphysics??
#    surface tension in N/m = J/m^2
#    depends on T and not significantly on pressure (see Massoudi 1974)
#    formula from IAPWS 2014
#    note that the surface tension is in gen. dep. on 
#    the mass fraction of the solution (kg solute/ kg solution)
#    which is not considered!
@vectorize("float64(float64)") 
def compute_surface_tension_water(temperature_):
    tau = 1 - temperature_ / 647.096
    return 0.2358 * tau**(1.256) * (1 - 0.625 * tau)

# latent enthalpy of vaporazation in J/kg
# formula by Dake 1972
# (in Henderson Sellers 1984
# (HS has better formula but with division -> numerical slow) )
# formula valid for 0 °C to 35 °C
# At NTP: 2.452E6 # J/kg
@vectorize("float64(float64)")
def compute_heat_of_vaporization(temperature_):
    return 1.0E3 * ( 2500.82 - 2.358 * (temperature_ - 273.0) ) 

# IN WORK: e_vs = saturation pressure gas <->
# liquid -> approx with Clausius relation...
# Approximation by Rogers and Yau 1989 (in Lohmann p.50)
# returns pressure in Pa = N/m^2
# from XX ? in Lohmann 2016
@vectorize("float64(float64)")
def compute_saturation_pressure_vapor_liquid(temperature_):
    return 2.53E11 * np.exp( -5420.0 / temperature_ )
@vectorize("float64(float64)", target="parallel") 
def compute_saturation_pressure_vapor_liquid_par(temperature_):
    return 2.53E11 * np.exp( -5420.0 / temperature_ )

### conversion dry potential temperature
c_pv_over_c_pd = c.specific_heat_capacity_water_vapor_20C \
                 / c.specific_heat_capacity_air_dry_NTP
kappa_factor = 1.0 / (1.0 - kappa_air_dry)
kappa_factor2 = -kappa_air_dry * kappa_factor
@njit()
def compute_p_dry_over_p_ref(grid_mass_density_air_dry,
                             grid_potential_temperature,
                             p_ref_inv):
    return ( grid_mass_density_air_dry * grid_potential_temperature \
             * c.specific_gas_constant_air_dry * p_ref_inv )**kappa_factor
# def compute_p_dry_over_p_ref(grid):
#     return ( grid.mass_density_air_dry * grid.potential_temperature \
#              * c.specific_gas_constant_air_dry*grid.p_ref_inv )**kappa_factor

# Theta/T from Theta and rho_dry 
# NOTE THAT kappa factor 2 is negative
# grid.p_ref_inv needs to be right (default is p_ref = 1.0E5)
@njit()
def compute_Theta_over_T(grid_mass_density_air_dry, grid_potential_temperature,
                         p_ref_inv):
    return (p_ref_inv * c.specific_gas_constant_air_dry\
            * grid_mass_density_air_dry * grid_potential_temperature )\
            **kappa_factor2
# def compute_Theta_over_T(grid):
#     return (grid.p_ref_inv * c.specific_gas_constant_air_dry\
#             * grid.mass_density_air_dry * grid.potential_temperature )\
#             **kappa_factor2
#def compute_Theta_over_T(grid, p_ref):
#    return (p_ref / ( specific_gas_constant_air_dry\
#    * grid.mass_density_air_dry * grid.potential_temperature) )**kappa_factor2