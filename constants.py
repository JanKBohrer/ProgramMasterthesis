#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:23:12 2019

@author: jdesk
"""

import math

# pyhsical universal constants

# Boltzmann
k_B = 1.380649E-13 # J/K
# Avogadro
avogadro = 6.02214076E23 # 1/mol
# universal gas constant
univ_gas = avogadro * k_B # J/(K mol)

# COMPUTE MASS FROM DIAMETER AND DENSITY
# 1/3
one_third = 1.0 / 3.0
# 4/3 * pi
pi_times_4_over_3 = 4.0 * math.pi / 3.0
four_pi_over_three = 4.0 * math.pi / 3.0
# 3 / (4 * pi)
pi_times_4_over_3_inv = 0.75 / math.pi

# volume to radius:
# R = (3/4/pi)^0.3333 V^1/3
volume_to_radius = (pi_times_4_over_3_inv)**(one_third)

# Constants
# Avogadro constant
# NIST: https://physics.nist.gov/cgi-bin/cuu/Value?na
avogadro_constant = 6.0221409E23  # 1/mol
           
# NTP == normal temperature and pressure by NIST
# T = 20 °C, p = 1.01325 x 10^5 Pa
            
# standard gravity m/s^2
# NIST special publication 330 2008 Ed
# Ed.: Taylor, Thompson, National Institute of Standards and Technology 
# Gaithersburg, MD  20899 
# Therin declaration of 3rd CGPM 1901
# ALSO ISO ISA 1975! as normal gravity
# note that Grabowski defines gravity as 9.72 m/s^2
# in vocals v3 (test case 1 ICMW 2012 fortran CODE)
# m/s^2 Taylor, Barry N. (March 2008).
# NIST special publication 330, 2008 edition:
earth_gravity = 9.80665 
a_gravity = -9.80665 # m/s^2

### Gas constants
# NIST
# https://physics.nist.gov/cgi-bin/cuu/Value?r
# note that CRC deviates significantly relative to NIST(?)
# R_CRC = 8.314472
# So does the standard atmosphere ISO 1975:
# ISO standard atmosphere 1975
# universal_gas_constant = 8.31432 # J/(mol K)
# NIST (website s.a.):
universal_gas_constant = 8.3144598 # J/(mol K)
# R_v = R*/M_v using the NIST values...
# the exact values are
# NIST
# CRC
# ISO
#461.5298251457119
#461.5305023591452
#461.52206494587847
#287.0578986618055
#287.05831986852826
#287.0530720470647
specific_gas_constant_water_vapor = 461.53 # J/(kg K)
specific_gas_constant_air_dry = 287.06 # J/(kg K)

### Densities and molar masses
mass_density_water_liquid_NTP = 998.2 # kg/m^3
mass_density_air_dry_NTP = 1.2041 # kg/m^3
# CRC 2005:
mass_density_NaCl_dry = 2163.0 # kg/m^3
# this is what wiki says for CRC 2011:
#mass_density_NaCl_dry = 2170.0 # kg/m^3
# CRC 2005:
mass_density_AS_dry = 1774.0 # kg/m^3
# US Standard Atmosphere 1976, US Government. Printing
# Office, Washington DC, pp. 3 and 33, 1976. 
# page 9, below table 8, this is the mass at see level...
# ALSO:
# ISO 1975 Int. Standard Atmosphere
molar_mass_air_dry = 28.9644E-3 # kg/mol
# CRC 2005
molar_mass_water = 18.015E-3 # kg/mol
# CRC 2005
molar_mass_NaCl = 58.4428E-3 # kg/mol
# https://pubchem.ncbi.nlm.nih.gov/compound/Ammonium-sulfate
molar_mass_AS = 132.14

### Heat capacities
# molar_heat_capacity_dry_air_NTP = 20.8 # J/(mol K)
# from engineering toolbox at 300 K
# isobaric spec. heat. cap. dry air 
# from Tables of Thermal Properties of Gases", NBS Circular 564,1955
# in https://www.ohio.edu/mechanical/thermo/property_tables/air/air_cp_cv.html
# ratio c_p/c_v = 1.4 at 300 K
#specific_heat_capacity_dry_air_NTP = 1005 # J/(kg K)
### NEW
# isochoric heat capacity of ideal gas = C_V = DOF/2 * R*,
# DOF of the gas molecules
# NOTE that the heat capacity does not vary with T and p using an ideal gas!
# assume 2-atomic molecules (N_2, O_2)
# C_V = 5/2 * R* (R* of NIST)
isochoric_molar_heat_capacity_air_dry_NTP = 20.7861 # J/(mol K)
# C_p = 7/2 * R* (R* of NIST)
molar_heat_capacity_air_dry_NTP = 29.1006 # J/(mol K)
# c_p = C_p / M_dry
specific_heat_capacity_air_dry_NTP = 1004.71 # J/(kg K)
# c_p / c_v = 7/5
adiabatic_index_air_dry = 1.4

# Lemmon 2015 in Lohmann 2016
specific_heat_capacity_water_vapor_20C = 1906 # J/(kg K)

# isobaric heat capacity water
# of data from Sabbah 1999, converted from molar with molar_mass_water
# NOTE that the graph describes a "parabolic" curve with minimum at 308 K
# and varies ca. 1 % from 0 to 100 °C
# a linear fit is not satisfactory, it lowers the error to 0.5%
# HOWEVER comparing data sources (CRC and Sabbah)
# leads to deviations of about 0.5 % anyways...
# THUS:
# just take NPT value from Sabbah 1999 for comparibility
# the average value from 0 .. 60 C from Sabbah
# shifts the error to be from -0.2 % to 0.8 %
specific_heat_capacity_water = 4187.9 # J/(kg K)
specific_heat_capacity_water_NTP = 4183.8 # J/(kg K)
# CRC:
#specific_heat_capacity_water_NTP = 4181.8 # J/(kg K)
                
####################################
# FORCES

# drag coefficient
drag_coefficient_high_Re_p = 0.44

### Material properties

# COMPUTE MASS FROM DIAMETER AND DENSITY
# 1/3
one_third = 1.0 / 3.0
# 4/3 * pi
pi_times_4_over_3 = 4.0 * math.pi / 3.0
# 3 / (4 * pi)
pi_times_4_over_3_inv = 0.75 / math.pi

# volume to radius:
# R = (3/4/pi)^0.3333 V^1/3
const_volume_to_radius = (pi_times_4_over_3_inv)**(one_third)