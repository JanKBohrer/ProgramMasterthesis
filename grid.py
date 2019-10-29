"""
created 29.04.19
copied everything from /python/grid_class.py
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from numba import njit, vectorize
# from numba import jit
# from help_functions import *
# from physical_relations_and_constants import *
import constants as c
from atmosphere import compute_heat_of_vaporization,\
                       compute_thermal_conductivity_air,\
                       compute_diffusion_constant,\
                       compute_specific_heat_capacity_air_moist,\
                       compute_viscosity_air,\
                       compute_surface_tension_water
from plotting import plot_scalar_field_2D
#from analysis import plot_scalar_field_2D

omega = 0.3
def u_rot_field(x_, y_):
    return -omega * y_
def v_rot_field(x_,y_):
    return omega * x_

# x, y in meter
# lapse rate = 6.5 K / km
T_ref = 288.15 # K
adiabatic_lapse_rate_dry = 0.0065 # K/m
def temperature_field_linear(x_, y_):
    return T_ref - adiabatic_lapse_rate_dry * y_

p_ref = 101325.0 # Pa
def pressure_field_exponential(x_, y_):
    return p_ref * np.exp( -y_ * c.earth_gravity * c.molar_mass_air_dry\
                           / ( T_ref * c.universal_gas_constant ) )
# @vectorize()
# def compute_cell(x, y, dx, dy):
#     i = int(math.floor(x/dx))
#     j = int(math.floor(y/dy))
#     return i, j

@njit()
def compute_cell_and_relative_position(pos, grid_ranges, grid_steps):
    x = pos[0]
    y = pos[1]
    cells = np.empty( (2,len(x)) , dtype = np.int64)
    rel_pos = np.empty( (2,len(x)) , dtype = np.float64 )
    # gridranges = arr [[x_min, x_max], [y_min, y_max]]
    rel_pos[0] = x - grid_ranges[0,0]
    rel_pos[1] = y - grid_ranges[1,0]
    cells[0] = np.floor(x/grid_steps[0]).astype(np.int64)
    cells[1] = np.floor(y/grid_steps[1]).astype(np.int64)
    
    rel_pos[0] = rel_pos[0] / grid_steps[0] - cells[0]
    rel_pos[1] = rel_pos[1] / grid_steps[1] - cells[1]
    return cells, rel_pos
    # cell_list = np.empty( (2,len(x)) )
    # rel_pos = np.empty( (2,len(x)) )
    # x = x - x_min # gridranges = arr [[x_min, x_max], [y_min, y_max]]
    # y = y - y_min
    # i = np.floor(x/dx).astype(np.int64)
    # j = np.floor(y/dy).astype(np.int64)
    
    # cell_list[0] = i
    # cell_list[1] = j
    # rel_pos[0] = x / dx - i
    # rel_pos[1] = y / dy - j
    # return i, j, x / dx - i, y / dy - j

@njit()
def weight_velocities_linear(i, j, a, b, u_n, v_n):
    return a * u_n[i + 1, j] + (1 - a) * u_n[i, j], \
           b * v_n[i, j + 1] + (1 - b) * v_n[i, j]
         
# f is a 2D scalar array, giving values for the grid cell [i,j]
# i.e. f[i,j] = scalar value of cell [i,j]
# the interpolation is given for the normalized position [a,b]
# in a cell with 4 corners 
# [i, j+1]    [i+1, j+1]
# [i, j]      [i+1, j]
# where a = 0..1,  b = 0..1   
@njit()
def bilinear_weight(i, j, a, b, f):
    return a * (b * f[i+1, j+1] + (1 - b) * f[i+1, j]) + \
            (1 - a) * (b * f[i, j+1] + (1 - b) * f[i, j])

# NOTE: this is adjusted for PBC in x and solid BC in z
# function was tested versus the grid.interpol... function            
@njit()
def interpolate_velocity_from_cell_bilinear(cells, rel_pos,
        grid_vel, grid_no_cells):
    no_pt = len(rel_pos[0])
    vel_ipol = np.empty( (2, no_pt), dtype = np.float64 )
    # vel_x = np.empty(len(x_rel), dtype = np.float64)
    # vel_y = np.empty(len(x_rel), dtype = np.float64)
    # vel_y = np.zeros_like(x_rel)
    # print("np.shape(vel_ipol)")
    # print(vel_ipol.shape)
    u, v = (0., 0.)
    for n in range( no_pt ):
        i = cells[0,n]
        j = cells[1,n]
        # i = i_list[n]
        # j = j_list[n]
        weight_x = rel_pos[0,n]
        weight_y = rel_pos[1,n]
        if j >= 0:
            if ( j == 0 and weight_y <= 0.5 ):
                u, v = weight_velocities_linear(i, j, weight_x, weight_y,
                                                grid_vel[0], grid_vel[1])
            elif ( j == (grid_no_cells[1] - 1) and weight_y >= 0.5 ):
                u, v = weight_velocities_linear(i, j, weight_x, weight_y,
                                                grid_vel[0], grid_vel[1])
            else:
                if weight_y > 0.5:
                    u = bilinear_weight(i, j,
                                        weight_x, weight_y - 0.5, grid_vel[0])
                else:
                    u = bilinear_weight(i, j - 1,
                                        weight_x, weight_y + 0.5, grid_vel[0])
            if weight_x > 0.5:
                v = bilinear_weight(i, j,
                                    weight_x - 0.5, weight_y, grid_vel[1])
            else:
                v = bilinear_weight(i - 1, j,
                                    weight_x + 0.5, weight_y, grid_vel[1])
        # vel_x[n] = u
        # vel_y[n] = v
        
        vel_ipol[0,n] = u
        vel_ipol[1,n] = v
    return vel_ipol
    # for n in range( len(x_rel) ):
    #     i = i_list[n]
    #     j = j_list[n]
    #     weight_x = x_rel[n]
    #     weight_y = y_rel[n]
    #     if ( j == 0 and weight_y <= 0.5 ):
    #         u, v = weight_velocities_linear(i, j, weight_x, weight_y,
    #                                         grid_vel[0], grid_vel[1])
    #     elif ( j == (grid_no_cells[1] - 1) and weight_y >= 0.5 ):
    #         u, v = weight_velocities_linear(i, j, weight_x, weight_y,
    #                                         grid_vel[0], grid_vel[1])
    #     else:
    #         if weight_y > 0.5:
    #             u = bilinear_weight(i, j,
    #                                 weight_x, weight_y - 0.5, grid_vel[0])
    #         else:
    #             u = bilinear_weight(i, j - 1,
    #                                 weight_x, weight_y + 0.5, grid_vel[0])
    #     if weight_x > 0.5:
    #         v = bilinear_weight(i, j,
    #                             weight_x - 0.5, weight_y, grid_vel[1])
    #     else:
    #         v = bilinear_weight(i - 1, j,
    #                             weight_x + 0.5, weight_y, grid_vel[1])
    #     # vel_x[n] = u
    #     # vel_y[n] = v
        
    #     vel_ipol[0,n] = u
    #     vel_ipol[1,n] = v
    # return vel_ipol
    # vel_x, vel_y

# function was tested versus the grid.interpol.. function
@njit()
def interpolate_velocity_from_position_bilinear(pos,
        grid_vel, grid_no_cells, grid_ranges, grid_steps):
    cells, rel_pos = compute_cell_and_relative_position(pos,grid_ranges,
                                                            grid_steps)
    # return cells, rel_pos
    return interpolate_velocity_from_cell_bilinear(cells, rel_pos,
                grid_vel, grid_no_cells)

def update_grid_r_l_np(m_w, xi, cells, grid_r_l, grid_mass_dry_inv, active_ids,
                       id_list):
#    no_sp = len(m_w)
    grid_r_l.fill(0.0)
    for ID in id_list[active_ids]:
        # cell = tuple(cells[0,ID], cells[1,ID])
        grid_r_l[cells[0,ID], cells[1,ID]] += m_w[ID] * xi[ID]
    grid_r_l *= 1.0E-18 * grid_mass_dry_inv
# from tests: the pure njit version (without parallel) is much faster
# than vanilla python (loop) and faster than njit parallel:
# for 112500 super particles: python, njit, njit_para
# 99.8 ms ± 775 µs per loop (mean ± std. dev. of 5 runs, 10 loops each)
# 233 µs ± 3.3 µs per loop (mean ± std. dev. of 5 runs, 1000 loops each)
# 339 µs ± 6.9 µs per loop (mean ± std. dev. of 5 runs, 1000 loops each)
# 98886.20279962197
# 229.0452829984133
# 329.4672499978333
# for 22500 particles: python, njit, njit_para
# 19.2 ms ± 91.7 µs per loop (mean ± std. dev. of 5 runs, 10 loops each)
# 45.4 µs ± 438 ns per loop (mean ± std. dev. of 5 runs, 1000 loops each)
# 80.9 µs ± 5.1 µs per loop (mean ± std. dev. of 5 runs, 1000 loops each)
# 19015.197699627606
# 44.7909360009362
# 73.54685899917968
update_grid_r_l = njit()(update_grid_r_l_np)
update_grid_r_l_par = njit(parallel = True)(update_grid_r_l_np)
        
def compute_no_grid_cells_from_step_sizes( gridranges_list_, stepsizes_list_ ):
    no_cells = []
    for i, range_i in enumerate(gridranges_list_):
        no_cells.append(
            int(np.ceil( (range_i[1] - range_i[0]) / stepsizes_list_[i] ) ) )
    return np.array(no_cells)

# grid_scalar_fields = np.array([T, p, Theta, rho_dry, r_v, r_l, S, e_s])
# grid_mat_prop = np.array([K, D_v, L, sigma_w, c_p_f, mu_f, rho_f])

class Grid:
    ranges = np.array( [ [-10.0, 10.0] , [-10.0,10.0] ] )
#     sizes = np.array( [ self.ranges[0,1] - self.ranges[0,0],
#                         self.ranges[1,1] - self.ranges[1,0] ]  )
    no_cells = [10, 10]
    sizes = np.array( [ ranges[0,1] - ranges[0,0], ranges[1,1] - ranges[1,0] ] )
    steps = [ sizes[0] / no_cells[0], sizes[1] / no_cells[1] ]
    
################################### NEW INIT

    def __init__(self,
                 grid_ranges_, # (m), as list [ [x_min, x_max], [z_min, z_max] ] 
                 grid_steps_, # in meter as list [dx, dz]
                 dy_, # in meter
                 u_field = u_rot_field, v_field = v_rot_field,
                 temperature_field_ = temperature_field_linear,
                 pressure_field_ = pressure_field_exponential): # m/s
        self.no_cells =\
            np.array( compute_no_grid_cells_from_step_sizes(grid_ranges_,
                                                            grid_steps_) )
        self.no_cells_tot = self.no_cells[0] * self.no_cells[1]
        self.steps = np.array( grid_steps_ )
        self.step_y = dy_
        self.volume_cell = grid_steps_[0] * grid_steps_[1] * dy_
        self.ranges = np.array( grid_ranges_ )
        self.ranges[:,1] = self.ranges[:,0] + self.steps * self.no_cells
        self.sizes = np.array( [ self.ranges[0,1] - self.ranges[0,0],
                                 self.ranges[1,1] - self.ranges[1,0] ]  )
        corners_x = np.linspace(0.0, self.sizes[0], self.no_cells[0] + 1)\
                    + self.ranges[0,0]
        corners_y = np.linspace(0.0, self.sizes[1], self.no_cells[1] + 1)\
                    + self.ranges[1,0]
        self.corners = np.array(
                           np.meshgrid(corners_x, corners_y, indexing = 'ij'))
        # get the grid centers (in 2D)
        self.centers = [self.corners[0][:-1,:-1] + 0.5 * self.steps[0],
                        self.corners[1][:-1,:-1] + 0.5 * self.steps[1]]
        self.pressure = np.zeros_like(self.centers[0])
        self.temperature = np.zeros_like(self.centers[0])
        self.potential_temperature = np.zeros_like(self.centers[0])
        self.mass_density_air_dry = np.zeros_like(self.centers[0])
        self.mixing_ratio_water_vapor = np.zeros_like(self.centers[0])
        self.mixing_ratio_water_liquid = np.zeros_like(self.centers[0])
        self.saturation_pressure = np.zeros_like(self.centers[0])
        self.saturation = np.zeros_like(self.centers[0])
        # for the normal velocities in u-direction, 
        # take the x-positions and shift the y-positions by half a y-step etc.
        pos_vel_u = [self.corners[0], self.corners[1] + 0.5 * self.steps[1]]
        pos_vel_w = [self.corners[0] + 0.5 * self.steps[0], self.corners[1]]
        # self.surface_centers[0] =
        # position where the normal velocity in x is projected onto the cell
        # self.surface_centers[1] =
        # position where of normal velocity in z is projected onto the cell
        self.surface_centers = [ pos_vel_u, pos_vel_w ]
        self.set_analytic_velocity_field_and_discretize(u_field, v_field)
        self.mass_flux_air_dry = np.zeros_like(self.velocity)
        # if the temperature field is given as discrete grid,
        # set default field first and change grid.pressure manually later
        self.set_analytic_temperature_field_and_discretize(temperature_field_)
        # if the pressure field is given as discrete grid,
        # set default field first and change grid.pressure manually later
        self.set_analytic_pressure_field_and_discretize(pressure_field_)
        
        ### new 18.02.19: material properties
        self.heat_of_vaporization = np.zeros_like(self.centers[0])
        self.thermal_conductivity = np.zeros_like(self.centers[0])
        self.diffusion_constant = np.zeros_like(self.centers[0])
        self.surface_tension = np.zeros_like(self.centers[0])
        self.specific_heat_capacity = np.zeros_like(self.centers[0])
        self.viscosity = np.zeros_like(self.centers[0])
        self.mass_density_fluid = np.zeros_like(self.centers[0])
        self.rho_dry_inv = np.zeros_like(self.centers[0])
        self.mass_dry_inv = np.zeros_like(self.centers[0])
        
        self.p_ref = 1.0E5
        self.p_ref_inv = 1.0E-5
        
######################## CONVERSIONS cell <-> location
#     For now, we have a rect. grid with constant gridsteps step_x, step_y
#     for all cells, i.e. the cell number can be calc. from a position (x,y)
    def compute_cell(self, x, y):
        # gridranges = arr [[x_min, x_max], [y_min, y_max]]
        x = x - self.ranges[0,0]
        y = y - self.ranges[1,0]

        return np.array(
            [math.floor(x/self.steps[0]) , math.floor(y/self.steps[1])])
    
    def compute_cell_and_relative_location(self, x, y):
        # gridranges = arr [[x_min, x_max], [y_min, y_max]]
        x = x - self.ranges[0,0] 
        y = y - self.ranges[1,0]
        i = np.floor(x/self.steps[0]).astype(int)
        j = np.floor(y/self.steps[1]).astype(int)

        return np.array( [i, j] ) , np.array( [ x / self.steps[0] - i,
                                                y / self.steps[1] - j] )
    
    # function to get the particle location from cell number and rel. loc.
    def compute_location(self, i, j, rloc_x, rloc_y):
        x = (i + rloc_x) * self.steps[0] + self.ranges[0][0]
        y = (j + rloc_y) * self.steps[1] + self.ranges[1][0]
        return np.array( [x, y] )
    
########################## VELOCITY INTERPOLATION
#     "Standard field"
    def analytic_velocity_field_u(self, x, y):
        omega = 0.3
        return -omega * y
    
    def analytic_velocity_field_v(self, x, y):
        omega = 0.3
        return omega * x
    
    def set_analytic_temperature_field_and_discretize(self, T_field_):
        self.analytic_temperature_field = T_field_
        self.temperature = T_field_( *self.centers )
        
    def set_analytic_pressure_field_and_discretize(self, p_field_):
        self.analytic_temperature_field = p_field_
        self.pressure = p_field_( *self.centers )
    
#     u_field and v_field have to be functions of (x,y)
    def set_analytic_velocity_field_and_discretize(self, u_field_, v_field_):
        self.analytic_velocity_field = [u_field_, v_field_]
        self.velocity =\
            np.array([self.analytic_velocity_field[0](*self.surface_centers[0]),
              self.analytic_velocity_field[1](*self.surface_centers[1])])
    
    def interpolate_velocity_from_location_linear(self, x, y):
        n, rloc = self.compute_cell_and_relative_location(x, y)
        u, v = weight_velocities_linear(*n, *rloc, *self.velocity)
        return u, v    
    
    def interpolate_velocity_from_cell_linear(self,i,j,rloc_x,rloc_y):
        return weight_velocities_linear(i, j, rloc_x, rloc_y, *self.velocity)

# NOTE: this is adjusted for PBC in x and solid BC in z
    def interpolate_velocity_from_cell_bilinear(self, i, j, weight_x, weight_y):
        if ( j == 0 and weight_y <= 0.5):
            u, v = self.interpolate_velocity_from_cell_linear(
                            i, j, weight_x, weight_y)
        elif ( j == (self.no_cells[1] - 1) and weight_y >= 0.5):
            u, v = self.interpolate_velocity_from_cell_linear(
                            i, j, weight_x, weight_y)
        else:
            if weight_y > 0.5:
                u = bilinear_weight(i, j,
                                    weight_x, weight_y - 0.5, self.velocity[0])
            else:
                u = bilinear_weight(i, j - 1,
                                    weight_x, weight_y + 0.5, self.velocity[0])
        if weight_x > 0.5:
            v = bilinear_weight(i, j,
                                weight_x - 0.5, weight_y, self.velocity[1])
        else:
            v = bilinear_weight(i - 1, j,
                                weight_x + 0.5, weight_y, self.velocity[1])
        return u, v
        
    def interpolate_velocity_from_location_bilinear(self, x, y):
        n, rloc = self.compute_cell_and_relative_location(x, y)
        return self.interpolate_velocity_from_cell_bilinear(*n, *rloc)

    # update 
    def update_material_properties(self):
        self.thermal_conductivity =\
            compute_thermal_conductivity_air(self.temperature)
        self.diffusion_constant =\
            compute_diffusion_constant(self.temperature, self.pressure)
        self.heat_of_vaporization =\
            compute_heat_of_vaporization(self.temperature)
        self.surface_tension = compute_surface_tension_water(self.temperature)        
        self.specific_heat_capacity = compute_specific_heat_capacity_air_moist(
                                          self.mixing_ratio_water_vapor)
        self.viscosity = compute_viscosity_air(self.temperature)
        self.mass_density_fluid = self.mass_density_air_dry\
                                  * (1 + self.mixing_ratio_water_vapor)
    
########################## PLOTTING
        
    def plot_thermodynamic_scalar_profiles_vertical_average(self):
        fields = [self.pressure, self.temperature, self.mass_density_air_dry,
                  self.saturation,
                  self.mixing_ratio_water_vapor, self.mixing_ratio_water_liquid]
                  
        field_names = ['pressure', 'temperature',
                       'mass_density_air_dry', 'saturation', 
                       'mixing_ratio_water_vapor', 'mixing_ratio_water_liquid']
        nfields = len(fields)
        ncols = 2
        nrows = int(np.ceil( nfields/ncols ))
        
        fields_avg = []
        
        for field in fields:
            fields_avg.append( field.mean(axis=0) )
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (10,5*nrows))
        
        n = 0
        for i in range(nrows):
            for j in range(ncols):
                field = fields_avg[n]
#                contours = ax[i,j].contour(grid_centers_x_, grid_centers_y_,
#                       field, no_contour_lines_, colors = 'black')
#                ax[i,j].clabel(contours, inline=True, fontsize=8)
#                CS = ax[i,j].contourf( grid_centers_x_, grid_centers_y_,
#                                 field,
#                                 levels = no_contour_colors_,
#                                 vmax = field.max(),
#                                 vmin = field.min(),
#                                cmap = plt.cm.coolwarm)
                ax[i,j].plot( field, self.centers[1][0] )
#                ax[i,j].set_xticks( np.linspace( tick_ranges_[0,0],
#                                                 tick_ranges_[0,1],
#                                                 no_ticks_[0] ) )
                ax[i,j].set_title( field_names[n] )
                # ax[i,j].set_yticks( np.linspace( tick_ranges_[1,0],
                #                                  tick_ranges_[1,1],
                #                                  no_ticks_[1] ) )
                # plt.colorbar(CS, fraction=colorbar_fraction_ ,
                #              pad=colorbar_pad_, ax=ax[i,j])
                ax[i,j].grid()
                n += 1
              
        fig.tight_layout()
    
#    def plot_thermodynamic_scalar_fields(self, no_ticks_ = [5,5],
#                         no_contour_colors_ = 10, no_contour_lines_ = 5,
#                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
#        fields = [self.pressure, self.temperature, self.mass_density_air_dry,
#                  self.saturation, self.mixing_ratio_water_vapor,
#                  self.mixing_ratio_water_liquid]
#                  
#        field_names = ['pressure', 'temperature', 'mass_density_air_dry',
#                       'saturation', 
#                       'mixing_ratio_water_vapor', 'mixing_ratio_water_liquid']
#        nfields = len(fields)
#        ncols = 2
#        nrows = int(np.ceil( nfields/ncols ))
#        
#        grid_centers_x_, grid_centers_y_ = self.centers[0], self.centers[1]
#        tick_ranges_ = self.ranges
#        
#        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (10,5*nrows))
#        n = 0
#        for i in range(nrows):
#            for j in range(ncols):
#                field = fields[n]
#                
#                contours = ax[i,j].contour(grid_centers_x_, grid_centers_y_,
#                       field, no_contour_lines_, colors = 'black')
#                ax[i,j].clabel(contours, inline=True, fontsize=8)
#                CS = ax[i,j].contourf( grid_centers_x_, grid_centers_y_,
#                                 field,
#                                 levels = no_contour_colors_,
#                                 vmax = field.max(),
#                                 vmin = field.min(),
#                                cmap = plt.cm.coolwarm)
#                ax[i,j].set_xticks( np.linspace( tick_ranges_[0,0],
#                                                 tick_ranges_[0,1],
#                                                 no_ticks_[0] ) )
#                ax[i,j].set_yticks( np.linspace( tick_ranges_[1,0],
#                                                 tick_ranges_[1,1],
#                                                 no_ticks_[1] ) )
#                ax[i,j].set_title( field_names[n] )
#                plt.colorbar(CS, fraction=colorbar_fraction_ ,
#                             pad=colorbar_pad_, ax=ax[i,j])
#                n += 1
#              
#        fig.tight_layout()

    def plot_thermodynamic_scalar_fields(self, no_ticks_ = [5,5],
                                              t = 0, fig_path = None):
#                         no_contour_colors_ = 10, no_contour_lines_ = 5,
#                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
        fields = [self.pressure * 0.01, self.temperature,
                  self.potential_temperature, self.mass_density_air_dry,
                  self.saturation, self.saturation_pressure * 0.01,
                  self.mixing_ratio_water_vapor*1000,
                  self.mixing_ratio_water_liquid*1000]
                  
        field_names = ['pressure', 'temperature', 'potential temperature',
                       'mass_density_air_dry',
                       'saturation', 'saturation pressure',
                       'mixing ratio water vapor', 'mixing ratio water liquid']
        unit_names = ['hPa', 'K', 'K', r'$\mathrm{kg/m^3}$', '-',
                      'hPa', 'g/kg', 'g/kg']
        nfields = len(fields)
        ncols = 2
        nrows = int(np.ceil( nfields/ncols ))
        
        # grid_centers_x_, grid_centers_y_ = self.centers[0], self.centers[1]
        tick_ranges_ = self.ranges
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (10,4*nrows))
        n = 0
        for i in range(nrows):
            for j in range(ncols):
                field = fields[n]
                if n == 7:
                    field_min = 0.001
                else:
                    field_min = field.min()
                field_max = field.max()
                if n in [0,1,2,3,5]:
                    cmap = "coolwarm"
                    alpha = None
                else:
                    cmap = "rainbow"
                    alpha = 0.7
                    
#                contours = ax[i,j].contour(grid_centers_x_, grid_centers_y_,
#                       field, no_contour_lines_, colors = 'black')
#                ax[i,j].clabel(contours, inline=True, fontsize=8)
                CS = ax[i,j].pcolorfast(*self.corners, field, cmap=cmap,
                                        alpha=alpha, edgecolor="face",
                                        vmin=field_min, vmax=field_max)
                CS.cmap.set_under("white")
                ax[i,j].set_title( field_names[n] + ' (' + unit_names[n] + ')' )
                ax[i,j].set_xticks( np.linspace( tick_ranges_[0,0],
                                                 tick_ranges_[0,1],
                                                 no_ticks_[0] ) )
                ax[i,j].set_yticks( np.linspace( tick_ranges_[1,0],
                                                 tick_ranges_[1,1],
                                                 no_ticks_[1] ) )
                if n == 7:
                    cbar = fig.colorbar(CS, ax=ax[i,j], extend = "min")
                else: cbar = fig.colorbar(CS, ax=ax[i,j])
                n += 1
              
        fig.tight_layout()
        if fig_path is not None:
            fig.savefig(fig_path + f"scalar_fields_grid_t_{int(t)}.png")
    
    def plot_scalar_field_2D(self, field_,
                         no_ticks_ = [5,5],
                         no_contour_colors_ = 10, no_contour_lines_ = 5,
                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
        
        tick_ranges_ = self.ranges
        plot_scalar_field_2D( *self.centers, field_,
                         tick_ranges_, no_ticks_,
                         no_contour_colors_, no_contour_lines_,
                         colorbar_fraction_, colorbar_pad_)
        
    
    
    # velocity = [ velocity_x[i,j], velocity_z[i,j] ] for 2D
    def plot_velocity_field_at_cell_surface(
            self, no_major_xticks=10, no_major_yticks=10, 
            no_arrows_u=10, no_arrows_v=10, ARROW_SCALE = 40.0,
            ARROW_WIDTH= 0.002, gridopt = 'minor'):
        # FILL IN START

        # say the x-grid ranges from x_min to x_max and x_range = x_max-x_min
        # now, you want labeled, major x-ticks, but not too many..
        # this number will be adjusted slightly to fit the geometry (see below)
        # the graphical grid will be plotted for all corners of the vel. grid
        # by using the minor ticks
#         no_major_xticks = 10
#         no_major_yticks = 10

        # also enter how many velocity arrows should be drawn
        # this number will be adjusted slightly to fit the geometry (see below)
#         no_arrows_u = 20 
#         no_arrows_v = 20

        # FILL IN END

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label cell the left corner
        # of cell 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label "-" of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            # take no_major_xticks - 1 to get the right spacing
            # in dimension of full cells widths
            tick_every_x = self.no_cells[0] // (no_major_xticks - 1)
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // (no_major_yticks - 1)
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1

        # velocity_grid_xticks = corners_x[::tick_every_x]
        # velocity_grid_yticks = corners_y[::tick_every_y]

        vel_pos_u = self.surface_centers[0]
        vel_pos_w = self.surface_centers[1]
        u_n = self.velocity[0]
        w_n = self.velocity[1]
        
        LW = 2.0
#         ARROW_SCALE = 40.0
        fig = plt.figure(figsize=(8,8), dpi = 81)
        ax = plt.gca()
        # ax.scatter(cell_corners[0], cell_corners[1], c = 'k',s = 5)
        # ax.scatter(vel_pos_v[0], vel_pos_v[1] , c = 'orange',s = 5)
        ax.quiver(vel_pos_u[0][::arrow_every_y,::arrow_every_x],
                  vel_pos_u[1][::arrow_every_y,::arrow_every_x],          
                  u_n[::arrow_every_y,::arrow_every_x],
                  np.zeros_like(u_n[::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE,
                  zorder = 3)
        ax.quiver(vel_pos_w[0][::arrow_every_y,::arrow_every_x],
                  vel_pos_w[1][::arrow_every_y,::arrow_every_x],
                  np.zeros_like(w_n[::arrow_every_y,::arrow_every_x]),
                  w_n[::arrow_every_y,::arrow_every_x], pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE,
                  zorder = 3)
        
        ax.set_xticks(self.corners[0][::tick_every_x,0])
        ax.set_yticks(self.corners[1][0,::tick_every_y])
        ax.set_xticks(self.corners[0][:,0], minor = True)
        ax.set_yticks(self.corners[1][0,:], minor = True)
        # ax.set_xticks(np.arange(-10,12,gridsteps[0]))
        # ax.set_yticks(np.arange(-10,12, gridsteps[1]))

        # np.arange(gridrange_x[0],gridrange_x[1]+gridsteps[0],gridsteps[0])

        # ax.minorticks_on()
        if gridopt == 'minor':
            ax.grid(which='minor', zorder=0)
        else:
            ax.grid(which='major', zorder=0)
        # ax.grid(which='minor', color="blue", linestyle='dashed', alpha=0.5)
        # ax.grid(which='major', color="c")
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')
#     trajectories must be a list or array of arrays:
#     trajectories[ [x1, y1  ] , [x2, y2 ] , [...] ]
        
#        ax.set_xticks(self.corners[0][0,::tick_every_x])
#        ax.set_yticks(self.corners[1][::tick_every_y,0])
#        ax.set_xticks(self.corners[0][0], minor = True)
#        ax.set_yticks(self.corners[1][:,0], minor = True)
        # ax.set_xticks(np.arange(-10,12,gridsteps[0]))
        # ax.set_yticks(np.arange(-10,12, gridsteps[1]))

        # np.arange(gridrange_x[0],gridrange_x[1]+gridsteps[0],gridsteps[0])

#        # ax.minorticks_on()
#        ax.grid(which='minor')
#        # ax.grid(which='minor', color="blue", linestyle='dashed', alpha=0.5)
#        # ax.grid(which='major', color="c")
#        ax.set_xlabel('horiz. pos. [m]')
#        ax.set_ylabel('vert. pos. [m]')
        
######################################################################

    def plot_velocity_field_centered(
            self, no_major_xticks=10, no_major_yticks=10, 
            no_arrows_u=10, no_arrows_v=10, ARROW_SCALE = 40.0,
            ARROW_WIDTH= 0.002, gridopt = 'minor'):
        
        centered_u_field = ( self.velocity[0][0:-1,0:-1]\
                             + self.velocity[0][1:,0:-1] ) * 0.5
        centered_w_field = ( self.velocity[1][0:-1,0:-1]\
                             + self.velocity[1][0:-1,1:] ) * 0.5
        
#        centered_velocity_field = [cent ]
        
        self.plot_external_field_list_output_centered(
                 [centered_u_field, centered_w_field],
                 no_major_xticks, no_major_yticks,
                 no_arrows_u, no_arrows_v, ARROW_SCALE, ARROW_WIDTH, gridopt)       

    def plot_mass_flux_field_centered(self, no_major_xticks=10,
                                      no_major_yticks=10, 
                                      no_arrows_u=10, no_arrows_v=10,
                                      ARROW_SCALE = 40.0, ARROW_WIDTH= 0.002,
                                      gridopt = 'minor'):
        
        centered_u_field = ( self.mass_flux_air_dry[0][0:-1,0:-1]\
                             + self.mass_flux_air_dry[0][1:,0:-1] ) * 0.5
        centered_w_field = ( self.mass_flux_air_dry[1][0:-1,0:-1]\
                             + self.mass_flux_air_dry[1][0:-1,1:] ) * 0.5
        
#        centered_velocity_field = [cent ]
        
        self.plot_external_field_list_output_centered( [centered_u_field,
                                                        centered_w_field],
                                                       no_major_xticks,
                                                       no_major_yticks,
                                                       no_arrows_u,
                                                       no_arrows_v,
                                                       ARROW_SCALE,
                                                       ARROW_WIDTH,
                                                       gridopt)     
          
        
#######################################################
    def plot_velocity_field_old(self, no_major_xticks=10, no_major_yticks=10, 
                            no_arrows_u=10, no_arrows_v=10, ARROW_SCALE = 40.0):
        # FILL IN START

        # say the x-grid ranges from x_min to x_max and x_range = x_max-x_min
        # now, you want labeled, major x-ticks, but not too many..
        # this number will be adjusted slightly to fit the geometry (see below)
        # the graphical grid will be plotted for all corners of the vel. grid
        # by using the minor ticks
#         no_major_xticks = 10
#         no_major_yticks = 10

        # also enter how many velocity arrows should be drawn
        # this number will be adjusted slightly to fit the geometry (see below)
#         no_arrows_u = 20 
#         no_arrows_v = 20

        # FILL IN END

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label cell the left corner
        # of cell 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label "-" of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            tick_every_x = self.no_cells[0] // no_major_xticks
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // no_major_yticks
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // no_arrows_u
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // no_arrows_v
        else:
            arrow_every_y = 1

        # velocity_grid_xticks = corners_x[::tick_every_x]
        # velocity_grid_yticks = corners_y[::tick_every_y]

        LW = 2.0
#         ARROW_SCALE = 40.0
        fig = plt.figure(figsize=(8,8), dpi = 92)
        ax = plt.gca()
        # ax.scatter(cell_corners[0], cell_corners[1], c = 'k',s = 5)
        # ax.scatter(vel_pos_v[0], vel_pos_v[1] , c = 'orange',s = 5)
        ax.quiver(self.vel_pos_u[0][::arrow_every_y,::arrow_every_x],
                  self.vel_pos_u[1][::arrow_every_y,::arrow_every_x],          
                  self.u_n[::arrow_every_y,::arrow_every_x],
                  np.zeros_like(self.u_n[::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = 0.002, scale = ARROW_SCALE, zorder = 3 )
        ax.quiver(self.vel_pos_v[0][::arrow_every_y,::arrow_every_x],
                  self.vel_pos_v[1][::arrow_every_y,::arrow_every_x],
                  np.zeros_like(self.v_n[::arrow_every_y,::arrow_every_x]),
                  self.v_n[::arrow_every_y,::arrow_every_x], pivot = 'mid',
                  width = 0.002, scale = ARROW_SCALE, zorder = 3 )
        ax.set_xticks(self.corners[0][0,::tick_every_x])
        ax.set_yticks(self.corners[1][::tick_every_y,0])
        ax.set_xticks(self.corners[0][0], minor = True)
        ax.set_yticks(self.corners[1][:,0], minor = True)
        # ax.set_xticks(np.arange(-10,12,gridsteps[0]))
        # ax.set_yticks(np.arange(-10,12, gridsteps[1]))

        # np.arange(gridrange_x[0],gridrange_x[1]+gridsteps[0],gridsteps[0])

        # ax.minorticks_on()
        ax.grid(which='minor', zorder=0)
        # ax.grid(which='minor', color="blue", linestyle='dashed', alpha=0.5)
        # ax.grid(which='major', color="c")
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')



###########################################################

#    field f returns f_x, f_y
    def plot_external_field_function_list_output(self, f,
                                     no_major_xticks=10, no_major_yticks=10, 
                                     no_arrows_u=10, no_arrows_v=10,
                                     ARROW_SCALE=40, ARROW_WIDTH=0.002,
                                     gridopt = 'minor'):

        # FILL IN START

        # say the x-grid ranges from x_min to x_max and x_range = x_max-x_min
        # now, you want labeled, major x-ticks, but not too many..
        # this number will be adjusted slightly to fit the geometry (see below)
        # the graphical grid will be plotted for all corners of the vel. grid
        # by using the minor ticks
#         no_major_xticks = 10
#         no_major_yticks = 10

        # also enter how many velocity arrows should be drawn
        # this number will be adjusted slightly to fit the geometry (see below)
#         no_arrows_u = 20 
#         no_arrows_v = 20

        # FILL IN END

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label cell the left corner
        # of cell 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label "-" of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            # take no_major_xticks - 1 to get the right spacing
            # in dimension of full cells widths
            tick_every_x = self.no_cells[0] // (no_major_xticks - 1)
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // (no_major_yticks - 1)
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1

        # velocity_grid_xticks = corners_x[::tick_every_x]
        # velocity_grid_yticks = corners_y[::tick_every_y]
#        width = ARROW_WIDTH
        LW = 2.0
#        ARROW_SCALE = 40.0
        fig = plt.figure(figsize=(8,8), dpi = 92)
        ax = plt.gca()
        # ax.scatter(cell_corners[0], cell_corners[1], c = 'k',s = 5)
        # ax.scatter(vel_pos_v[0], vel_pos_v[1] , c = 'orange',s = 5)
        ax.quiver(
            self.corners[0][::arrow_every_y,::arrow_every_x],
            self.corners[1][::arrow_every_y,::arrow_every_x],
            *f(self.corners[0][::arrow_every_y,::arrow_every_x],
               self.corners[1][::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )
        ax.set_xticks(self.corners[0][::tick_every_x,0])
        ax.set_yticks(self.corners[1][0,::tick_every_y])
        ax.set_xticks(self.corners[0][:,0], minor = True)
        ax.set_yticks(self.corners[1][0,:], minor = True)

        # np.arange(gridrange_x[0],gridrange_x[1]+gridsteps[0],gridsteps[0])

        # ax.minorticks_on()
        if gridopt == 'minor':
            ax.grid(which='minor', zorder=0)
        else:
            ax.grid(which='major', zorder=0)
        # ax.grid(which='minor', color="blue", linestyle='dashed', alpha=0.5)
        # ax.grid(which='major', color="c")
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')
#     trajectories must be a list or array of arrays:
#     trajectories[ [x1, y1  ] , [x2, y2 ] , [...] ]
        

###############################################################        

#    field f returns ARRAYS f_x[i,j], f_y[i,j]
    def plot_external_field_list_output_centered(self, f,
                                     no_major_xticks=10, no_major_yticks=10, 
                                     no_arrows_u=10, no_arrows_v=10,
                                     ARROW_SCALE=40, ARROW_WIDTH=0.002,
                                     gridopt = 'minor'):

        # FILL IN START

        # say the x-grid ranges from x_min to x_max and x_range = x_max-x_min
        # now, you want labeled, major x-ticks, but not too many..
        # this number will be adjusted slightly to fit the geometry (see below)
        # the graphical grid will be plotted for all corners of the vel. grid
        # by using the minor ticks
#         no_major_xticks = 10
#         no_major_yticks = 10

        # also enter how many velocity arrows should be drawn
        # this number will be adjusted slightly to fit the geometry (see below)
#         no_arrows_u = 20 
#         no_arrows_v = 20

        # FILL IN END

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label cell the left corner
        # of cell 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label "-" of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            # take no_major_xticks - 1 to get the right spacing
            # in dimension of full cells widths
            tick_every_x = self.no_cells[0] // (no_major_xticks - 1)
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // (no_major_yticks - 1)
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1

        # velocity_grid_xticks = corners_x[::tick_every_x]
        # velocity_grid_yticks = corners_y[::tick_every_y]
#        width = ARROW_WIDTH
        LW = 2.0
#        ARROW_SCALE = 40.0
        fig = plt.figure(figsize=(8,8), dpi = 92)
        ax = plt.gca()
        # ax.scatter(cell_corners[0], cell_corners[1], c = 'k',s = 5)
        # ax.scatter(vel_pos_v[0], vel_pos_v[1] , c = 'orange',s = 5)
        ax.quiver(
            self.centers[0][::arrow_every_y,::arrow_every_x],
            self.centers[1][::arrow_every_y,::arrow_every_x],
            f[0][::arrow_every_y,::arrow_every_x],
            f[1][::arrow_every_y,::arrow_every_x],
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )
#         ax.quiver(self.vel_pos_v[0][::arrow_every_y,::arrow_every_x],
#                   self.vel_pos_v[1][::arrow_every_y,::arrow_every_x],
#                   np.zeros_like(self.v_n[::arrow_every_y,::arrow_every_x]),
#                   self.v_n[::arrow_every_y,::arrow_every_x], pivot = 'mid',
#                   width = 0.002, scale = ARROW_SCALE )
        ax.set_xticks(self.corners[0][::tick_every_x,0])
        ax.set_yticks(self.corners[1][0,::tick_every_y])
        ax.set_xticks(self.corners[0][:,0], minor = True)
        ax.set_yticks(self.corners[1][0,:], minor = True)
        ax.set_xlim(self.corners[0][0,0], self.corners[0][-1,0])
        ax.set_ylim(self.corners[1][0,0], self.corners[1][0,-1])
        # ax.set_xticks(np.arange(-10,12,gridsteps[0]))
        # ax.set_yticks(np.arange(-10,12, gridsteps[1]))

        # np.arange(gridrange_x[0],gridrange_x[1]+gridsteps[0],gridsteps[0])

        # ax.minorticks_on()
        if gridopt == 'minor':
            ax.grid(which='minor', zorder=0)
        else:
            ax.grid(which='major', zorder=0)
        # ax.grid(which='minor', color="blue", linestyle='dashed', alpha=0.5)
        # ax.grid(which='major', color="c")
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')
#     trajectories must be a list or array of arrays:
#     trajectories[ [x1, y1  ] , [x2, y2 ] , [...] ]
        
########################################################################        

    # the components of the external field f = (f_x, f_y) 
    # must be defined python functions
    # of the spatial variables (x,y): f_x(x,y), f_y (x,y)
    def plot_external_field_function(self,f_x, f_y, 
                                     no_major_xticks=10, no_major_yticks=10, 
                                     no_arrows_u=10, no_arrows_v=10,
                                     ARROW_SCALE=40, ARROW_WIDTH=0.002,
                                     gridopt = 'minor'):

        # FILL IN START

        # say the x-grid ranges from x_min to x_max and x_range = x_max-x_min
        # now, you want labeled, major x-ticks, but not too many..
        # this number will be adjusted slightly to fit the geometry (see below)
        # the graphical grid will be plotted for all corners of the vel. grid
        # by using the minor ticks
#         no_major_xticks = 10
#         no_major_yticks = 10

        # also enter how many velocity arrows should be drawn
        # this number will be adjusted slightly to fit the geometry (see below)
#         no_arrows_u = 20 
#         no_arrows_v = 20

        # FILL IN END

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label cell the left corner
        # of cell 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label "-" of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            # take no_major_xticks - 1 to get the right spacing
            # in dimension of full cells widths
            tick_every_x = self.no_cells[0] // (no_major_xticks - 1)
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // (no_major_yticks - 1)
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1

        # velocity_grid_xticks = corners_x[::tick_every_x]
        # velocity_grid_yticks = corners_y[::tick_every_y]
#        width = ARROW_WIDTH
        LW = 2.0
#        ARROW_SCALE = 40.0
        fig = plt.figure(figsize=(8,8), dpi = 92)
        ax = plt.gca()
        # ax.scatter(cell_corners[0], cell_corners[1], c = 'k',s = 5)
        # ax.scatter(vel_pos_v[0], vel_pos_v[1] , c = 'orange',s = 5)
        ax.quiver(
            self.corners[0][::arrow_every_y,::arrow_every_x],
            self.corners[1][::arrow_every_y,::arrow_every_x],
            f_x(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
            f_y(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )
#         ax.quiver(self.vel_pos_v[0][::arrow_every_y,::arrow_every_x],
#                   self.vel_pos_v[1][::arrow_every_y,::arrow_every_x],
#                   np.zeros_like(self.v_n[::arrow_every_y,::arrow_every_x]),
#                   self.v_n[::arrow_every_y,::arrow_every_x], pivot = 'mid',
#                   width = 0.002, scale = ARROW_SCALE )
        ax.set_xticks(self.corners[0][::tick_every_x,0])
        ax.set_yticks(self.corners[1][0,::tick_every_y])
        ax.set_xticks(self.corners[0][:,0], minor = True)
        ax.set_yticks(self.corners[1][0,:], minor = True)
        # ax.set_xticks(np.arange(-10,12,gridsteps[0]))
        # ax.set_yticks(np.arange(-10,12, gridsteps[1]))

        # np.arange(gridrange_x[0],gridrange_x[1]+gridsteps[0],gridsteps[0])

        # ax.minorticks_on()
        if gridopt == 'minor':
            ax.grid(which='minor', zorder=0)
        else:
            ax.grid(which='major', zorder=0)
        # ax.grid(which='minor', color="blue", linestyle='dashed', alpha=0.5)
        # ax.grid(which='major', color="c")
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')
#     trajectories must be a list or array of arrays:
#     trajectories[ [x1, y1  ] , [x2, y2 ] , [...] ]



    def plot_field_and_trajectories(self,f_x, f_y, trajectories,
                                     no_major_xticks=10, no_major_yticks=10, 
                                     no_arrows_u=10, no_arrows_v=10, 
                                    fig_n = 1, LW = 0.9, ARROW_SCALE = 80.0):
        
        if no_major_xticks < self.no_cells[0]:
            tick_every_x = self.no_cells[0] // no_major_xticks
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // no_major_yticks
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // no_arrows_u
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // no_arrows_v
        else:
            arrow_every_y = 1

        # velocity_grid_xticks = corners_x[::tick_every_x]
        # velocity_grid_yticks = corners_y[::tick_every_y]

#         LW = 2.0
        # ARROW_SCALE = 80.0
        if (self.sizes[0] == self.sizes[1]):
            figsize_x = 8
            figsize_y = 8
        else:
            size_ratio = self.sizes[1] / self.sizes[0]
            figsize_x = 8
            figsize_y = figsize_x * size_ratio
            #else:
            #    figsize_y = 8
            #    figsize_x = figsize_y / size_ratio
            print(figsize_x, figsize_y)
        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi = 92)
#        fig = plt.figure(fig_n, figsize=(figsize_x, figsize_y), dpi = 92)
        ax = plt.gca()
        # ax.scatter(cell_corners[0], cell_corners[1], c = 'k',s = 5)
        # ax.scatter(vel_pos_v[0], vel_pos_v[1] , c = 'orange',s = 5)
        ax.quiver(
            self.corners[0][::arrow_every_y,::arrow_every_x],
            self.corners[1][::arrow_every_y,::arrow_every_x],
            f_x(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
            f_y(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = 0.002, scale = ARROW_SCALE )
#         ax.quiver(self.vel_pos_v[0][::arrow_every_y,::arrow_every_x],
#                   self.vel_pos_v[1][::arrow_every_y,::arrow_every_x],
#                   np.zeros_like(self.v_n[::arrow_every_y,::arrow_every_x]),
#                   self.v_n[::arrow_every_y,::arrow_every_x], pivot = 'mid',
#                   width = 0.002, scale = ARROW_SCALE )
        for data_x_y in trajectories:
            ax.plot(data_x_y[0], data_x_y[1], linewidth = LW, linestyle='--')
        ax.set_xticks(self.corners[0][0,::tick_every_x])
        ax.set_yticks(self.corners[1][::tick_every_y,0])
        ax.set_xticks(self.corners[0][0], minor = True)
        ax.set_yticks(self.corners[1][:,0], minor = True)
        # ax.set_xticks(np.arange(-10,12,gridsteps[0]))
        # ax.set_yticks(np.arange(-10,12, gridsteps[1]))

        # np.arange(gridrange_x[0],gridrange_x[1]+gridsteps[0],gridsteps[0])

        # ax.minorticks_on()
        ax.grid(which='minor', zorder=0)
        # ax.grid(which='minor', color="blue", linestyle='dashed', alpha=0.5)
        # ax.grid(which='major', color="c")
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')

    def plot_field_and_trajectories_with_particle_size(self,
            f_x, f_y, trajectories, particle_sizes,
            no_major_xticks=10, no_major_yticks=10, 
            no_arrows_u=10, no_arrows_v=10, circle_every = 100,
            fig_n = 1, LW = 0.9, ARROW_SCALE = 80.0,
            fig_title = '', fig_name = 'trajectory.pdf'):
        
        if no_major_xticks < self.no_cells[0]:
            tick_every_x = self.no_cells[0] // no_major_xticks
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // no_major_yticks
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // no_arrows_u
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // no_arrows_v
        else:
            arrow_every_y = 1

        # velocity_grid_xticks = corners_x[::tick_every_x]
        # velocity_grid_yticks = corners_y[::tick_every_y]

#         LW = 2.0
        # ARROW_SCALE = 80.0
        if (self.sizes[0] == self.sizes[1]):
            figsize_x = 8
            figsize_y = 8
        else:
            size_ratio = self.sizes[1] / self.sizes[0]
            figsize_x = 8
            figsize_y = figsize_x * size_ratio
            #else:
            #    figsize_y = 8
            #    figsize_x = figsize_y / size_ratio
            print(figsize_x, figsize_y)
#        fig = plt.figure(fig_n, figsize=(figsize_x, figsize_y), dpi = 92)
        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi = 92)
        ax = plt.gca() 
        # ax.scatter(cell_corners[0], cell_corners[1], c = 'k',s = 5)
        # ax.scatter(vel_pos_v[0], vel_pos_v[1] , c = 'orange',s = 5)
        ax.quiver(
            self.corners[0][::arrow_every_y,::arrow_every_x],
            self.corners[1][::arrow_every_y,::arrow_every_x],
            f_x(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
            f_y(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = 0.002, scale = ARROW_SCALE )
#         ax.quiver(self.vel_pos_v[0][::arrow_every_y,::arrow_every_x],
#                   self.vel_pos_v[1][::arrow_every_y,::arrow_every_x],
#                   np.zeros_like(self.v_n[::arrow_every_y,::arrow_every_x]),
#                   self.v_n[::arrow_every_y,::arrow_every_x], pivot = 'mid',
#                   width = 0.002, scale = ARROW_SCALE )
        for cnt, data_x_y in enumerate(trajectories):
            if cnt == 0:
                cmap = 'Blues'
            elif cnt == 1:
                cmap = 'Oranges'
            elif cnt == 2:
                cmap = 'Greens'
            elif cnt == 3:
                cmap = 'Reds'
            elif cnt == 4:
                cmap = 'Purples'
            elif cnt == 5:
                cmap = 'YlOrBr'
            elif cnt == 6:
                cmap = 'RdPu'
            else:
                cmap = 'Greys'
            ax.plot(data_x_y[0], data_x_y[1], linewidth = LW, linestyle='--')
            ax.scatter(data_x_y[0][::circle_every], data_x_y[1][::circle_every],
                       linewidth = LW,
                       s = particle_sizes[cnt][::circle_every]**2,
                       marker = 'o', alpha = 0.5,
                       c = particle_sizes[cnt][::circle_every], cmap = 'cool')
            ax.scatter(data_x_y[0][::circle_every], data_x_y[1][::circle_every],
                       linewidth = LW,
                       s = particle_sizes[cnt][::circle_every]**2, marker = 'o', 
                       edgecolors = 'k',
                       facecolors = 'none' )
        ax.set_xticks(self.corners[0][0,::tick_every_x])
        ax.set_yticks(self.corners[1][::tick_every_y,0])
        ax.set_xticks(self.corners[0][0], minor = True)
        ax.set_yticks(self.corners[1][:,0], minor = True)
        # ax.set_xticks(np.arange(-10,12,gridsteps[0]))
        # ax.set_yticks(np.arange(-10,12, gridsteps[1]))

        # np.arange(gridrange_x[0],gridrange_x[1]+gridsteps[0],gridsteps[0])

        # ax.minorticks_on()
        ax.grid(which='minor')
        # ax.grid(which='minor', color="blue", linestyle='dashed', alpha=0.5)
        # ax.grid(which='major', color="c")
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')
        ax.set_title(fig_title, fontsize = 9)
        fig.savefig(fig_name)
    
    def print_info(self):
        print('')
        print('grid information:')
        print('grid ranges [x_min, x_max] [z_min, z_max]:')
        print(self.ranges)
        print('number of cells:', self.no_cells)
        print('grid steps:', self.steps)
