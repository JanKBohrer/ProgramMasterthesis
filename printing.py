#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:33:56 2019

@author: jdesk
"""

# function to get the particle position from cell number and weights
def compute_position_from_weights(i,j,weight_x,weight_y, grid_steps, grid_ranges):
    x = (i + weight_x) * grid_steps[0] + grid_ranges[0][0]
    y = (j + weight_y) * grid_steps[1] + grid_ranges[1][0]
    return x, y

# print to file
# note that there must be an open file handle present
# e.g. position_file = open("position_vs_time.txt","w")
# which must be closed at some time later
def print_position_from_weights_vs_time_to_file(t, i, j, weight_x, weight_y,
                                                gridsteps, gridranges,
                                                file_handle):
    pos_x, pos_y = compute_position_from_weights(i, j, weight_x, weight_y,
                                                 gridsteps, gridranges)
    
    string = f'{t:.6} {pos_x:.6} {pos_y:.6}\n'
    file_handle.write(string)

def print_position_vs_time_to_file(t, x, y, file_handle):
    
    string = f'{t:.6} {x:.6} {y:.6}\n'
    file_handle.write(string)

def print_data_vs_time_to_file(t, x, y, u, v, file_handle):
    
    string = f'{t:.6} {x:.6} {y:.6} {u:.6} {v:.6}\n'
    file_handle.write(string)