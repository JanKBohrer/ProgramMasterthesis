#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:34:40 2019

@author: jdesk
"""

#import numpy.sqrt
import math.sqrt

from numba import njit

# VECTOR LENGTH
# vector must be a list or an array of more than one component
# the components may be scalars or np.arrays
@njit()
def vector_length(vector):
    r = 0.0
    for el in vector:
        r += el*el
    return math.sqrt(r)

# MAGNITUDE OF THE DEVIATION BETWEEN TWO VECTORS
# dev = sqrt( (v1 - v2)**2 )
# v1 and v2 may be lists or np.arrays of the same dimensions > 1
@njit()
def deviation_magnitude_between_vectors(v1,v2):
    dev = 0.0
    
    for i, comp1 in enumerate(v1):
        dev += (v2[i]-comp1)*(v2[i]-comp1)
    
    return math.sqrt(dev)