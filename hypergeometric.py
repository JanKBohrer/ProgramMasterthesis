#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:21:02 2019

@author: jdesk
"""

import numpy as np
#from mpmath import *
import mpmath as mpm
mpm.mp.dps = 25; mpm.mp.pretty = True
A = mpm.hyp1f1(2, (-1,3), 3.25)

print(A)

print(mpm.hyp1f1(3, 4, 1E20))

B = np.ones(100, dtype = np.float128)


print(B)

for n in range(100):
    B[n] = mpm.exp(10*n)
    print(mpm.exp(20*(-n)))    
    print(mpm.exp(20*(-n))*mpm.exp(20*(n)))    
    
print(B)

C = np.arange(10)

#print(mpm.exp(C))

D = mpm.besseli()