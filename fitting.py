#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:30:47 2019

@author: jdesk
"""

### fitting

# fitting function of two variables
# rho_sol(w_s, T)
# fit for values between
# T = 0 .. 60°C
# w_s = 0 .. 0.22


def rho_NaCl_fit(X, *p):
    w,T = X
    return p[0] + p[1] * w + p[2] * T + p[3] * T*T + p[4] * w * T 

def rho_NaCl_fit2(X, *p):
    w,T = X
    return p[0] + p[1] * w + p[2] * T + p[3] * w*w + p[4] * w * T + p[5] * T*T

def lin_fit(x,a,b):
    return a + b*x

def quadratic_fit(x,a,b,c):
    return a + b*x + c*x*x

def quadratic_fit_shift(x,y0,a,x0):
    return y0 + a * (x - x0) * (x - x0)
# def quadratic_fit_shift_set(T,a):
#     return rho0h + a * (T - T0h) * (T - T0h)
# def quadratic_fit_shift_set2(T,a,rhomax):
#     return rhomax + a * (T - T0h) * (T - T0h)
def cubic_fit(x, a,b,c,d):
    return a + b*x + c*x**2 + d*x**3