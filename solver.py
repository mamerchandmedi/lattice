#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:37:31 2022

@author: Tomasz Krajewski
"""

# %% Importing procedure for solving nonlinear equations.
from scipy.optimize import fsolve
# end: Importing procedure for solving nonlinear equations.

# %% Import termodynamically corrected quantities.
from thermo import p, w
# end: Import termodynamically corrected quantities.


def solve_func(x, t, f, z):
    return t + p(f, x[0]) - 0.5 * (w(f, x[0])+(w(f, x[0])**2 + 4 * z**2)**0.5)


def tauZphi2Tpv(tau, Z, phi, T, P, v):
    sol = fsolve(lambda x: solve_func(x, tau, phi, Z), T)
    T = sol[0]
    P = p(phi, T)
    v = Z / (tau + P)
    return [T, P, v]
