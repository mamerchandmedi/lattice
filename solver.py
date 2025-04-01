#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:37:31 2022

@author: Tomasz Krajewski
"""

# %% Importing procedure for solving nonlinear equations.
from scipy.optimize import fsolve
from scipy.optimize import newton
# end: Importing procedure for solving nonlinear equations.

# %% Import termodynamically corrected quantities.
from thermo import w, p, dpdT, dwdT
# end: Import termodynamically corrected quantities.


def solve_func(x, t, f1, f2, z):
    return t + p(f1, f2, x) - 0.5 * (w(f1, f2, x)+(w(f1, f2, x)**2 + 4 * z**2)**0.5)


def solve_func_prime(x, t, f, z, v):
    return dpdT(f, x) - 0.5 * (dwdT(f, x) + 0.5 * (2 * w(f, x) * dwdT(f, x)) *
                               (w(f, x)**2 + 4 * z**2)**(-0.5))


def tauZphi2Tpv_old(tau, Z, f1, f2, T, P, v):
    sol = fsolve(lambda x: solve_func(x, tau, f1, f2, Z), T)
    T = sol[0]
    P = p(f1, f2, T)
    v = Z / (tau + P)
    return [T, P, v]


def tauZphi2Tpv_new(tau, Z, f1, f2, T, P, v):
    sol = newton(lambda x: solve_func(x, tau, f1, f2, Z), T,
                 fprime=lambda x: solve_func_prime(x, tau, f1, f2, Z, v), tol=10**(-10))
    T = sol
    P = p(f1, f2, T)
    v = Z / (tau + P)
    return [T, P, v]


def tauZphi2Tpv(tau, Z, f1, f2, T, P, v):
    try:
        return tauZphi2Tpv_new(tau, Z, f1, f2, T, P, v)
    except RuntimeError:
        print("Problem with newton solver detected. Trying fsolve.")
        return tauZphi2Tpv_old(tau, Z, f1, f2, T, P, v)
