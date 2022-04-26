#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:49:33 2022

@author: Tomasz Krajewski
"""

# %% Import termodynamically corrected quantities.
from thermo import gst, a, VT, dVT, eps, p, w, gam
# end: Import termodynamically corrected quantities.

# %% Importing procedure for solving algebraic relation
#    T, p, v as function of tau, Z, phi.
from solver import tauZphi2Tpv
# end: Importing procedure for solving algebraic relation
#      T, p, v as function of tau, Z, phi.

# %% Defining rescaling factor.
GeV = 0.01  # Value of GeV i therms of simulation units.
# end: Defining rescaling factor.


# %% Defining the rescaled potential.
def VT_mod(field, temp):
    return VT(field / GeV, temp / GeV) * GeV**4
# end: Defining the rescaled potential.


# %% Defining rescaled derivative of the potential.
def dVT_mod(field, temp):
    return dVT(field / GeV, temp / GeV) * GeV**3
# end: Defining rescaled derivative of the potential.


def tauZphi2Tpv_mod(tau, Z, phi, T, P, v):
    T, P, v = tauZphi2Tpv(tau / GeV**4, Z / GeV**4, phi / GeV, T / GeV,
                          P / GeV**4, v)
    return [T * GeV, P * GeV**4, v]