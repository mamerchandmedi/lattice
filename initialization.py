#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:23:02 2022

@author: Tomasz Krajewski
"""

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Importing parameters of the model.
from parameters import Tc, L, sigma, lc, T0, alpha, gamma, lam, Tn
# end: Importing parameters of the model.

# %% Import critical bubble calculated by CosmoTransitions.
from bubble import vt, profile, fphi, phi_tw
# end: Import critical bubble calculated by CosmoTransitions.

# %% Import termodynamically corrected quantities.
from thermo import gst, a, VT, dVT, eps, p, w, gam
# end: Import termodynamically corrected quantities.

# %% Import rescaled quantities.
from scaled import GeV
# end: Import rescaled quantities.

# %% Import matrices of the integration scheme.
from matrices_low_order import dr, r,  cr1, cr2, cF0, cFm1, \
    cGUp1, cGU0, cGUm1, den
# end: Import matrices of the integration scheme.

phi0 = phi_tw
phi = phi0
N = len(phi)-1
T = Tn * np.ones(r.size)
v = 0. * np.ones(r.size)
P = p(phi0, T)
tau = w(phi0, T) * gam(v)**2 - P
#tau = tau[0]*np.exp(-2*(r-5)**2)

Z = w(phi0, T) * gam(v)**2 * v
P_off = (p(vt, Tn)) * np.ones(r.size)

phi *= GeV
T *= GeV
P *= GeV**4
tau *= GeV**4
Z *= GeV**4
P_off *= GeV**4
