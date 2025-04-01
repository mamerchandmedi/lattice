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
from potential import Tn
# end: Importing parameters of the model.

# %% Import termodynamically corrected quantities.
from thermo import p, w, gam
# end: Import termodynamically corrected quantities.

# %% Import rescaled quantities.
#from scaled import GeV
# end: Import rescaled quantities.

# %% Import matrices of the integration scheme.
from lattice import r
# end: Import matrices of the integration scheme.

try:
	# %% Import critical bubble calculated by CosmoTransitions.
	from bubble import vlow, uhigh, phi_tw_h, phi_tw_s
	# end: Import critical bubble calculated by CosmoTransitions.
	h0 = phi_tw_h
	s0 = phi_tw_s
except:
	profiles = np.load("critical_bubble.npz")
	h0 = profiles["h"]
	s0 = profiles["s"]

h = h0
s = s0
#N = len(phi)-1
T = Tn * np.ones(r.size)
v = 0. * np.ones(r.size)
P = p(h0, s0, T)
tau = w(h0, s0, T) * gam(v)**2 - P

Z = w(h0, s0, T) * gam(v)**2 * v
P_off = 0 * np.ones(r.size)

print("Initialization: ", h[0], s[0], P[0], tau[0], Z[0], v[0])
#phi *= GeV
#T *= GeV
#P *= GeV**4
#tau *= GeV**4
#Z *= GeV**4
#P_off *= GeV**4
