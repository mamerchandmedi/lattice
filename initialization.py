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
from parameters import Tn
# end: Importing parameters of the model.


# %% Import matrices of the integration scheme.
from lattice import r, GeV
# end: Import matrices of the integration scheme.

try:
	# %% Import critical bubble calculated by CosmoTransitions.
	from bubble import vt, phi_tw
	# end: Import critical bubble calculated by CosmoTransitions.
	phi0 = phi_tw
except:
	#data = np.load("critical_bubble.npz")
	#phi0 = data["phi"]
	#phi0 = 0*np.ones(r.size)
	print("error")

phi = phi0
#N = len(phi)-1
T = Tn * np.ones(r.size)
v = 0. * np.ones(r.size)
try:
	# %% Import termodynamically corrected quantities.
	from thermo import p, w, gam
	# end: Import termodynamically corrected quantities.
	P = p(phi0, T)
	tau = w(phi0, T) * gam(v)**2 - P
	Z = w(phi0, T) * gam(v)**2 * v
	np.savez_compressed('critical_plasma', r=r, P=P, tau = tau, Z=Z)
except:
	#data = np.load("critical_plasma.npz")
	#P = data["P"]
	#Z = data["Z"]
	#tau = data["tau"]
	print("error")

P_off = 0 * np.ones(r.size)

if __name__=='__main__':
    print(T)
    print(Z)
    print(P)
