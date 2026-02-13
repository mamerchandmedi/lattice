#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:23:02 2022

@author: Tomasz Krajewski
"""

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

import re

# %% Import interpolation procedure.
from scipy import interpolate
# end: Import interpolation procedure.

# %% Importing parameters of the model.
from parameters import Tn
# end: Importing parameters of the model.


# %% Import matrices of the integration scheme.
from lattice import r, GeV
# end: Import matrices of the integration scheme.

# try:
# 	# %% Import critical bubble calculated by CosmoTransitions.
# 	from bubble import vt, phi_tw
# 	# end: Import critical bubble calculated by CosmoTransitions.
# 	phi0 = phi_tw
# except:
# 	#data = np.load("critical_bubble.npz")
# 	#phi0 = data["phi"]
# 	#phi0 = 0*np.ones(r.size)
# 	print("error")
data = list()
with open('4Dbubble_a_30.txt') as file:
    match_data = None
    while not match_data:
        line = file.readline()
        match = re.search(r'(\s*)#define(\s+)profile_points(\s*)(\d+)', line)
        if match:
            profile_points = int(match.group(4))
        match = re.search(r'(\s*)#define(\s+)profile_step(\s*)([-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?)', line)
        if match:
            profile_step = float(match.group(4))
        match = re.search(r'(\s*)#define(\s+)profile_start(\s*)([-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?)', line)
        if match:
            profile_start = float(match.group(4))
        match_data = re.search(r'(\s*)/*(\s+)!data!', line)
    for i in range(profile_points):
        line = file.readline()
        data.append(float(line))
    data = np.asarray(data)

covered_r = np.arange(profile_start, profile_points * profile_step, profile_step)
interpolator = interpolate.InterpolatedUnivariateSpline(covered_r, data)

phi0 = np.full(r.size, data[-1])
for j in (i for i in range(r.size-1) if r[i+1] <= covered_r[-1]):
    phi0[j] = interpolator.integral(r[j], r[j+1]) / (r[j+1] - r[j])
    
phi = phi0
#N = len(phi)-1
T = Tn * np.ones(r.size)
v = 0. * np.ones(r.size)
#try:
	# %% Import termodynamically corrected quantities.
from thermo import p, w, gam
	# end: Import termodynamically corrected quantities.
P = p(phi, T)
tau = w(phi, T) * gam(v)**2 - P
Z = w(phi, T) * gam(v)**2 * v
np.savez_compressed('critical_plasma', r=r, P=P, tau = tau, Z=Z)
#except:
	#data = np.load("critical_plasma.npz")
	#P = data["P"]
	#Z = data["Z"]
	#tau = data["tau"]
print("error")
	
P_off = 0 * np.ones(r.size)
print(T)
print(Z)
print(P)

