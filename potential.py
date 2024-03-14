#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:26:59 2022

@author: Tomasz Krajewski
"""

# %% Importing parameters of the model.
from parameters import Tn, lam, v, y, etaV
# end: Importing parameters of the model.
from thermo import VT, dVTdphi, dVTdT
import matplotlib.pyplot as plt
import numpy as np


# %% Defining potential of the model.
def V(phi):
    return VT(phi, Tn)
# end: Defining potential of the model.


# %% Defining derivative of the potential of the model.
def dV(phi):
    return dVTdphi(phi, Tn)
# end: Defining derivative of the potential of the model.

def dVdT(phi):
    return dVTdT(phi, Tn)
    
vec_phi = np.array(range(14000)) * 0.01
vec_V = []
vec_dV = []
vec_w = []
for i in range(len(vec_phi)):
	vec_V.append(V(vec_phi[i]))
	vec_dV.append(dV(vec_phi[i]))
	vec_w.append(-Tn*dVdT(vec_phi[i]))
plt.figure()
plt.plot(vec_phi, vec_V)
plt.savefig('tunneling_potential.png')

plt.figure()
plt.plot(vec_phi, vec_dV)
plt.savefig('tunneling_potential_derivative.png')

plt.figure()
plt.plot(vec_phi, vec_w)
plt.savefig('tunneling_enthalpy.png')
