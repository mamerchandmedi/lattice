#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:26:59 2022

@author: Tomasz Krajewski
"""

# %% Importing parameters of the model.
from parameters import Tn, lam, m0, q
# end: Importing parameters of the model.
from thermo import VT, dVTdphi
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

vec_phi = np.array(range(100000)) * 0.001
vec_V = []
for i in range(len(vec_phi)):
	vec_V.append(V(vec_phi[i]))
plt.figure()
plt.plot(vec_phi, vec_V)
plt.savefig('tunneling_potential.png')
