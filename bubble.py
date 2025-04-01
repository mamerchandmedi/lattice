#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:33:07 2022

@author: Tomasz Krajewski
"""

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import procedure for interpolation.
from scipy import interpolate
# end: Import procedure for interpolation.


from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#from scaled import GeV
from lattice import r, GeV

# %% Importing the potetnial of the model.
from potential import phi, R, vlow, uhigh
# end: Importing the potetnial of the model.



def thin_wall_h(r, v0, L, r0):
    return v0/2*(-np.tanh((r-r0)/L) + 1.)
    
def thin_wall_s(r, v0, L, r0, s0,):
    return v0/2*(np.tanh((r-r0)/L) + s0)


fit1 = curve_fit(thin_wall_h, R, phi[:,0],
                bounds=(0, [1.5 * vlow, 1, 1]))[0]

fit2 = curve_fit(thin_wall_s, R, phi[:,1],
                bounds=(0, [uhigh, 10, 1,uhigh]))[0]
                
plt.figure()                
plt.plot(r, thin_wall_h(r, fit1[0], fit1[1], fit1[2]),
         label="$h(r)$", c = "deepskyblue")
plt.scatter(R, phi[:,0], c="dimgray", s=1, label="Numerical profile: $h(r)$")

plt.plot(r, thin_wall_s(r, fit2[0], fit2[1], fit2[2], fit2[3]),
         label="$s(r)$", c = "crimson")
plt.scatter(R, phi[:,1], c="dimgray", s=1, label="Numerical profile: $s(r)$")
plt.xlim(0, 1)
plt.legend()
plt.savefig('phi0.pdf')
plt.close()
phi_tw_h = []
phi_tw_s = []
for i in range(len(r)):
    phi_tw_h.append(thin_wall_h(r[i] / 1.01 / GeV**(-1), fit1[0], fit1[1], fit1[2]))
    phi_tw_s.append(thin_wall_s(r[i] / 1.01 / GeV**(-1), fit2[0], fit2[1], fit2[2], fit2[3]))
phi_tw_h = np.array(phi_tw_h)
phi_tw_s = np.array(phi_tw_s)
np.savez_compressed('critical_bubble', r=r, h=phi_tw_h, s=phi_tw_s)
