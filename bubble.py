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

# %% Import CosmoTransistions.
from cosmoTransitions.tunneling1D import SingleFieldInstanton
# end: Import CosmoTransistions.

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scaled import GeV
from matrices_low_order import r

# %% Importing the potetnial of the model.
from potential import V, dV
# end: Importing the potetnial of the model.

xi = np.array(range(2500000)) * 0.0001
Vmin = np.amin(V(xi))
vt = xi[np.where(V(xi) == Vmin)[0]][0]  # Find location of the global minimum.
profile = SingleFieldInstanton(vt, 0.0, V, dV).findProfile(npoints=1000)
fphi = interpolate.interp1d(profile.R, profile.Phi,
                            fill_value=(0), bounds_error=False)


def thin_wall(r, v0, L, r0):
    return v0/2*(-np.tanh((r-r0)/L) + 1.)


fit = curve_fit(thin_wall, profile.R, profile.Phi,
                bounds=(0, [1.5 * vt, 1, 1]))[0]
dt = 0.002
plt.plot(r / GeV, thin_wall(r / GeV, fit[0], fit[1], fit[2]),
         label="thin-wall approximation")
plt.scatter(profile.R, profile.Phi, c="red", s=1, label="Numerical profile")
plt.xlim(0, 0.2)
plt.legend()
plt.savefig('phi0')
plt.close()
phi_tw = []
for i in range(len(r)):
    #if r[i]/1.01 > profile.R[-1]:
    phi_tw.append(thin_wall(r[i] / 1.01 / GeV**(-1), fit[0], fit[1], fit[2]))
    #else:
    #phi_tw.append(fphi(r[i]/1.01))
phi_tw = np.array(phi_tw)
