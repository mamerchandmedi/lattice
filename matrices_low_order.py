#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:58:23 2022

@author: Tomasz Krajewski
"""

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import rescaled quantities.
from scaled import GeV
# end: Import rescaled quantities.

rmin = 0. * GeV**(-1)
rmax = 10. * GeV**(-1)
dr = 0.01 * GeV**(-1)

N = int(np.floor((rmax - rmin) / dr))

r = np.array(range(0, N + 1)) * dr

cr1 = np.empty_like(r)
cr2 = np.empty_like(r)

for j in range(N):
    cr1[j] = 2 * (r[j+1]**3 - r[j]**3) * \
        (r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1]*r[j-1] +
         2*r[j]*r[j+1])**(-1) / dr**3
    cr2[j] = 2 * (r[j]**3 - r[j-1]**3) * \
        (r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1]*r[j-1] +
         2*r[j]*r[j+1])**(-1)/dr**3

cr1[1] = (r[2]**3-r[1]**3) * \
    (r[1]**3 - r[0]**3 + (r[2] - r[1]) / 4 * (3 * r[1]**2 + 2 * r[1] * r[2] +
                                              r[2]**2))**(-1) / dr**2
cr2[1] = 0.

cF0 = np.empty_like(r)
cFm1 = np.empty_like(r)
for j in range(1, N):
    cF0[j] = 4 * (r[j+1]**2 + r[j+1] * r[j] + r[j]**2)
    cFm1[j] = -4 * (r[j]**2 + r[j] * r[j-1] + r[j-1]**2)

cF0[0] = 4 * (r[0]**2 + r[0] * r[1] + r[1]**2)
cF0[N] = 12*r[N]**2
cFm1[N] = -4 * (r[N-1]**2 + r[N-1] * r[N] + r[N]**2)

cGUp1 = np.empty_like(r)
cGU0 = np.empty_like(r)
cGUm1 = np.empty_like(r)
den = np.empty_like(r)
for j in range(N):
    cGU0[j] = 2 * dr * (r[j + 1] + 6 * r[j] + r[j-1]) - \
        ((3 * r[j]**2 + 2 * r[j] * r[j+1] + r[j+1]**2) -
         (r[j-1]**2 + 2 * r[j-1] * r[j] + 3 * r[j]**2))
    cGUp1[j] = 2 * dr * (r[j+1] + r[j]) - (r[j]**2 + 2*r[j]*r[j+1] +
                                           3 * r[j+1]**2)
    cGUm1[j] = 2 * dr * (r[j] + r[j-1]) + (3 * r[j-1]**2 + 2 * r[j-1]*r[j] +
                                           r[j]**2)
    den[j] = (r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1]*r[j-1] +
              r[j+1]*r[j] + r[j]*r[j-1])

advection_matrix = np.ndarray((N + 1, 2))
sources_matrix = np.ndarray((N + 1, 3))
mass_matrix = np.ndarray((N + 1, ))
laplacian_matrix = np.ndarray((N + 1, 3))
kinetic_matrix = np.ndarray((N + 1, ))
for j in range(1, N):
    advection_matrix[j][-1] = -4 * (r[j]**2 + r[j] * r[j-1] + r[j-1]**2)
    advection_matrix[j][0] = 4 * (r[j+1]**2 + r[j+1] * r[j] + r[j]**2)
    mass_matrix[j] = (r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1] * r[j-1] +
                      r[j+1] * r[j] + r[j] * r[j-1]) * (r[j+1] - r[j-1])
    sources_matrix[j][-1] = -2 * (r[j] - r[j-1]) * (r[j] + r[j-1])
    sources_matrix[j][0] = -2 * ((r[j+1] - r[j]) * (3 * r[j] + r[j+1]) +
                                 (r[j] - r[j-1]) * (3 * r[j] + r[j-1]))
    sources_matrix[j][1] = -2 * (r[j+1] - r[j]) * (r[j] + r[j+1])
    laplacian_matrix[j][-1] = -(r[j]**3 - r[j-1]**3) / (r[j] - r[j-1])**2
    laplacian_matrix[j][0] = 0.
    laplacian_matrix[j][1] = (r[j+1]**3 - r[j]**3) / (r[j+1] - r[j])**2
    kinetic_matrix[j] = (r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1] * r[j-1] +
                         r[j] * r[j+1] + r[j] * r[j-1]) * (r[j+1] - r[j-1]) / 4

advection_matrix[0][-1] = 0.
advection_matrix[0][0] = 4 * (r[0]**2 + r[0] * r[1] + r[1]**2)
mass_matrix[0] = (3 * r[0]**2 + 2 * r[0] * r[1] + r[1]**2) * (r[1] - r[0])
sources_matrix[0][-1] = 0.
sources_matrix[0][0] = -2 * (r[1] - r[0]) * (3 * r[0] + r[1])
sources_matrix[0][1] = -2 * (r[1] - r[0]) * (r[0] + r[1])
laplacian_matrix[1][-1] = 0.
laplacian_matrix[1][0] = 0.
laplacian_matrix[1][1] = (r[2]**3 - r[1]**3) / (r[2] - r[1])**2
kinetic_matrix[1] = r[1]**3 - r[0]**3 + \
    (3 * r[1]**2 + 2 * r[1] * r[2] + r[2]**2) * (r[2] - r[1]) / 4

advection_matrix[N][-1] = -4 * (r[N]**2 + r[N] * r[N-1] + r[N-1]**2)
advection_matrix[N][0] = 12 * r[N]**2
mass_matrix[N] = (r[N-1]**2 + 2 * r[N-1] * r[N] + 3 * r[N]**2) * \
    (r[N] - r[N-1])
sources_matrix[N][-1] = -2 * (r[N] - r[N-1]) * (r[N] + r[N-1])
sources_matrix[N][0] = -2 * (r[N] - r[N-1]) * (3 * r[N] + r[N-1])
sources_matrix[N][1] = 0.
