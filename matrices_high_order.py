#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:06:46 2022

@author: Tomasz Krajewski
"""

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import rescaled quantities.
from lattice import r, N
# end: Import rescaled quantities.

advection_matrix = np.ndarray((N + 1, 3))
pressure_matrix = np.ndarray((N + 1, 3))
sources_matrix = np.ndarray((N + 1, 3))
mass_matrix = np.ndarray((N + 1, ))
for j in range(1, N):
    mass_matrix[j] = (r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1] * r[j-1] +
                      r[j+1] * r[j] + r[j] * r[j-1]) * (r[j+1] - r[j-1])
    advection_matrix[j][-1] = -(3 * r[j-1]**2 + 2 * r[j-1]*r[j] + r[j]**2)
    advection_matrix[j][0] = (3 * r[j]**2 + 2 * r[j] * r[j+1] + r[j+1]**2) - \
        (r[j-1]**2 + 2 * r[j-1] * r[j] + 3 * r[j]**2)
    advection_matrix[j][1] = (r[j]**2 + 2 * r[j]*r[j+1] + 3 * r[j+1]**2)
    pressure_matrix[j][-1] = advection_matrix[j][-1] - \
        2 * (r[j] - r[j-1]) * (r[j] + r[j-1])
    pressure_matrix[j][0] = advection_matrix[j][0] - \
        2 * ((r[j+1] - r[j]) * (3 * r[j] + r[j+1]) +
             (r[j] - r[j-1]) * (3 * r[j] + r[j-1]))
    pressure_matrix[j][1] = advection_matrix[j][1] - \
        2 * (r[j+1] - r[j]) * (r[j] + r[j+1])
    sources_matrix[j][-1] = - ((r[j] - r[j-1]) *
                               (r[j-1]**2 + 2 * r[j-1] * r[j] +
                                3 * r[j]**2))
    sources_matrix[j][0] = 0
    sources_matrix[j][1] = - ((r[j+1] - r[j]) *
                              (3 * r[j]**2 + 2 * r[j] * r[j+1] + r[j+1]**2))

mass_matrix[0] = (3 * r[0]**2 + 2 * r[0] * r[1] + r[1]**2) * (r[1] - r[0])
advection_matrix[0][-1] = 0.
advection_matrix[0][0] = (3 * r[0]**2 + 2 * r[0] * r[1] + r[1]**2)
advection_matrix[0][1] = (r[0]**2 + 2 * r[0] * r[1] + 3 * r[1]**2)
pressure_matrix[0][-1] = advection_matrix[0][-1] + 0.
pressure_matrix[0][0] = advection_matrix[0][0] - \
    2 * (r[1] - r[0]) * (3 * r[0] + r[1])
pressure_matrix[0][1] = advection_matrix[0][1] - \
    2 * (r[1] - r[0]) * (r[0] + r[1])
sources_matrix[0][0] = - (3 * r[0]**2 + 2 * r[0] * r[1] + r[1]**2) * \
                        (r[1] - r[0])

mass_matrix[N] = (r[N-1]**2 + 2 * r[N-1] * r[N] + 3 * r[N]**2) * \
    (r[N] - r[N-1])
advection_matrix[N][-1] = - (3 * r[N-1]**2 + 2 * r[N-1]*r[N] + r[N]**2)
advection_matrix[N][0] = - (r[N-1]**2 + 2 * r[N-1] * r[N] + 3 * r[N]**2) + \
    12 * r[N]**2
advection_matrix[N][1] = 0.
pressure_matrix[N][-1] = advection_matrix[N][-1] - \
    2 * (r[N] - r[N-1]) * (r[N] + r[N-1])
pressure_matrix[N][0] = advection_matrix[N][0] - \
    2 * (r[N] - r[N-1]) * (3 * r[N] + r[N-1])
pressure_matrix[N][1] = advection_matrix[N][1] + 0.
sources_matrix[N][0] = - (r[N-1]**2 + 2 * r[N-1] * r[N] + 3 * r[N]**2) * \
    (r[N] - r[N-1])
