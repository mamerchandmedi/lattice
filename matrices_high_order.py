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
from lattice import r, N, c_alpha
# end: Import rescaled quantities.

advection_matrix = np.ndarray((N + 1, 3))
pressure_matrix = np.ndarray((N + 1, 3))
sources_matrix = np.ndarray((N + 1, 3))
mass_matrix = np.ndarray((N + 1, ))
kinetic_matrix = np.ndarray((N + 1, ))
gradient_matrix_plus = np.ndarray((N + 1, ))
gradient_matrix_minus = np.ndarray((N + 1, ))
for j in range(1, N-1):
    mass_matrix[j] = (r[j+1]**3 - r[j]**3)/3
    advection_matrix[j][-1] = - 0.5 * r[j]**2 * (1 - c_alpha)
    advection_matrix[j][0] = 0.5 * r[j+1]**2 * (1 - c_alpha) - 0.5 * r[j]**2 * (1 + c_alpha)
    advection_matrix[j][1] = 0.5 * r[j+1]**2 * (1 + c_alpha)
    pressure_matrix[j][-1] = advection_matrix[j][-1] + 0.
    pressure_matrix[j][0] = advection_matrix[j][0] #- (r[j+1]**2 - r[j]**2)
    pressure_matrix[j][1] = advection_matrix[j][1] + 0.
    sources_matrix[j][-1] = 0.
    sources_matrix[j][0] = - (r[j+1]**3 - r[j]**3)/3
    sources_matrix[j][1] =  0.
    gradient_matrix_plus[j] = (r[j+1]**2 + 2 * r[j+1] * r[j] + 3 * r[j]**2) * (r[j+1] - r[j]) / 12
    gradient_matrix_minus[j] = (3 * r[j+1]**2 + 2 * r[j+1] * r[j] + r[j]**2) * (r[j+1] - r[j]) / 12

mass_matrix[0] = (r[1]**3 - r[0]**3)/3
advection_matrix[0][-1] = 0.
advection_matrix[0][0] = 0.5 * r[1]**2 * (1 - c_alpha) 
advection_matrix[0][1] = 0.5 * r[1]**2 * (1 - c_alpha) 
pressure_matrix[0][-1] = advection_matrix[0][-1] + 0.
pressure_matrix[0][0] = advection_matrix[0][0] #- (r[1]**2 - r[0]**2)
pressure_matrix[0][1] = advection_matrix[0][1] + 0.
sources_matrix[0][0] = - (r[1]**3 - r[0]**3)/3
gradient_matrix_plus[0] = 0.
gradient_matrix_minus[0] = (3 * r[1]**2 + 2 * r[1] * r[0] + r[0]**2) * (r[1] - r[0]) / 12

mass_matrix[N-1] = (r[N]**3 - r[N-1]**3)/3
advection_matrix[N-1][-1] =  - 0.5 * r[N-1]**2 * (1 - c_alpha) 
advection_matrix[N-1][0] = - 0.5 * r[N-1]**2 * (1 + c_alpha) + r[N]**2
advection_matrix[N-1][1] = 0.
pressure_matrix[N-1][-1] = advection_matrix[N-1][-1] + 0.
pressure_matrix[N-1][0] = advection_matrix[N-1][0] #- (r[N]**2 - r[N-1]**2)
pressure_matrix[N-1][1] = advection_matrix[N-1][1] + 0.
sources_matrix[N-1][0] = - (r[N]**3 - r[N-1]**3)/3
gradient_matrix_plus[N-1] = (r[N]**2 + 2 * r[N] * r[N-1] + 3 * r[N-1]**2) * (r[N] - r[N-1]) / 12
gradient_matrix_minus[N-1] = 0.

mass_matrix[N] = 0.
