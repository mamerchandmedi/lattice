#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:09:20 2022

@author: Tomasz Krajewski
"""

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import rescaled quantities.
from lattice import r, N
# end: Import rescaled quantities.

laplacian_matrix = np.ndarray((N + 1, 3))
kinetic_matrix = np.ndarray((N + 1, ))
for j in range(1, N):
    laplacian_matrix[j][-1] = -(r[j]**3 - r[j-1]**3) / (r[j] - r[j-1])**2
    laplacian_matrix[j][0] = 0.
    laplacian_matrix[j][1] = (r[j+1]**3 - r[j]**3) / (r[j+1] - r[j])**2
    kinetic_matrix[j] = (r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1] * r[j-1] +
                         r[j] * r[j+1] + r[j] * r[j-1]) * (r[j+1] - r[j-1]) / 4

laplacian_matrix[1][-1] = 0.
laplacian_matrix[1][0] = 0.
laplacian_matrix[1][1] = (r[2]**3 - r[1]**3) / (r[2] - r[1])**2
kinetic_matrix[1] = r[1]**3 - r[0]**3 + \
    (3 * r[1]**2 + 2 * r[1] * r[2] + r[2]**2) * (r[2] - r[1]) / 4
