#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import rescaled quantities.
from lattice import r, N, c_alpha, c_beta, c_gamma
# end: Import rescaled quantities.

laplacian_matrix = np.ndarray((N + 1, 3))
kinetic_matrix = np.ndarray((N + 1, ))
gradient_matrix = np.ndarray((N + 1, 3))
derivative_matrix_plus = np.ndarray((N + 1, ))
derivative_matrix_minus = np.ndarray((N + 1, ))
for j in range(1, N-1):
    laplacian_matrix[j][-1] = -3*(1-c_alpha)*(1-c_beta)*r[j]**4/((r[j+1]-r[j])*(3*r[j]**2+2*r[j]*r[j+1]+r[j+1]**2)) -3*(1+c_alpha)*(1+c_beta)*r[j]**4/((r[j]-r[j-1])*(r[j-1]**2+2*r[j-1]*r[j]+3*r[j]**2)) + 0.5*c_gamma*r[j]**2
    laplacian_matrix[j][0] = 0.
    laplacian_matrix[j][1] = 3*(1-c_alpha)*(1-c_beta)*r[j+1]**4/((r[j+2]-r[j+1])*(3*r[j+1]**2+2*r[j+1]*r[j+2]+r[j+2]**2)) +3*(1+c_alpha)*(1+c_beta)*r[j+1]**4/((r[j+1]-r[j])*(r[j]**2+2*r[j]*r[j+1]+3*r[j+1]**2)) - 0.5*c_gamma*r[j]**2
    derivative_matrix_minus[j] = 6*(1+c_alpha)*r[j]**2/(r[j]-r[j-1])/(r[j-1]**2 + 2*r[j-1]*r[j] +3*r[j]**2)
    derivative_matrix_plus[j] = 6*(1-c_alpha)*r[j]**2/(r[j+1]-r[j])/(3*r[j]**2 + 2*r[j]*r[j+1] + r[j+1]**2)
    gradient_matrix[j][-1] = r[j-1]**2 + 2 * r[j-1] * r[j] + 3 * r[j]**2 
    gradient_matrix[j][0] = 0.
    gradient_matrix[j][1] = (r[j+1]**2 + 2 * r[j+1] * r[j] + 3 * r[j]**2)
    kinetic_matrix[j] = (r[j+1]**3 - r[j]**3) / 3

laplacian_matrix[0][-1] = 0.
laplacian_matrix[0][0] = 0.
laplacian_matrix[0][1] = 3*(1-c_alpha)*(1-c_beta)*r[1]**4/((r[2]-r[1])*(3*r[1]**2+2*r[1]*r[2]+r[2]**2)) +3*(1+c_alpha)*(1+c_beta)*r[1]**4/((r[1]-r[0])*(r[0]**2+2*r[0]*r[1]+3*r[1]**2)) - 0.5*c_gamma*r[1]**2
derivative_matrix_plus[0] = 0.
derivative_matrix_minus[0] = 0.
gradient_matrix[1][-1] = 0.
gradient_matrix[1][0] = 0.
gradient_matrix[0][1] = 3 * r[1]**2 + 2 * r[1] * r[2] + r[2]**2
kinetic_matrix[0] = (r[1]**3 - r[0]**3) / 3

laplacian_matrix[N-1][-1] = -3*(1-c_alpha)*(1-c_beta)*r[N-1]**4/((r[N]-r[N-1])*(3*r[N-1]**2+2*r[N-1]*r[N]+r[N]**2)) -3*(1+c_alpha)*(1+c_beta)*r[N-1]**4/((r[N-1]-r[N-2])*(r[N-2]**2+2*r[N-2]*r[N-1]+3*r[N-1]**2)) + 0.5*c_gamma*r[N-1]**2
laplacian_matrix[N-1][0] = 0.
laplacian_matrix[N-1][1] = 0.
derivative_matrix_minus[N-1] = 6*(1+c_alpha)*r[N-1]**2/(r[N-1]-r[N-2])/(r[N-2]**2 + 2*r[N-2]*r[N-1] +3*r[N-1]**2)
derivative_matrix_plus[N-1] = 6*(1-c_alpha)*r[N-1]**2/(r[N]-r[N-1])/(3*r[N-1]**2 + 2*r[N-1]*r[N] + r[N]**2)
gradient_matrix[N-1][-1] = 0.
gradient_matrix[N-1][0] = 0.
gradient_matrix[N-1][1] = 0.
kinetic_matrix[N-1] = (r[N]**3 - r[N-1]**3) / 3

derivative_matrix_minus[N] = 0.
derivative_matrix_plus[N] = 0.
kinetic_matrix[N] = 0.

