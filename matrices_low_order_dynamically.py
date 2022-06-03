# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import rescaled quantities.
from lattice import N
# end: Import rescaled quantities.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import advection_matrix as high_order_advection_matrix
# end: Import matrices of the high order integration scheme.


def advection(velocity):
    KH = np.ndarray((N + 1, 3))
    KH[0][0] = high_order_advection_matrix[0][0] * velocity[0]
    KH[0][1] = high_order_advection_matrix[0][1] * velocity[1]
    for j in range(1, N):
        KH[j][-1] = high_order_advection_matrix[j][-1] * velocity[j-1]
        KH[j][0] = high_order_advection_matrix[j][0] * velocity[j]
        KH[j][1] = high_order_advection_matrix[j][1] * velocity[j+1]
    KH[N][-1] = high_order_advection_matrix[N][-1] * velocity[N-1]
    KH[N][0] = high_order_advection_matrix[N][0] * velocity[N]

    diffusion_matrix = np.ndarray((N + 1, 3))
    diffusion_matrix[0][1] = max(0, -KH[0][1], -KH[1][-1])
    diffusion_matrix[0][0] = - diffusion_matrix[0][1]
    for j in range(1, N):
        diffusion_matrix[j][-1] = max(0, -KH[j][-1], -KH[j-1][1])
        diffusion_matrix[j][1] = max(0, -KH[j][1], -KH[j+1][-1])
    for j in range(1, N):
        diffusion_matrix[j][0] = - diffusion_matrix[j][-1] - \
            diffusion_matrix[j][1]
    diffusion_matrix[N][-1] = max(0, -KH[N][-1], -KH[N-1][1])
    diffusion_matrix[N][0] = - diffusion_matrix[N][-1]

    advection_matrix = KH + diffusion_matrix

    return advection_matrix
