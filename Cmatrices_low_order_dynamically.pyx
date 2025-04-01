# %% Import numpy.
import numpy as np
cimport numpy as np
# end: Import numpy.

# %% Importing used function.
from libc.math cimport fmax
# end: Importing used functions.

# %% Import rescaled quantities.
from lattice import N
# end: Import rescaled quantities.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import advection_matrix as high_order_advection_matrix
# end: Import matrices of the high order integration scheme.

cdef int n_N = N

high_order_advection_num_np = np.array(high_order_advection_matrix, dtype=float)
cdef double[:, ::1] advection_num = high_order_advection_num_np

def advection(np.ndarray[double, ndim=1] velocity,
              np.ndarray[double, ndim=2] KL):
    cdef double D01 = fmax(fmax(0, -advection_num[0][1] * velocity[1]),
                           -advection_num[1][-1] * velocity[0])
    KL[0][0] = advection_num[0][0] * velocity[0] - D01
    KL[0][1] = advection_num[0][1] * velocity[1] + D01
    cdef int j
    cdef double Djm1
    cdef double Dj0
    cdef double Djp1
    for j in range(1, n_N):
        Djm1 = fmax(fmax(0, -advection_num[j][-1] * velocity[j - 1]),
                    -advection_num[j - 1][1] * velocity[j])
        Djp1 = fmax(fmax(0, -advection_num[j][1] * velocity[j + 1]),
                    -advection_num[j + 1][-1] * velocity[j])
        Dj0 = - Djm1 - Djp1
        KL[j][-1] = advection_num[j][-1] + Djm1
        KL[j][0] = advection_num[j][0] + Dj0
        KL[j][1] = advection_num[j][1] + Djp1
    cdef double DNNm1 = fmax(fmax(0, -advection_num[n_N][-1] * velocity[n_N - 1]),
                             -advection_num[n_N - 1][1] * velocity[n_N])
    KL[n_N][-1] = advection_num[n_N][-1] * velocity[n_N - 1] + DNNm1
    KL[n_N][0] = advection_num[n_N][0] * velocity[n_N - 1] - DNNm1
