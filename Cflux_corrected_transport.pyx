# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.

# %% Importing Cython array support.
#from cpython cimport array
#from cython cimport view
#import array
# end: Importing Cython array support.

# %% Import dynamic allocation functions.
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# end: Import dynamic allocation functions.

# %% Importing used function.
from libc.math cimport fmax, fmin
# end: Importing used functions.

# %% Importing Cython array support.
from cpython cimport array
from cython cimport view
import array
# end: Importing Cython array support.

# %% Import numpy.
import numpy as np
cimport numpy as np
# end: Import numpy.

# %% Import rescaled quantities.
from lattice import N, r
# end: Import rescaled quantities.

# %% Import rescaled quantities.
from Cscaled cimport tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import advection_matrix, mass_matrix
# end: Import matrices of the high order integration scheme.

# %% Importing characteristic variables.
from Ccharacteristic_variables cimport T_matrices
# end: Importing characteristic variables.

# %% Fixing a datatype for numpy arrays.
DTYPE = np.double
# end: Fixing a datatype for numpy arrays.

cdef int n_N = N

advection_num_np = np.array(advection_matrix, dtype=float)
cdef double[:, ::1] advection_num = advection_num_np

mass_num_np = np.array(mass_matrix, dtype=float)
cdef double[::1] mass_num = mass_num_np


@cython.boundscheck(False)  # Deactivate bounds checking
def antidiffusive_matrix(np.ndarray[double, ndim=2] KL,
                         np.ndarray[double, ndim=1] FH,
                         np.ndarray[double, ndim=1] FL,
                         double dt,
                         np.ndarray[double, ndim=2] A):
    A[0][-1] = 0
    A[0][0] = -dt * (advection_num[0][0] * FH[0] - KL[0][0] * FL[0])
    A[0][1] = -dt * (advection_num[0][1] * FH[1] - KL[0][1] * FL[1])
    cdef int j
    for j in range(1, n_N):
        A[j][-1] = -dt * (advection_num[j][-1] * FH[j - 1] -
                         KL[j][-1] * FL[j - 1])
        A[j][0] = -dt * (advection_num[j][0] * FH[j] -
                         KL[j][0] * FL[j])
        A[j][1] = -dt * (advection_num[j][1] * FH[j + 1]-
                        KL[j][1] * FL[j + 1])
    A[n_N][-1] =  -dt * (advection_num[n_N][-1] * FH[n_N - 1] -
                      KL[n_N][-1] * FL[n_N - 1])
    A[n_N][0] = -dt * (advection_num[n_N][0] * FH[n_N] -
                      KL[n_N][0] * FL[n_N])
    A[n_N][1] = 0


@cython.boundscheck(False)  # Deactivate bounds checking
def trans_antidiffusive_matrix(np.ndarray[double, ndim=2] A1,
                               np.ndarray[double, ndim=2] A2,
                               np.ndarray[double, ndim=2] tau,
                               np.ndarray[double, ndim=2] Z,
                               np.ndarray[double, ndim=2] phi,
                               np.ndarray[double, ndim=2] AT1,
                               np.ndarray[double, ndim=2] AT2):
    cdef double T
    cdef double P
    cdef double v
    cdef double *T_mat = <double*> PyMem_Malloc(4 * sizeof(double))
    cdef double *T_inv = <double*> PyMem_Malloc(4 * sizeof(double))
    tauZphi2Tpv_mod(tau[0][0], Z[0][0], phi[0][0], &T, &P, &v)
    T_matrices(v, phi[0][0], T, T_mat, T_inv)
    AT1[0][0] = T_inv[0] * A1[0][0] + T_inv[1] * A2[0][0]
    AT2[0][0] = T_inv[2] * A1[0][0] + T_inv[3] * A2[0][0]
    tauZphi2Tpv_mod(tau[0][1], Z[0][1], phi[0][1], &T, &P, &v)
    T_matrices(v, phi[0][1], T, T_mat, T_inv)
    AT1[0][1] = T_inv[0] * A1[0][1] + T_inv[1] * A2[0][1]
    AT2[0][1] = T_inv[2] * A1[0][1] + T_inv[3] * A2[0][1]
    cdef int i
    cdef int j
    for i in range(1, N):
        for j in [-1, 0, 1]:
            tauZphi2Tpv_mod(tau[i][j], Z[i][j], phi[i][j], &T, &P, &v)
            T_matrices(v, phi[i][j], T, T_mat, T_inv)
            AT1[i][j] = T_inv[0] * A1[i][j] + T_inv[1] * A2[i][j]
            AT2[i][j] = T_inv[2] * A1[i][j] + T_inv[3] * A2[i][j]
    tauZphi2Tpv_mod(tau[N][-1], Z[N][-1], phi[N][-1], &T, &P, &v)
    T_matrices(v, phi[N][-1], T, T_mat, T_inv)
    AT1[N][-1] = T_inv[0] * A1[N][-1] + T_inv[1] * A2[N][-1]
    AT2[N][-1] = T_inv[2] * A1[N][-1] + T_inv[3] * A2[N][-1]
    tauZphi2Tpv_mod(tau[N][0], Z[N][0], phi[N][0], &T, &P, &v)
    T_matrices(v, phi[N][0], T, T_mat, T_inv)
    AT1[N][0] = T_inv[0] * A1[N][0] + T_inv[1] * A2[N][0]
    AT2[N][0] = T_inv[2] * A1[N][0] + T_inv[3] * A2[N][0]

    PyMem_Free(T_mat)
    PyMem_Free(T_inv)

cdef array.array P1_plus_array = array.array('d')
cdef array.array P1_minus_array = array.array('d')
cdef array.array P2_plus_array = array.array('d')
cdef array.array P2_minus_array = array.array('d')
array.resize(P1_plus_array, n_N + 1)
array.resize(P1_minus_array, n_N + 1)
array.resize(P2_plus_array, n_N + 1)
array.resize(P2_minus_array, n_N + 1)
cdef double[:] P1_plus = P1_plus_array
cdef double[:] P1_minus = P1_minus_array
cdef double[:] P2_plus = P2_plus_array
cdef double[:] P2_minus = P2_minus_array

cdef array.array R1_plus_array = array.array('d')
cdef array.array R1_minus_array = array.array('d')
cdef array.array R2_plus_array = array.array('d')
cdef array.array R2_minus_array = array.array('d')
array.resize(R1_plus_array, n_N + 1)
array.resize(R1_minus_array, n_N + 1)
array.resize(R2_plus_array, n_N + 1)
array.resize(R2_minus_array, n_N + 1)
cdef double[:] R1_plus = R1_plus_array
cdef double[:] R1_minus = R1_minus_array
cdef double[:] R2_plus = R2_plus_array
cdef double[:] R2_minus = R2_minus_array

cdef array.array tau_prim_array = array.array('d', [0, 0, 0])
cdef array.array Z_prim_array = array.array('d', [0, 0, 0])
cdef double[:] tau_prim = tau_prim_array
cdef double[:] Z_prim = Z_prim_array

cdef array.array tau_old_array = array.array('d', [0, 0, 0])
cdef array.array Z_old_array = array.array('d', [0, 0, 0])
cdef double[:] tau_old = tau_old_array
cdef double[:] Z_old = Z_old_array

@cython.boundscheck(False)  # Deactivate bounds checking
def limit_fluxes(np.ndarray[double, ndim=2] A1_prim,
                 np.ndarray[double, ndim=2] A2_prim,
                 np.ndarray[double, ndim=1] tau,
                 np.ndarray[double, ndim=1] Z,
                 np.ndarray[double, ndim=1] phi,
                 np.ndarray[double, ndim=1] tau_L,
                 np.ndarray[double, ndim=1] Z_L,
                 np.ndarray[double, ndim=1] phi_L,
                 # np.ndarray[double, ndim=2] tau_bar,
                 # np.ndarray[double, ndim=2] Z_bar,
                 # np.ndarray[double, ndim=2] phi_bar,
                 np.ndarray[double, ndim=2] alpha_limiter1,
                 np.ndarray[double, ndim=2] alpha_limiter2,
                 np.ndarray[double, ndim=2] limited_flux_tau,
                 np.ndarray[double, ndim=2] limited_flux_Z):
    P_vectors(A1_prim, P1_plus, P1_minus)
    P_vectors(A2_prim, P2_plus, P2_minus)

    # cdef double *T_mat = <double*> PyMem_Malloc(4 * sizeof(double))
    # cdef double *T_inv = <double*> PyMem_Malloc(4 * sizeof(double))
    cdef int i
    cdef int j
    cdef int k
    cdef double T
    cdef double P
    cdef double v
    cdef double Q1_plus
    cdef double Q1_minus
    cdef double Q2_plus
    cdef double Q2_minus
    for i in range(n_N + 1):
        if i == 0:
            indexes = [0, 1]
            neighbours = [0, 1]
        elif i == N:
            indexes = [-1, 0]
            neighbours = [-1, 0]
        else:
            indexes = [-1, 0, 1]
            neighbours = [-1, 0, 1]
        # for j in indexes:
        #     tauZphi2Tpv_mod(tau_bar[i][j], Z_bar[i][j], phi_bar[i][j],
        #                     &T, &P, &v)
        #     T_matrices(v, phi_bar[i][j], T, T_mat, T_inv)
        for k in neighbours:
            tau_prim[k] = tau_L[i + k] * mass_num[i + k]
            tau_old[k] = tau[i + k] * mass_num[i + k]
            Z_prim[k] = Z_L[i + k] * mass_num[i + k]
            Z_old[k] = Z[i + k] * mass_num[i + k]
        Q1_plus = fmax(Q_values_plus(tau_prim, i),
                       Q_values_plus(tau_old, i)) / mass_num[i]
        Q1_minus = fmin(Q_values_minus(tau_prim, i),
                        Q_values_minus(tau_old, i)) / mass_num[i]
        Q2_plus = fmax(Q_values_plus(Z_prim, i),
                       Q_values_plus(Z_old, i)) / mass_num[i]
        Q2_minus = fmin(Q_values_minus(Z_prim, i),
                        Q_values_minus(Z_old, i)) / mass_num[i]
        R1_plus[i] = R_values(Q1_plus, P1_plus[i])
        R1_minus[i] = R_values(Q1_minus, P1_minus[i])
        R2_plus[i] = R_values(Q2_plus, P2_plus[i])
        R2_minus[i] = R_values(Q2_minus, P2_minus[i])

    for i in range(0, n_N + 1):
        if i == 0:
            indexes = [0, 1]
        elif i == N:
            indexes = [-1, 0]
        else:
            indexes = [-1, 0, 1]
        for j in indexes:
            # tauZphi2Tpv_mod(tau_bar[i][j], Z_bar[i][j], phi_bar[i][j],
            #                 &T, &P, &v)
            # T_matrices(v, phi_bar[i][j], T, T_mat, T_inv)
            if A1_prim[i][j] >= 0:
                alpha_limiter1[i][j] = fmin(R1_plus[i], R1_minus[i + j])
            else:
                alpha_limiter1[i][j] = fmin(R1_minus[i], R1_plus[i + j])
            if A2_prim[i][j] >= 0:
                alpha_limiter2[i][j] = fmin(R2_plus[i], R2_minus[i + j])
            else:
                alpha_limiter2[i][j] = fmin(R2_minus[i], R2_plus[i + j])
            limited_flux_tau[i][j] = A1_prim[i][j] * alpha_limiter1[i][j]
            limited_flux_Z[i][j] = A2_prim[i][j] * alpha_limiter2[i][j]
    # PyMem_Free(T_mat)
    # PyMem_Free(T_inv)

@cython.boundscheck(False)  # Deactivate bounds checking
cdef inline void P_vectors(np.ndarray[double, ndim=2] A,
                           double[:] P_plus,
                           double[:] P_minus):
    P_plus[0] = fmax(0, A[0][1])
    P_minus[0] = fmin(0, A[0][1])
    cdef int j
    for j in range(1, n_N):
        P_plus[j] = fmax(0, A[j][-1]) + fmax(0, A[j][1])
        P_minus[j] = fmin(0, A[j][-1]) + fmin(0, A[j][1])
    P_plus[n_N] = fmax(0, A[n_N][-1])
    P_minus[n_N] = fmin(0, A[n_N][-1])


cdef inline double Q_values_plus(double[:] U, int i):
    if i == 0:
        return fmax(U[0], U[1]) - U[0]
    elif i == n_N:
        return fmax(U[-1], U[0]) - U[0]
    else:
        return fmax(fmax(U[-1], U[0]), U[1]) - U[0]


cdef inline double Q_values_minus(double[:] U, int i):
    if i == 0:
        return fmin(U[0], U[1]) - U[0]
    elif i == n_N:
        return fmin(U[-1], U[0]) - U[0]
    else:
        return fmin(fmin(U[-1], U[0]), U[1]) - U[0]


cdef inline double R_values(double Q, double P):
    cdef double R
    if P != 0:
        R = fmin(1, Q / P)
    else:
        R = 1
    return R



# def Ubar(U):
#     U_matrix = np.ndarray((N + 1, 3))
#     U_matrix[0][0] = U[0]
#     U_matrix[0][1] = 0.5 * (U[0] + U[1])
#     for j in range(1, N):
#         U_matrix[j][-1] = 0.5 * (U[j] + U[j-1])
#         U_matrix[j][0] = U[j]
#         U_matrix[j][1] = 0.5 * (U[j] + U[j+1])
#     U_matrix[N][-1] = 0.5 * (U[N] + U[N-1])
#     U_matrix[N][0] = U[N]
#     return U_matrix
