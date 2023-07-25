# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.

# %% Import numpy.
import numpy as np
cimport numpy as np
# end: Import numpy.

# %% Importing Cython array support.
from cpython cimport array
from cython cimport view
import array
# end: Importing Cython array support.

# %% Importing used function.
from libc.math cimport fabs, sqrt
# end: Importing used functions.

# %% Import termodynamically corrected quantities.
from Cthermo cimport a, dVTdT, d2VTdT, w, gam, dpdT
# end: Import termodynamically corrected quantities.


cdef inline double dtaudT(double v, double field, double T):
    return gam(v)**2 * (16 * a * T**3 - dVTdT(field, T) -
                        T * d2VTdT(field, T)) - dpdT(field, T)


cdef inline double dtaudv(double v, double field, double T):
    return 2 * w(field, T) * v * gam(v)**4


cdef inline double dZdT(double v, double field, double T):
    return v * gam(v)**2 * (16 * a * T**3 - dVTdT(field, T) -
                            T * d2VTdT(field, T))


cdef inline double dZdv(double v, double field, double T):
    return w(field, T) * (1 + v**2) * gam(v)**4


cdef inline double determinant(double v, double field, double T):
    return dtaudv(v, field, T) * dZdT(v, field, T) - \
        dtaudT(v, field, T) * dZdv(v, field, T)


@cython.cdivision(True)
cdef inline double dvdtau(double v, double field, double T):
    return dZdT(v, field, T) / determinant(v, field, T)


@cython.cdivision(True)
cdef inline double dvdZ(double v, double field, double T):
    return -dtaudT(v, field, T) / determinant(v, field, T)


@cython.cdivision(True)
cdef inline double dTdtau(double v, double field, double T):
    return -dZdv(v, field, T) / determinant(v, field, T)


@cython.cdivision(True)
cdef inline double dTdZ(double v, double field, double T):
    return dtaudv(v, field, T) / determinant(v, field, T)


@cython.cdivision(True)
cdef void T_matrices(double v,
                     double field,
                     double T,
                     double* T_mat,
                     double* T_inv):
    cdef double temp = dpdT(field, T) * dTdZ(v, field, T) / \
        (dpdT(field, T) * dTdtau(v, field, T))
    T_mat[0] = -temp # [0][0]
    T_mat[1] = 1 # [0][1]
    T_mat[2] = 1 # [1][0]
    T_mat[3] = 0 # [1][1]

    T_inv[0] = 0 # [0][0]
    T_inv[1] = 1 # [0][1]
    T_inv[2] = 1 # [1][0]
    T_inv[3] = temp # [1][1]
