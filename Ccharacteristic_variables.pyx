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



cdef inline double dtaudT(double v, double f1, double f2, double T):
    return gam(v)**2 * (16 * a * T**3 - dVTdT(f1, f2, T) -
                        T * d2VTdT(f1, f2, T)) - dpdT(f1, f2, T)


cdef inline double dtaudv(double v, double f1, double f2, double T):
    return 2 * w(f1, f2, T) * v * gam(v)**4


cdef inline double dZdT(double v, double f1, double f2, double T):
    return v * gam(v)**2 * (16 * a * T**3 - dVTdT(f1, f2, T) -
                            T * d2VTdT(f1, f2, T))


cdef inline double dZdv(double v, double f1, double f2, double T):
    return w(f1, f2, T) * (1 + v**2) * gam(v)**4


cdef inline double determinant(double v, double f1, double f2, double T):
    return dtaudv(v, f1, f2, T) * dZdT(v, f1, f2, T) - \
        dtaudT(v, f1, f2, T) * dZdv(v, f1, f2, T)


@cython.cdivision(True)
cdef inline double dvdtau(double v, double f1, double f2, double T):
    return dZdT(v, f1, f2, T) / determinant(v, f1, f2, T)


@cython.cdivision(True)
cdef inline double dvdZ(double v, double f1, double f2, double T):
    return -dtaudT(v, f1, f2, T) / determinant(v, f1, f2, T)


@cython.cdivision(True)
cdef inline double dTdtau(double v, double f1, double f2, double T):
    return -dZdv(v, f1, f2, T) / determinant(v, f1, f2, T)


@cython.cdivision(True)
cdef inline double dTdZ(double v, double f1, double f2, double T):
    return dtaudv(v, f1, f2, T) / determinant(v, f1, f2, T)


@cython.cdivision(True)
cdef void T_matrices(double v,
                     double h,
                     double s,
                     double T,
                     double* T_mat,
                     double* T_inv):
    cdef double temp = dpdT(h, s, T) * dTdZ(v, h, s, T) / \
        (dpdT(h, s, T) * dTdtau(v, h, s, T))
    T_mat[0] = -temp # [0][0]
    T_mat[1] = 1 # [0][1]
    T_mat[2] = 1 # [1][0]
    T_mat[3] = 0 # [1][1]

    T_inv[0] = 0 # [0][0]
    T_inv[1] = 1 # [0][1]
    T_inv[2] = 1 # [1][0]
    T_inv[3] = temp # [1][1]
