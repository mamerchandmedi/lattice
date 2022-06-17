"""
Created on Tue Apr 26 12:37:31 2022

@author: Tomasz Krajewski
"""

# %% Importing procedure for solving nonlinear equations.
from scipy.optimize import fsolve
# end: Importing procedure for solving nonlinear equations.

# %% Importing used function.
from libc.math cimport fabs, sqrt
# end: Importing used functions.

# %% Import termodynamically corrected quantities.
from Cthermo cimport w, p, dpdT, dwdT
# end: Import termodynamically corrected quantities.


cdef inline double solve_func(double x, double t, double f, double z):
    return t + p(f, x) - 0.5 * (w(f, x) + sqrt(w(f, x) * w(f, x) + 4 * z * z))


cdef inline double solve_func_prime(double x, double t, double f, double z):
    return dpdT(f, x) - 0.5 * (dwdT(f, x) + (w(f, x) * dwdT(f, x)) /
                               sqrt(w(f, x) * w(f, x) + 4 * z * z))


cdef void tauZphi2Tpv_new(double tau, double Z, double phi,
                          double* T, double* P, double* v):
    cdef int maxiter = 50
    cdef double tol=1.48e-8
    cdef double rtol=0.0

    cdef double x = T[0]
    cdef double x0 = T[0]

    cdef int itr
    cdef double fval
    cdef double fder
    cdef double newton_step

    # Newton-Raphson method
    for itr in range(maxiter):
        # first evaluate fval
        fval = solve_func(x0, tau, phi, Z)
        # If fval is 0, a root has been found, then terminate
        if fval == 0.:
            break
        fder = solve_func_prime(x0, tau, phi, Z)
        if fder == 0:
            msg = "Derivative was zero."
            raise RuntimeError(msg)
        newton_step = fval / fder
        x = x0 - newton_step
        if fabs(x - x0) <= tol + rtol * fabs(x):
            break
        x0 = x

    T[0] = x
    P[0] = p(phi, x)
    v[0] = Z / (tau + P[0])
