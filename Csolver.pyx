"""
Created on Tue Apr 26 12:37:31 2022

@author: Tomasz Krajewski
"""
# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.

# %% Importing used function.
from libc.math cimport fabs, sqrt
# end: Importing used functions.

# %% Import termodynamically corrected quantities.
from Cthermo cimport w, p, dpdT, dwdT, d2pdT, d2wdT
# end: Import termodynamically corrected quantities.


cdef inline double solve_func(double x, double t, double f, double z):
    return t + p(f, x) - 0.5 * (w(f, x) + sqrt(w(f, x) * w(f, x) + 4 * z * z))


@cython.cdivision(True)
cdef inline double solve_func_prime(double x, double t, double f, double z):
    return dpdT(f, x) - 0.5 * (dwdT(f, x) + (w(f, x) * dwdT(f, x)) /
                               sqrt(w(f, x) * w(f, x) + 4 * z * z))


@cython.cdivision(True)
cdef inline double solve_func_prime2(double x, double t, double f, double z):
    return d2pdT(f, x) - 0.5 * (d2wdT(f, x) +
                                ((w(f, x) * d2wdT(f, x) + dwdT(f, x)**2) *
                                 sqrt(w(f, x) * w(f, x) + 4 * z * z) -
                                 (dwdT(f, x) + (w(f, x) * dwdT(f, x))) *
                                                (w(f, x) * dwdT(f, x)) /
                               sqrt(w(f, x) * w(f, x) + 4 * z * z)) /
                               (w(f, x) * w(f, x) + 4 * z * z))


@cython.cdivision(True)
cdef void tauZphi2Tpv_new(double tau, double Z, double phi,
                          double* T, double* P, double* v):
    cdef int maxiter = 100
    cdef double tol=1.11e-16
    cdef double rtol=1.48e-8

    cdef double x = T[0]
    cdef double x0 = T[0]

    cdef int itr
    cdef double fval
    cdef double fder
    cdef double fder2
    cdef double newton_step
    cdef double adj

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
        fder2 = solve_func_prime2(x0, tau, phi, Z)
        # Halley's method:
        #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
        # Only do it if denominator stays close enough to 1
        # Rationale: If 1-adj < 0, then Halley sends x in the
        # opposite direction to Newton. Doesn't happen if x is close
        # enough to root.
        adj = newton_step * fder2 / fder / 2
        if fabs(adj) < 1.:
            newton_step /= 1.0 - adj
        x = x0 - newton_step
        if fabs(x - x0) <= tol + rtol * fabs(x):
            break
        x0 = x

    T[0] = x
    P[0] = p(phi, x)
    v[0] = Z / (tau + P[0])
