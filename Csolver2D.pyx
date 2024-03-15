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
from Cthermo2D cimport w, p, dpdT, dwdT, d2pdT2, d2wdT, dpdmu, dwdmu,\
    d2pdTdmu, d2pdmu2
# end: Import termodynamically corrected quantities.


cdef inline (double, double) solve_func(double x, double t, double f, double z,\
                                        double n, double mu):
    return (t + p(f, x, mu) - 0.5 * (w(f, x, mu) + sqrt(w(f, x, mu) * w(f, x, mu) + 4 * z * z)),
            z * dpdmu(f, x, mu) - n * (t + p(f, x, mu))) 


@cython.cdivision(True)
cdef inline double[:,:] solve_func_prime(double x, double t, double f, double z,\
                                         double n, double mu):
    cdef double output[2][2]
    output[0][0] = dpdT(f, x, mu) - 0.5 * (dwdT(f, x, mu) + (w(f, x, mu) * dwdT(f, x, mu)) /
                               sqrt(w(f, x, mu) * w(f, x, mu) + 4 * z * z))
    output[0][1] = dpdmu(f, x, mu) - 0.5 * (dwdmu(f, x, mu) + (w(f, x, mu) * dwdmu(f, x, mu)) /
                               sqrt(w(f, x, mu) * w(f, x, mu) + 4 * z * z))
    output[1][0] = z * d2pdTdmu(f, x, mu) - n * dpdT(f, x, mu)
    output[1][1] = z * d2pdmu2(f, x, mu) - n * dpdmu(f, x, mu)
    return output


@cython.cdivision(True)
cdef void tauZphin2Tpvmu(double tau, double Z, double phi, double n,
                         double* T, double* P, double* v, double* mu):
    cdef int maxiter = 100
    cdef double tol=1.11e-16
    cdef double rtol=1.48e-8

    cdef (double, double) x = (T[0], mu[0])
    cdef (double, double) x0 = (T[0], mu[0])

    cdef int itr
    cdef (double, double) fval
    cdef double[:, :] fder
    cdef (double, double) newton_step
    cdef double adj

    # Newton-Raphson method
    for itr in range(maxiter):
        # first evaluate fval
        fval = solve_func(x0[0], tau, phi, Z, n, x0[1])
        # If fval is 0, a root has been found, then terminate
        if fval == (0., 0.):
            break
        fder = solve_func_prime(x0[0], tau, phi, Z, n, x0[1])
        detfder = fder[0][0] * fder[1][1] - fder[0][1] * fder[1][0]
        if detfder == 0:
            msg = "Derivative was zero."
            raise RuntimeError(msg)
        newton_step = ((fder[1][1] * fval[0] - fder[0][1] * fval[1]) / detfder,\
                       (-fder[1][0] * fval[0] - fder[0][0] * fval[1]) / detfder)
        x = x0 - newton_step
        if fabs(x[0] - x0[0]) <= tol + rtol * fabs(x[0]) and \
            fabs(x[1] - x0[1]) <= tol + rtol * fabs(x[1]):
            break
        x0 = x

    T[0] = x[0]
    P[0] = p(phi, x[0], x[1])
    v[0] = Z / (tau + P[0])
    mu[0] = x[1]
