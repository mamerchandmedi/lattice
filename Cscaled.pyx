"""
Created on Tue Apr 26 10:49:33 2022

@author: Tomasz Krajewski
"""
# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.

# %% Import termodynamically corrected quantities.
from Cthermo cimport VT, dVTdh, dVTds
# end: Import termodynamically corrected quantities.

# %% Importing procedure for solving algebraic relation
#    T, p, v as function of tau, Z, phi.
from Csolver cimport tauZphi2Tpv_new
# end: Importing procedure for solving algebraic relation
#      T, p, v as function of tau, Z, phi.

# %% Defining rescaling factor.
cdef double GeV = 1.  # Value of GeV in terms of simulation units.
# end: Defining rescaling factor.


# %% Defining the rescaled potential.
@cython.cdivision(True)
cdef double VT_mod(double f1, double f2, double temp):
    return VT(f1 / GeV, f2 / GeV, temp / GeV) * GeV**4
# end: Defining the rescaled potential.


# %% Defining rescaled derivative of the potential.
@cython.cdivision(True)
cdef double dVTdh_mod(double f1, double f2, double temp):
    return dVTdh(f1 / GeV, f2 / GeV, temp / GeV) * GeV**3
    
@cython.cdivision(True)
cdef double dVTds_mod(double f1, double f2, double temp):
    return dVTds(f1 / GeV, f2 / GeV, temp / GeV) * GeV**3
# end: Defining rescaled derivative of the potential.


@cython.cdivision(True)
cdef void tauZphi2Tpv_mod(double tau, double Z, double f1, double f2,
                          double* T, double* P, double* v):
    cdef double newT = T[0] / GeV
    cdef double newP = P[0] / GeV**4
    cdef double newv = v[0]
    tauZphi2Tpv_new(tau / GeV**4, Z / GeV**4, f1 / GeV, f2 / GeV, &newT, &newP, &newv)
    T[0] = newT * GeV
    P[0] = newP * GeV**4
    v[0] = newv
