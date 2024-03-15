"""
Created on Tue Apr 26 10:49:33 2022

@author: Tomasz Krajewski
"""
# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.

# %% Import termodynamically corrected quantities.
from Cpotential cimport VT, dVTdphi, dVTdT, d2VTdphidT, d2VTdT2, d3VTdT3
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
cdef double VT_mod(const double field, const double temp):
    return VT(field / GeV, temp / GeV) * GeV * GeV * GeV * GeV
# end: Defining the rescaled potential.


# %% Defining rescaled derivative of the potential.
@cython.cdivision(True)
cdef double dVTdphi_mod(const double field, const double temp):
    return dVTdphi(field / GeV, temp / GeV) * GeV * GeV * GeV
# end: Defining rescaled derivative of the potential.


# %% Defining rescaled derivative of the potential.
@cython.cdivision(True)
cdef double dVTdT_mod(const double field, const double temp):
    return dVTdT(field / GeV, temp / GeV) * GeV * GeV * GeV
# end: Defining rescaled derivative of the potential.


# %% Defining rescaled second derivative of the potential.
@cython.cdivision(True)
cdef double d2VTdphidT_mod(const double field, const double temp):
    return d2VTdphidT(field / GeV, temp / GeV) * GeV * GeV
# end: Defining rescaled second derivative of the potential.


# %% Defining rescaled second derivative of the potential.
@cython.cdivision(True)
cdef double d2VTdT2_mod(const double field, const double temp):
    return d2VTdT2(field / GeV, temp / GeV) * GeV * GeV
# end: Defining rescaled second derivative of the potential.


# %% Defining rescaled derivative of the potential.
@cython.cdivision(True)
cdef double d3VTdT3_mod(const double field, const double temp):
    return d3VTdT3(field / GeV, temp / GeV) * GeV
# end: Defining rescaled derivative of the potential.


@cython.cdivision(True)
cdef void tauZphi2Tpv_mod(const double tau, const double Z, const double phi,
                          double* T, double* P, double* v):
    cdef double GeV4 = GeV * GeV * GeV * GeV
    cdef double newT = T[0] / GeV
    cdef double newP = P[0] / GeV4
    cdef double newv = v[0]
    tauZphi2Tpv_new(tau / GeV4, Z / GeV4, phi / GeV, &newT, &newP, &newv)
    T[0] = newT * GeV
    P[0] = newP * GeV4
    v[0] = newv
