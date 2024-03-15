"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.

# %% Importing mathematical constants.
from libc.math cimport pi
from libc.math cimport log
from libc.math cimport fabs
# end: Importing mathematical constants.

# %% Importing parameters of the model.
from parameters import eta as pyeta
# end: Importing parameters of the model.

# %% Importing used function.
from libc.math cimport sqrt
# end: Importing used functions.

# %% Import thermally corrected potential and its derivatives.
from Cpotential cimport V0, VT, dVTdphi, dVTdT, d2VTdphidT, d2VTdT2, d3VTdT3
# end: Import thermally corrected potential and its derivatives.

cdef double eta = pyeta


# %% Defining energy density of the system.
cdef inline double eps(const double field, const double temp):
    return VT(field, temp) - temp * dVTdT(field, temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
cdef inline double p(const double field, const double temp):
    return - VT(field, temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
cdef inline double w(const double field, const double temp):
    return eps(field, temp) + p(field, temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
cdef inline double dpdT(const double field, const double temp):
    return - dVTdT(field, temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining seond derivative of the pressure with respect to temperature.
cdef inline double d2pdT(const double field, const double temp):
    return - d2VTdT2(field, temp)
# end: Defining second derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
cdef inline double depsdT(const double field, const double temp):
    return - temp * d2VTdT2(field, temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining second derivative of the energy density with respect to temperature.
cdef inline double d2epsdT(const double field, const double temp):
    return - d2VTdT2(field, temp) - temp * d3VTdT3(field, temp)
# end: Defining second derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
cdef inline double dwdT(const double field, const double temp):
    return depsdT(field, temp) + dpdT(field, temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining second derivative of the barotropic parameter
#    with respect to temperature.
cdef inline double d2wdT(const double field, const double temp):
    return d2epsdT(field, temp) + d2pdT(field, temp)
# end: Defining second derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining relativistic gamma factor.
@cython.cdivision(True)
cdef inline double gam(const double vel):
    return 1 / sqrt(1 - vel**2)
# end: Defining relativistic gamma factor.

# %% Defining first derivative of the gamma factor.
cdef inline double dgamdv(const double v):
    return v * gam(v) * gam(v) * gam(v)
# end: Defining first derivative of the gamma factor.

# %% Defining friction source for tau.
cdef inline double Gf_tau(const double phi, const double dphidt, 
                          const double dphidr, const double T, const double v):
    return dVTdphi(phi, T) * dphidt + \
        eta * gam(v) * (dphidt + v * dphidr) * dphidt
# end: Defining friction source for tau.


# %% Defining friction source for Z.
cdef inline double Gf_Z(const double phi, const double dphidt,
                        const double dphidr, const double T, const double v):
    return -dVTdphi(phi, T) * dphidr - \
        eta * gam(v) * (dphidt + v * dphidr) * dphidr
# end: Defining friction source for Z.

# %% Defining tau derivative with respect to p.
@cython.cdivision(True)
cdef inline double dtaudp(const double v, const double field, const double temp):
    return gam(v)**2 * dwdT(field, temp) / dpdT(field, temp) - 1
# end: Defining tau derivative with respect to p.

# %% Defining Z derivative with respect to p.
cdef inline double dZdp(const double v, const double field, const double temp):
    return gam(v)**2 * v * dwdT(field, temp) / dpdT(field, temp)
# end: Defining Z derivative with respect to p.

# %% Defining tau derivative with respect to T.
cdef inline double dtaudT(const double v, const double field, const double temp):
    return dwdT(field, temp) * gam(v) * gam(v)  - dpdT(field, temp)
# end: Defining tau derivative with respect to T.

# %% Defining tau derivative with respect to v.
cdef inline double dtaudv(const double v, const double field, const double temp, const double P):
    return (eps(field, temp) + P) * 2 * v * gam(v) * gam(v) * gam(v) * gam(v)
# end: Defining tau derivative with respect to v.

# %% Defining v derivative with respect to tau along Z = const.
@cython.cdivision(True)
cdef inline double dvdtau(const double v, const double field, const double temp, const double P):
    return 1/(dtaudv(v, field, temp, P) - dtaudT(v, field, temp)*dZdT(v, field, temp)**(-1)*dZdv(v, field, temp))
# end: Defining v derivative with respect to tau along Z = const.

# %% Defining v derivative with respect to Z along tau = const.
@cython.cdivision(True)
cdef inline double dvdZ(const double v, const double field, const double temp, const double P):
    return 1/(dZdv(v, field, temp) - dZdT(v, field, temp)*dtaudT(v, field, temp)**(-1)*dtaudv(v, field, temp, P))
# end: Defining v derivative with respect to Z along tau = const.

# %% Defining T derivative with respect to tau along Z = const.
@cython.cdivision(True)
cdef inline double dTdtau(const double v, const double field, const double temp, const double P):
    return 1/(dtaudT(v, field, temp) - dtaudv(v, field, temp, P)*dZdv(v, field, temp)**(-1)*dZdT(v, field, temp))
# end: Defining T derivative with respect to tau along Z = const.

# %% Defining T derivative with respect to Z along tau = const.
@cython.cdivision(True)
cdef inline double dTdZ(const double v, const double field, const double temp, const double P):
    return 1/(dZdT(v, field, temp) - dZdv(v, field, temp)*dtaudv(v, field, temp, P)**(-1)*dtaudT(v, field, temp))
# end: Defining T derivative with respect to Z along tau = const.

# %% Defining p derivative with respect to tau along Z = const.
@cython.cdivision(True)
cdef inline double dpdtau(const double v, const double field, const double temp, const double P):
    return 1/(dtaudp(v, field, temp) - dtaudv(v, field, temp, P)*dZdv(v, field, temp)**(-1)*dZdp(v, field, temp))
    #return dpdT(field, temp) * dTdtau(v, field, temp, P)
# end: Defining p derivative with respect to tau along Z = const.

# %% Defining p derivative with respect to Z along tau = const.
@cython.cdivision(True)
cdef inline double dpdZ(const double v, const double field, const double temp, const double P):
    return 1 / (dZdp(v, field, temp) - dZdv(v, field, temp) * dtaudp(v, field, temp) / dtaudv(v, field, temp, P))
    #return dpdT(field, temp) * dTdZ(v, field, temp, P)
# end: Defining p derivative with respect to Z along tau = const.

# %% Defining Z derivative with respect to v.
@cython.cdivision(True)
cdef inline double dZdv(const double v, const double field, const double temp):
    return 2 * (eps(field, temp) + p(field,temp)) * gam(v) * dgamdv(v) +\
        (eps(field, temp) + p(field,temp)) * gam(v) * gam(v)
# end: Defining Z derivative with respect to v.

# %% Defining Z derivative with respect to T.
cdef inline double dZdT(const double v, const double field, const double temp):
    return dwdT(field, temp) * gam(v) * gam(v) * v
# end: Defining Z derivative with respect to T.

# %% Defining Ftau derivative with respect to tau.
cdef inline double dFtaudtau(const double v, const double field, const double temp, const double P):
    return 1 + dpdtau(v, field, temp, P)
# end: Defining Ftau derivative with respect to tau.

# %% Defining Ftau derivative with respect to Z.
cdef inline double dFtaudZ(const double v, const double field, const double temp, const double P):
    return dpdZ(v, field, temp, P)
# end: Defining Ftau derivative with respect to Z.

# %% Defining Gtau derivative with respect to tau.
cdef inline double dGtaudtau(const double v, const double phi, const double T, const double P, const double dphidt, const double dphidr):
    return d2VTdphidT(phi, T) * dTdtau(v, phi, T, P) * dphidt +\
        eta * dgamdv(v) * dvdtau(v, phi, T, P) * (dphidt + v * dphidr) * dphidt +\
        eta * gam(v) * dvdtau(v, phi, T, P) * dphidr * dphidt
# end: Defining Gtau derivative with respect to tau.

# %% Defining Gtau derivative with respect to Z.
cdef inline double dGtaudZ(const double v, const double phi, const double T, const double P, const double dphidt, const double dphidr):
    return d2VTdphidT(phi, T) * dTdZ(v, phi, T, P) * dphidt +\
        eta * dgamdv(v) * dvdZ(v, phi, T, P) * (dphidt + v * dphidr) * dphidt +\
        eta * gam(v) * dvdZ(v, phi, T, P) * dphidr * dphidt
# end: Defining Gtau derivative with respect to Z.

# %% Defining GZ derivative with respect to tau.
cdef inline double dGZdtau(const double v, const double phi, const double T, const double P, const double dphidt, const double dphidr):
    return -d2VTdphidT(phi, T) * dTdtau(v, phi, T, P) * dphidt -\
        eta * dgamdv(v) * dvdtau(v, phi, T, P) * (dphidt + v * dphidr) * dphidr -\
        eta * gam(v) * dvdtau(v, phi, T, P) * dphidr * dphidr
# end: Defining Gtau derivative with respect to tau.

# %% Defining GZ derivative with respect to Z.
cdef inline double dGZdZ(const double v, const double phi, const double T, const double P, const double dphidt, const double dphidr):
    return -d2VTdphidT(phi, T) * dTdZ(v, phi, T, P) * dphidt -\
        eta * dgamdv(v)*dvdZ(v, phi, T, P) * (dphidt + v * dphidr) * dphidr -\
        eta * gam(v) * dvdZ(v, phi, T, P) * dphidr * dphidr
# end: Defining GZ derivative with respect to Z.
