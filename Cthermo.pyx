"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.

# %% Importing mathematical constants.
from libc.math cimport pi
# end: Importing mathematical constants.

# %% Importing parameters of the model.
from parameters import T0, alpha, gamma, lam, eta
# end: Importing parameters of the model.

cdef double gst = 34.25
cdef double a = gst * pi**2 / 90

# %% Defining thermally corrected potential of the model.
cdef inline double VT(double field, double temp):
    return 0.5 * gamma * (temp**2 - T0**2) * field**2 - \
        (1 / 3) * alpha * temp * field**3 + 0.25 * lam * field**4
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
cdef inline double dVTdphi(double field, double temp):
    return gamma * (temp**2 - T0**2) * field - alpha * temp * field**2 + \
        lam * field**3
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef inline double dVTdT(double field, double temp):
    return gamma * temp * field**2 - 1/3 * alpha * field**3
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef inline double d2VTdT(double field, double temp):
    return gamma * field**2
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining energy density of the system.
cdef inline double eps(double field, double temp):
    return 3 * a * temp**4 + VT(field, temp) - temp * dVTdT(field, temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
cdef inline double p(double field, double temp):
    return a * temp**4 - VT(field, temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
cdef inline double w(double field, double temp):
    return eps(field, temp) + p(field, temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
cdef inline double dpdT(double field, double temp):
    return 4 * a * temp**3 - dVTdT(field, temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
cdef inline double depsdT(double field, double temp):
    return 12 * a * temp**3 - temp * d2VTdT(field, temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
cdef inline double dwdT(double field, double temp):
    return depsdT(field, temp) + dpdT(field, temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining relativistic gamma factor.
cdef inline double gam(double vel):
    return 1 / (1 - vel**2)**0.5
# end: Defining relativistic gamma factor.


# %% Defining friction source for tau.
cdef inline double Gf_tau(double phi, double dphidt, double dphidr,
                          double T, double v):
    return dVTdphi(phi, T) * dphidt + \
        eta * gam(v) * (dphidt + v * dphidr) * dphidt
# end: Defining friction source for tau.


# %% Defining friction source for Z.
cdef inline double Gf_Z(double phi, double dphidt, double dphidr,
                        double T, double v):
    return -dVTdphi(phi, T) * dphidr - \
        eta * gam(v) * (dphidt + v * dphidr) * dphidr
# end: Defining friction source for Z.
