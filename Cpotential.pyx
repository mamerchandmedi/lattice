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
from parameters import A_parameter
from parameters import V_parameter
from parameters import etaV as pyetaV
from parameters import y as pyy
from parameters import Tc as pyTc
from parameters import eta as pyeta
# end: Importing parameters of the model.

import numpy as np
#from thermal_functions.Cthermal_functions cimport Jf, oldJf, J
from thermal_functions.Cthermal_functions cimport J0_interpolated, J1_interpolated, \
 J2_interpolated, J3_interpolated

cdef double V_PARAMETER = V_parameter
cdef double A_PARAMETER = A_parameter
cdef double etaV = pyetaV
cdef double y = pyy
cdef double eta = pyeta
cdef double Tc = pyTc
cdef double epsilon

# %% Defining zero-temperature potential of the model.
cpdef inline double V0(const double x) noexcept:
    cdef double x2 = x * x;
    return (V_PARAMETER * V_PARAMETER * V_PARAMETER * V_PARAMETER + A_PARAMETER * V_PARAMETER * V_PARAMETER * x2 - (2 * A_PARAMETER + 4) * V_PARAMETER * x2 * x + (A_PARAMETER + 3) * x2 * x2);
# end: Defining zero-temperature potential of the model.


cpdef inline double dV0dphi(const double x) noexcept:
    cdef double x2 = x * x;
    return (2 * A_PARAMETER * V_PARAMETER * V_PARAMETER * x - 3 * (2 * A_PARAMETER + 4) * V_PARAMETER * x2 + 4 * (A_PARAMETER + 3) * x2 * x);


cpdef inline double mf2(const double phi) noexcept:
    return y**2*phi**2


cpdef inline double dmf2dphi(const double phi) noexcept:
    return 2*y**2*phi


# %% Defining thermal corrections to the potential.
@cython.cdivision(True)
cpdef inline double Vtherm(const double phi, const double T) noexcept:
    #return T**4/(2*pi**2) * Jf(mf2(phi,T)/T**2)
    return T**4/(2*pi**2) * J0_interpolated(mf2(phi)/T**2)
# end: Defining thermal corrections to the potential.


# %% Defining thermally corrected potential of the model.
cpdef inline double VT(const double field, const double temp) noexcept:
    return V0(field)
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double dVTdphi(const double field, const double temp) noexcept:
    return dV0dphi(field)
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double dVTdT(const double field, const double temp) noexcept:
    return 0
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining second mixed derivative of thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d2VTdphidT(const double field, const double temp):
    return 0
# end: Defining second mixed derivative of thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d2VTdT2(const double field, const double temp) noexcept:
    return 0    
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining third derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d3VTdT3(const double field, const double temp) noexcept:
    return 0
# end: Defining third derivative with respect to the temperature of
#      thermally corrected potential of the model.
