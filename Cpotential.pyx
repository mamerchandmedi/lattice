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
from parameters import lam as pylam
from parameters import etaV as pyetaV
from parameters import y as pyy
from parameters import v as pyv
from parameters import Tc as pyTc
from parameters import eta as pyeta
# end: Importing parameters of the model.

import numpy as np
#from thermal_functions.Cthermal_functions cimport Jf, oldJf, J
from thermal_functions.Cthermal_functions cimport J0_interpolated, J1_interpolated, \
 J2_interpolated, J3_interpolated

cdef double lam = pylam
cdef double etaV = pyetaV
cdef double y = pyy
cdef double eta = pyeta
cdef double v = pyv
cdef double Tc = pyTc
cdef double epsilon

# %% Defining zero-temperature potential of the model.
cpdef inline double V0(const double phi) noexcept:
    return lam*(0.5*etaV*v**2*phi**2 - (1+etaV)/3*v*phi**3 + 0.25*phi**4)
# end: Defining zero-temperature potential of the model.


cpdef inline double dV0dphi(const double phi) noexcept:
    return lam*(v**2*etaV*phi - v*(1+etaV)*phi**2 + phi**3)


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
    return V0(field) + Vtherm(field, temp)
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double dVTdphi(const double field, const double temp) noexcept:
    return dV0dphi(field) + temp**2/(2*pi**2)*dmf2dphi(field) * J1_interpolated(mf2(field)/temp**2)
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double dVTdT(const double field, const double temp) noexcept:
    return 2*temp**3/(pi**2) * J0_interpolated(mf2(field)/temp**2) - temp/(pi**2)*mf2(field) * J1_interpolated(mf2(field)/temp**2)
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining second mixed derivative of thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d2VTdphidT(const double field, const double temp):
    if(mf2(field)/temp**2 >0.):
        return dmf2dphi(field) * ( temp/(pi**2) * J1_interpolated(mf2(field)/temp**2) - mf2(field)/(temp*pi**2) * J2_interpolated(mf2(field)/temp**2)  )
    else:
        return dmf2dphi(field) * temp/(pi**2) * J1_interpolated(mf2(field)/temp**2)
# end: Defining second mixed derivative of thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d2VTdT2(const double field, const double temp) noexcept:
    if(mf2(field)/temp**2 > 0.):
        return 6*temp**2/(pi**2) * J0_interpolated(mf2(field)/temp**2) - 5*mf2(field)/(pi**2) * J1_interpolated(mf2(field)/temp**2) + mf2(field)**2/(2*temp**2*pi**2) * J2_interpolated(mf2(field)/temp**2)
    else:
        return 6*temp**2/(pi**2) * J0_interpolated(mf2(field)/temp**2) - 5*mf2(field)/(pi**2) * J1_interpolated(mf2(field)/temp**2)
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining third derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d3VTdT3(const double field, const double temp) noexcept:
    if(mf2(field)/temp**2 > 0.):
        return 12*temp/(pi**2) * J0_interpolated(mf2(field)/temp**2) -  12/(temp*pi**2)*mf2(field) * J1_interpolated(mf2(field)/temp**2) + 9*mf2(field)**2/(temp**3*pi**2) * J2_interpolated(mf2(field)/temp**2) - mf2(field)**3/(temp**5*pi**2) * J3_interpolated(mf2(field)/temp**2)
    else:
        return 12*temp/(pi**2) * J0_interpolated(mf2(field)/temp**2) -  12/(temp*pi**2)*mf2(field) * J1_interpolated(mf2(field)/temp**2)
# end: Defining third derivative with respect to the temperature of
#      thermally corrected potential of the model.
