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
from thermal_functions.Cthermal_functions cimport Jf, oldJf, J

cdef double lam = pylam
cdef double etaV = pyetaV
cdef double y = pyy
cdef double eta = pyeta
cdef double v = pyv
cdef double Tc = pyTc
cdef double epsilon

# %% Defining zero-temperature potential of the model.
cpdef inline double V0(const double phi):
    return lam*(0.5*etaV*v**2*phi**2 - (1+etaV)/3*v*phi**3 + 0.25*phi**4)
# end: Defining zero-temperature potential of the model.
	

cpdef inline double mf2(const double phi, const double T):
    return y**2*phi**2
	
# %% Defining thermal corrections to the potential.
cpdef inline double Vtherm(const double phi, const double T): 
    #return T**4/(2*pi**2) * Jf(mf2(phi,T)/T**2)
    return T**4/(2*pi**2) * J(mf2(phi,T)/T**2)
# end: Defining thermal corrections to the potential.

	
# %% Defining thermally corrected potential of the model.
cpdef inline double VT(const double field, const double temp):
    return V0(field) + Vtherm(field, temp)
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
cpdef inline double dVTdphi(const double field, const double temp):
    epsilon = 0.001*v
    #return (-VT(field+2*epsilon, temp) + 8*VT(field+epsilon, temp) - 8*VT(field-epsilon, temp) + VT(field-2*epsilon, temp))/(12*epsilon)
    return (-VT(field-epsilon, temp) + VT(field+epsilon, temp)) / (2*epsilon) 
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
cpdef inline double dVTdT(const double field, const double temp):
    epsilon = 0.001*Tc
    #return (-VT(field, temp+2*epsilon) + 8*VT(field, temp+epsilon) - 8*VT(field, temp-epsilon) + VT(field, temp-2*epsilon))/(12*epsilon)
    return (-VT(field, temp-epsilon) + VT(field, temp+epsilon)) / (2*epsilon)
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining second mixed derivative of thermally corrected potential of the model.
cpdef inline double d2VTdphidT(const double field, const double temp):
    epsilon = 0.001*Tc
    #return (-dVTdphi(field, temp+2*epsilon) + 8*dVTdphi(field, temp+epsilon) - 8*dVTdphi(field, temp-epsilon) + dVTdphi(field, temp-2*epsilon))/(12*epsilon)
    return (-dVTdphi(field, temp-epsilon) + dVTdphi(field, temp+epsilon)) / (2*epsilon)
# end: Defining second mixed derivative of thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
cpdef inline double d2VTdT2(const double field, const double temp):
    epsilon = 0.001*Tc
    #return (-dVTdT(field, temp+2*epsilon) + 8*dVTdT(field, temp+epsilon) - 8*dVTdT(field, temp-epsilon) + dVTdT(field, temp-2*epsilon))/(12*epsilon)
    return (-dVTdT(field, temp-epsilon)  + dVTdT(field, temp+epsilon)) / (2*epsilon)
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining third derivative with respect to the temperature of
#    thermally corrected potential of the model.
cpdef inline double d3VTdT3(const double field, const double temp):
    epsilon = 0.001*Tc
    #return (-d2VTdT(field, temp+2*epsilon) + 8*d2VTdT(field, temp+epsilon) - 8*d2VTdT(field, temp-epsilon) + d2VTdT(field, temp-2*epsilon))/(12*epsilon)
    return (-d2VTdT2(field, temp-epsilon) + d2VTdT2(field, temp+epsilon)) / (2*epsilon)
# end: Defining third derivative with respect to the temperature of
#      thermally corrected potential of the model.
