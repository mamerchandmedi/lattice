#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

# %% Importing mathematical constants.
from numpy import pi
# end: Importing mathematical constants.

# %% Importing parameters of the model.
from parameters import T0, alpha, gamma, lam, eta
# end: Importing parameters of the model.

import matplotlib.pyplot as plt
import numpy as np

#gst = 106.75
gst = 34.25
a = gst * pi**2 / 90



# %% Defining thermally corrected potential of the model.
def VT(field, temp):
    return 0.5 * gamma * (temp**2 - T0**2) * field**2 - \
        (1 / 3) * alpha * temp * field**3 + 0.25 * lam * field**4
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
def dVTdphi(field, temp):
    return gamma * (temp**2 - T0**2) * field - alpha * temp * field**2 + \
        lam * field**3
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
def dVTdT(field, temp):
    return gamma * temp * field**2 - 1/3 * alpha * field**3
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
def d2VTdT(field, temp):
    return gamma * field**2
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining second mixed derivative of
#    thermally corrected potential of the model.
def d2VTdphidT(field, temp):
    return 2 * gamma * field - alpha * field**2
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining energy density of the system.
def eps(field, temp):
    return 3 * a * temp**4 + VT(field, temp) - temp * dVTdT(field, temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
def p(field, temp):
    return a * temp**4 - VT(field, temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
def w(field, temp):
    return eps(field, temp) + p(field, temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
def dpdT(field, temp):
    return 4 * a * temp**3 - dVTdT(field, temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
def depsdT(field, temp):
    return 12 * a * temp**3 - temp * d2VTdT(field, temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
def dwdT(field, temp):
    return depsdT(field, temp) + dpdT(field, temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining relativistic gamma factor.
def gam(vel):
    return 1 / (1 - vel**2)**0.5
# end: Defining relativistic gamma factor.


# %% Defining friction source for tau.
def Gf_tau(phi, dpdt, dpdr, T, v):
    G = dVTdphi(phi, T) * dpdt + eta * gam(v) * (dpdt + v * dpdr) * dpdt
    return G


# %% Defining friction source for Z.
def Gf_Z(phi, dpdt, dpdr, T, v):
    G = - dVTdphi(phi, T) * dpdr - eta * gam(v) * (dpdt + v * dpdr) * dpdr
    return G
    
##########################################

def dvdtau(field, temp, v, p):
    return (1 - v**2)**2 / (2 * (eps(field, temp) + p) * v)
   
def dpdtau(v):
    return (1 - v**2) / (v**2)
    
def dTdtau(field, temp, v):
    return (gam(v)**2 * (depsdT(field, temp) + dpdT(field, temp)))**(-1)

def dFtaudtau(v):
    return 1 + dpdtau(v)
    
def dGtaudtau(field, temp, v, p, dfdt, dfdr):
    return (d2VTdphidT(field, temp) * dTdtau(field, temp, v) * dfdt 
           + eta * 2*v/(1 - v**2)**2 * dvdtau(field, temp, v, p) * (dfdt + v * dfdr) * dfdt
           + eta * gam(v) * (dfdt + dvdtau(field, temp, v, p) * dfdr) * dfdt)

