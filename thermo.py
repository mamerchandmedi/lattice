#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

# %% Importing mathematical constants.
from numpy import pi
# end: Importing mathematical constants.

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import package for dynamical compilation of Cython code.
import pyximport
pyximport.install(inplace=True, language_level='3', build_in_temp=False)
# end: Import package for dynamical compilation of Cython code.

# %% Import thermally corrected potential and its derivatives.
from potential import VT, dVTdphi, dVTdT
# end: Import thermally corrected potential and its derivatives.


# %% Defining energy density of the system.
def eps(field, temp):
    return VT(field, temp) - temp * dVTdT(field, temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
def p(field, temp):
    return - VT(field, temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
def w(field, temp):
    return eps(field, temp) + p(field, temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
def dpdT(field, temp):
    return - dVTdT(field, temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
def depsdT(field, temp):
    return - temp * d2VTdT2(field, temp)
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

#vec_phi = np.arange(0,800,0.1)
#vec_V = VT(vec_phi, Tn)
#vec_dVdphi = dVTdphi(vec_phi,Tn)
#vec_dVdT = dVTdT(vec_phi,Tn)
#np.savez_compressed('potential_check', phi = vec_phi, V = vec_V, dVdphi = vec_dVdphi, dVdT = vec_dVdT)
