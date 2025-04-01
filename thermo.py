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
from parameters import muh2, mus2, lamh, lamhs, lams, ms, mh, v, ch, cs, Tn
# end: Importing parameters of the model.

import numpy as np
from scipy import interpolate

data = np.loadtxt( 'satoshi_dof.dat' )
Temperature_d=(data.T)[0][900:3900]
dof_d=(data.T)[1][900:3900]
g_star = interpolate.interp1d(Temperature_d, dof_d, kind='cubic')
a = g_star(Tn) * pi**2 / 90



# %% Defining potential of the model.
def VT(f1, f2, temp):
	return (0.5 * ( muh2 + ch*temp**2 ) * f1**2 + 0.25*lamh * f1**4 +
	 0.25*lamhs * f1**2 * f2**2 + 0.5 * (mus2 + cs*temp**2) * f2**2 + 0.25 * lams * f2**4)
# end: Defining potential of the model.



# %% Defining gradient of the potential of the model.
def dVTdh(f1, f2, temp):
	return ( muh2 + ch*temp**2 ) * f1 + lamh * f1**3 + 0.5 * lamhs * f1 * f2**2

def dVTds(f1, f2, temp):
	return  0.5 * lamhs * f1**2 * f2 + (mus2 + cs*temp**2) * f2 + lams * f2**3
# end: Defining derivative of the potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
def dVTdT(f1, f2, temp):
	return  ch * temp * f1**2 + cs * temp * f2**2 
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
def d2VTdT(f1, f2, temp):
    return  ch * f1**2 + cs * f2**2 
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining second mixed derivative of
#    thermally corrected potential of the model.
def d2VTdhdT(f1, f2, temp):
    return 2 *ch*temp * f1

def d2VTdsdT(f1, f2, temp):
    return 2 *cs*temp * f2
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining energy density of the system.
def eps(f1, f2, temp):
    return 3 * a * temp**4 + VT(f1, f2, temp) - temp * dVTdT(f1, f2, temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
def p(f1, f2, temp):
    return a * temp**4 - VT(f1, f2, temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
def w(f1, f2, temp):
    return eps(f1, f2, temp) + p(f1, f2, temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
def dpdT(f1, f2, temp):
    return 4 * a * temp**3 - dVTdT(f1, f2, temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
def depsdT(f1, f2, temp):
    return 12 * a * temp**3 - temp * d2VTdT(f1, f2, temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
def dwdT(f1, f2, temp):
    return depsdT(f1, f2, temp) + dpdT(f1, f2, temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining relativistic gamma factor.
def gam(vel):
    return 1 / (1 - vel**2)**0.5
# end: Defining relativistic gamma factor.

#####NOT YET READY
# %% Defining friction source for tau.
def Gf_tau(phi, dpdt, dpdr, T, v):
    G = dVTdphi(phi, T) * dpdt + eta * gam(v) * (dpdt + v * dpdr) * dpdt
    return G


# %% Defining friction source for Z.
def Gf_Z(phi, dpdt, dpdr, T, v):
    G = - dVTdphi(phi, T) * dpdr - eta * gam(v) * (dpdt + v * dpdr) * dpdr
    return G
    
##########################################

def dvdtau(f1, f2, temp, v, p):
    return (1 - v**2)**2 / (2 * (eps(f1, f2, temp) + p) * v)
   
def dpdtau(v):
    return (1 - v**2) / (v**2)
    
def dTdtau(f1, f2, temp, v):
    return (gam(v)**2 * (depsdT(f1, f2, temp) + dpdT(f1, f2, temp)))**(-1)

def dFtaudtau(v):
    return 1 + dpdtau(v)
    
def dGtaudtau(f1, f2, temp, v, p, dfdt, dfdr):
    return (d2VTdphidT(f1, f2, temp) * dTdtau(f1, f2, temp, v) * dfdt 
           + eta * 2*v/(1 - v**2)**2 * dvdtau(field, temp, v, p) * (dfdt + v * dfdr) * dfdt
           + eta * gam(v) * (dfdt + dvdtau(field, temp, v, p) * dfdr) * dfdt)


