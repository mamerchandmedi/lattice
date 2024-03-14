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
from parameters import lam, etaV, y, v, Tn, Tc
# end: Importing parameters of the model.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from thermal_functions import Jf, Jb, J



##### Effective potential #####

m02 = lam*v**2*etaV
mv2 = lam*v**2*(1 - etaV)
def meff2(phi):
	return lam*(etaV*v**2 - 2*(1+etaV)*v*phi + 3*phi**2)
	
def m2(phi):
	#return max(meff2(phi), 0)
	return meff2(phi)
	
def V0(phi):
	return lam*(0.5*etaV*v**2*phi**2 - (1+etaV)/3*v*phi**3 + 0.25*phi**4)


def mf2(phi):
	return y**2*phi**2

		
def Vtherm(phi, T):
	#return  T**4/(2*np.pi**2) * Jf((mf2(phi))/T**2) 
	return  T**4/(2*np.pi**2) * J((mf2(phi))/T**2) 

#phi_test = np.linspace(0,120,100)


	
# %% Defining thermally corrected potential of the model.
def VT(field, temp):
    return V0(field) + Vtherm(field, temp)
#end: Defining thermally corrected potential of the model.


#def I(k, m):
#    return k**2*np.exp(-(m + k**2)**0.5)
    
#def J(x2):
#    integral_result, _ = quad(I, 0, np.inf, args=(x2,))
#    return -integral_result


#print(J(10))
#J_vec = np.vectorize(J)
	
#def Vtherm_Boltzmann(phi, T):
#	return T**4/(2*np.pi**2) * J_vec((mf2(phi))/T**2)


# %% Defining thermally corrected potential of the model.
#def VT_B(field, temp):
#    return V0(field) + Vtherm_Boltzmann(field, temp)
#end: Defining thermally corrected potential of the model.



#x2 = np.logspace(2.9, 3.1, 600)
#fx2 = []
#for x in x2:
#	fx2.append(J(x))
#fx2 = J_vec(x2)
#print(x2)
#np.savetxt('finiteT.txt', np.column_stack((x2, fx2)), fmt='%.18e', delimiter=' ')

#VT_test = VT(phi_test, 50)
#VT_test_Boltzmann = VT_B(phi_test, 50)

#plt.figure()
#plt.scatter(x2, fx2,s=1)
#plt.xscale("log")
#plt.plot(phi_test, VT_test_Boltzmann, lw = 0.5)
#plt.savefig("test_J.pdf")

	



# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
def dVTdphi(field, temp):
    epsilon = 0.001*v
    return (-VT(field+2*epsilon, temp) + 8*VT(field+epsilon, temp) - 8*VT(field-epsilon, temp) + VT(field-2*epsilon, temp))/(12*epsilon)
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
def dVTdT(field, temp):
    epsilon = 0.001*Tc
    return (-VT(field, temp+2*epsilon) + 8*VT(field, temp+epsilon) - 8*VT(field, temp-epsilon) + VT(field, temp-2*epsilon))/(12*epsilon)
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
def d2VTdT(field, temp):
    epsilon = 0.001*Tc
    return (-dVTdT(field, temp+2*epsilon) + 8*dVTdT(field, temp+epsilon) - 8*dVTdT(field, temp-epsilon) + dVTdT(field, temp-2*epsilon))/(12*epsilon)
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining second mixed derivative of
#    thermally corrected potential of the model.
def d2VTdphidT(field, temp):
    epsilon = 0.001*Tc
    return (-dVTdphi(field, temp+2*epsilon) + 8*dVTdphi(field, temp+epsilon) - 8*dVTdphi(field, temp-epsilon) + dVTdphi(field, temp-2*epsilon))/(12*epsilon)
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


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
    return - temp * d2VTdT(field, temp)
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
