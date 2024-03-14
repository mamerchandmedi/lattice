#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:26:59 2022

@author: Tomasz Krajewski
"""
# %% Import package for plotting.
import matplotlib.pyplot as plt
# end: Import package for plotting.

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import termodynamically corrected quantities.
from Cpotential import V0, VT, dVTdphi, dVTdT, d2VTdphidT, d2VTdT2, d3VTdT3
# end: Import termodynamically corrected quantities.

V0 = np.vectorize(V0)
VT = np.vectorize(VT)
dVTdphi = np.vectorize(dVTdphi)
dVTdT = np.vectorize(dVTdT)
d2VTdphidT = np.vectorize(d2VTdphidT)
d2VTdT2 = np.vectorize(d2VTdT2)
d3VTdT3 = np.vectorize(d3VTdT3)

# %% Importing parameters of the model.
from parameters import lam, etaV, y, v, Tn, Tc
# end: Importing parameters of the model.

# # %% Import termal functions.
# from thermal_functions.thermal_functions import Jf, Jb, J
# # end: %% Import termal functions.

# ##### Effective potential #####

# m02 = lam*v**2*etaV
# mv2 = lam*v**2*(1 - etaV)
# def meff2(phi):
# 	return lam*(etaV*v**2 - 2*(1+etaV)*v*phi + 3*phi**2)
# 	
# def m2(phi):
# 	#return max(meff2(phi), 0)
# 	return meff2(phi)

# # %% Defining zero-temperature potential of the model.
# def V0(phi):
# 	return lam*(0.5*etaV*v**2*phi**2 - (1+etaV)/3*v*phi**3 + 0.25*phi**4)
# # end: Defining zero-temperature potential of the model.


# def mf2(phi):
# 	return y**2*phi**2

# # %% Defining thermal corrections to the potential.
# def Vtherm(phi, T):
# 	#return  T**4/(2*np.pi**2) * Jf((mf2(phi))/T**2) 
# 	return  T**4/(2*np.pi**2) * J((mf2(phi))/T**2) 
# # end: Defining thermal corrections to the potential.


# # %% Defining thermally corrected potential of the model.
# def VT(field, temp):
#     return V0(field) + Vtherm(field, temp)
# #end: Defining thermally corrected potential of the model.


# #def I(k, m):
# #    return k**2*np.exp(-(m + k**2)**0.5)
    
# #def J(x2):
# #    integral_result, _ = quad(I, 0, np.inf, args=(x2,))
# #    return -integral_result


# #print(J(10))
# #J_vec = np.vectorize(J)
# 	
# #def Vtherm_Boltzmann(phi, T):
# #	return T**4/(2*np.pi**2) * J_vec((mf2(phi))/T**2)


# # %% Defining thermally corrected potential of the model.
# #def VT_B(field, temp):
# #    return V0(field) + Vtherm_Boltzmann(field, temp)
# #end: Defining thermally corrected potential of the model.



# #x2 = np.logspace(2.9, 3.1, 600)
# #fx2 = []
# #for x in x2:
# #	fx2.append(J(x))
# #fx2 = J_vec(x2)
# #print(x2)
# #np.savetxt('finiteT.txt', np.column_stack((x2, fx2)), fmt='%.18e', delimiter=' ')

# #VT_test = VT(phi_test, 50)
# #VT_test_Boltzmann = VT_B(phi_test, 50)

# #plt.figure()
# #plt.scatter(x2, fx2,s=1)
# #plt.xscale("log")
# #plt.plot(phi_test, VT_test_Boltzmann, lw = 0.5)
# #plt.savefig("test_J.pdf")

# # %% Defining derivative with respect to the field of
# #    thermally corrected potential of the model.
# def dVTdphi(field, temp):
#     epsilon = 0.001*v
#     return (-VT(field+2*epsilon, temp) + 8*VT(field+epsilon, temp) - 8*VT(field-epsilon, temp) + VT(field-2*epsilon, temp))/(12*epsilon)
# # end: Defining derivative with respect to the field of
# #      thermally corrected potential of the model.


# # %% Defining first derivative with respect to the temperature of
# #    thermally corrected potential of the model.
# def dVTdT(field, temp):
#     epsilon = 0.001*Tc
#     return (-VT(field, temp+2*epsilon) + 8*VT(field, temp+epsilon) - 8*VT(field, temp-epsilon) + VT(field, temp-2*epsilon))/(12*epsilon)
# # end: Defining first derivative with respect to the temperature of
# #      thermally corrected potential of the model.


# # %% Defining second derivative with respect to the temperature of
# #    thermally corrected potential of the model.
# def d2VTdT2(field, temp):
#     epsilon = 0.001*Tc
#     return (-dVTdT(field, temp+2*epsilon) + 8*dVTdT(field, temp+epsilon) - 8*dVTdT(field, temp-epsilon) + dVTdT(field, temp-2*epsilon))/(12*epsilon)
# # end: Defining second derivative with respect to the temperature of
# #      thermally corrected potential of the model.

# # %% Defining second mixed derivative of
# #    thermally corrected potential of the model.
# def d2VTdphidT(field, temp):
#     epsilon = 0.001*Tc
#     return (-dVTdphi(field, temp+2*epsilon) + 8*dVTdphi(field, temp+epsilon) - 8*dVTdphi(field, temp-epsilon) + dVTdphi(field, temp-2*epsilon))/(12*epsilon)
# # end: Defining second derivative with respect to the temperature of
# #      thermally corrected potential of the model.


#%% Defining potential of the model.
def V(phi):
    return VT(phi, Tn)
# end: Defining potential of the model.


# %% Defining derivative of the potential of the model.
def dV(phi):
    return dVTdphi(phi, Tn)
# end: Defining derivative of the potential of the model.

def dVdT(phi):
    return dVTdT(phi, Tn)
    
if __name__=='__main__':
    vec_phi = np.array(range(14000)) * 0.01
    vec_V = []
    vec_dV = []
    vec_w = []
    for i in range(len(vec_phi)):
    	vec_V.append(V(vec_phi[i]))
    	vec_dV.append(dV(vec_phi[i]))
    	vec_w.append(-Tn*dVdT(vec_phi[i]))
    plt.figure()
    plt.plot(vec_phi, vec_V)
    plt.savefig('tunneling_potential.png')
    
    plt.figure()
    plt.plot(vec_phi, vec_dV)
    plt.savefig('tunneling_potential_derivative.png')
    
    plt.figure()
    plt.plot(vec_phi, vec_w)
    plt.savefig('tunneling_enthalpy.png')
