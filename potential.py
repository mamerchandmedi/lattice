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

# %% Import thermal functions for verification.
from thermal_functions.Cthermal_functions import J0_interpolated, \
 J1_interpolated, J2_interpolated, J3_interpolated
from thermal_functions.Cbosonic_thermal_functions import Jb0_interpolated, \
 Jb1_interpolated, Jb2_interpolated, Jb3_interpolated
from thermal_functions.Cfermionic_thermal_functions import Jf0_interpolated, \
 Jf1_interpolated, Jf2_interpolated, Jf3_interpolated
# end: Import thermal functions for verification.

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
    vec_x = np.logspace(-4,4)
    vec_J0 = []
    vec_J1 = []
    vec_J2 = []
    vec_J3 = []
    vec_Jb0 = []
    vec_Jb1 = []
    vec_Jb2 = []
    vec_Jb3 = []
    vec_Jf0 = []
    vec_Jf1 = []
    vec_Jf2 = []
    vec_Jf3 = []
    for i in range(len(vec_phi)):
    	vec_V.append(V(vec_phi[i]))
    	vec_dV.append(dV(vec_phi[i]))
    	vec_w.append(-Tn*dVdT(vec_phi[i]))
    for i in range(len(vec_x)):
    	vec_J0.append(J0_interpolated(vec_x[i]))
    	vec_J1.append(J1_interpolated(vec_x[i]))
    	vec_J2.append(J2_interpolated(vec_x[i]))
    	vec_J3.append(J3_interpolated(vec_x[i]))
    	vec_Jb0.append(Jb0_interpolated(vec_x[i]))
    	vec_Jb1.append(Jb1_interpolated(vec_x[i]))
    	vec_Jb2.append(Jb2_interpolated(vec_x[i]))
    	vec_Jb3.append(Jb3_interpolated(vec_x[i]))
    	vec_Jf0.append(Jf0_interpolated(vec_x[i]))
    	vec_Jf1.append(Jf1_interpolated(vec_x[i]))
    	vec_Jf2.append(Jf2_interpolated(vec_x[i]))
    	vec_Jf3.append(Jf3_interpolated(vec_x[i]))
    plt.figure()
    plt.plot(vec_phi, vec_V)
    plt.savefig('tunneling_potential_test.png')
    
    plt.figure()
    plt.plot(vec_phi, vec_dV)
    plt.savefig('tunneling_potential_derivative_test.png')
    
    plt.figure()
    plt.plot(vec_phi, vec_w)
    plt.savefig('tunneling_enthalpy_test.png')
    
    plt.figure()
    plt.plot(vec_x, vec_J0)
    plt.xscale("log")
    plt.savefig('CJ0.png')
    plt.figure()
    plt.plot(vec_x, vec_J1)
    plt.xscale("log")
    plt.savefig('CJ1.png')
    plt.figure()
    plt.plot(vec_x, vec_J2)
    plt.xscale("log")
    plt.savefig('CJ2.png')
    plt.figure()
    plt.plot(vec_x, vec_J3)
    plt.xscale("log")
    plt.savefig('CJ3.png')
    plt.figure()
    plt.plot(vec_x, vec_Jb0)
    plt.xscale("log")
    plt.savefig('CJb0.png')
    plt.figure()
    plt.plot(vec_x, vec_Jb1)
    plt.xscale("log")
    plt.savefig('CJb1.png')
    plt.figure()
    plt.plot(vec_x, vec_Jb2)
    plt.xscale("log")
    plt.savefig('CJb2.png')
    plt.figure()
    plt.plot(vec_x, vec_Jb3)
    plt.xscale("log")
    plt.savefig('CJb3.png')
    plt.figure()
    plt.plot(vec_x, vec_Jf0)
    plt.xscale("log")
    plt.savefig('CJf0.png')
    plt.figure()
    plt.plot(vec_x, vec_Jf1)
    plt.xscale("log")
    plt.savefig('CJf1.png')
    plt.figure()
    plt.plot(vec_x, vec_Jf2)
    plt.xscale("log")
    plt.savefig('CJf2.png')
    plt.figure()
    plt.plot(vec_x, vec_Jf3)
    plt.xscale("log")
    plt.savefig('CJf3.png')
