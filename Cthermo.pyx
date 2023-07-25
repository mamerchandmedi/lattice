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
from parameters import m0 as pym0
from parameters import q as pyq
from parameters import Tc as pyTc
from parameters import eta as pyeta
# end: Importing parameters of the model.

# %% Import package for dynamical compilation of Cython code.
#import pyximport
#pyximport.install(inplace=True, language_level='3')
# end: Import package for dynamical compilation of Cython code.

import numpy as np
from Cthermal_functions cimport Jb

cdef double lam = pylam
cdef double m0 = pym0
cdef double q = pyq
cdef double eta = pyeta
cdef double mu2 = m0**2/2
cdef double v = (mu2/lam)**(0.5)
cdef double mA = q*v
cdef double Tc = pyTc
cdef double epsilon


cdef inline double V0(double phi):
    return -0.5*mu2*phi**2 + 0.25*lam*phi**4 
	
cdef inline double m2(double phi, double T):
    return -mu2 + 3*lam*phi**2 + (lam/3 + 0.25*q**2)*T**2
	
cdef inline double mAl2(double phi, double T):
    return q**2*phi**2 + 1/3 * q**2 * T**2
	
cdef inline double mAt2(double phi, double T):
    return q**2*phi**2 

cdef inline double Vtherm(double phi, double T):
    return T**4/(2*np.pi**2) * Jb(m2(phi,T)/T**2) +  T**4/(2*np.pi**2) * Jb(mAl2(phi,T)/T**2) + 2*T**4/(2*np.pi**2) * Jb(mAt2(phi,T)/T**2)
cdef inline double VCW(double phi, double T):
    return 1/(64*pi**2) * (m2(phi,T)**2*(log(fabs(m2(phi,T)/m0**2) + 1e-100 ) - 1.5) + 2*m2(phi,T)*m0**2) + 1/(64*pi**2) * (mAl2(phi,T)**2*(log(mAl2(phi,T)/(mA**2) + 1e-100) - 1.5) + 2*mAl2(phi,T)*(mA**2)) +  2/(64*pi**2) * (mAt2(phi,T)**2*(log(mAt2(phi,T)/(mA**2) + 1e-100) - 1.5) + 2*mAt2(phi,T)*(mA**2))
	
	
# %% Defining thermally corrected potential of the model.
cdef inline double VT(double field, double temp):
    return V0(field) + Vtherm(field, temp) + VCW(field, temp)
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
cdef inline double dVTdphi(double field, double temp):
    epsilon = 0.001*v
    return (-VT(field+2*epsilon, temp) + 8*VT(field+epsilon, temp) - 8*VT(field-epsilon, temp) + VT(field-2*epsilon, temp))/(12*epsilon)
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef inline double dVTdT(double field, double temp):
    epsilon = 0.001*Tc
    return (-VT(field, temp+2*epsilon) + 8*VT(field, temp+epsilon) - 8*VT(field, temp-epsilon) + VT(field, temp-2*epsilon))/(12*epsilon)
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining second mixed derivative of thermally corrected potential of the model.
cdef inline double d2VTdphidT(double field, double temp):
    epsilon = 0.001*Tc
    return (-dVTdphi(field, temp+2*epsilon) + 8*dVTdphi(field, temp+epsilon) - 8*dVTdphi(field, temp-epsilon) + dVTdphi(field, temp-2*epsilon))/(12*epsilon)
# end: Defining second mixed derivative of thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef inline double d2VTdT(double field, double temp):
    epsilon = 0.001*Tc
    return (-dVTdT(field, temp+2*epsilon) + 8*dVTdT(field, temp+epsilon) - 8*dVTdT(field, temp-epsilon) + dVTdT(field, temp-2*epsilon))/(12*epsilon)
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining third derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef inline double d3VTdT(double field, double temp):
    epsilon = 0.001*Tc
    return (-d2VTdT(field, temp+2*epsilon) + 8*d2VTdT(field, temp+epsilon) - 8*d2VTdT(field, temp-epsilon) + d2VTdT(field, temp-2*epsilon))/(12*epsilon)
# end: Defining third derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining energy density of the system.
cdef inline double eps(double field, double temp):
    return VT(field, temp) - temp * dVTdT(field, temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
cdef inline double p(double field, double temp):
    return - VT(field, temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
cdef inline double w(double field, double temp):
    return eps(field, temp) + p(field, temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
cdef inline double dpdT(double field, double temp):
    return - dVTdT(field, temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining seond derivative of the pressure with respect to temperature.
cdef inline double d2pdT(double field, double temp):
    return - d2VTdT(field, temp)
# end: Defining second derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
cdef inline double depsdT(double field, double temp):
    return - temp * d2VTdT(field, temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining second derivative of the energy density with respect to temperature.
cdef inline double d2epsdT(double field, double temp):
    return - d2VTdT(field, temp) - temp * d3VTdT(field, temp)
# end: Defining second derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
cdef inline double dwdT(double field, double temp):
    return depsdT(field, temp) + dpdT(field, temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining second derivative of the barotropic parameter
#    with respect to temperature.
cdef inline double d2wdT(double field, double temp):
    return d2epsdT(field, temp) + d2pdT(field, temp)
# end: Defining second derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining relativistic gamma factor.
@cython.cdivision(True)
cdef inline double gam(double vel):
    return 1 / (1 - vel**2)**0.5
# end: Defining relativistic gamma factor.

# %% Defining first derivative of the gamma factor.
cdef inline double dgamdv(double v):
    return v*gam(v)**3
# end: Defining first derivative of the gamma factor.

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

# %% Defining tau derivative with respect to p.
cdef inline double dtaudp(double v, double field, double temp):
    return gam(v)**2 * dwdT(field, temp) / dpdT(field, temp) - 1
# end: Defining tau derivative with respect to p.

# %% Defining Z derivative with respect to p.
cdef inline double dZdp(double v, double field, double temp):
    return gam(v)**2 * v * dwdT(field, temp) / dpdT(field, temp)
# end: Defining Z derivative with respect to p.

# %% Defining tau derivative with respect to T.
cdef inline double dtaudT(double v, double field, double temp):
    return dwdT(field, temp) * gam(v)**2 - dpdT(field, temp)
# end: Defining tau derivative with respect to T.

# %% Defining tau derivative with respect to v.
cdef inline double dtaudv(double v, double field, double temp, double P):
    return (eps(field, temp) + P) * 2 * v * gam(v)**4
# end: Defining tau derivative with respect to v.

# %% Defining v derivative with respect to tau along Z = const.
cdef inline double dvdtau(double v, double field, double temp, double P):
    return 1/(dtaudv(v, field, temp, P) - dtaudT(v, field, temp)*dZdT(v, field, temp)**(-1)*dZdv(v, field, temp))
# end: Defining v derivative with respect to tau along Z = const.

# %% Defining v derivative with respect to Z along tau = const.
cdef inline double dvdZ(double v, double field, double temp, double P):
    return 1/(dZdv(v, field, temp) - dZdT(v, field, temp)*dtaudT(v, field, temp)**(-1)*dtaudv(v, field, temp, P))
# end: Defining v derivative with respect to Z along tau = const.

# %% Defining T derivative with respect to tau along Z = const.
cdef inline double dTdtau(double v, double field, double temp, double P):
    return 1/(dtaudT(v, field, temp) - dtaudv(v, field, temp, P)*dZdv(v, field, temp)**(-1)*dZdT(v, field, temp))
# end: Defining T derivative with respect to tau along Z = const.

# %% Defining T derivative with respect to Z along tau = const.
cdef inline double dTdZ(double v, double field, double temp, double P):
    return 1/(dZdT(v, field, temp) - dZdv(v, field, temp)*dtaudv(v, field, temp, P)**(-1)*dtaudT(v, field, temp))
# end: Defining T derivative with respect to Z along tau = const.

# %% Defining p derivative with respect to tau along Z = const.
cdef inline double dpdtau(double v, double field, double temp, double P):
    return 1/(dtaudp(v, field, temp) - dtaudv(v, field, temp, P)*dZdv(v, field, temp)**(-1)*dZdp(v, field, temp))
    #return dpdT(field, temp) * dTdtau(v, field, temp, P)
# end: Defining p derivative with respect to tau along Z = const.

# %% Defining p derivative with respect to Z along tau = const.
cdef inline double dpdZ(double v, double field, double temp, double P):
    return 1/(dZdp(v, field, temp) - dZdv(v, field, temp)*dtaudv(v, field, temp, P)**(-1)*dtaudp(v, field, temp))
    #return dpdT(field, temp) * dTdZ(v, field, temp, P)
# end: Defining p derivative with respect to Z along tau = const.

# %% Defining Z derivative with respect to v.
cdef inline double dZdv(double v, double field, double temp):
    return 2 * (eps(field, temp) + p(field,temp))*gam(v)*dgamdv(v) + (eps(field, temp) + p(field,temp))*gam(v)**2
# end: Defining Z derivative with respect to v.

# %% Defining Z derivative with respect to T.
cdef inline double dZdT(double v, double field, double temp):
    return dwdT(field, temp) * gam(v)**2 * v
# end: Defining Z derivative with respect to T.

# %% Defining Ftau derivative with respect to tau.
cdef inline double dFtaudtau(double v, double field, double temp, double P):
    return 1 + dpdtau(v, field, temp, P)
# end: Defining Ftau derivative with respect to tau.

# %% Defining Ftau derivative with respect to Z.
cdef inline double dFtaudZ(double v, double field, double temp, double P):
    return dpdZ(v, field, temp, P)
# end: Defining Ftau derivative with respect to Z.

# %% Defining Gtau derivative with respect to tau.
cdef inline double dGtaudtau(double v, double phi, double T, double P, double dphidt, double dphidr):
    return d2VTdphidT(phi, T)*dTdtau(v, phi, T, P)*dphidt + eta*dgamdv(v)*dvdtau(v, phi, T, P)*(dphidt + v*dphidr)*dphidt + eta*gam(v)*dvdtau(v, phi, T, P)*dphidr*dphidt
# end: Defining Gtau derivative with respect to tau.

# %% Defining Gtau derivative with respect to Z.
cdef inline double dGtaudZ(double v, double phi, double T, double P, double dphidt, double dphidr):
    return d2VTdphidT(phi, T)*dTdZ(v, phi, T, P)*dphidt + eta*dgamdv(v)*dvdZ(v, phi, T, P)*(dphidt + v*dphidr)*dphidt + eta*gam(v)*dvdZ(v, phi, T, P)*dphidr*dphidt
# end: Defining Gtau derivative with respect to Z.

# %% Defining GZ derivative with respect to tau.
cdef inline double dGZdtau(double v, double phi, double T, double P, double dphidt, double dphidr):
    return -d2VTdphidT(phi, T)*dTdtau(v, phi, T, P)*dphidt - eta*dgamdv(v)*dvdtau(v, phi, T, P)*(dphidt + v*dphidr)*dphidr - eta*gam(v)*dvdtau(v, phi, T, P)*dphidr**2
# end: Defining Gtau derivative with respect to tau.

# %% Defining GZ derivative with respect to Z.
cdef inline double dGZdZ(double v, double phi, double T, double P, double dphidt, double dphidr):
    return -d2VTdphidT(phi, T)*dTdZ(v, phi, T, P)*dphidt - eta*dgamdv(v)*dvdZ(v, phi, T, P)*(dphidt + v*dphidr)*dphidr - eta*gam(v)*dvdZ(v, phi, T, P)*dphidr**2
# end: Defining GZ derivative with respect to Z.
