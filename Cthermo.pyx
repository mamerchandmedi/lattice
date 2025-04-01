"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""
from scipy import interpolate
# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.

# %% Importing mathematical constants.
from libc.math cimport pi
# end: Importing mathematical constants.

# %% Importing parameters of the model.
from parameters import muh2 as pymuh2
from parameters import mus2 as pymus2
from parameters import lamh as pylamh
from parameters import lamhs as pylamhs
from parameters import lams as pylams
from parameters import ch as pych
from parameters import cs as pycs

from thermo import a as pya

from parameters_friction import eta_h as pyeta_h
from parameters_friction import eta_s as pyeta_s
# end: Importing parameters of the model.



cdef double muh2 = pymuh2
cdef double mus2 = pymus2
cdef double lamh = pylamh
cdef double lamhs = pylamhs
cdef double lams = pylams
cdef double ch = pych
cdef double cs = pycs
cdef double a = pya

cdef double eta_h = pyeta_h
cdef double eta_s = pyeta_s


# %% Defining thermally corrected potential of the model.
cdef inline double VT(double f1, double f2, double temp):
    return (0.5 * ( muh2 + ch*temp**2 ) * f1**2 + 0.25*lamh * f1**4 +
        0.25*lamhs * f1**2 * f2**2 + 0.5 * (mus2 + cs*temp**2) * f2**2 + 0.25 * lams * f2**4)
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
cdef inline double dVTdh(double f1, double f2, double temp):
    return ( muh2 + ch*temp**2 ) * f1 + lamh * f1**3 + 0.5 * lamhs * f1 * f2**2
cdef inline double dVTds(double f1, double f2, double temp):
    return  0.5 * lamhs * f1**2 * f2 + (mus2 + cs*temp**2) * f2 + lams * f2**3  
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef inline double dVTdT(double f1, double f2, double temp):
    return  ch * temp * f1**2 + cs * temp * f2**2 
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining second mixed derivative of thermally corrected potential of the model.
cdef inline double d2VTdhdT(double f1, double f2, double temp):
    return 2 *ch*temp * f1
cdef inline double d2VTdsdT(double f1, double f2, double temp):
    return 2 *cs*temp * f2
# end: Defining second mixed derivative of thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef inline double d2VTdT(double f1, double f2, double temp):
    return ch * f1**2 + cs * f2**2 
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining third derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef inline double d3VTdT(double f1, double f2, double temp):
    return 0.
# end: Defining third derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining energy density of the system.
cdef inline double eps(double f1, double f2, double temp):
    return 3 * a * temp**4 + VT(f1, f2, temp) - temp * dVTdT(f1, f2, temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
cdef inline double p(double f1, double f2, double temp):
    return a * temp**4 - VT(f1, f2, temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
cdef inline double w(double f1, double f2, double temp):
    return eps(f1, f2, temp) + p(f1, f2, temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
cdef inline double dpdT(double f1, double f2, double temp):
    return 4 * a * temp**3 - dVTdT(f1, f2, temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining seond derivative of the pressure with respect to temperature.
cdef inline double d2pdT(double f1, double f2, double temp):
    return 12 * a * temp**2 - d2VTdT(f1, f2, temp)
# end: Defining second derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
cdef inline double depsdT(double f1, double f2, double temp):
    return 12 * a * temp**3 - temp * d2VTdT(f1, f2, temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining second derivative of the energy density with respect to temperature.
cdef inline double d2epsdT(double f1, double f2, double temp):
    return 36 * a * temp**2 - d2VTdT(f1, f2, temp) - temp * d3VTdT(f1, f2, temp)
# end: Defining second derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
cdef inline double dwdT(double f1, double f2, double temp):
    return depsdT(f1, f2, temp) + dpdT(f1, f2, temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining second derivative of the barotropic parameter
#    with respect to temperature.
cdef inline double d2wdT(double f1, double f2, double temp):
    return d2epsdT(f1, f2, temp) + d2pdT(f1, f2, temp)
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
cdef inline double Gf_tau(double h, double s, double dhdt, double dsdt, double dhdr, double dsdr,
                          double T, double v):
    return dVTdh(h, s, T) * dhdt + dVTds(h, s, T) * dsdt + \
        eta_h * gam(v) * (dhdt + v * dhdr) * dhdt + eta_s * gam(v) * (dsdt + v * dsdr) * dsdt
# end: Defining friction source for tau.


# %% Defining friction source for Z.
cdef inline double Gf_Z(double h, double s, double dhdt, double dsdt, double dhdr, double dsdr,
                          double T, double v):
    return -dVTdh(h, s, T) * dhdr - dVTds(h, s, T) * dsdr - \
        eta_h * gam(v) * (dhdt + v * dhdr) * dhdr - eta_s * gam(v) * (dsdt + v * dsdr) * dsdr
# end: Defining friction source for Z.

# %% Defining tau derivative with respect to p.
cdef inline double dtaudp(double v, double f1, double f2, double temp):
    return gam(v)**2 * dwdT(f1, f2, temp) / dpdT(f1, f2, temp) - 1
# end: Defining tau derivative with respect to p.

# %% Defining Z derivative with respect to p.
cdef inline double dZdp(double v, double f1, double f2, double temp):
    return gam(v)**2 * v * dwdT(f1, f2, temp) / dpdT(f1, f2, temp)
# end: Defining Z derivative with respect to p.

# %% Defining tau derivative with respect to T.
cdef inline double dtaudT(double v, double f1, double f2, double temp):
    return dwdT(f1, f2, temp) * gam(v)**2 - dpdT(f1, f2, temp)
# end: Defining tau derivative with respect to T.

# %% Defining tau derivative with respect to v.
cdef inline double dtaudv(double v, double f1, double f2, double temp, double P):
    return (eps(f1, f2, temp) + P) * 2 * v * gam(v)**4
# end: Defining tau derivative with respect to v.

# %% Defining v derivative with respect to tau along Z = const.
cdef inline double dvdtau(double v, double f1, double f2, double temp, double P):
    return 1/(dtaudv(v, f1, f2, temp, P) - dtaudT(v, f1, f2, temp)*dZdT(v, f1, f2, temp)**(-1)*dZdv(v, f1, f2, temp))
# end: Defining v derivative with respect to tau along Z = const.

# %% Defining v derivative with respect to Z along tau = const.
cdef inline double dvdZ(double v, double f1, double f2, double temp, double P):
    return 1/(dZdv(v, f1, f2, temp) - dZdT(v, f1, f2, temp)*dtaudT(v, f1, f2, temp)**(-1)*dtaudv(v, f1, f2, temp, P))
# end: Defining v derivative with respect to Z along tau = const.

# %% Defining T derivative with respect to tau along Z = const.
cdef inline double dTdtau(double v, double f1, double f2, double temp, double P):
    return 1/(dtaudT(v, f1, f2, temp) - dtaudv(v, f1, f2, temp, P)*dZdv(v, f1, f2, temp)**(-1)*dZdT(v, f1, f2, temp))
# end: Defining T derivative with respect to tau along Z = const.

# %% Defining T derivative with respect to Z along tau = const.
cdef inline double dTdZ(double v, double f1, double f2, double temp, double P):
    return 1/(dZdT(v, f1, f2, temp) - dZdv(v, f1, f2, temp)*dtaudv(v, f1, f2, temp, P)**(-1)*dtaudT(v, f1, f2, temp))
# end: Defining T derivative with respect to Z along tau = const.

# %% Defining p derivative with respect to tau along Z = const.
cdef inline double dpdtau(double v, double f1, double f2, double temp, double P):
    return 1/(dtaudp(v, f1, f2, temp) - dtaudv(v, f1, f2, temp, P)*dZdv(v, f1, f2, temp)**(-1)*dZdp(v, f1, f2, temp))
    #return dpdT(f1, f2, temp) * dTdtau(v, f1, f2, temp, P)
# end: Defining p derivative with respect to tau along Z = const.

# %% Defining p derivative with respect to Z along tau = const.
cdef inline double dpdZ(double v, double f1, double f2, double temp, double P):
    return 1/(dZdp(v, f1, f2, temp) - dZdv(v, f1, f2, temp)*dtaudv(v, f1, f2, temp, P)**(-1)*dtaudp(v, f1, f2, temp))
    #return dpdT(f1, f2, temp) * dTdZ(v, f1, f2, temp, P)
# end: Defining p derivative with respect to Z along tau = const.

# %% Defining Z derivative with respect to v.
cdef inline double dZdv(double v, double f1, double f2, double temp):
    return 2 * (eps(f1, f2, temp) + p(f1, f2, temp))*gam(v)*dgamdv(v) + (eps(f1, f2, temp) + p(f1, f2, temp))*gam(v)**2
# end: Defining Z derivative with respect to v.

# %% Defining Z derivative with respect to T.
cdef inline double dZdT(double v, double f1, double f2, double temp):
    return dwdT(f1, f2, temp) * gam(v)**2 * v
# end: Defining Z derivative with respect to T.

# %% Defining Ftau derivative with respect to tau.
cdef inline double dFtaudtau(double v, double f1, double f2, double temp, double P):
    return 1 + dpdtau(v, f1, f2, temp, P)
# end: Defining Ftau derivative with respect to tau.

# %% Defining Ftau derivative with respect to Z.
cdef inline double dFtaudZ(double v, double f1, double f2, double temp, double P):
    return dpdZ(v, f1, f2, temp, P)
# end: Defining Ftau derivative with respect to Z.

# %% Defining Gtau derivative with respect to tau.
cdef inline double dGtaudtau(double v, double h, double s, double T, double P, double dhdt, double dsdt, double dhdr, double dsdr):
    return d2VTdhdT(h, s, T)*dTdtau(v, h, s, T, P)*dhdt + d2VTdsdT(h, s, T)*dTdtau(v, h, s, T, P)*dsdt + eta_h*dgamdv(v)*dvdtau(v, h, s, T, P)*(dhdt + v*dhdr)*dhdt + eta_h*gam(v)*dvdtau(v, h, s, T, P)*dhdr*dhdt + eta_s*dgamdv(v)*dvdtau(v, h, s, T, P)*(dsdt + v*dsdr)*dsdt + eta_s*gam(v)*dvdtau(v, h, s, T, P)*dsdr*dsdt
# end: Defining Gtau derivative with respect to tau.

# %% Defining Gtau derivative with respect to Z.
cdef inline double dGtaudZ(double v, double h, double s, double T, double P, double dhdt, double dsdt, double dhdr, double dsdr):
    return d2VTdhdT(h, s, T)*dTdZ(v, h, s, T, P)*dhdt + d2VTdsdT(h, s, T)*dTdZ(v, h, s, T, P)*dsdt + eta_h*dgamdv(v)*dvdZ(v, h, s, T, P)*(dhdt + v*dhdr)*dhdt + eta_h*gam(v)*dvdZ(v, h, s, T, P)*dhdr*dhdt + eta_s*dgamdv(v)*dvdZ(v, h, s, T, P)*(dsdt + v*dsdr)*dsdt + eta_s*gam(v)*dvdZ(v, h, s, T, P)*dsdr*dsdt
# end: Defining Gtau derivative with respect to Z.

# %% Defining GZ derivative with respect to tau.
cdef inline double dGZdtau(double v, double h, double s, double T, double P, double dhdt, double dsdt, double dhdr, double dsdr):
    return -d2VTdhdT(h, s, T)*dTdtau(v, h, s, T, P)*dhdt - d2VTdsdT(h, s, T)*dTdtau(v, h, s, T, P)*dsdt - eta_h*dgamdv(v)*dvdtau(v, h, s, T, P)*(dhdt + v*dhdr)*dhdr - eta_h*gam(v)*dvdtau(v, h, s, T, P)*dhdr**2 - eta_s*dgamdv(v)*dvdtau(v, h, s, T, P)*(dsdt + v*dsdr)*dsdr - eta_s*gam(v)*dvdtau(v, h, s, T, P)*dsdr**2
# end: Defining Gtau derivative with respect to tau.

# %% Defining GZ derivative with respect to Z.
cdef inline double dGZdZ(double v, double h, double s, double T, double P, double dhdt, double dsdt, double dhdr, double dsdr):
    return -d2VTdhdT(h, s, T)*dTdZ(v, h, s, T, P)*dhdt - d2VTdsdT(h, s, T)*dTdZ(v, h, s, T, P)*dsdt - eta_h*dgamdv(v)*dvdZ(v, h, s, T, P)*(dhdt + v*dhdr)*dhdr - eta_h*gam(v)*dvdZ(v, h, s, T, P)*dhdr**2 - eta_s*dgamdv(v)*dvdZ(v, h, s, T, P)*(dsdt + v*dsdr)*dsdr - eta_s*gam(v)*dvdZ(v, h, s, T, P)*dsdr**2
# end: Defining GZ derivative with respect to Z.
