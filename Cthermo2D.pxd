"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

cdef double eps(double field, double temp, double mu)

cdef double p(double field, double temp, double mu)

cdef double w(double field, double temp, double mu)

cdef double dpdT(double field, double temp, double mu)

cdef double dpdmu(double field, double temp, double mu)

cdef double d2pdT2(double field, double temp, double mu)

cdef double d2pdmu2(double field, double temp, double mu)

cdef double d2pdTdmu(double field, double temp, double mu)

cdef double depsdT(double field, double temp, double mu)

cdef double depsdmu(double field, double temp, double mu)

cdef double d2epsdT(double field, double temp, double mu)

cdef double dwdT(double field, double temp, double mu)

cdef double dwdmu(double field, double temp, double mu)

cdef double d2wdT(double field, double temp, double mu)

cdef double gam(double vel, double mu)

cdef double dgamdv(double v, double mu)

cdef double Gf_tau(double phi, double dphidt, double dphidr, double T, double v, double mu)

cdef double Gf_Z(double phi, double dphidt, double dphidr, double T, double v, double mu)

cdef double dtaudp(double v, double field, double temp, double mu)

cdef double dZdp(double v, double field, double temp, double mu)

cdef double dtaudT(double v, double field, double temp, double mu)

cdef double dtaudv(double v, double field, double temp, double p, double mu)

cdef double dvdtau(double v, double field, double temp, double p, double mu)    

cdef double dvdZ(double v, double field, double temp, double P, double mu)

cdef double dTdtau(double v, double field, double temp, double p, double mu)

cdef double dTdZ(double v, double field, double temp, double P, double mu)

cdef double dpdtau(double v, double field, double temp, double P, double mu)

cdef double dpdZ(double v, double field, double temp, double P, double mu)

cdef double dZdv(double v, double field, double temp, double mu)

cdef double dZdT(double v, double field, double temp, double mu)

cdef double dFtaudtau(double v, double field, double temp, double P, double mu)

cdef double dFtaudZ(double v, double field, double temp, double P, double mu)

cdef double dGtaudtau(double v, double phi, double T, double P, double dphidt, double dphidr, double mu)

cdef double dGtaudZ(double v, double phi, double T, double P, double dphidt, double dphidr, double mu)

cdef double dGZdtau(double v, double phi, double T, double P, double dphidt, double dphidr, double mu)

cdef double dGZdZ(double v, double phi, double T, double P, double dphidt, double dphidr, double mu)
