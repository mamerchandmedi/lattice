"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

cdef double eps(double field, double temp)

cdef double p(double field, double temp)

cdef double w(double field, double temp)

cdef double dpdT(double field, double temp)

cdef double d2pdT(double field, double temp)

cdef double depsdT(double field, double temp)

cdef double d2epsdT(double field, double temp)

cdef double dwdT(double field, double temp)

cdef double d2wdT(double field, double temp)

cdef double gam(double vel)

cdef double dgamdv(double v)

cdef double Gf_tau(double phi, double dphidt, double dphidr, double T, double v)

cdef double Gf_Z(double phi, double dphidt, double dphidr, double T, double v)

cdef double dtaudp(double v, double field, double temp)

cdef double dZdp(double v, double field, double temp)

cdef double dtaudT(double v, double field, double temp)

cdef double dtaudv(double v, double field, double temp, double p)

cdef double dvdtau(double v, double field, double temp, double p)    

cdef double dvdZ(double v, double field, double temp, double P)

cdef double dTdtau(double v, double field, double temp, double p)

cdef double dTdZ(double v, double field, double temp, double P)

cdef double dpdtau(double v, double field, double temp, double P)

cdef double dpdZ(double v, double field, double temp, double P)

cdef double dZdv(double v, double field, double temp)

cdef double dZdT(double v, double field, double temp)

cdef double dFtaudtau(double v, double field, double temp, double P)

cdef double dFtaudZ(double v, double field, double temp, double P)

cdef double dGtaudtau(double v, double phi, double T, double P, double dphidt, double dphidr)

cdef double dGtaudZ(double v, double phi, double T, double P, double dphidt, double dphidr)

cdef double dGZdtau(double v, double phi, double T, double P, double dphidt, double dphidr)

cdef double dGZdZ(double v, double phi, double T, double P, double dphidt, double dphidr)
