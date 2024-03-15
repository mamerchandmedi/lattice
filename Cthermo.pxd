"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

cdef double eps(const double field, const double temp)

cdef double p(const double field, const double temp)

cdef double w(const double field, const double temp)

cdef double dpdT(const double field, const double temp)

cdef double d2pdT(const double field, const double temp)

cdef double depsdT(const double field, const double temp)

cdef double d2epsdT(const double field, const double temp)

cdef double dwdT(const double field, const double temp)

cdef double d2wdT(const double field, const double temp)

cdef double gam(const double vel)

cdef double dgamdv(const double v)

cdef double Gf_tau(const double phi, const double dphidt,
                   const double dphidr, const double T, const double v)

cdef double Gf_Z(const double phi, const double dphidt, 
                 const double dphidr, const double T, const double v)

cdef double dtaudp(const double v, const double field, const double temp)

cdef double dZdp(const double v, const double field, const double temp)

cdef double dtaudT(const double v, const double field, const double temp)

cdef double dtaudv(const double v, const double field, const double temp, const double P)

cdef double dvdtau(const double v, const double field, const double temp, const double P)    

cdef double dvdZ(const double v, const double field, const double temp, const double P)

cdef double dTdtau(const double v, const double field, const double temp, const double P)

cdef double dTdZ(const double v, const double field, const double temp, const double P)

cdef double dpdtau(const double v, const double field, const double temp, const double P)

cdef double dpdZ(const double v, const double field, const double temp, const double P)

cdef double dZdv(const double v, const double field, const double temp)

cdef double dZdT(const double v, const double field, const double temp)

cdef double dFtaudtau(const double v, const double field, const double temp, const double P)

cdef double dFtaudZ(const double v, const double field, const double temp, const double P)

cdef double dGtaudtau(const double v, const double phi, const double T, const double P, const double dphidt, const double dphidr)

cdef double dGtaudZ(const double v, const double phi, const double T, const double P, const double dphidt, const double dphidr)

cdef double dGZdtau(const double v, const double phi, const double T, const double P, const double dphidt, const double dphidr)

cdef double dGZdZ(const double v, const double phi, const double T, const double P, const double dphidt, const double dphidr)
