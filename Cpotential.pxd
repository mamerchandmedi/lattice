"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

cpdef double V0(const double phi)

cpdef double dV0dphi(const double phi)

cpdef double mf2(const double phi)

cpdef double dmf2dphi(const double phi)

cpdef double Vtherm(const double phi, const double T)

cpdef double VT(const double field, const double temp)

cpdef double dVTdphi(const double field, const double temp)

cpdef double dVTdT(const double field, const double temp)

cpdef double d2VTdphidT(const double field, const double temp)

cpdef double d2VTdT2(const double field, const double temp)

cpdef double d3VTdT3(const double field, const double temp)
