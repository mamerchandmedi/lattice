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
from parameters import etaV as pyetaV
from parameters import v as pyv
from parameters import Tc as pyTc
from parameters import eta as pyeta

from parameters import gBL as pygBL
from parameters import nZp as pynZp
from parameters import num_fermion_dof as pynum_fermion_dof
from parameters import num_boson_dof as pynum_boson_dof
from parameters import renormScaleSq as pyrenormScaleSq
from parameters import nZ as pynZ
from parameters import nW as pynW
from parameters import muHsq as pymuHsq
from parameters import lHH as pylHH
from parameters import c2beta as pyc2beta
from parameters import g2sq as pyg2sq
from parameters import g1sq as pyg1sq
from parameters import yt2 as pyyt2
# end: Importing parameters of the model.

import numpy as np
from thermal_functions.Cbosonic_thermal_functions cimport Jb0_interpolated, \
 Jb1_interpolated, Jb2_interpolated, Jb3_interpolated
from thermal_functions.Cfermionic_thermal_functions cimport Jf0_interpolated, \
 Jf1_interpolated, Jf2_interpolated, Jf3_interpolated
 

cdef double etaV = pyetaV
cdef double eta = pyeta
cdef double v = pyv
cdef double Tc = pyTc
cdef double epsilon

cdef double gBL = pygBL
cdef double nZp = pynZp

cdef int num_fermion_dof = pynum_fermion_dof
cdef int num_boson_dof = pynum_boson_dof
cdef double renormScaleSq = pyrenormScaleSq

cdef double nh = 1.
cdef double nW = pynW 
cdef double nZ = pynZ
cdef double nt = 12.
cdef double muHsq = pymuHsq
cdef double lHH = pylHH
cdef double c2beta = pyc2beta
cdef double g2sq = pyg2sq
cdef double g1sq = pyg1sq
cdef double yt2 = pyyt2

# %% Defining zero-temperature potential of the model.
cpdef inline double V0(const double phi) noexcept:
    if phi > 1.:
        return nZp * gBL**4 / (4 * pi**2) * phi**4 * (log(phi**2 / renormScaleSq) - 0.5)
    elif phi > 0:
        return nZp * gBL**4 / (4 * pi**2) * phi**2 * (log((phi**2 / renormScaleSq)**(phi**2)) - 0.5 * phi**2)
    else:
        return 0.

# end: Defining zero-temperature potential of the model.


cpdef inline double dV0dphi(const double phi) noexcept:
    if phi > 1.:
        return (nZp * gBL**4 / pi**2) * phi**3 * log(phi**2 / renormScaleSq)
    elif phi > 0:
        return (nZp * gBL**4 / pi**2) * phi * log((phi**2 / renormScaleSq)**(phi**2))
    else:
        return 0.

cpdef inline double mh2(const double phi) noexcept:
    return muHsq  + 3 * lHH * c2beta * phi**2

cpdef inline double MW2(const double phi) noexcept:
    return 0.25 * g2sq * c2beta * phi**2

cpdef inline double MZ2(const double phi) noexcept:
        cdef double alfa = 0.25 * g2sq * c2beta * phi**2
        cdef double beta = 0.25 * g1sq * c2beta * phi**2
        cdef double Alfa = 0.5 * (alfa + beta)
        cdef double Beta = 0.5 * np.sqrt((alfa - beta)**2 + 0.25 * g2sq * g1sq * (c2beta * phi**2)**2)
        return Alfa + Beta
    
cpdef inline double MZp2(const double phi) noexcept:
     return 4 * gBL**2 * phi**2

cpdef inline double Mt2(const double phi) noexcept:
    return 0.5 * yt2 * c2beta * phi**2

cpdef inline double dmh2dphi(const double phi) noexcept:
    return 6 * lHH * c2beta * phi

cpdef inline double dMW2dphi(const double phi) noexcept:
    return 0.5 * g2sq * c2beta * phi

cpdef inline double dMZ2dphi(const double phi) noexcept:
    return 0.5 * (g1sq + g2sq) * c2beta * phi

cpdef inline double dMZp2dphi(const double phi) noexcept:
    return 8 * gBL**2 * phi

cpdef inline double dMt2dphi(const double phi) noexcept:
    return yt2 * c2beta * phi


# %% Defining thermal corrections to the potential.
@cython.cdivision(True)
cpdef inline double Vtherm(const double phi, const double T) noexcept:
    cdef double T2 = T * T
    cdef double T4 = T2 * T2
    cdef double y = 0
    y += nZp * Jb0_interpolated(MZp2(phi) / T2)# + \
    #     nh * Jb0_interpolated(mh2(phi) / T2) + \
    #     nW * Jb0_interpolated(MW2(phi) / T2) + \
    #     nZ * Jb0_interpolated(MZ2(phi) / T2)
     # y += nt * Jf0_interpolated(Mt2(phi)/T2)
    cdef double nb = num_boson_dof - nZp# - nh - nW - nZ
    y -= nb * pi**4 / 45.
    cdef double nf = num_fermion_dof# - nt
    y -= nf * 7 * pi**4 / 360.
    return y * T4 / (2 * pi * pi)
# end: Defining thermal corrections to the potential.


# %% Defining thermally corrected potential of the model.
cpdef inline double VT(const double field, const double temp) noexcept:
    return V0(field) + Vtherm(field, temp)
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double dVTdphi(const double field, const double T) noexcept:
    cdef double T2 = T * T
    cdef double y = 0
    y += nZp * (T2 / (2 * pi**2)) * dMZp2dphi(field) * Jb1_interpolated(MZp2(field) / T2)# + \
    # nh * (T2 / (2 * pi**2)) * dmh2dphi(field) * Jb1_interpolated(mh2(field) / T2) + \
    # nW * (T2 / (2 * pi**2)) * dMW2dphi(field) * Jb1_interpolated(MW2(field) / T2) + \
    # nZ * (T2 / (2 * pi**2)) * dMZ2dphi(field) * Jb1_interpolated(MZ2(field) / T2) + \
    # nt * (T2 / (2 * pi**2)) * dMt2dphi(field) * Jf1_interpolated(Mt2(field) / T2)
    return dV0dphi(field) + y
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double dVTdT(const double field, const double T) noexcept:
    cdef double nb = num_boson_dof - nZp# - nh - nW - nZ
    cdef double nf = num_fermion_dof# - nt
    cdef double T2 = T * T
    cdef double T3 = T2 * T
    cdef double y = 0
    y += nZp * (2 * T3 / (pi**2)) * Jb0_interpolated(MZp2(field) / T2) - \
    nZp * (T / pi**2) * MZp2(field) * Jb1_interpolated(MZp2(field) / T2)# + \
    # nh * 2 * T3 / pi**2 * Jb0_interpolated(mh2(field) / T2) - \
    # nh * (T / pi**2) * mh2(field) * Jb1_interpolated(mh2(field) / T2) + \
    # nW * 2 * T3 / pi**2 * Jb0_interpolated(MW2(field) / T2) - \
    # nW * T / pi**2 * MW2(field) * Jb1_interpolated(MW2(field) / T2) + \
    # nZ * 2 * T3 / pi**2 * Jb0_interpolated(MZ2(field) / T2) - \
    # nZ * (T / pi**2) * MZ2(field) * Jb1_interpolated(MZ2(field) / T2) + \
    # nt * 2 * T3 / pi**2 * Jf0_interpolated(Mt2(field) / T2) - \
    # nt * (T / pi**2) * Mt2(field) * Jf1_interpolated(Mt2(field) / T2)
    return y - 4 * T3 * nb * pi**4 / 45. - 4 * T3 * nf * 7 * pi**4 / 360.
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining second mixed derivative of thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d2VTdphidT(const double field, const double T):
    cdef double T2 = T * T
    cdef double y = 0
    if field > 0:
        y += nZp * dMZp2dphi(field) * (T/(pi**2)) * Jb1_interpolated(MZp2(field)/T2) - \
        nZp * MZp2(field)/(T*pi**2) * Jb2_interpolated(MZp2(field)/T2)# + \
        # nh * dmh2dphi(field) * (T/(pi**2) * Jb1_interpolated(mh2(field)/T2) - \
        # nh * mh2(field)/(T*pi**2) * Jb2_interpolated(mh2(field)/T2)) + \
        # nW * dMW2dphi(field) * (T/(pi**2) * Jb1_interpolated(MW2(field)/T2) - \
        # nW * MW2(field)/(T*pi**2) * Jb2_interpolated(MW2(field)/T2)) + \
        # nZ * dMZ2dphi(field) * (T/(pi**2) * Jb1_interpolated(MZ2(field)/T2) - \
        # nZ * MZ2(field)/(T*pi**2) * Jb2_interpolated(MZ2(field)/T2)) + \
        # nt * dMt2dphi(field) * (T/(pi**2) * Jf1_interpolated(Mt2(field)/T2) - \
        # nt * Mt2(field)/(T*pi**2) * Jf2_interpolated(Mt2(field)/T2)) 
    else:
        y += nZp * dMZp2dphi(field) * (T/(pi**2)) * Jb1_interpolated(MZp2(field)/T2)
    return y
# end: Defining second mixed derivative of thermally corrected potential of the model.


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d2VTdT2(const double field, const double T) noexcept:
    cdef double T2 = T * T
    cdef double y = 0
    if field > 0:
        y += nZp * 6 * (T2 / pi**2) * Jb0_interpolated(MZp2(field) / T2) - \
        nZp * MZp2(field) * (5 / pi**2) * Jb1_interpolated(MZp2(field) / T2) + \
        nZp * MZp2(field)**2 / (2 * T2 * pi**2) * Jb2_interpolated(MZp2(field) / T2)# + \
        # nh * 6 * (T2 / pi**2) * Jb0_interpolated(mh2(field) / T2) - \
        # nh * mh2(field) * (5 / pi**2) * Jb1_interpolated(mh2(field) / T2) + \
        # nh * mh2(field)**2 / (2 * T2 * pi**2) * Jb2_interpolated(mh2(field) / T2) + \
        # nW * 6 * (T2 / pi**2) * Jb0_interpolated(MW2(field) / T2) - \
        # nW * MW2(field) * (5 / pi**2) * Jb1_interpolated(MW2(field) / T2) + \
        # nW * MW2(field)**2 / (2 * T2 * pi**2) * Jb2_interpolated(MW2(field) / T2) + \
        # nZ * 6 * (T2 / pi**2) * Jb0_interpolated(MZ2(field) / T2) - \
        # nZ * MZ2(field) * (5 / pi**2) * Jb1_interpolated(MZ2(field) / T2) + \
        # nZ * MZ2(field)**2 / (2 * T2 * pi**2) * Jb2_interpolated(MZ2(field) / T2) + \
        # nt * 6 * (T2 / pi**2) * Jf0_interpolated(Mt2(field) / T2) - \
        # nt * Mt2(field) * (5 / pi**2) * Jf1_interpolated(Mt2(field) / T2) + \
        # nt * Mt2(field)**2 / (2 * T2 * pi**2) * Jf2_interpolated(Mt2(field) / T2)
    else:
        y += nZp * 6 * (T2 / pi**2) * Jb0_interpolated(MZp2(field) / T2)
    return y
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining third derivative with respect to the temperature of
#    thermally corrected potential of the model.
@cython.cdivision(True)
cpdef inline double d3VTdT3(const double field, const double T) noexcept:
    cdef double T2 = T * T
    cdef double T3 = T2 * T
    cdef double T5 = T2 * T3
    cdef double y = 0
    if field > 0:
        y += nZp * 12 * (T / pi**2) * Jb0_interpolated(MZp2(field) / T2) - \
        nZp * (12 / (T * pi**2)) * MZp2(field) * Jb1_interpolated(MZp2(field) / T2) + \
        nZp * MZp2(field)**2 * (9 / (T3 * pi**2)) * Jb2_interpolated(MZp2(field) / T2) - \
        nZp * MZp2(field)**3 * (1 / (T5 * pi**2)) * Jb3_interpolated(MZp2(field) / T2)# + \
        # nh * 12 * (T / pi**2) * Jb0_interpolated(mh2(field) / T2) - \
        # nh * (12 / (T * pi**2)) * mh2(field) * Jb1_interpolated(mh2(field) / T2) + \
        # nh * mh2(field)**2 * (9 / (T3 * pi**2)) * Jb2_interpolated(mh2(field) / T2) - \
        # nh * mh2(field)**3 * (1 / (T5 * pi**2)) * Jb3_interpolated(mh2(field) / T2) + \
        # nW * 12 * (T / pi**2) * Jb0_interpolated(MW2(field) / T2) - \
        # nW * (12 / (T * pi**2)) * MW2(field) * Jb1_interpolated(MW2(field) / T2) + \
        # nW * MW2(field)**2 * (9 / (T3 * pi**2)) * Jb2_interpolated(MW2(field) / T2) - \
        # nW * MW2(field)**3 * (1 / (T5 * pi**2)) * Jb3_interpolated(MW2(field) / T2) + \
        # nZ * 12 * (T / pi**2) * Jb0_interpolated(MZ2(field) / T2) - \
        # nZ * (12 / (T * pi**2)) * MZ2(field) * Jb1_interpolated(MZ2(field) / T2) + \
        # nZ * MZ2(field)**2 * (9 / (T3 * pi**2)) * Jb2_interpolated(MZ2(field) / T2) - \
        # nZ * MZ2(field)**3 * (1 / (T5 * pi**2)) * Jb3_interpolated(MZ2(field) / T2) + \
        # nt * 12 * (T / pi**2) * Jf0_interpolated(Mt2(field) / T2) - \
        # nt * (12 / (T * pi**2)) * MZ2(field) * Jf1_interpolated(Mt2(field) / T2) + \
        # nt * MZp2(field)**2 * (9 / (T3 * pi**2)) * Jf2_interpolated(Mt2(field) / T2) - \
        # nt * MZp2(field)**3 * (1 / (T5 * pi**2)) * Jf3_interpolated(Mt2(field) / T2)
    else:
        y += nZp * 12 * (T / pi**2) * Jb0_interpolated(MZp2(field) / T2) - \
        nZp * (12 / (T * pi**2)) * MZp2(field) * Jb1_interpolated(MZp2(field) / T2)
    return y
# end: Defining third derivative with respect to the temperature of
#      thermally corrected potential of the model.
