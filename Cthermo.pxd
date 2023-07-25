"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""

cdef double V0(double phi)
cdef double m2(double phi,double T)
cdef double mAl2(double phi, double T)
cdef double mAt2(double phi, double T)
cdef double Vtherm(double phi, double T)
cdef double VCW(double phi, double T)
# %% Defining thermally corrected potential of the model.
cdef double VT(double field, double temp)
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
cdef double dVTdphi(double field, double temp)
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef double dVTdT(double field, double temp)
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining second mixed derivative of thermally corrected potential of the model.
cdef double d2VTdphidT(double field, double temp)
# end: Defining second mixed derivative of thermally corrected potential of the model.

# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef double d2VTdT(double field, double temp)
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining third derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef double d3VTdT(double field, double temp)
# end: Defining third derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining energy density of the system.
cdef double eps(double field, double temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
cdef double p(double field, double temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
cdef double w(double field, double temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
cdef double dpdT(double field, double temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining second derivative of the pressure with respect to temperature.
cdef double d2pdT(double field, double temp)
# end: Defining second derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
cdef double depsdT(double field, double temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining second derivative of the energy density with respect to temperature.
cdef double d2epsdT(double field, double temp)
# end: Defining second derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
cdef double dwdT(double field, double temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining second derivative of the barotropic parameter
#    with respect to temperature.
cdef double d2wdT(double field, double temp)
# end: Defining second derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining relativistic gamma factor.
cdef double gam(double vel)
# end: Defining relativistic gamma factor.

# %% Defining first derivative of the gamma factor.
cdef double dgamdv(double v)
# end: Defining first derivative of the gamma factor.

# %% Defining friction source for tau.
cdef double Gf_tau(double phi, double dphidt, double dphidr, double T,
                   double v)
# end: Defining friction source for tau.


# %% Defining friction source for Z.
cdef double Gf_Z(double phi, double dphidt, double dphidr, double T, double v)
# end: Defining friction source for Z.

# %% Defining tau derivative with respect to p.
cdef double dtaudp(double v, double field, double temp)
# end: Defining tau derivative with respect to p.

# %% Defining Z derivative with respect to p.
cdef double dZdp(double v, double field, double temp)
# end: Defining Z derivative with respect to p.

# %% Defining tau derivative with respect to T.
cdef double dtaudT(double v, double field, double temp)
# end: Defining tau derivative with respect to T.

# %% Defining tau derivative with respect to v.
cdef double dtaudv(double v, double field, double temp, double p)
# end: Defining tau derivative with respect to v.

# %% Defining v derivative with respect to tau along Z = const.
cdef double dvdtau(double v, double field, double temp, double p)    
# end: Defining v derivative with respect to tau along Z = const.

# %% Defining v derivative with respect to Z along tau = const.
cdef double dvdZ(double v, double field, double temp, double P)
# end: Defining v derivative with respect to Z along tau = const.

# %% Defining T derivative with respect to tau along Z = const.
cdef double dTdtau(double v, double field, double temp, double p)
# end: Defining T derivative with respect to tau along Z = const.

# %% Defining T derivative with respect to Z along tau = const.
cdef double dTdZ(double v, double field, double temp, double P)
# end: Defining T derivative with respect to Z along tau = const.

# %% Defining p derivative with respect to tau along Z = const.
cdef double dpdtau(double v, double field, double temp, double P)
# end: Defining p derivative with respect to tau along Z = const.

# %% Defining p derivative with respect to Z along tau = const.
cdef double dpdZ(double v, double field, double temp, double P)
# end: Defining p derivative with respect to Z along tau = const.

# %% Defining Z derivative with respect to v.
cdef double dZdv(double v, double field, double temp)
# end: Defining Z derivative with respect to v.

# %% Defining Z derivative with respect to T.
cdef double dZdT(double v, double field, double temp)
# end: Defining Z derivative with respect to T.

# %% Defining Ftau derivative with respect to tau.
cdef double dFtaudtau(double v, double field, double temp, double P)
# end: Defining Ftau derivative with respect to tau.

# %% Defining Ftau derivative with respect to Z.
cdef double dFtaudZ(double v, double field, double temp, double P)
# end: Defining Ftau derivative with respect to Z.

# %% Defining Gtau derivative with respect to tau.
cdef double dGtaudtau(double v, double phi, double T, double P, double dphidt, double dphidr)
# end: Defining Gtau derivative with respect to tau.

# %% Defining Gtau derivative with respect to Z.
cdef double dGtaudZ(double v, double phi, double T, double P, double dphidt, double dphidr)
# end: Defining Gtau derivative with respect to Z.

# %% Defining GZ derivative with respect to tau.
cdef double dGZdtau(double v, double phi, double T, double P, double dphidt, double dphidr)
# end: Defining Gtau derivative with respect to tau.

# %% Defining GZ derivative with respect to Z.
cdef double dGZdZ(double v, double phi, double T, double P, double dphidt, double dphidr)
# end: Defining GZ derivative with respect to Z.

cdef double a
