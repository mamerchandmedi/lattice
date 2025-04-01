"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""
# %% Defining thermally corrected potential of the model.
cdef double VT(double f1, double f2, double temp)
# end: Defining thermally corrected potential of the model.


# %% Defining derivative with respect to the field of
#    thermally corrected potential of the model.
cdef double dVTdh(double f1, double f2, double temp)
cdef double dVTds(double f1, double f2, double temp)
# end: Defining derivative with respect to the field of
#      thermally corrected potential of the model.


# %% Defining first derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef double dVTdT(double f1, double f2, double temp)
# end: Defining first derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining second mixed derivative of thermally corrected potential of the model.
cdef double d2VTdhdT(double f1, double f2, double temp)
cdef double d2VTdsdT(double f1, double f2, double temp)
# end: Defining second mixed derivative of thermally corrected potential of the model.

# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef double d2VTdT(double f1, double f2, double temp)
# end: Defining second derivative with respect to the temperature of
#      thermally corrected potential of the model.

# %% Defining third derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef double d3VTdT(double f1, double f2, double temp)
# end: Defining third derivative with respect to the temperature of
#      thermally corrected potential of the model.


# %% Defining energy density of the system.
cdef double eps(double f1, double f2, double temp)
# end: Defining energy density of the system.


# %% Defining pressure of the system.
cdef double p(double f1, double f2, double temp)
# end: Defining pressure of the system.


# %% Defining barotropic parameter of the system.
cdef double w(double f1, double f2, double temp)
# end: Defining barotropic parameter of the system.


# %% Defining derivative of the pressure with respect to temperature.
cdef double dpdT(double f1, double f2, double temp)
# end: Defining derivative of the pressure with respect to temperature.


# %% Defining second derivative of the pressure with respect to temperature.
cdef double d2pdT(double f1, double f2, double temp)
# end: Defining second derivative of the pressure with respect to temperature.


# %% Defining derivative of the energy density with respect to temperature.
cdef double depsdT(double f1, double f2, double temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining second derivative of the energy density with respect to temperature.
cdef double d2epsdT(double f1, double f2, double temp)
# end: Defining second derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
cdef double dwdT(double f1, double f2, double temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining second derivative of the barotropic parameter
#    with respect to temperature.
cdef double d2wdT(double f1, double f2, double temp)
# end: Defining second derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining relativistic gamma factor.
cdef double gam(double vel)
# end: Defining relativistic gamma factor.

# %% Defining first derivative of the gamma factor.
cdef double dgamdv(double v)
# end: Defining first derivative of the gamma factor.

# %% Defining friction source for tau.
cdef double Gf_tau(double h, double s, double dhdt, double dsdt, double dhdr, double dsdr, double T,
                   double v)
# end: Defining friction source for tau.


# %% Defining friction source for Z.
cdef double Gf_Z(double h, double s, double dhdt, double dsdt, double dhdr, double dsdr, double T,
                   double v)
# end: Defining friction source for Z.

# %% Defining tau derivative with respect to p.
cdef double dtaudp(double v, double f1, double f2, double temp)
# end: Defining tau derivative with respect to p.

# %% Defining Z derivative with respect to p.
cdef double dZdp(double v, double f1, double f2, double temp)
# end: Defining Z derivative with respect to p.

# %% Defining tau derivative with respect to T.
cdef double dtaudT(double v, double f1, double f2, double temp)
# end: Defining tau derivative with respect to T.

# %% Defining tau derivative with respect to v.
cdef double dtaudv(double v, double f1, double f2, double temp, double p)
# end: Defining tau derivative with respect to v.

# %% Defining v derivative with respect to tau along Z = const.
cdef double dvdtau(double v, double f1, double f2, double temp, double p)    
# end: Defining v derivative with respect to tau along Z = const.

# %% Defining v derivative with respect to Z along tau = const.
cdef double dvdZ(double v, double f1, double f2, double temp, double P)
# end: Defining v derivative with respect to Z along tau = const.

# %% Defining T derivative with respect to tau along Z = const.
cdef double dTdtau(double v, double f1, double f2, double temp, double p)
# end: Defining T derivative with respect to tau along Z = const.

# %% Defining T derivative with respect to Z along tau = const.
cdef double dTdZ(double v, double f1, double f2, double temp, double P)
# end: Defining T derivative with respect to Z along tau = const.

# %% Defining p derivative with respect to tau along Z = const.
cdef double dpdtau(double v, double f1, double f2, double temp, double P)
# end: Defining p derivative with respect to tau along Z = const.

# %% Defining p derivative with respect to Z along tau = const.
cdef double dpdZ(double v, double f1, double f2, double temp, double P)
# end: Defining p derivative with respect to Z along tau = const.

# %% Defining Z derivative with respect to v.
cdef double dZdv(double v, double f1, double f2, double temp)
# end: Defining Z derivative with respect to v.

# %% Defining Z derivative with respect to T.
cdef double dZdT(double v, double f1, double f2, double temp)
# end: Defining Z derivative with respect to T.

# %% Defining Ftau derivative with respect to tau.
cdef double dFtaudtau(double v, double f1, double f2, double temp, double P)
# end: Defining Ftau derivative with respect to tau.

# %% Defining Ftau derivative with respect to Z.
cdef double dFtaudZ(double v, double f1, double f2, double temp, double P)
# end: Defining Ftau derivative with respect to Z.

# %% Defining Gtau derivative with respect to tau.
cdef double dGtaudtau(double v, double h, double s, double T, double P, double dhdt, double dsdt, double dhdr, double dsdr)
# end: Defining Gtau derivative with respect to tau.

# %% Defining Gtau derivative with respect to Z.
cdef double dGtaudZ(double v, double h, double s, double T, double P, double dhdt, double dsdt, double dhdr, double dsdr)
# end: Defining Gtau derivative with respect to Z.

# %% Defining GZ derivative with respect to tau.
cdef double dGZdtau(double v, double h, double s, double T, double P, double dhdt, double dsdt, double dhdr, double dsdr)
# end: Defining Gtau derivative with respect to tau.

# %% Defining GZ derivative with respect to Z.
cdef double dGZdZ(double v, double h, double s, double T, double P, double dhdt, double dsdt, double dhdr, double dsdr)
# end: Defining GZ derivative with respect to Z.

cdef double a
