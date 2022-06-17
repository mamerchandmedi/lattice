"""
Created on Tue Apr 26 10:37:54 2022

@author: Tomasz Krajewski
"""
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


# %% Defining second derivative with respect to the temperature of
#    thermally corrected potential of the model.
cdef double d2VTdT(double field, double temp)
# end: Defining second derivative with respect to the temperature of
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


# %% Defining derivative of the energy density with respect to temperature.
cdef double depsdT(double field, double temp)
# end: Defining derivative of the energy density with respect to temperature.


# %% Defining derivative of the barotropic parameter
#    with respect to temperature.
cdef double dwdT(double field, double temp)
# end: Defining derivative of the barotropic parameter
#      with respect to temperature.


# %% Defining relativistic gamma factor.
cdef double gam(double vel)
# end: Defining relativistic gamma factor.


# %% Defining friction source for tau.
cdef double Gf_tau(double phi, double dphidt, double dphidr, double T,
                   double v)
# end: Defining friction source for tau.


# %% Defining friction source for Z.
cdef double Gf_Z(double phi, double dphidt, double dphidr, double T, double v)
# end: Defining friction source for Z.
