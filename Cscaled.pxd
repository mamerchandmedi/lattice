cdef double VT_mod(double field, double temp)

cdef double dVTdT_mod(double field, double temp)

cdef double dVTdphi_mod(double field, double temp)

cdef double d2VTdphidT_mod(double field, double temp)

cdef double d2VTdT2_mod(double field, double temp)

cdef double d3VTdT3_mod(double field, double temp)

cdef void tauZphi2Tpv_mod(double tau, double Z, double phi,
                          double* T, double* P, double* v)
