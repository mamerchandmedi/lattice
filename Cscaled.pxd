cdef double VT_mod(const double field, const double temp)

cdef double dVTdT_mod(const double field, const double temp)

cdef double dVTdphi_mod(const double field, const double temp)

cdef double d2VTdphidT_mod(const double field, const double temp)

cdef double d2VTdT2_mod(const double field, const double temp)

cdef double d3VTdT3_mod(const double field, const double temp)

cdef void tauZphi2Tpv_mod(const double tau, const double Z, const double phi,
                          double* T, double* P, double* v)
