# Declare the functions

cdef int bisection(const double x)

cdef double interpolated_function_0(const double z)
cdef double interpolated_function_1(const double z)
cdef double interpolated_function_2(const double z)
cdef double interpolated_function_3(const double z)

cdef double I0(const double k, const double m)
cdef double I1(const double k, const double m)
cdef double I2(const double k, const double m)
cdef double I3(const double k, const double m)

cdef double J0_full(const double x2)
cdef double J1_full(const double x2)
cdef double J2_full(const double x2)
cdef double J3_full(const double x2)

cdef double J0_low(const double x)
cdef double J1_low(const double x)
cdef double J2_low(const double x)
cdef double J3_low(const double x)

cdef double J0_high(const double x)
cdef double J1_high(const double x)
cdef double J2_high(const double x)
cdef double J3_high(const double x)

cpdef double J0_interpolated(const double z)
cpdef double J1_interpolated(const double z)
cpdef double J2_interpolated(const double z)
cpdef double J3_interpolated(const double z)
