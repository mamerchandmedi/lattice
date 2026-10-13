# Declare the functions

cdef double Jf0_low(const double x)
cdef double Jf1_low(const double x)
cdef double Jf2_low(const double x)
cdef double Jf3_low(const double x)

cdef double Jf0_high(const double x)
cdef double Jf1_high(const double x)
cdef double Jf2_high(const double x)
cdef double Jf3_high(const double x)

cpdef double Jf0_interpolated(const double z)
cpdef double Jf1_interpolated(const double z)
cpdef double Jf2_interpolated(const double z)
cpdef double Jf3_interpolated(const double z)
