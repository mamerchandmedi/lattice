# Declare the functions

cdef double Jb0_low(const double x)
cdef double Jb1_low(const double x)
cdef double Jb2_low(const double x)
cdef double Jb3_low(const double x)

cdef double Jb0_high(const double x)
cdef double Jb1_high(const double x)
cdef double Jb2_high(const double x)
cdef double Jb3_high(const double x)

cpdef double Jb0_interpolated(const double z)
cpdef double Jb1_interpolated(const double z)
cpdef double Jb2_interpolated(const double z)
cpdef double Jb3_interpolated(const double z)
