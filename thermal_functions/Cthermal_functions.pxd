
# Declare the functions
cdef double Jf_low(const double x)
cdef double Jb_low(const double x)
cdef double J_low(const double x)
cdef double Jf_high(const double x)
cdef double Jb_high(const double x)
cdef double J_high(const double x)


cpdef double Jf(const double x)
cdef double oldJf(const double x)
cpdef double Jb(const double x)
cpdef double J(const double x)

