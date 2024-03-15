# %% Importing Cython for decorators.
cimport cython
# end: Importing Cython for decorators.


# %% Importing Cython array support.
from cpython cimport array
from cython cimport view
import array
# end: Importing Cython array support.

# %% Import numpy.
import numpy as np
cimport numpy as np
# end: Import numpy.

from libc.math cimport pi
from libc.math cimport exp
from libc.math cimport sqrt
from libc.math cimport floor
from libc.math cimport fabs
from libc.math cimport asinh

# %% Fixing a datatype for numpy arrays.
DTYPE = np.double
ctypedef np.double_t DTYPE_t
# end: Fixing a datatype for numpy arrays.

Jf_dat_path = "thermal_functions/finiteT_f.txt"
dataf = np.loadtxt(Jf_dat_path, dtype = DTYPE)
xf0  = dataf[:, 0]
yf0  = dataf[:, 1]
Jb_dat_path = "thermal_functions/finiteT_b.txt"
datab = np.loadtxt(Jb_dat_path, dtype = DTYPE)
xb0  = datab[:, 0]
yb0  = datab[:, 1]
J_dat_path = "thermal_functions/finiteT.txt"
x0, y0 = np.loadtxt(J_dat_path, dtype = DTYPE, unpack=True)
#x0  = datab[:, 0]
#y0  = datab[:, 1]

# Define the low and high functions
cdef double Jf_low(const double x):
    return -7.0 / 8.0 * pi ** 4 / 45.0 - pi ** 2 * x ** 2 / 24.0

cdef double Jb_low(const double x):
    return -pi ** 4 / 45.0 + pi ** 2 * x ** 2 / 12.0

cdef double Jf_high(const double x):
    return sqrt(pi / 2) * x ** (3.0 / 2.0) * exp(x)

cdef double Jb_high(const double x):
    return sqrt(pi / 2) * x ** (3.0 / 2.0) * exp(x)
    
cdef double J_low(const double x):
    return -2 + x/2
	
cdef double J_high(const double x):
    return  -sqrt(pi/2) * x**(3.0/4.0)*exp(-sqrt(x))

# Spline fitting, Jf
cdef double xfmin = -6.82200203  # -11.2403168
cdef double xfmax = 1.35e3    
# Define the memory views for xf and yf arrays
xf0_np = np.array(xf0, dtype = DTYPE)
cdef double[::1] xf = xf0_np
yf0_np = np.array(yf0, dtype = DTYPE)
cdef double[::1] yf = yf0_np


cdef double slope 
cdef double x_interp 
cdef double coeff = 20 / fabs(xfmin)
cdef double xf_lband = asinh(coeff * xf[0])
cdef double xf_rband = asinh(coeff * xf[-1])
cdef double dxf = (xf_rband - xf_lband)/(len(xf) - 1)
cdef double idxf = 1/dxf
cdef double x_temp


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.initializedcheck(False)
@cython.binding(False)
@cython.nonecheck(False) 
@cython.wraparound(False)
cdef inline double oldJf(const double x):
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double dx 
    cdef double dy 
    if xfmin < x < xfmax:
        x_temp = asinh(x * coeff)
        i = int(floor((x_temp - xf_lband) * idxf))
        j = i + 1
        dx = xf[i] - xf[j]
        dy = yf[i] - yf[j]
        slope = dy / dx
        return yf[j] + slope * (x - xf[j])

    elif x < xfmin:
        return Jf_low(x)
    else:
        return Jf_high(x)


@cython.boundscheck(False)  # Disable bounds-checking for faster array access
cpdef inline double Jf(const double x):
    cdef Py_ssize_t i = 300
    cdef Py_ssize_t j = 999
    cdef Py_ssize_t mid
    cdef double dx 
    cdef double dy
    if xfmin < x < xfmax:
        while i < j:
            mid = (i + j) // 2
            if xf[mid] < x:
                i = mid + 1
            else:
                j = mid

        i = max(i - 1, 0)
        j = i + 1

        dx = xf[i] - xf[j]
        dy = yf[i] - yf[j]
        slope = dy / dx
        return yf[j] + slope * (x - xf[j])

    elif x < xfmin:
        return Jf_low(x)
    else:
        return Jf_high(x)

# Spline fitting, Jb
cdef double xbmin = -3.72402637
cdef double xbmax = 1.41e3

# Define the memory views for xb and yb arrays
xb0_np = np.array(xb0, dtype = DTYPE)
cdef double[::1] xb = xb0_np
yb0_np = np.array(yb0, dtype = DTYPE)
cdef double[::1] yb = yb0_np


@cython.boundscheck(False)  # Disable bounds-checking for faster array access
cpdef inline double Jb(const double x):
    cdef int i = 0
    cdef int j = len(xb) - 1
    cdef int mid
    if xbmin < x < xbmax:
        while i < j:
            mid = (i + j) // 2
            if xb[mid] < x:
                i = mid + 1
            else:
                j = mid

        i = max(i - 1, 0)
        j = i + 1

        dx = xb[i] - xb[j]
        dy = yb[i] - yb[j]
        slope = dy / dx
        x_interp = yb[j] + slope * (x - xb[j])
        return x_interp

    elif x < xbmin:
        return Jb_low(x)
    else:
        return Jb_high(x)

# Spline fitting, J
cdef double xmin = 1.e-3
cdef double xmax = 1.e3


# Define the memory views for x and y arrays
x0_np = np.array(x0, dtype = DTYPE)
cdef double[::1] xi = x0_np
y0_np = np.array(y0, dtype = DTYPE)
cdef double[::1] yi = y0_np

@cython.boundscheck(False)  # Disable bounds-checking for faster array access
cpdef inline double J(const double x):
    cdef int i = 0
    cdef int j = len(xi) - 1
    cdef int mid
    if xmin < x < xmax:
        while i < j:
            mid = (i + j) // 2
            if xi[mid] < x:
                i = mid + 1
            else:
                j = mid

        i = max(i - 1, 0)
        j = i + 1

        dx = xi[i] - xi[j]
        dy = yi[i] - yi[j]
        slope = dy / dx
        x_interp = yi[j] + slope * (x - xi[j])
        return x_interp

    elif x < xmin:
        return J_low(x)
    else:
        return J_high(x)


