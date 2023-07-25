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

# %% Import package for dynamical compilation of Cython code.
#import pyximport
#pyximport.install(inplace=True, language_level='3')
# end: Import package for dynamical compilation of Cython code.

from initialization import xf0, yf0, xb0, yb0

# %% Fixing a datatype for numpy arrays.
DTYPE = np.double
ctypedef np.double_t DTYPE_t
# end: Fixing a datatype for numpy arrays.

# Define the low and high functions
cdef double Jf_low(double x):
    return -7.0 / 8.0 * pi ** 4 / 45.0 - pi ** 2 * x ** 2 / 24.0

cdef double Jb_low(double x):
    return -pi ** 4 / 45.0 + pi ** 2 * x ** 2 / 12.0

cdef double Jf_high(double x):
    return sqrt(pi / 2) * x ** (3.0 / 2.0) * exp(sqrt(x))

cdef double Jb_high(double x):
    return -sqrt(pi / 2) * x ** (3.0 / 2.0) * exp(-sqrt(x))

# Spline fitting, Jf
cdef double xfmin = -6.82200203  # -11.2403168
cdef double xfmax = 1.35e3    
# Define the memory views for xf and yf arrays
xf0_np = np.array(xf0, dtype = DTYPE)
cdef double[::1] xf = xf0_np
yf0_np = np.array(yf0, dtype = DTYPE)
cdef double[::1] yf = yf0_np

cdef double dx 
cdef double dy 
cdef double slope 
cdef double x_interp 


@cython.boundscheck(False)  # Disable bounds-checking for faster array access
cdef inline double Jf(double x):
    cdef int i = 0
    cdef int j = len(xf) - 1
    cdef int mid
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
        x_interp = yf[j] + slope * (x - xf[j])
        return x_interp

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
cdef inline double Jb(double x):
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



