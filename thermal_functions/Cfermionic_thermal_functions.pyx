# cython: infer_types=True

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

from pathlib import Path

from scipy.integrate import quad


from libc.math cimport pi
from libc.math cimport exp
from libc.math cimport sqrt
from libc.math cimport pow
from libc.math cimport floor
from libc.math cimport fabs
from libc.math cimport log
from libc.math cimport asinh

cdef extern from "numpy/npy_math.h":
    double inf "NPY_INFINITY"

from .fermionic_thermal_functions import zmin, zmax, intervals, rescaling


from .fermionic_thermal_functions_jacobian cimport jacobian_1_0, jacobian_1_1, jacobian_2_0, jacobian_2_1, jacobian_2_2, jacobian_3_0, jacobian_3_1, jacobian_3_2, jacobian_3_3


# %% Fixing a datatype for numpy arrays.
DTYPE = np.double
ctypedef np.double_t DTYPE_t
# end: Fixing a datatype for numpy arrays.

file_path = str(Path(__file__).parent)
data_path = file_path + '/fermionic_interpolation_coefficients.npy'
coefficients_data = np.load(data_path)
coefficients = coefficients_data[0]
coefficients_0 = coefficients_data[1]
coefficients_1 = coefficients_data[2]
coefficients_2 = coefficients_data[3]
coefficients_3 = coefficients_data[4]
cdef double[::1] coefficients_num = coefficients
cdef double[::1] coefficients_num_0 = coefficients_0
cdef double[::1] coefficients_num_1 = coefficients_1
cdef double[::1] coefficients_num_2 = coefficients_2
cdef double[::1] coefficients_num_3 = coefficients_3

intervals_num_np = np.array(intervals, dtype=DTYPE)
cdef double[::1] intervals_num = intervals_num_np
cdef int intervals_len = len(intervals_num_np)


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline int bisection(const double x) noexcept:
    cdef int i = 0
    cdef int j = intervals_len - 1
    cdef int mid
    i = 0
    j = intervals_len - 1
    mid = (i+j) // 2
    while j - i > 1:
        if x >= intervals_num[mid]:
            i = mid
        else:
            j = mid
        mid = (i+j) // 2
    return i


cdef int p = 8
cdef double rescaling_num = rescaling
cdef double zmin_num = zmin
cdef double zmax_num = zmax


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double interpolated_function_0(const double z) noexcept:
    cdef int i, j
    i = bisection(z)
    j = i+1
    cdef double arg = (z - intervals_num[i]) / (intervals_num[j] - intervals_num[j-1])
    cdef double result = 0
    cdef int m
    for m in range(p+1):
        result += coefficients_num_0[(p+1) * i + m] * arg**m
    return result


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double interpolated_function_1(const double z) noexcept:
    cdef int i, j
    i = bisection(z)
    j = i+1
    cdef double arg = (z - intervals_num[i]) / (intervals_num[j] - intervals_num[j-1])
    cdef double result = 0
    cdef int m
    for m in range(1, p+1):
        result += coefficients_num_1[(p+1) * i + m] * arg**(m-1)
    return result


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double interpolated_function_2(const double z) noexcept:
    cdef int i, j
    i = bisection(z)
    j = i+1
    cdef double arg = (z - intervals_num[i]) / (intervals_num[j] - intervals_num[j-1])
    cdef double result = 0
    cdef int m
    for m in range(2, p+1):
        result += coefficients_num_2[(p+1) * i + m] * arg**(m-2)
    return result


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double interpolated_function_3(const double z) noexcept:
    cdef int i, j
    i = bisection(z)
    j = i+1
    cdef double arg = (z - intervals_num[i]) / (intervals_num[j] - intervals_num[j-1])
    cdef double result = 0
    cdef int m
    for m in range(3, p+1):
        result += coefficients_num_3[(p+1) * i + m] * arg**(m-3)
    return result


@cython.cdivision(True)
cdef double Jf0_low(const double x) noexcept:
    return sqrt(pi/2.) * pow(x,0.75) * exp(-sqrt(x))


@cython.cdivision(True)
cdef double Jf1_low(const double x) noexcept:
    return sqrt(pi/2)/4. * exp(-sqrt(x)) * (-3. + 2. * sqrt(x)) / pow(x, 0.25)


@cython.cdivision(True)
cdef double Jf2_low(const double x) noexcept:
    return sqrt(pi/2.)/16. * exp(-sqrt(x)) * (3. + 8. * sqrt(x) - 4. * x) / pow(x, 1.25)


@cython.cdivision(True)
cdef double Jf3_low(const double x) noexcept:
    return sqrt(pi/2.)/64. * exp(-sqrt(x)) * (-15. - 30. * sqrt(x) - 12. * x + 8. * pow(x,1.5)) / pow(x,2.25)


cdef double E_gamma = 0.57721566490153286
cdef double a_f = pi**2 * exp(1.5 - 2 * E_gamma)

@cython.cdivision(True)
cdef double Jf0_high(const double x) noexcept:
    if(x>1):
        return -7 * pi**4 / 360. + x * pi**2 / 24. + x**2 * log(x / a_f) / 32.
    elif(x>0):
        return -7 * pi**4 / 360. + x * pi**2 / 24. + x * log((x / a_f)**x) / 32.
    else:
        return -7 * pi**4 / 360.


@cython.cdivision(True)
cdef double Jf1_high(const double x) noexcept:
    if(x>1):
        return pi**2 / 24. + x * log(x / a_f) / 16. + x / 32.
    elif(x>0):
        return pi**2 / 24. + log((x / a_f)**x) / 16. + x / 32.
    else:
        return pi**2 / 24.


@cython.cdivision(True)
cdef double Jf2_high(const double x) noexcept:
    return log(x / a_f) / 16. + 3. / 32.


@cython.cdivision(True)
cdef double Jf3_high(const double x) noexcept:
    return 1. / x / 16.


cdef double rescaling_num_total = rescaling_num/zmin_num


@cython.cdivision(True)
cpdef double Jf0_interpolated(const double z) noexcept:
    if (z <= zmin_num):
        return Jf0_high(z)
    elif (z >= zmax_num):
        return Jf0_low(z)
    else:
        return interpolated_function_0(asinh(rescaling_num_total * z))


@cython.cdivision(True)
cpdef double Jf1_interpolated(const double z) noexcept:
    if (z <= zmin_num):
        return Jf1_high(z)
    elif (z >= zmax_num):
        return Jf1_low(z)
    else:
        return interpolated_function_0(asinh(rescaling_num_total * z))*jacobian_1_0(z) + interpolated_function_1(asinh(rescaling_num_total * z))*jacobian_1_1(z)


@cython.cdivision(True)
cpdef double Jf2_interpolated(const double z) noexcept:
    if (z <= zmin_num):
        return Jf2_high(z)
    elif (z >= zmax_num):
        return Jf2_low(z)
    else:
        return interpolated_function_0(asinh(rescaling_num_total * z))*jacobian_2_0(z) + interpolated_function_1(asinh(rescaling_num_total * z))*jacobian_2_1(z) + interpolated_function_2(asinh(rescaling_num_total * z))*jacobian_2_2(z)


@cython.cdivision(True)
cpdef double Jf3_interpolated(const double z) noexcept:
    if (z <= zmin_num):
        return Jf3_high(z)
    elif (z >= zmax_num):
        return Jf3_low(z)
    else:
        return interpolated_function_0(asinh(rescaling_num_total * z))*jacobian_3_0(z) + interpolated_function_1(asinh(rescaling_num_total * z))*jacobian_3_1(z) + interpolated_function_2(asinh(rescaling_num_total * z))*jacobian_3_2(z) + interpolated_function_3(asinh(rescaling_num_total * z))*jacobian_3_3(z)
