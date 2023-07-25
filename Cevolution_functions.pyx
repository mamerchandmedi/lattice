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

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
import copy as cp

# %% Importing parameters of the model.
from parameters import eta
# end: Importing parameters of the model.
# %% Import matrices of lattice definition.
from lattice import dt, r, N
# end: Import matrices of lattice definition.

# %% Import matrices of wave equation integration scheme.
from matrices_field import laplacian_matrix, kinetic_matrix, gradient_matrix, derivative_matrix_plus, derivative_matrix_minus
# end: Import matrices of wave equation integration scheme.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import pressure_matrix, mass_matrix, \
    sources_matrix, advection_matrix, gradient_matrix_plus, gradient_matrix_minus
# end: Import matrices of the high order integration scheme.

# %% Import package for dynamical compilation of Cython code.
#import pyximport
#pyximport.install(inplace=True, language_level='3')
# end: Import package for dynamical compilation of Cython code.

# %% Import termodynamically corrected quantities.
from Cthermo cimport gam, Gf_tau, Gf_Z, dFtaudtau, dFtaudZ, dGtaudtau, dGtaudZ, dGZdtau, dGZdZ, dpdtau, dpdZ
# end: Import termodynamically corrected quantities.

# %% Import rescaled quantities.
from Cscaled cimport dVTdphi_mod
from Cscaled cimport tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Fixing a datatype for numpy arrays.
DTYPE = np.double
ctypedef np.double_t DTYPE_t
# end: Fixing a datatype for numpy arrays.

cdef int n_N = N
cdef double n_dt = dt
cdef double n_eta = eta


r_num_np = np.array(r, dtype=float)
cdef double[::1] r_num = r_num_np

laplacian_num_np = np.array(laplacian_matrix, dtype=DTYPE)
cdef double[:, ::1] laplacian_num = laplacian_num_np

kinetic_num_np = np.array(kinetic_matrix, dtype=DTYPE)
cdef double[::1] kinetic_num = kinetic_num_np

derivative_plus_num_np = np.array(derivative_matrix_plus, dtype=DTYPE)
cdef double[::1] derivative_plus_num = derivative_plus_num_np

derivative_minus_num_np = np.array(derivative_matrix_minus, dtype=DTYPE)
cdef double[::1] derivative_minus_num = derivative_minus_num_np

gradient_np = np.array(gradient_matrix, dtype=DTYPE)
cdef double[:, ::1] gradient_num = gradient_np

pressure_num_np = np.array(pressure_matrix, dtype=DTYPE)
cdef double[:, ::1] pressure_num = pressure_num_np

mass_num_np = np.array(mass_matrix, dtype=DTYPE)
cdef double[::1] mass_num = mass_num_np

gradient_num_plus_np = np.array(gradient_matrix_plus, dtype=DTYPE)
cdef double[::1] gradient_num_plus = gradient_num_plus_np

gradient_num_minus_np = np.array(gradient_matrix_minus, dtype=DTYPE)
cdef double[::1] gradient_num_minus = gradient_num_minus_np

sources_num_np = np.array(sources_matrix, dtype=DTYPE)
cdef double[:, ::1] sources_num = sources_num_np

advection_num_np = np.array(advection_matrix, dtype=DTYPE)
cdef double[:, ::1] advection_num = advection_num_np

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def update_Tpv(np.ndarray[DTYPE_t, ndim=1] T,
               np.ndarray[DTYPE_t, ndim=1] P,
               np.ndarray[DTYPE_t, ndim=1] v,
               np.ndarray[DTYPE_t, ndim=1] phi,
               np.ndarray[DTYPE_t, ndim=1] tau,
               np.ndarray[DTYPE_t, ndim=1] Z):
    cdef int j
    for j in range(n_N):
        tauZphi2Tpv_mod(tau[j], Z[j], phi[j], &(T[j]), &(P[j]), &(v[j]))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def evaluate_fluxes(np.ndarray[DTYPE_t, ndim=1] tau,
                    np.ndarray[DTYPE_t, ndim=1] Z,
                    np.ndarray[DTYPE_t, ndim=1] P,
                    np.ndarray[DTYPE_t, ndim=1] v,
                    np.ndarray[DTYPE_t, ndim=1] Ftau,
                    np.ndarray[DTYPE_t, ndim=1] FZ):
    cdef int j
    for j in range(n_N):
        Ftau[j] = (tau[j] + P[j]) * v[j]
        FZ[j] = Z[j] * v[j]

#@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def evaluate_psi(np.ndarray[DTYPE_t, ndim=1] phi,
                 np.ndarray[DTYPE_t, ndim=1] psi):
    cdef int j
    cdef double rp 
    rp = (3/4) * (r[1]**4 - r[0]**4) / (r[1]**3 - r[0]**3)
    psi[0] = derivative_minus_num[1]*(phi[1]-phi[0])*(rp - r[0])/(r[1] - r[0])
    for j in range(1, n_N-1):
        rp = (3/4) * (r[j+1]**4 - r[j]**4) / (r[j+1]**3 - r[j]**3)
        psi[j] = (derivative_plus_num[j]*(phi[j]-phi[j-1])*(r[j+1] - rp)/(r[j+1] - r[j]) +
                  derivative_minus_num[j+1]*(phi[j+1]-phi[j])*(rp - r[j])/(r[j+1] - r[j]))
    rp = (3/4) * (r[n_N]**4 - r[n_N-1]**4) / (r[n_N]**3 - r[n_N-1]**3)
    psi[n_N-1] = derivative_plus_num[n_N-1]*(phi[n_N-1]-phi[n_N-2])*(r[n_N] - rp)/(r[n_N] - r[n_N-1])
    

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
def ho_update_taudot(np.ndarray[DTYPE_t, ndim=1] phi,
                     np.ndarray[DTYPE_t, ndim=1] phidot,
                     np.ndarray[DTYPE_t, ndim=1] psi,
                     np.ndarray[DTYPE_t, ndim=1] T,
                     np.ndarray[DTYPE_t, ndim=1] v,
                     np.ndarray[DTYPE_t, ndim=1] Ftau,
                     np.ndarray[DTYPE_t, ndim=1] taudot,
                     np.ndarray[DTYPE_t, ndim=1] Gtau):
                     
    Gtau[0] = sources_num[0][0] * Gf_tau(phi[0], phidot[0], psi[0], T[0], v[0])
    taudot[0] = (advection_num[0][0] * Ftau[0] +
                 advection_num[0][1] * Ftau[1] +
                 Gtau[0]) / mass_num[0]
    cdef int j
    for j in range(1, n_N-1):
        Gtau[j] = sources_num[j][0] * Gf_tau(phi[j], phidot[j], psi[j], T[j], v[j])
        taudot[j] = (advection_num[j][-1] * Ftau[j - 1] +
                     advection_num[j][0] * Ftau[j] +
                     advection_num[j][1] * Ftau[j + 1] +
                     Gtau[j]) / mass_num[j]
    Gtau[n_N-1] = sources_num[n_N-1][0] * \
        Gf_tau(phi[n_N-1], 0, psi[n_N-1], T[n_N-1], v[n_N-1])
    taudot[n_N-1] = (advection_num[n_N-1][-1] * Ftau[n_N - 2] +
                 advection_num[n_N-1][0] * Ftau[n_N-1] +
                 Gtau[n_N-1]) / mass_num[n_N-1]


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
def ho_update_Zdot(np.ndarray[DTYPE_t, ndim=1] phi,
                   np.ndarray[DTYPE_t, ndim=1] phidot,
                   np.ndarray[DTYPE_t, ndim=1] psi,
                   np.ndarray[DTYPE_t, ndim=1] T,
                   np.ndarray[DTYPE_t, ndim=1] P,
                   np.ndarray[DTYPE_t, ndim=1] v,
                   np.ndarray[DTYPE_t, ndim=1] FZ,
                   np.ndarray[DTYPE_t, ndim=1] Zdot,
                   np.ndarray[DTYPE_t, ndim=1] GZ):

    GZ[0] = sources_num[0][0] * Gf_Z(phi[0], phidot[0], psi[0], T[0], v[0])
    Zdot[0] = ((advection_num[0][0] * FZ[0] +
               advection_num[0][1] * FZ[1] +
               gradient_num_minus[0] * derivative_minus_num[1] * (P[1] - P[0]) +
               GZ[0]) / mass_num[0])
    cdef int j
    for j in range(1, n_N-1):
        GZ[j] = sources_num[j][0] * Gf_Z(phi[j], phidot[j], psi[j], T[j], v[j])
        Zdot[j] = ((advection_num[j][-1] * FZ[j - 1] +
                   advection_num[j][0] * FZ[j] +
                   advection_num[j][1] * FZ[j + 1] +
                   gradient_num_plus[j] * derivative_plus_num[j] * (P[j] - P[j-1]) +
                   gradient_num_minus[j] * derivative_minus_num[j+1] * (P[j+1] - P[j]) +  
                   GZ[j]) / mass_num[j])
    GZ[n_N-1] = (sources_num[n_N-1][0] *
             Gf_Z(phi[n_N-1], 0, psi[n_N-1], T[n_N-1], v[n_N-1]))
    Zdot[n_N-1] = ((advection_num[n_N-1][-1] * FZ[n_N - 2] +
               advection_num[n_N-1][0] * FZ[n_N-1] +
               gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (P[n_N-1] - P[n_N-2]) + 
               GZ[n_N-1]) / mass_num[n_N-1])


@cython.cdivision(True)
#@cython.boundscheck(False)  # Deactivate bounds checking
def lo_update_taudot(np.ndarray[DTYPE_t, ndim=1] phi,
                     np.ndarray[DTYPE_t, ndim=1] phidot,
                     np.ndarray[DTYPE_t, ndim=1] psi,
                     np.ndarray[DTYPE_t, ndim=1] T,
                     np.ndarray[DTYPE_t, ndim=1] v,
                     np.ndarray[DTYPE_t, ndim=1] Ftau,
                     np.ndarray[DTYPE_t, ndim=2] lo_advection_matrix,
                     np.ndarray[DTYPE_t, ndim=1] taudot,
                     np.ndarray[DTYPE_t, ndim=1] Gtau):
    Gtau[0] = sources_num[0][0] * Gf_tau(phi[0], phidot[0], 0, T[0], v[0])
    taudot[0] = (lo_advection_matrix[0][0] * Ftau[0] +
                 lo_advection_matrix[0][1] * Ftau[1] +
                 Gtau[0]) / mass_num[0]
    cdef int j
    for j in range(1, n_N-1):
        Gtau[j] = sources_num[j][0] * Gf_tau(phi[j], phidot[j], psi[j], T[j], v[j])
        taudot[j] = (lo_advection_matrix[j][-1] * Ftau[j - 1] +
                     lo_advection_matrix[j][0] * Ftau[j] +
                     lo_advection_matrix[j][1] * Ftau[j + 1] +
                     Gtau[j]) / mass_num[j]
    Gtau[n_N-1] = (sources_num[n_N-1][0] *
                 Gf_tau(phi[n_N-1], 0, psi[n_N-1], T[n_N-1], v[n_N-1]))
    taudot[n_N-1] = (lo_advection_matrix[n_N-1][-1] * Ftau[n_N - 2] +
                 lo_advection_matrix[n_N-1][0] * Ftau[n_N-1] +
                 Gtau[n_N-1]) / mass_num[n_N-1]


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
def lo_update_Zdot(np.ndarray[DTYPE_t, ndim=1] phi,
                   np.ndarray[DTYPE_t, ndim=1] phidot,
                   np.ndarray[DTYPE_t, ndim=1] psi,
                   np.ndarray[DTYPE_t, ndim=1] T,
                   np.ndarray[DTYPE_t, ndim=1] P,
                   np.ndarray[DTYPE_t, ndim=1] v,
                   np.ndarray[DTYPE_t, ndim=1] FZ,
                   np.ndarray[DTYPE_t, ndim=2] lo_advection_matrix,
                   np.ndarray[DTYPE_t, ndim=1] Zdot,
                   np.ndarray[DTYPE_t, ndim=1] GZ):
    GZ[0] = sources_num[0][0] * Gf_Z(phi[0], phidot[0], psi[0], T[0], v[0])
    Zdot[0] = (lo_advection_matrix[0][0] * FZ[0] +
               lo_advection_matrix[0][1] * FZ[1] +
               gradient_num_minus[0] * derivative_minus_num[1] * (P[1] - P[0]) +
               GZ[0]) / mass_num[0]
    cdef int j
    for j in range(1, n_N-1):
        GZ[j] = sources_num[j][0] * Gf_Z(phi[j], phidot[j], psi[j], T[j], v[j])
        Zdot[j] = (lo_advection_matrix[j][-1] * FZ[j - 1] +
                   lo_advection_matrix[j][0] * FZ[j] +
                   lo_advection_matrix[j][1] * FZ[j + 1] +
                   gradient_num_plus[j] * derivative_plus_num[j] * (P[j] - P[j-1]) +
                   gradient_num_minus[j] * derivative_minus_num[j+1] * (P[j+1] - P[j]) +  
                   GZ[j]) / mass_num[j]
    GZ[n_N-1] = sources_num[n_N-1][0] * Gf_Z(phi[n_N-1], 0, psi[n_N-1], T[n_N-1], v[n_N-1])
    Zdot[n_N-1] = (lo_advection_matrix[n_N-1][-1] * FZ[n_N - 2] +
               lo_advection_matrix[n_N-1][0] * FZ[n_N-1] +
               gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (P[n_N-1] - P[n_N-2]) + 
               GZ[n_N-1]) / mass_num[n_N-1]

cdef array.array U_old_array = array.array('d')
array.resize(U_old_array, 2*n_N)
cdef double[::1] U_old = U_old_array

cdef array.array U_array = array.array('d')
array.resize(U_array, 2*n_N)
cdef double[::1] U = U_array

cdef array.array fU_array = array.array('d')
array.resize(fU_array, 2*n_N)
cdef double[::1] fU = fU_array

cdef array.array dU_array = array.array('d')
array.resize(dU_array, 2*n_N)
cdef double[::1] dU = dU_array


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
def lo_newton(np.ndarray[DTYPE_t, ndim=1] tau_old,
              np.ndarray[DTYPE_t, ndim=1] tau,
              np.ndarray[DTYPE_t, ndim=1] taudot,
              np.ndarray[DTYPE_t, ndim=1] Z_old,
              np.ndarray[DTYPE_t, ndim=1] Z,
              np.ndarray[DTYPE_t, ndim=1] Zdot,
              np.ndarray[DTYPE_t, ndim=2] Tf,
              np.ndarray[DTYPE_t, ndim=2] Tf_banded,
              np.ndarray[DTYPE_t, ndim=1] phi,
              np.ndarray[DTYPE_t, ndim=1] phidot,
              np.ndarray[DTYPE_t, ndim=1] psi,
              np.ndarray[DTYPE_t, ndim=1] T,
              np.ndarray[DTYPE_t, ndim=1] v,
              np.ndarray[DTYPE_t, ndim=1] P,
              np.ndarray[DTYPE_t, ndim=2] lo_advection_matrix):
              
    #cdef double[::1] U_old = np.zeros(2*n_N)
    #cdef double[::1] U = np.zeros(2*n_N)
    #cdef double[::1] fU = np.zeros(2*n_N)
    cdef int j 
    for j in range(n_N):
        U_old[2*j] = tau_old[j]
        U_old[2*j + 1] = Z_old[j]
        U[2*j] = tau[j]
        U[2*j+1] = Z[j]
        fU[2*j] = tau[j] - tau_old[j] + dt * taudot[j]  
        fU[2*j + 1] = Z[j] - Z_old[j] + dt * Zdot[j]

    Tf[0, 0] = 1 + dt * (lo_advection_matrix[0][0] * dFtaudtau(v[0], phi[0], T[0], P[0]) +
                  sources_num[0][0] * dGtaudtau(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    Tf[1, 0] = dt * (lo_advection_matrix[0][0] * dFtaudZ(v[0], phi[0], T[0], P[0]) +
                  sources_num[0][0] * dGtaudZ(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    Tf[2, 0] = dt * (lo_advection_matrix[0][1] * dFtaudtau(v[1], phi[1], T[1], P[1]))/mass_num[0]
    Tf[3, 0] = dt * (lo_advection_matrix[0][1] * dFtaudZ(v[1], phi[1], T[1], P[1]))/mass_num[0]
        
     
    Tf[0, 1] = dt * (gradient_num_plus[0] * derivative_plus_num[0] * (dpdtau(v[0], phi[0], T[0], P[0])) + gradient_num_minus[0] * derivative_minus_num[1] * (- dpdtau(v[0], phi[0], T[0], P[0])) + sources_num[0][0] * dGZdtau(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    Tf[1, 1] = 1 + dt * (lo_advection_matrix[0][0] * 1  + gradient_num_plus[0] * derivative_plus_num[0] * (dpdZ(v[0], phi[0], T[0], P[0])) + gradient_num_minus[0] * derivative_minus_num[1] * (- dpdZ(v[0], phi[0], T[0], P[0])) + sources_num[0][0] * dGZdZ(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    Tf[2, 1] = dt * ((gradient_num_minus[0] * derivative_minus_num[1] * (dpdtau(v[1], phi[1], T[1], P[1])))/mass_num[0])
    Tf[3, 1] = dt * (lo_advection_matrix[0][1] * 1 +
                            (gradient_num_minus[0] * derivative_minus_num[1] * (dpdZ(v[1], phi[1], T[1], P[1]))) / mass_num[0])                 
    for j in range(1, n_N-1):
        Tf[2*j-2, 2*j] = dt * (lo_advection_matrix[j][-1] * dFtaudtau(v[j-1], phi[j-1], T[j-1], P[j-1]))/mass_num[j]
        Tf[2*j-1, 2*j] = dt * (lo_advection_matrix[j][-1] * dFtaudZ(v[j-1], phi[j-1], T[j-1], P[j-1]))/mass_num[j]
        Tf[2*j, 2*j] = 1 + dt * (lo_advection_matrix[j][0] * dFtaudtau(v[j], phi[j], T[j], P[j]) +
                  sources_num[j][0] * dGtaudtau(v[j], phi[j], T[j], P[j], phidot[j], psi[j]))/mass_num[j]
        Tf[2*j+1, 2*j] = dt * (lo_advection_matrix[j][0] * dFtaudZ(v[j], phi[j], T[j], P[j]) +
                  sources_num[j][0] * dGtaudZ(v[j], phi[j], T[j], P[j], phidot[j], psi[j]))/mass_num[j]
        Tf[2*j+2, 2*j] = dt * (lo_advection_matrix[j][1] * dFtaudtau(v[j+1], phi[j+1], T[j+1], P[j+1]))/mass_num[j]
        Tf[2*j+3, 2*j] = dt * (lo_advection_matrix[j][1] * dFtaudZ(v[j+1], phi[j+1], T[j+1], P[j+1]))/mass_num[j]
        
        Tf[2*j-2, 2*j+1] = dt * (gradient_num_plus[j] * derivative_plus_num[j] * (-dpdtau(v[j-1], phi[j-1], T[j-1], P[j-1])))/mass_num[j]
        Tf[2*j-1, 2*j+1] = dt * (lo_advection_matrix[j][-1] * 1 +
                            gradient_num_plus[j] * derivative_plus_num[j] * (- dpdZ(v[j-1], phi[j-1], T[j-1], P[j-1]))) / mass_num[j]
        Tf[2*j, 2*j+1] = dt * (gradient_num_plus[j] * derivative_plus_num[j] * (dpdtau(v[j], phi[j], T[j], P[j])) + gradient_num_minus[j] * derivative_minus_num[j+1] * (- dpdtau(v[j], phi[j], T[j], P[j])) + sources_num[j][0] * dGZdtau(v[j], phi[j], T[j], P[j], phidot[j], psi[j]))/mass_num[j]
        Tf[2*j+1, 2*j+1] = 1 + dt * (lo_advection_matrix[j][0] * 1  + gradient_num_plus[j] * derivative_plus_num[j] * (dpdZ(v[j], phi[j], T[j], P[j])) + gradient_num_minus[j] * derivative_minus_num[j+1] * (- dpdZ(v[j], phi[j], T[j], P[j])) + sources_num[j][0] * dGZdZ(v[j], phi[j], T[j], P[j], phidot[j], psi[j]))/mass_num[j]
        Tf[2*j+2, 2*j+1] = dt * ((gradient_num_minus[j] * derivative_minus_num[j+1] * (dpdtau(v[j+1], phi[j+1], T[j+1], P[j+1])))/mass_num[j])
        Tf[2*j+3, 2*j+1] = dt * (lo_advection_matrix[j][1] * 1 +
                            (gradient_num_minus[j] * derivative_minus_num[j+1] * (dpdZ(v[j+1], phi[j+1], T[j+1], P[j+1]))) / mass_num[j])
    
    Tf[2*(n_N-1)-2, 2*(n_N-1)] = dt*(lo_advection_matrix[n_N-1][-1] * dFtaudtau(v[n_N-2], phi[n_N-2], T[n_N-2], P[n_N-2]))/mass_num[n_N-1] 
    Tf[2*(n_N-1)-1, 2*(n_N-1)] = dt*(lo_advection_matrix[n_N-1][-1] * dFtaudZ(v[n_N-2], phi[n_N-2], T[n_N-2], P[n_N-2]))/mass_num[n_N-1]
    Tf[2*(n_N-1), 2*(n_N-1)] = 1 + dt * (lo_advection_matrix[n_N-1][0] * dFtaudtau(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1]) +
                   sources_num[n_N-1][0] * dGtaudtau(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1], phidot[n_N-1], psi[n_N-1]))/mass_num[n_N-1]
    Tf[2*(n_N-1)+1, 2*(n_N-1)] = dt * (lo_advection_matrix[n_N-1][0] * dFtaudZ(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1]) +
                   sources_num[n_N-1][0] * dGtaudZ(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1], phidot[n_N-1], psi[n_N-1]))/mass_num[n_N-1]
    
    Tf[2*(n_N-1)-2, 2*(n_N-1)+1] = dt * (gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (-dpdtau(v[n_N-2], phi[n_N-2], T[n_N-2], P[n_N-2])))/mass_num[n_N-1]
    Tf[2*(n_N-1)-1, 2*(n_N-1)+1] = dt * (lo_advection_matrix[n_N-1][-1] * 1 +
                            gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (- dpdZ(v[n_N-2], phi[n_N-2], T[n_N-2], P[n_N-2]))) / mass_num[n_N-1]         
    Tf[2*(n_N-1), 2*(n_N-1)+1] = dt * ((gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (dpdtau(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1]))) + sources_num[n_N-1][0] * dGZdtau(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1], phidot[n_N-1], psi[n_N-1]))/mass_num[n_N-1]
    Tf[2*(n_N-1)+1, 2*(n_N-1)+1] = 1 + dt * (lo_advection_matrix[n_N-1][0] * 1 +
                        gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (dpdZ(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1])) + sources_num[n_N-1][0] * dGZdZ(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1], phidot[n_N-1], psi[n_N-1]))/mass_num[n_N-1]


    for j in range(2*n_N):
        for k in range(-3,4):
            if j >= -k and j < 2*n_N - k:
                Tf_banded[3+k,j] = Tf[j+k,j]
            else: 
                Tf_banded[3+k,j] = 0  
    dU = solve_banded((3, 3), Tf_banded, fU)
    for j in range(n_N):
       tau[j] = U[2*j] - dU[2*j]
       Z[j] = U[2*j + 1] - dU[2*j + 1]



@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
def lo_newton_fast(np.ndarray[DTYPE_t, ndim=1] tau_old,
              np.ndarray[DTYPE_t, ndim=1] tau,
              np.ndarray[DTYPE_t, ndim=1] taudot,
              np.ndarray[DTYPE_t, ndim=1] Z_old,
              np.ndarray[DTYPE_t, ndim=1] Z,
              np.ndarray[DTYPE_t, ndim=1] Zdot,
              np.ndarray[DTYPE_t, ndim=2] Tf_banded,
              np.ndarray[DTYPE_t, ndim=1] phi,
              np.ndarray[DTYPE_t, ndim=1] phidot,
              np.ndarray[DTYPE_t, ndim=1] psi,
              np.ndarray[DTYPE_t, ndim=1] T,
              np.ndarray[DTYPE_t, ndim=1] v,
              np.ndarray[DTYPE_t, ndim=1] P,
              np.ndarray[DTYPE_t, ndim=2] lo_advection_matrix):
              
    #cdef double[::1] U_old = np.zeros(2*n_N)
    #cdef double[::1] U = np.zeros(2*n_N)
    #cdef double[::1] fU = np.zeros(2*n_N)
    cdef int j 
    for j in range(n_N):
        U_old[2*j] = tau_old[j]
        U_old[2*j + 1] = Z_old[j]
        U[2*j] = tau[j]
        U[2*j+1] = Z[j]
        fU[2*j] = tau[j] - tau_old[j] + dt * taudot[j]  
        fU[2*j + 1] = Z[j] - Z_old[j] + dt * Zdot[j]    
    
    Tf_banded[0,0] = 0
    Tf_banded[1,0] = 0
    Tf_banded[2,0] = 0
    Tf_banded[3,0] = 1 + dt * (lo_advection_matrix[0][0] * dFtaudtau(v[0], phi[0], T[0], P[0]) +
                  sources_num[0][0] * dGtaudtau(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    Tf_banded[4, 0] = dt * (lo_advection_matrix[0][0] * dFtaudZ(v[0], phi[0], T[0], P[0]) +
                  sources_num[0][0] * dGtaudZ(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    Tf_banded[5, 0] = dt * (lo_advection_matrix[0][1] * dFtaudtau(v[1], phi[1], T[1], P[1]))/mass_num[0]
    Tf_banded[6, 0] = dt * (lo_advection_matrix[0][1] * dFtaudZ(v[1], phi[1], T[1], P[1]))/mass_num[0]
    
    Tf_banded[0,1] = 0
    Tf_banded[1,1] = 0
    Tf_banded[2,1] = dt * (gradient_num_plus[0] * derivative_plus_num[0] * (dpdtau(v[0], phi[0], T[0], P[0])) + gradient_num_minus[0] * derivative_minus_num[1] * (- dpdtau(v[0], phi[0], T[0], P[0])) + sources_num[0][0] * dGZdtau(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    Tf_banded[3,1] = 1 + dt * (lo_advection_matrix[0][0] * 1  + gradient_num_plus[0] * derivative_plus_num[0] * (dpdZ(v[0], phi[0], T[0], P[0])) + gradient_num_minus[0] * derivative_minus_num[1] * (- dpdZ(v[0], phi[0], T[0], P[0])) + sources_num[0][0] * dGZdZ(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    Tf_banded[4, 1] = dt * ((gradient_num_minus[0] * derivative_minus_num[1] * (dpdtau(v[1], phi[1], T[1], P[1])))/mass_num[0])
    Tf_banded[5, 1] = dt * (lo_advection_matrix[0][1] * 1 +
                            (gradient_num_minus[0] * derivative_minus_num[1] * (dpdZ(v[1], phi[1], T[1], P[1]))) / mass_num[0])                 
    Tf_banded[6, 1] = 0
    
    for j in range(1, n_N-1):
        Tf_banded[0,2*j] = 0
        Tf_banded[1,2*j] = dt * (lo_advection_matrix[j][-1] * dFtaudtau(v[j-1], phi[j-1], T[j-1], P[j-1]))/mass_num[j]
        Tf_banded[2,2*j] = dt * (lo_advection_matrix[j][-1] * dFtaudZ(v[j-1], phi[j-1], T[j-1], P[j-1]))/mass_num[j]
        Tf_banded[3,2*j] = 1 + dt * (lo_advection_matrix[j][0] * dFtaudtau(v[j], phi[j], T[j], P[j]) +
                  sources_num[j][0] * dGtaudtau(v[j], phi[j], T[j], P[j], phidot[j], psi[j]))/mass_num[j]
        Tf_banded[4,2*j] = dt * (lo_advection_matrix[j][0] * dFtaudZ(v[j], phi[j], T[j], P[j]) +
                  sources_num[j][0] * dGtaudZ(v[j], phi[j], T[j], P[j], phidot[j], psi[j]))/mass_num[j]
        Tf_banded[5,2*j] = dt * (lo_advection_matrix[j][1] * dFtaudtau(v[j+1], phi[j+1], T[j+1], P[j+1]))/mass_num[j]
        Tf_banded[6,2*j] = dt * (lo_advection_matrix[j][1] * dFtaudZ(v[j+1], phi[j+1], T[j+1], P[j+1]))/mass_num[j]
        
        Tf_banded[0,2*j+1] = dt * (gradient_num_plus[j] * derivative_plus_num[j] * (-dpdtau(v[j-1], phi[j-1], T[j-1], P[j-1])))/mass_num[j]
        Tf_banded[1, 2*j+1] =  dt * (lo_advection_matrix[j][-1] * 1 +
                            gradient_num_plus[j] * derivative_plus_num[j] * (- dpdZ(v[j-1], phi[j-1], T[j-1], P[j-1]))) / mass_num[j]
        Tf_banded[2, 2*j+1] = dt * (gradient_num_plus[j] * derivative_plus_num[j] * (dpdtau(v[j], phi[j], T[j], P[j])) + gradient_num_minus[j] * derivative_minus_num[j+1] * (- dpdtau(v[j], phi[j], T[j], P[j])) + sources_num[j][0] * dGZdtau(v[j], phi[j], T[j], P[j], phidot[j], psi[j]))/mass_num[j]
        Tf_banded[3, 2*j+1] = 1 + dt * (lo_advection_matrix[j][0] * 1  + gradient_num_plus[j] * derivative_plus_num[j] * (dpdZ(v[j], phi[j], T[j], P[j])) + gradient_num_minus[j] * derivative_minus_num[j+1] * (- dpdZ(v[j], phi[j], T[j], P[j])) + sources_num[j][0] * dGZdZ(v[j], phi[j], T[j], P[j], phidot[j], psi[j]))/mass_num[j]
        Tf_banded[4, 2*j+1] = dt * ((gradient_num_minus[j] * derivative_minus_num[j+1] * (dpdtau(v[j+1], phi[j+1], T[j+1], P[j+1])))/mass_num[j])
        Tf_banded[5, 2*j+1] = dt * (lo_advection_matrix[j][1] * 1 +
                            (gradient_num_minus[j] * derivative_minus_num[j+1] * (dpdZ(v[j+1], phi[j+1], T[j+1], P[j+1]))) / mass_num[j])
        Tf_banded[6, 2*j+1] = 0
        
    Tf_banded[0, 2*(n_N-1)] = 0
    Tf_banded[1, 2*(n_N-1)] = dt*(lo_advection_matrix[n_N-1][-1] * dFtaudtau(v[n_N-2], phi[n_N-2], T[n_N-2], P[n_N-2]))/mass_num[n_N-1] 
    Tf_banded[2, 2*(n_N-1)] = dt*(lo_advection_matrix[n_N-1][-1] * dFtaudZ(v[n_N-2], phi[n_N-2], T[n_N-2], P[n_N-2]))/mass_num[n_N-1]
    Tf_banded[3, 2*(n_N-1)] = 1 + dt * (lo_advection_matrix[n_N-1][0] * dFtaudtau(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1]) +
                   sources_num[n_N-1][0] * dGtaudtau(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1], phidot[n_N-1], psi[n_N-1]))/mass_num[n_N-1]
    Tf_banded[4, 2*(n_N-1)] = dt * (lo_advection_matrix[n_N-1][0] * dFtaudZ(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1]) +
                   sources_num[n_N-1][0] * dGtaudZ(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1], phidot[n_N-1], psi[n_N-1]))/mass_num[n_N-1]
    Tf_banded[5, 2*(n_N-1)] = 0
    Tf_banded[6, 2*(n_N-1)] = 0
    Tf_banded[0, 2*(n_N-1)+1] = dt * (gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (-dpdtau(v[n_N-2], phi[n_N-2], T[n_N-2], P[n_N-2])))/mass_num[n_N-1]
    Tf_banded[1, 2*(n_N-1)+1] = dt * (lo_advection_matrix[n_N-1][-1] * 1 +
                            gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (- dpdZ(v[n_N-2], phi[n_N-2], T[n_N-2], P[n_N-2]))) / mass_num[n_N-1]         
    Tf_banded[2, 2*(n_N-1)+1] = dt * ((gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (dpdtau(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1]))) + sources_num[n_N-1][0] * dGZdtau(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1], phidot[n_N-1], psi[n_N-1]))/mass_num[n_N-1]
    Tf_banded[3, 2*(n_N-1)+1] = 1 + dt * (lo_advection_matrix[n_N-1][0] * 1 +
                        gradient_num_plus[n_N-1] * derivative_plus_num[n_N-1] * (dpdZ(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1])) + sources_num[n_N-1][0] * dGZdZ(v[n_N-1], phi[n_N-1], T[n_N-1], P[n_N-1], phidot[n_N-1], psi[n_N-1]))/mass_num[n_N-1]
    Tf_banded[4, 2*(n_N-1)+1] = 0
    Tf_banded[5, 2*(n_N-1)+1] = 0
    Tf_banded[6, 2*(n_N-1)+1] = 0
        
    dU = solve_banded((3, 3), Tf_banded, fU, check_finite=False, overwrite_ab=True, overwrite_b=True)
    for j in range(n_N):
       tau[j] = U[2*j] - dU[2*j]
       Z[j] = U[2*j + 1] - dU[2*j + 1]
       
"""
@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
def lo_newton_thomas(np.ndarray[DTYPE_t, ndim=1] tau_old,
              np.ndarray[DTYPE_t, ndim=1] tau,
              np.ndarray[DTYPE_t, ndim=1] taudot,
              np.ndarray[DTYPE_t, ndim=1] Ftau,
              np.ndarray[DTYPE_t, ndim=1] Z_old,
              np.ndarray[DTYPE_t, ndim=1] Z,
              np.ndarray[DTYPE_t, ndim=1] Zdot,
              np.ndarray[DTYPE_t, ndim=1] FZ,
              np.ndarray[DTYPE_t, ndim=1] phi,
              np.ndarray[DTYPE_t, ndim=1] phidot,
              np.ndarray[DTYPE_t, ndim=1] psi,
              np.ndarray[DTYPE_t, ndim=1] T,
              np.ndarray[DTYPE_t, ndim=1] v,
              np.ndarray[DTYPE_t, ndim=1] P,
              np.ndarray[DTYPE_t, ndim=2] lo_advection_matrix):
              
    #cdef double[::1] U_old = np.zeros(2*n_N)
    #cdef double[::1] U = np.zeros(2*n_N)
    #cdef double[::1] fU = np.zeros(2*n_N)
    
    cdef double[::1] U_i = np.zeros(2)
    cdef double[::1] U_ip = np.zeros(2)
    cdef double[::1] U_im = np.zeros(2)
    cdef double[::1] fU_i = np.zeros(2)
    cdef double[::1] dU_i = np.zeros(2)
    cdef double[:, ::1] T_i = [[0,0],[0,0]]
    cdef double[:, ::1] T_im = [[0,0],[0,0]]
    cdef double[:, ::1] T_ip = [[0,0],[0,0]]
    cdef double det_i
    

    
    #i=0
    U_i[0] = tau[0]
    U_i[1] = Z[0]
    U_i_p[0] = tau[1]
    U_i_p[1] = Z[1]
    
    T_i[0,0] = (lo_advection_matrix[0][0]*dFtaudtau(v[0], phi[0], T[0], P[0]) +
                  sources_num[0][0] * dGtaudtau(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    T_i[1,0] = (lo_advection_matrix[0][0]*dFtaudZ(v[0], phi[0], T[0], P[0]) +
                  sources_num[0][0] * dGtaudZ(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    T_i[0,1] = (lo_advection_matrix[0][0]*dFZdtau(v[0], phi[0], T[0], P[0]) +
                  sources_num[0][0] * dGZdtau(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
    T_i[1,1] = (lo_advection_matrix[0][0]*dFZdZ(v[0], phi[0], T[0], P[0]) +
                  sources_num[0][0] * dGZdZ(v[0], phi[0], T[0], P[0], phidot[0], psi[0]))/mass_num[0]
                  
    T_ip[0,0] = (lo_advection_matrix[0][1]*dFtaudtau(v[1], phi[1], T[1], P[1]))/mass_num[0]
    T_ip[1,0] = (lo_advection_matrix[0][1]*dFtaudZ(v[1], phi[1], T[1], P[1]))/mass_num[0]
    T_ip[0,1] = (lo_advection_matrix[0][1]*dFZdtau(v[1], phi[1], T[1], P[1]))/mass_num[0]
    T_ip[1,1] = (lo_advection_matrix[0][1]*dFZdZ(v[1], phi[1], T[1], P[1]))/mass_num[0]
    
    det_i = T_i[0,0] * T_i[1,1]  - T_i[1,0] * T_i[0,1]        
    T_i_inv[0,0] = T_i[1,1]/det_i
    T_i_inv[0,1] = - T_i[0,1]/det_i
    T_i_inv[1,0] = - T_i[1,0]/det_i
    T_i_inv[1,1] = T_i[0,0]/det_i
    
    fU_i[0] = (tau[0] - tau_old[0] - dt*(lo_advection_matrix[0][0] * Ftau[0] + lo_advection_matrix[0][1] * Ftau[1] - source_matrix[0][0] * Gtau[0])/mass_num[0])/dt
    fU_i[1] = (Z[0] - Z_old[0] - dt*(lo_advection_matrix[0][0] * FZ[0] + lo_advection_matrix[0][1] * FZ[1] - source_matrix[0][0] * GZ[0])/mass_num[0])/dt
    
    dU_i = - np.matmul(np.matmul(T_i_inv, T_ip), U_ip) + np.matmul(T_i_inv, fU_i)/dt
    
    tau[0] = U_i[0] - dU_i[0]
    Z[0] = U_i[1] - dU_i[1]
    
    #i
    U_i_m[0] = tau[i-1]
    U_i_m[1] = Z[i-1]
    U_i[0] = tau[i]
    U_i[1] = Z[i]
    U_i_p[0] = tau[i+1]
    U_i_p[1] = Z[i+1]
    
    T_im[0,0] = (lo_advection_matrix[i][-1]*dFtaudtau(v[i-1], phi[i-1], T[i-1], P[i-1]))/mass_num[i]
    T_im[1,0] = (lo_advection_matrix[i][-1]*dFtaudZ(v[i-1], phi[i-1], T[i-1], P[i-1]))/mass_num[i]
    T_im[0,1] = (lo_advection_matrix[i][-1]*dFZdtau(v[i-1], phi[i-1], T[i-1], P[i-1]))/mass_num[i]
    T_im[1,1] = (lo_advection_matrix[i][-1]*dFZdZ(v[i-1], phi[i-1], T[i-1], P[i-1]))/mass_num[i]
    
    
    T_i[0,0] = (lo_advection_matrix[i][0]*dFtaudtau(v[i], phi[i], T[i], P[i]) +
                  sources_num[i][0] * dGtaudtau(v[i], phi[i], T[i], P[i], phidot[i], psi[i]))/mass_num[i]
    T_i[1,0] = (lo_advection_matrix[i][0]*dFtaudZ(v[i], phi[i], T[i], P[i]) +
                  sources_num[i][0] * dGtaudZ(v[i], phi[i], T[i], P[i], phidot[i], psi[i]))/mass_num[i]
    T_i[0,1] = (lo_advection_matrix[i][0]*dFZdtau(v[i], phi[i], T[i], P[i]) +
                  sources_num[i][0] * dGZdtau(v[i], phi[i], T[i], P[i], phidot[i], psi[i]))/mass_num[i]
    T_i[1,1] = (lo_advection_matrix[i][0]*dFZdZ(v[i], phi[i], T[i], P[i]) +
                  sources_num[i][0] * dGZdZ(v[i], phi[i], T[i], P[i], phidot[i], psi[i]))/mass_num[i]
                  
    T_ip[0,0] = (lo_advection_matrix[i][1]*dFtaudtau(v[i+1], phi[i+1], T[i+1], P[i+1]))/mass_num[i]
    T_ip[1,0] = (lo_advection_matrix[i][1]*dFtaudZ(v[i+1], phi[i+1], T[i+1], P[i+1]))/mass_num[i]
    T_ip[0,1] = (lo_advection_matrix[i][1]*dFZdtau(v[i+1], phi[i+1], T[i+1], P[i+1]))/mass_num[i]
    T_ip[1,1] = (lo_advection_matrix[i][1]*dFZdZ(v[i+1], phi[i+1], T[i+1], P[i+1]))/mass_num[i]
     
    
    cdef int j 
    for j in range(n_N):
        U_old[2*j] = tau_old[j]
        U_old[2*j + 1] = Z_old[j]
        U[2*j] = tau[j]
        U[2*j+1] = Z[j]
        fU[2*j] = tau[j] - tau_old[j] + dt * taudot[j]  
        fU[2*j + 1] = Z[j] - Z_old[j] + dt * Zdot[j]

    
    for j in range(n_N):
       tau[j] = U[2*j] - dU[2*j]
       Z[j] = U[2*j + 1] - dU[2*j + 1]

"""
@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
def update_phidot(np.ndarray[DTYPE_t, ndim=1] phi,
                  np.ndarray[DTYPE_t, ndim=1] phidot,
                  np.ndarray[DTYPE_t, ndim=1] psi,
                  np.ndarray[DTYPE_t, ndim=1] v,
                  np.ndarray[DTYPE_t, ndim=1] T):
    phidot[0] = ((1 + 0.5 * n_dt * n_eta * gam(v[0]))**(-1) *
                     ((1 - 0.5 * n_dt * n_eta * gam(v[0])) * phidot[0] -
                      dt * (dVTdphi_mod(phi[0], T[0]) -
                      (laplacian_num[0][1] * (phi[1] - phi[0]) -
                       n_eta * gam(v[0]) * v[0] * kinetic_num[0] * psi[0]) /
                      kinetic_num[0])))
    
    cdef int j
    for j in range(1, n_N-1):
        phidot[j] = ((1 + 0.5 * n_dt * n_eta * gam(v[j]))**(-1) *
                     ((1 - 0.5 * n_dt * n_eta * gam(v[j])) * phidot[j] -
                      dt * (dVTdphi_mod(phi[j], T[j]) -
                      (laplacian_num[j][1] * (phi[j+1] - phi[j]) +
                       laplacian_num[j][-1] * (phi[j] - phi[j-1]) -
                       n_eta * gam(v[j]) * v[j] *kinetic_num[j]* psi[j]) /
                      kinetic_num[j])))
                      
    phidot[n_N-1] = ((1 + 0.5 * n_dt * n_eta * gam(v[n_N-1]))**(-1) *
                     ((1 - 0.5 * n_dt * n_eta * gam(v[n_N-1])) * phidot[n_N-1] -
                      dt * (dVTdphi_mod(phi[n_N-1], T[n_N-1]) -
                      (laplacian_num[n_N-1][-1] * (phi[n_N-1] - phi[n_N-2]) -
                       n_eta * gam(v[n_N-1]) * v[n_N-1] * kinetic_num[n_N-1] * psi[n_N-1]
                       ) /
                      kinetic_num[n_N-1])))
                      
