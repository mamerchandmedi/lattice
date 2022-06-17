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

# %% Importing parameters of the model.
from parameters import eta
# end: Importing parameters of the model.

# %% Import matrices of lattice definition.
from lattice import dt, r, N
# end: Import matrices of lattice definition.

# %% Import matrices of wave equation integration scheme.
from matrices_field import laplacian_matrix, kinetic_matrix, gradient_matrix
# end: Import matrices of wave equation integration scheme.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import pressure_matrix, mass_matrix, \
    sources_matrix, advection_matrix
# end: Import matrices of the high order integration scheme.

# %% Import package for dynamical compilation of Cython code.
#import pyximport
#pyximport.install(inplace=True, language_level='3')
# end: Import package for dynamical compilation of Cython code.

# %% Import termodynamically corrected quantities.
from Cthermo cimport gam, Gf_tau, Gf_Z
# end: Import termodynamically corrected quantities.

# %% Import rescaled quantities.
from Cscaled cimport dVTdphi_mod
from Cscaled cimport tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Fixing a datatype for numpy arrays.
DTYPE = np.double
# end: Fixing a datatype for numpy arrays.

cdef int n_N = N
cdef double n_dt = dt
cdef double n_eta = eta


r_num_np = np.array(r, dtype=float)
cdef double[::1] r_num = r_num_np

laplacian_num_np = np.array(laplacian_matrix, dtype=float)
cdef double[:, ::1] laplacian_num = laplacian_num_np

kinetic_num_np = np.array(kinetic_matrix, dtype=float)
cdef double[::1] kinetic_num = kinetic_num_np

gradient_np = np.array(gradient_matrix, dtype=float)
cdef double[:, ::1] gradient_num = gradient_np

pressure_num_np = np.array(pressure_matrix, dtype=float)
cdef double[:, ::1] pressure_num = pressure_num_np

mass_num_np = np.array(mass_matrix, dtype=float)
cdef double[::1] mass_num = mass_num_np

sources_num_np = np.array(sources_matrix, dtype=float)
cdef double[:, ::1] sources_num = sources_num_np

advection_num_np = np.array(advection_matrix, dtype=float)
cdef double[:, ::1] advection_num = advection_num_np

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def update_Tpv(np.ndarray[double, ndim=1] T,
               np.ndarray[double, ndim=1] P,
               np.ndarray[double, ndim=1] v,
               np.ndarray[double, ndim=1] phi,
               np.ndarray[double, ndim=1] tau,
               np.ndarray[double, ndim=1] Z):
    cdef int j
    for j in range(n_N + 1):
        tauZphi2Tpv_mod(tau[j], Z[j], phi[j], &(T[j]), &(P[j]), &(v[j]))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def evaluate_fluxes(np.ndarray[double, ndim=1] tau,
                    np.ndarray[double, ndim=1] Z,
                    np.ndarray[double, ndim=1] P,
                    np.ndarray[double, ndim=1] v,
                    np.ndarray[double, ndim=1] Ftau,
                    np.ndarray[double, ndim=1] FZ):
    cdef int j
    for j in range(n_N + 1):
        Ftau[j] = (tau[j] + P[j]) * v[j]
        FZ[j] = Z[j] * v[j]


@cython.boundscheck(False)  # Deactivate bounds checking
def ho_update_taudot(np.ndarray[double, ndim=1] phi,
                     np.ndarray[double, ndim=1] phidot,
                     np.ndarray[double, ndim=1] T,
                     np.ndarray[double, ndim=1] v,
                     np.ndarray[double, ndim=1] Ftau,
                     np.ndarray[double, ndim=1] taudot,
                     np.ndarray[double, ndim=1] Gtau):
    Gtau[0] = sources_num[0][0] * Gf_tau(phi[0], phidot[0], 0, T[0], v[0])
    taudot[0] = (advection_num[0][0] * Ftau[0] +
                 advection_num[0][1] * Ftau[1] +
                 Gtau[0]) / mass_num[0]
    cdef int j
    for j in range(1, n_N):
        Gtau[j] = (sources_num[j][-1] * Gf_tau(phi[j], phidot[j],
                   (phi[j] - phi[j - 1])/(r_num[j] - r_num[j - 1]), T[j], v[j]) +
                   sources_num[j][1] * Gf_tau(phi[j], phidot[j],
                   (phi[j + 1] - phi[j])/(r_num[j + 1] - r_num[j]), T[j], v[j]))
        taudot[j] = (advection_num[j][-1] * Ftau[j - 1] +
                     advection_num[j][0] * Ftau[j] +
                     advection_num[j][1] * Ftau[j + 1] +
                     Gtau[j]) / mass_num[j]
    Gtau[n_N] = sources_num[n_N][0] * \
        Gf_tau(phi[n_N], 0, (phi[n_N] - phi[n_N - 1]) /
               (r_num[n_N] - r_num[n_N - 1]), T[n_N], v[n_N])
    taudot[n_N] = (advection_num[n_N][-1] * Ftau[n_N - 1] +
                 advection_num[n_N][0] * Ftau[n_N] +
                 Gtau[n_N]) / mass_num[n_N]

@cython.boundscheck(False)  # Deactivate bounds checking
def ho_update_Zdot(np.ndarray[double, ndim=1] phi,
                   np.ndarray[double, ndim=1] phidot,
                   np.ndarray[double, ndim=1] T,
                   np.ndarray[double, ndim=1] P,
                   np.ndarray[double, ndim=1] v,
                   np.ndarray[double, ndim=1] FZ,
                   np.ndarray[double, ndim=1] Zdot,
                   np.ndarray[double, ndim=1] GZ):

    GZ[0] = ((pressure_num[0][0]) * P[0] +
             (pressure_num[0][1]) * P[1] +
             sources_num[0][0] * Gf_Z(phi[0], phidot[0], 0, T[0], v[0]))
    Zdot[0] = ((advection_num[0][0] * FZ[0] +
               advection_num[0][1] * FZ[1] +
               GZ[0]) / mass_num[0])
    cdef int j
    for j in range(1, n_N):
        GZ[j] = ((pressure_num[j][-1]) * P[j - 1] +
                 (pressure_num[j][0]) * P[j] +
                 (pressure_num[j][1]) * P[j + 1] +
                 sources_num[j][-1] * Gf_Z(phi[j], phidot[j],
                 (phi[j] - phi[j - 1]) / (r_num[j] - r_num[j - 1]), T[j], v[j]) +
                 sources_num[j][1] * Gf_Z(phi[j], phidot[j],
                 (phi[j + 1] - phi[j])/(r_num[j + 1] - r_num[j]), T[j], v[j]))
        Zdot[j] = ((advection_num[j][-1] * FZ[j-1] +
                   advection_num[j][0] * FZ[j] +
                   advection_num[j][1] * FZ[j + 1] +
                   GZ[j]) / mass_num[j])
    GZ[n_N] = ((pressure_num[n_N][-1]) * P[n_N - 1] +
             (pressure_num[n_N][0]) * P[n_N] +
             sources_num[n_N][0] *
             Gf_Z(phi[n_N], 0, (phi[n_N] - phi[n_N - 1]) /
             (r_num[n_N] - r_num[n_N - 1]), T[n_N], v[n_N]))
    Zdot[n_N] = ((advection_num[n_N][-1] * FZ[n_N - 1] +
               advection_num[n_N][0] * FZ[n_N] +
               GZ[N]) / mass_num[n_N])

@cython.boundscheck(False)  # Deactivate bounds checking
def lo_update_taudot(np.ndarray[double, ndim=1] phi,
                     np.ndarray[double, ndim=1] phidot,
                     np.ndarray[double, ndim=1] T,
                     np.ndarray[double, ndim=1] v,
                     np.ndarray[double, ndim=1] Ftau,
                     np.ndarray[double, ndim=2] lo_advection_matrix,
                     np.ndarray[double, ndim=1] taudot,
                     np.ndarray[double, ndim=1] Gtau):
    Gtau[0] = sources_matrix[0][0] * Gf_tau(phi[0], phidot[0], 0, T[0], v[0])
    taudot[0] = (lo_advection_matrix[0][0] * Ftau[0] +
                 lo_advection_matrix[0][1] * Ftau[1] +
                 Gtau[0]) / mass_num[0]
    cdef int j
    for j in range(1, n_N):
        Gtau[j] = (sources_num[j][-1] * Gf_tau(phi[j], phidot[j],
                   (phi[j] - phi[j-1])/(r_num[j] - r_num[j - 1]), T[j], v[j]) +
                   sources_num[j][1] * Gf_tau(phi[j], phidot[j],
                   (phi[j + 1] - phi[j])/(r_num[j + 1] - r_num[j]), T[j], v[j]))
        taudot[j] = (lo_advection_matrix[j][-1] * Ftau[j - 1] +
                     lo_advection_matrix[j][0] * Ftau[j] +
                     lo_advection_matrix[j][1] * Ftau[j + 1] +
                     Gtau[j]) / mass_matrix[j]
    Gtau[n_N] = (sources_num[n_N][0] *
                 Gf_tau(phi[n_N], 0, (phi[n_N] - phi[n_N - 1]) /
               (r_num[n_N] - r_num[n_N-1]), T[n_N], v[n_N]))
    taudot[n_N] = (lo_advection_matrix[n_N][-1] * Ftau[n_N - 1] +
                 lo_advection_matrix[n_N][0] * Ftau[n_N] +
                 Gtau[n_N]) / mass_matrix[n_N]

@cython.boundscheck(False)  # Deactivate bounds checking
def lo_update_Zdot(np.ndarray[double, ndim=1] phi,
                   np.ndarray[double, ndim=1] phidot,
                   np.ndarray[double, ndim=1] T,
                   np.ndarray[double, ndim=1] P,
                   np.ndarray[double, ndim=1] v,
                   np.ndarray[double, ndim=1] FZ,
                   np.ndarray[double, ndim=2] lo_advection_matrix,
                   np.ndarray[double, ndim=1] Zdot,
                   np.ndarray[double, ndim=1] GZ):
    GZ[0] = ((pressure_num[0][0]) * P[0] +
             (pressure_num[0][1]) * P[1] +
             sources_num[0][0] * Gf_Z(phi[0], phidot[0], 0, T[0], v[0]))
    Zdot[0] = (lo_advection_matrix[0][0] * FZ[0] +
               lo_advection_matrix[0][1] * FZ[1] +
               GZ[0]) / mass_num[0]
    cdef int j
    for j in range(1, n_N):
        GZ[j] = ((pressure_num[j][-1]) * P[j - 1] +
                 (pressure_num[j][0]) * P[j] +
                 (pressure_num[j][1]) * P[j + 1] +
                 sources_num[j][-1] * Gf_Z(phi[j], phidot[j],
                 (phi[j] - phi[j - 1]) / (r_num[j] - r_num[j - 1]), T[j], v[j]) +
                 sources_num[j][1] * Gf_Z(phi[j], phidot[j],
                 (phi[j + 1] - phi[j])/(r_num[j + 1] - r_num[j]), T[j], v[j]))
        Zdot[j] = (lo_advection_matrix[j][-1] * FZ[j - 1] +
                   lo_advection_matrix[j][0] * FZ[j] +
                   lo_advection_matrix[j][1] * FZ[j + 1] +
                   GZ[j]) / mass_num[j]
    GZ[n_N] = ((pressure_num[n_N][-1]) * P[n_N - 1] +
             (pressure_num[n_N][0]) * P[n_N] +
             sources_num[n_N][0] * Gf_Z(phi[n_N], 0, (phi[n_N] - phi[n_N - 1]) /
             (r_num[n_N] - r[n_N - 1]), T[n_N], v[n_N]))
    Zdot[n_N] = (lo_advection_matrix[n_N][-1] * FZ[n_N - 1] +
               lo_advection_matrix[n_N][0] * FZ[n_N] +
               GZ[n_N]) / mass_num[n_N]

@cython.boundscheck(False)  # Deactivate bounds checking
def update_phidot(np.ndarray[double, ndim=1] phi,
                  np.ndarray[double, ndim=1] phidot,
                  np.ndarray[double, ndim=1] v,
                  np.ndarray[double, ndim=1] T):
    phidot[1] = ((1 + 0.5 * n_dt * n_eta * kinetic_num[1]**(-1) *
                  ((r_num[1]**3 - r_num[0]**3) * gam(v[0]) +
                   0.25 * (3 * r_num[1]**2 + 2 * r_num[1] * r_num[2] + r_num[2]**2) *
                   (r_num[2] - r_num[1]) * gam(v[1])))**(-1) *
                 ((1 - 0.5 * n_dt * n_eta * kinetic_num[1]**(-1) *
                   ((r_num[1]**3 - r_num[0]**3) * gam(v[0]) +
                    0.25 * (3 * r_num[1]**2 + 2 * r_num[1] * r_num[2] + r_num[2]**2) *
                     (r[2] - r[1]) * gam(v[1]))) * phidot[1] -
                  n_dt * (dVTdphi_mod(phi[1], T[1]) -
                  (laplacian_num[1][1] * (phi[2] - phi[1]) -
                   eta * gam(v[1]) * v[1] *
                   (gradient_num[1][1] * (phi[2] - phi[1]))) /
                  kinetic_num[1])))
    phidot[0] = phidot[1]
    cdef int j
    for j in range(2, n_N):
        phidot[j] = ((1 + 0.5 * n_dt * n_eta * gam(v[j]))**(-1) *
                     ((1 - 0.5 * n_dt * n_eta * gam(v[j])) * phidot[j] -
                      dt * (dVTdphi_mod(phi[j], T[j]) -
                      (laplacian_num[j][1] * (phi[j+1] - phi[j]) +
                       laplacian_num[j][-1] * (phi[j] - phi[j-1]) -
                       n_eta * gam(v[j]) * v[j] *
                       (gradient_num[j][-1] * (phi[j] - phi[j-1]) +
                        gradient_num[j][1] * (phi[j+1] - phi[j]))) /
                      kinetic_num[j])))
