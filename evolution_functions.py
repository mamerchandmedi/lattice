import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve

# %% Importing parameters of the model.
from parameters import Tc, L, sigma, lc, T0, alpha, gamma, lam, Tn, eta
# end: Importing parameters of the model.

# %% Importing the potetnial of the model.
from potential import V, dV
# end: Importing the potetnial of the model.

# %% Import critical bubble calculated by CosmoTransitions.
from bubble import vt, profile, fphi, phi_tw
# end: Import critical bubble calculated by CosmoTransitions.

# %% Import termodynamically corrected quantities.
from thermo import gst, a, VT, dVTdphi, eps, p, w, gam, Gf_tau, Gf_Z
# end: Import termodynamically corrected quantities.

# %% Import rescaled quantities.
from scaled import GeV, VT_mod, dVTdphi_mod, tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Import matrices of lattice definition.
from lattice import dr, dt, r, N
# end: Import matrices of lattice definition.

# %% Import matrices of wave equation integration scheme.
from matrices_field import laplacian_matrix, kinetic_matrix, gradient_matrix
# end: Import matrices of wave equation integration scheme.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import sources_matrix
from matrices_high_order import advection_matrix as ho_advection_matrix
from matrices_high_order import mass_matrix
from matrices_high_order import friction_matrix
# end: Import matrices of the high order integration scheme.

# %% Import matrices of the low order integration scheme.
from matrices_low_order_dynamically import advection
# end: Import matrices of the low order integration scheme.

# %% Importing procedure for solving algebraic relation
#    T, p, v as function of tau, Z, phi.
from solver import tauZphi2Tpv
# end: Importing procedure for solving algebraic relation
#      T, p, v as function of tau, Z, phi.

def update_Tpv(T, P, v, phi, tau, Z):
    T_new = np.zeros(r.size)
    P_new = np.zeros(r.size)
    v_new = np.zeros(r.size)
    for j in range(N+1):
        T_new[j], P_new[j], v_new[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j], v[j])
    return: T_new, P_new, v_new


def evaluate_fluxes(tau, Z, P, v):
    Ftau = (tau + P)*v
    FZ = Z * v
    return Ftau, FZ


def ho_update_taudot(phi, phidot, T, v, Ftau):
    taudot = np.zeros(r.size)
    Gtau = np.zeros(r.size)
    Gtau[0] = friction_matrix[0][0] * Gf_tau(phi[0], phidot[0], 0, T[0], v[0])
    taudot[0] = (ho_advection_matrix[0][0] * Ftau[0] +
                 ho_advection_matrix[0][1] * Ftau[1] +
                 Gtau[0]) / mass_matrix[0]
    for j in range(1, N):
        Gtau[j] = (friction_matrix[j][-1] * Gf_tau(phi[j], phidot[j],
                   (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                   friction_matrix[j][1] * Gf_tau(phi[j], phidot[j],
                   (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]))
        taudot[j] = (ho_advection_matrix[j][-1] * Ftau[j-1] +
                     ho_advection_matrix[j][0] * Ftau[j] +
                     ho_advection_matrix[j][1] * Ftau[j+1] +
                     Gtau[j]) / mass_matrix[j]
    Gtau[N] = friction_matrix[N][0] * Gf_tau(phi[N], 0,
              (phi[N] - phi[N-1])/(r[N] - r[N-1]), T[N], v[N])
    taudot[N] = (ho_advection_matrix[N][-1] * Ftau[N-1] +
                 ho_advection_matrix[N][0] * Ftau[N] +
                 Gtau[N]) / mass_matrix[N]
    return taudot, Gtau


def ho_updade_Z(phi, phidot, T, P, v, FZ):
    GZ[0] = (sources_matrix[0][0] + ho_advection_matrix[0][0]) * P[0] +
            (sources_matrix[0][1] + ho_advection_matrix[0][1]) * P[1] +
            friction_matrix[0][0] * Gf_Z(phi[0], phidot[0], 0, T[0], v[0])
    Zdot[0] = (ho_advection_matrix[0][0] * FZ[0] +
               ho_advection_matrix[0][1] * FZ[1] +
               GZ[0]) / mass_matrix[0]
    for j in range(1, N):
        GZ[j] = (sources_matrix[j][-1] + ho_advection_matrix[j][-1]) * P[j-1] +
                 (sources_matrix[j][0] + ho_advection_matrix[j][0]) * P[j] +
                 (sources_matrix[j][1] + ho_advection_matrix[j][1]) * P[j+1] +
                 friction_matrix[j][-1] * Gf_Z(phi[j], phidot[j],
                 (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                 friction_matrix[j][1] * Gf_Z(phi[j], phidot[j],
                 (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j])
        Zdot[j] = (ho_advection_matrix[j][-1] * FZ[j-1] +
                   ho_advection_matrix[j][0] * FZ[j] +
                   ho_advection_matrix[j][1] * FZ[j+1] +
                   GZ[j] ) / mass_matrix[j]
    GZ[N] = (sources_matrix[N][-1] + ho_advection_matrix[N][-1]) * P[N-1] +
             (sources_matrix[N][0] + ho_advection_matrix[N][0]) * P[N] +
             friction_matrix[N][0] * Gf_Z(phi[N], 0, (phi[N] - phi[N-1]) /
             (r[N] - r[N-1]), T[N], v[N])
    Zdot[N] = (ho_advection_matrix[N][-1] * FZ[N-1] +
               ho_advection_matrix[N][0] * FZ[N] +
               GZ[N]) / mass_matrix[N]
    return Zdot, GZ


def update_phidot(phi, phidot, v, T):
    phidot[1] = ((1 + 0.5 * dt * eta * gam(v[1]))**(-1) *
                 ((1 - 0.5 * dt * eta * gam(v[1])) * phidot[1] -
                  dt * (dVTdphi_mod(phi[1], T[1]) -
                  (laplacian_matrix[1][1] * (phi[2] - phi[1]) -
                   eta * gam(v[1]) * v[1] *
                   (gradient_matrix[1][1] * (phi[2] - phi[1]))) /
                  kinetic_matrix[1])))
    for j in range(2, N):
        phidot[j] = ((1 + 0.5 * dt * eta * gam(v[j]))**(-1) *
                     ((1 - 0.5 * dt * eta * gam(v[j])) * phidot[j] -
                      dt * (dVTdphi(phi[j], T[j]) -
                      (laplacian_matrix[j][1] * (phi[j+1] - phi[j]) +
                       laplacian_matrix[j][-1] * (phi[j] - phi[j-1]) -
                       eta * gam(v[j]) * v[j] *
                       (gradient_matrix[j][-1] * (phi[j] - phi[j-1]) +
                        gradient_matrix[j][1] * (phi[j+1] - phi[j]))) /
                      kinetic_matrix[j])))
    return phidot
