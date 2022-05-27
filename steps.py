import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve
import matplotlib.pylab as pl
from tqdm import tqdm

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

# %% Importing initial values.
from initialization import P_off
# end: Importing initial values.


def high_order_step(fi, ti, zi, Ti, vi, Pi, fidot):
    phidot = fidot
    taudot = np.zeros(r.size)
    Zdot = np.zeros(r.size)
    T = Ti
    P = Pi
    v = vi
    phi = fi
    tau = ti
    Z = zi
    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])

    Ftau = (tau + P)*v
    FZ = Z * v

    taudot[0] = (ho_advection_matrix[0][0] * Ftau[0] +
                 ho_advection_matrix[0][1] * Ftau[1] +
                 friction_matrix[0][0] *
                 Gf_tau(phi[0], phidot[0], 0, T[0], v[0])) / mass_matrix[0]

    taudot[N] = ((ho_advection_matrix[N][-1] * Ftau[N-1] +
                 ho_advection_matrix[N][0] * Ftau[N]) / mass_matrix[N] +
                 Gf_tau(phi[N], 0, (phi[N] - phi[N-1]) /
                 (r[N] - r[N-1]), T[0], v[0]))

    Zdot[0] = (ho_advection_matrix[0][0] * FZ[0] +
               ho_advection_matrix[0][1] * FZ[1] +
               (sources_matrix[0][0] + ho_advection_matrix[0][0]) *
               (P[0] - P_off[0]) +
               (sources_matrix[0][1] + ho_advection_matrix[0][1]) *
               (P[1] - P_off[1]) +
               friction_matrix[0][0] *
               Gf_Z(phi[0], phidot[0], 0, T[0], v[0])) / mass_matrix[0]
    Zdot[N] = ((ho_advection_matrix[N][-1] * FZ[N-1] +
               ho_advection_matrix[N][0] * FZ[N] +
               (sources_matrix[N][-1] + ho_advection_matrix[N][-1]) *
               (P[N-1] - P_off[N-1]) +
               (sources_matrix[N][0] + ho_advection_matrix[N][0]) *
               (P[N] - P_off[N])) / mass_matrix[N] -
               Gf_Z(phi[N], 0, (phi[N] - phi[N-1]) /
               (r[N] - r[N-1]), T[N], v[N]))

    for j in range(1, N):
        taudot[j] = (ho_advection_matrix[j][-1] * Ftau[j-1] +
                     ho_advection_matrix[j][0] * Ftau[j] +
                     ho_advection_matrix[j][1] * Ftau[j+1] +
                     friction_matrix[j][-1] *
                     Gf_tau(phi[j], phidot[j],
                     (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                     friction_matrix[j][1] *
                     Gf_tau(phi[j], phidot[j],
                     (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j])) / \
            mass_matrix[j]
        Zdot[j] = (ho_advection_matrix[j][-1] * FZ[j-1] +
                   ho_advection_matrix[j][0] * FZ[j] +
                   ho_advection_matrix[j][1] * FZ[j+1] -
                   friction_matrix[j][-1] *
                   Gf_Z(phi[j], phidot[j],
                   (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                   friction_matrix[j][1] *
                   Gf_Z(phi[j], phidot[j],
                   (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]) +
                   (sources_matrix[j][-1] + ho_advection_matrix[j][-1]) *
                   P[j-1] +
                   (sources_matrix[j][0] + ho_advection_matrix[j][0]) *
                   P[j] +
                   (sources_matrix[j][1] + ho_advection_matrix[j][1]) *
                   P[j+1]) / mass_matrix[j]

    # ------tau, Z evolution (half step)
    tau_temp = tau - 0.5 * dt * taudot
    Z_temp = Z - 0.5 * dt * Zdot

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau_temp[j], Z_temp[j], phi[j],
                                           T[j], P[j], v[j])

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

    Ftau = (tau + P)*v
    FZ = Z*v
    Gtau = np.zeros(r.size)
    GZ = np.zeros(r.size)

    taudot[0] = (ho_advection_matrix[0][0] * Ftau[0] +
                 ho_advection_matrix[0][1] * Ftau[1] +
                 friction_matrix[0][0] *
                 Gf_tau(phi[0], phidot[0], 0, T[0], v[0])) / mass_matrix[0]
    Gtau[0] = friction_matrix[0][0] * Gf_tau(phi[0], phidot[0], 0, T[0], v[0])
    taudot[N] = (((ho_advection_matrix[N][-1] * Ftau[N-1] +
                  ho_advection_matrix[N][0] * Ftau[N]) +
                  friction_matrix[N][0] * Gf_tau(phi[N], 0,
                  (phi[N] - phi[N-1]) / (r[N] - r[N-1]), T[N], v[N])) /
                 mass_matrix[N])
    Gtau[N] = (friction_matrix[N][0] * Gf_tau(phi[N], 0, (phi[N] - phi[N-1]) /
               (r[N] - r[N-1]), T[N], v[N]))
    Zdot[0] = (ho_advection_matrix[0][0] * FZ[0] +
               ho_advection_matrix[0][1] * FZ[1] +
               (sources_matrix[0][0] + ho_advection_matrix[0][0]) *
               (P[0] - P_off[0]) +
               (sources_matrix[0][1] + ho_advection_matrix[0][1]) *
               (P[1] - P_off[1]) +
               friction_matrix[0][0] *
               Gf_Z(phi[0], phidot[0], 0, T[0], v[0])) / mass_matrix[0]
    GZ[0] = ((sources_matrix[0][0] + ho_advection_matrix[0][0]) * P[0] +
             (sources_matrix[0][1] + ho_advection_matrix[0][1]) * P[1] +
             friction_matrix[0][0] * Gf_Z(phi[0], phidot[0], 0, T[0], v[0]))

    Zdot[N] = (ho_advection_matrix[N][-1] * FZ[N-1] +
               ho_advection_matrix[N][0] * FZ[N] +
               (sources_matrix[N][-1] + ho_advection_matrix[N][-1]) *
               (P[N-1] - P_off[N-1]) +
               (sources_matrix[N][0] + ho_advection_matrix[N][0]) *
               (P[N] - P_off[N]) +
               friction_matrix[N][0] * Gf_Z(phi[N], 0, (phi[N] - phi[N-1]) /
               (r[N] - r[N-1]), T[N], v[N])) / mass_matrix[N]
    GZ[N] = ((sources_matrix[N][-1] + ho_advection_matrix[N][-1]) * P[N-1] +
             (sources_matrix[N][0] + ho_advection_matrix[N][0]) * P[N] +
             friction_matrix[N][0] * Gf_Z(phi[N], 0, (phi[N] - phi[N-1]) /
             (r[N] - r[N-1]), T[N], v[N]))

    for j in range(1, N):
        taudot[j] = (ho_advection_matrix[j][-1] * Ftau[j-1] +
                     ho_advection_matrix[j][0] * Ftau[j] +
                     ho_advection_matrix[j][1] * Ftau[j+1] +
                     friction_matrix[j][-1] *
                     Gf_tau(phi[j], phidot[j],
                     (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                     friction_matrix[j][1] *
                     Gf_tau(phi[j], phidot[j],
                     (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j])) / \
            mass_matrix[j]
        Gtau[j] = (friction_matrix[j][-1] * Gf_tau(phi[j], phidot[j],
                   (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                   friction_matrix[j][1] * Gf_tau(phi[j], phidot[j],
                   (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]))
        Zdot[j] = (ho_advection_matrix[j][-1] * FZ[j-1] +
                   ho_advection_matrix[j][0] * FZ[j] +
                   ho_advection_matrix[j][1] * FZ[j+1] -
                   friction_matrix[j][-1] *
                   Gf_Z(phi[j], phidot[j],
                   (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                   friction_matrix[j][1] *
                   Gf_Z(phi[j], phidot[j],
                   (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]) +
                   (sources_matrix[j][-1] + ho_advection_matrix[j][-1]) *
                   P[j-1] +
                   (sources_matrix[j][0] + ho_advection_matrix[j][0]) *
                   P[j] +
                   (sources_matrix[j][1] + ho_advection_matrix[j][1]) *
                   P[j+1]) / mass_matrix[j]
        GZ[j] = ((sources_matrix[j][-1] + ho_advection_matrix[j][-1]) *
                 P[j-1] +
                 (sources_matrix[j][0] + ho_advection_matrix[j][0]) *
                 P[j] +
                 (sources_matrix[j][1] + ho_advection_matrix[j][1]) *
                 P[j+1] +
                 friction_matrix[j][-1] * Gf_Z(phi[j], phidot[j],
                 (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                 friction_matrix[j][1] * Gf_Z(phi[j], phidot[j],
                 (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]))

    # ------tau, Z evolution (half step)
    tau -= dt * taudot
    Z -= dt * Zdot

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]
    return phi, tau, Z, T, v, P, phidot, Ftau, FZ, Gtau, GZ, taudot


def low_order_step(fi, ti, zi, Ti, vi, Pi, fidot):
    phidot = fidot
    taudot = np.zeros(r.size)
    Zdot = np.zeros(r.size)
    T = Ti
    P = Pi
    v = vi
    phi = fi
    tau = ti
    Z = zi
    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])

    Ftau = (tau + P)
    FZ = Z
    Gtau = np.zeros(r.size)
    GZ = np.zeros(r.size)

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    # -------evaluation of advection matrix
    lo_advection_matrix = advection(v, ho_advection_matrix)

    taudot[0] = (lo_advection_matrix[0][0] * Ftau[0] +
                 lo_advection_matrix[0][1] * Ftau[1] +
                 friction_matrix[0][0] *
                 Gf_tau(phi[0], phidot[0], 0, T[0], v[0])) / mass_matrix[0]
    Gtau[0] = friction_matrix[0][0] * Gf_tau(phi[0], phidot[0], 0, T[0], v[0])
    taudot[N] = ((lo_advection_matrix[N][-1] * Ftau[N-1] +
                 lo_advection_matrix[N][0] * Ftau[N] +
                 friction_matrix[N][0] * Gf_tau(phi[N], 0,
                 (phi[N] - phi[N-1]) / (r[N] - r[N-1]), T[N], v[N])) /
                 mass_matrix[N])
    Gtau[N] = (friction_matrix[N][0] * Gf_tau(phi[N], 0, (phi[N] - phi[N-1]) /
               (r[N] - r[N-1]), T[N], v[N]))
    Zdot[0] = (lo_advection_matrix[0][0] * FZ[0] +
               lo_advection_matrix[0][1] * FZ[1] +
               (sources_matrix[0][0] + ho_advection_matrix[0][0]) *
               (P[0] - P_off[0]) +
               (sources_matrix[0][1] + ho_advection_matrix[0][1]) *
               (P[1] - P_off[1]) +
               friction_matrix[0][0] *
               Gf_Z(phi[0], phidot[0], 0, T[0], v[0])) / mass_matrix[0]
    GZ[0] = ((sources_matrix[0][0] + ho_advection_matrix[0][0]) * P[0] +
             (sources_matrix[0][1] + ho_advection_matrix[0][1]) * P[1] +
             friction_matrix[0][0] * Gf_Z(phi[0], phidot[0], 0, T[0], v[0]))
    Zdot[N] = (lo_advection_matrix[N][-1] * FZ[N-1] +
               lo_advection_matrix[N][0] * FZ[N] +
               (sources_matrix[N][-1] + ho_advection_matrix[N][-1]) *
               (P[N-1] - P_off[N-1]) +
               (sources_matrix[N][0] + ho_advection_matrix[N][0]) *
               (P[N] - P_off[N]) +
               friction_matrix[N][0] * Gf_Z(phi[N], 0, (phi[N] - phi[N-1]) /
               (r[N] - r[N-1]), T[N], v[N])) / mass_matrix[N]
    GZ[N] = ((sources_matrix[N][-1] + ho_advection_matrix[N][-1]) * P[N-1] +
             (sources_matrix[N][0] + ho_advection_matrix[N][0]) * P[N] +
             friction_matrix[N][0] * Gf_Z(phi[N], 0, (phi[N] - phi[N-1]) /
             (r[N] - r[N-1]), T[N], v[N]))

    for j in range(1, N):
        taudot[j] = (lo_advection_matrix[j][-1] * Ftau[j-1] +
                     lo_advection_matrix[j][0] * Ftau[j] +
                     lo_advection_matrix[j][1] * Ftau[j+1] +
                     friction_matrix[j][-1] *
                     Gf_tau(phi[j], phidot[j],
                     (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                     friction_matrix[j][1] *
                     Gf_tau(phi[j], phidot[j],
                     (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j])) / \
            mass_matrix[j]
        Gtau[j] = (friction_matrix[j][-1] * Gf_Z(phi[j], phidot[j],
                   (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                   friction_matrix[j][1] * Gf_Z(phi[j], phidot[j],
                   (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]))
        Zdot[j] = (lo_advection_matrix[j][-1] * FZ[j-1] +
                   lo_advection_matrix[j][0] * FZ[j] +
                   lo_advection_matrix[j][1] * FZ[j+1] -
                   friction_matrix[j][-1] *
                   Gf_Z(phi[j], phidot[j],
                   (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                   friction_matrix[j][1] *
                   Gf_Z(phi[j], phidot[j],
                   (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]) +
                   (sources_matrix[j][-1] +
                    ho_advection_matrix[j][-1]) * P[j-1] +
                   (sources_matrix[j][0] +
                    ho_advection_matrix[j][0]) * P[j] +
                   (sources_matrix[j][1] +
                    ho_advection_matrix[j][1]) * P[j+1]) / \
            mass_matrix[j]
        GZ[j] = ((sources_matrix[j][-1] + ho_advection_matrix[j][-1]) *
                 P[j-1] +
                 (sources_matrix[j][0] + ho_advection_matrix[j][0]) * P[j] +
                 (sources_matrix[j][1] + ho_advection_matrix[j][1]) * P[j+1] +
                 friction_matrix[j][-1] * Gf_Z(phi[j], phidot[j],
                 (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                 friction_matrix[j][1] * Gf_Z(phi[j], phidot[j],
                 (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]))

    # ------tau, Z evolution (half step)
    tau -= 0.5 * dt * taudot
    Z -= 0.5 * dt * Zdot

    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])

    phidot[1] = ((1 + 0.5 * dt * eta * gam(v[1]))**(-1) *
                 ((1 - 0.5 * dt * eta * gam(v[1])) * phidot[1] -
                  dt * (dVTdphi(phi[1], T[1]) -
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

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    # ------tau, Z evolution (half step)
    tau -= 0.5 * dt * taudot
    Z -= 0.5 * dt * Zdot
    return phi, tau, Z, T, v, P, phidot, Ftau, FZ, Gtau, GZ, lo_advection_matrix
