import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve
import matplotlib.pylab as pl
from tqdm import tqdm

# %% Importing parameters of the model.
from parameters import Tc, L, sigma, lc, T0, alpha, gamma, lam, Tn
# end: Importing parameters of the model.

# %% Importing the potetnial of the model.
from potential import V, dV
# end: Importing the potetnial of the model.

# %% Import critical bubble calculated by CosmoTransitions.
from bubble import vt, profile, fphi, phi_tw
# end: Import critical bubble calculated by CosmoTransitions.

# %% Import termodynamically corrected quantities.
from thermo import gst, a, VT, dVTdphi, eps, p, w, gam
# end: Import termodynamically corrected quantities.

# %% Import rescaled quantities.
from scaled import GeV, VT_mod, dVTdphi_mod, tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Import matrices of lattice definition.
from lattice import dr, dt, r, N
# end: Import matrices of lattice definition.

# %% Import matrices of wave equation integration scheme.
from matrices_field import laplacian_matrix, kinetic_matrix
# end: Import matrices of wave equation integration scheme.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import sources_matrix
from matrices_high_order import advection_matrix as ho_advection_matrix
from matrices_high_order import mass_matrix
# end: Import matrices of the high order integration scheme.

# %% Import matrices of the low order integration scheme.
from matrices_low_order_dynamically import advection
# end: Import matrices of the low order integration scheme.

# %% Importing procedure for solving algebraic relation
#    T, p, v as function of tau, Z, phi.
from solver import tauZphi2Tpv
# end: Importing procedure for solving algebraic relation
#      T, p, v as function of tau, Z, phi.


def high_order_step(fi, ti, zi, Ti, vi, Pi, fidot, tidot, zidot):
    phidot = fidot
    taudot = tidot
    Zdot = zidot
    T = Ti
    P = Pi
    v = vi
    phi = fi
    tau = ti
    Z = zi
    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])

    Ftau = (tau + P) * v
    FZ = Z * v + P

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    taudot[0] = (ho_advection_matrix[0][0] * Ftau[0] +
                 ho_advection_matrix[0][1] * Ftau[1]) / mass_matrix[0]
    taudot[N] = (ho_advection_matrix[N][-1] * Ftau[N-1] +
                 ho_advection_matrix[N][0] * Ftau[N]) / \
        mass_matrix[N]

    Zdot[0] = (ho_advection_matrix[0][0] * FZ[0] +
               ho_advection_matrix[0][1] * FZ[1] +
               sources_matrix[0][0] * P[0] +
               sources_matrix[0][1] * P[1]) / mass_matrix[0]
    Zdot[N] = (ho_advection_matrix[N][-1] * FZ[N-1] +
               ho_advection_matrix[N][0] * FZ[N] +
               sources_matrix[N][-1] * P[N-1] +
               sources_matrix[N][0] * P[N]) / \
        mass_matrix[N]

    for j in range(1, N):
        taudot[j] = (ho_advection_matrix[j][-1] * Ftau[j-1] +
                     ho_advection_matrix[j][0] * Ftau[j] +
                     ho_advection_matrix[j][1] * Ftau[j+1]) / \
            mass_matrix[j]
        Zdot[j] = (ho_advection_matrix[j][-1] * FZ[j-1] +
                   ho_advection_matrix[j][0] * FZ[j] +
                   ho_advection_matrix[j][1] * FZ[j+1] +
                   sources_matrix[j][-1] * P[j-1] +
                   sources_matrix[j][0] * P[j] +
                   sources_matrix[j][1] * P[j+1]) / \
            mass_matrix[j]

    # ------tau, Z evolution (half step)
    tau_temp = tau - 0.5 * dt * taudot
    Z_temp = Z - 0.5 * dt * Zdot

    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau_temp[j], Z_temp[j], phi[j],
                                           T[j], P[j], v[j])

    phidot[1] += dt * ((laplacian_matrix[1][1] * (phi[2] - phi[1])) /
                       kinetic_matrix[1] - dVTdphi_mod(phi[1], T[1]))
    for j in range(2, N):
        phidot[j] += dt * ((laplacian_matrix[j][1] * (phi[j+1] - phi[j]) +
                            laplacian_matrix[j][-1] * (phi[j] - phi[j-1])) /
                           kinetic_matrix[j] - dVTdphi_mod(phi[j], T[j]))

    Ftau = (tau + P)*v
    FZ = Z*v + P

    taudot[0] = (ho_advection_matrix[0][0] * Ftau[0] +
                 ho_advection_matrix[0][1] * Ftau[1]) / mass_matrix[0]
    taudot[N] = (ho_advection_matrix[N][-1] * Ftau[N-1] +
                 ho_advection_matrix[N][0] * Ftau[N]) / \
        mass_matrix[N]

    Zdot[0] = (ho_advection_matrix[0][0] * FZ[0] +
               ho_advection_matrix[0][1] * FZ[1] +
               sources_matrix[0][0] * P[0] +
               sources_matrix[0][1] * P[1]) / mass_matrix[0]
    Zdot[N] = (ho_advection_matrix[N][-1] * FZ[N-1] +
               ho_advection_matrix[N][0] * FZ[N] +
               sources_matrix[N][-1] * P[N-1] +
               sources_matrix[N][0] * P[N]) / \
        mass_matrix[N]

    for j in range(1, N):
        taudot[j] = (ho_advection_matrix[j][-1] * Ftau[j-1] +
                     ho_advection_matrix[j][0] * Ftau[j] +
                     ho_advection_matrix[j][1] * Ftau[j+1]) / \
            mass_matrix[j]
        Zdot[j] = (ho_advection_matrix[j][-1] * FZ[j-1] +
                   ho_advection_matrix[j][0] * FZ[j] +
                   ho_advection_matrix[j][1] * FZ[j+1] +
                   sources_matrix[j][-1] * P[j-1] +
                   sources_matrix[j][0] * P[j] +
                   sources_matrix[j][1] * P[j+1]) / \
            mass_matrix[j]

    # ------tau, Z evolution (half step)
    tau -= dt * taudot
    Z -= dt * Zdot

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]
    return phi, tau, Z, T, v, P, phidot, taudot, Zdot


def low_order_step(fi, ti, zi, Ti, vi, Pi, fidot, tidot, zidot):
    phidot = fidot
    taudot = tidot
    Zdot = zidot
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

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    # -------evaluation of advection matrix
    lo_advection_matrix = advection(v, ho_advection_matrix)

    taudot[0] = (lo_advection_matrix[0][0] * Ftau[0] +
                 lo_advection_matrix[0][1] * Ftau[1]) / \
        mass_matrix[0]
    taudot[N] = (lo_advection_matrix[N][-1] * Ftau[N-1] +
                 lo_advection_matrix[N][0] * Ftau[N]) / \
        mass_matrix[N]

    Zdot[0] = (lo_advection_matrix[0][0] * FZ[0] +
               lo_advection_matrix[0][1] * FZ[1] +
               (sources_matrix[0][0] + ho_advection_matrix[0][0]) * P[0] +
               (sources_matrix[0][1] + ho_advection_matrix[0][1]) * P[1]) / \
        mass_matrix[0]
    Zdot[N] = (lo_advection_matrix[N][-1] * FZ[N-1] +
               lo_advection_matrix[N][0] * FZ[N] +
               (sources_matrix[N][-1] + ho_advection_matrix[N][-1]) * P[N-1] +
               (sources_matrix[N][0] + ho_advection_matrix[N][0]) * P[N]) / \
        mass_matrix[N]

    for j in range(1, N):
        taudot[j] = (lo_advection_matrix[j][-1] * Ftau[j-1] +
                     lo_advection_matrix[j][0] * Ftau[j] +
                     lo_advection_matrix[j][1] * Ftau[j+1]) / \
            mass_matrix[j]
        Zdot[j] = (lo_advection_matrix[j][-1] * FZ[j-1] +
                   lo_advection_matrix[j][0] * FZ[j] +
                   lo_advection_matrix[j][1] * FZ[j+1] +
                   (sources_matrix[j][-1] +
                    ho_advection_matrix[j][-1]) * P[j-1] +
                   (sources_matrix[j][0] +
                    ho_advection_matrix[j][0]) * P[j] +
                   (sources_matrix[j][1] +
                    ho_advection_matrix[j][1]) * P[j+1]) / \
            mass_matrix[j]

    # ------tau, Z evolution (half step)
    tau -= 0.5 * dt * taudot
    Z -= 0.5 * dt * Zdot

    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])

    phidot[1] += dt * ((laplacian_matrix[1][1] * (phi[2] - phi[1])) /
                       kinetic_matrix[1] - dVTdphi_mod(phi[1], T[1]))
    for j in range(2, N):
        phidot[j] += dt * ((laplacian_matrix[j][1] * (phi[j+1] - phi[j]) +
                            laplacian_matrix[j][-1] * (phi[j] - phi[j-1])) /
                           kinetic_matrix[j] - dVTdphi_mod(phi[j], T[j]))

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    # ------tau, Z evolution (half step)
    tau -= 0.5 * dt * taudot
    Z -= 0.5 * dt * Zdot
    return phi, tau, Z, T, v, P, phidot, taudot, Zdot
