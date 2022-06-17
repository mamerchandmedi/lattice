#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve
import matplotlib.pylab as pl
from tqdm import tqdm
import copy as cp

# %% Import system dependent functionality.
import os
# end: Import system dependent functionality.

# %% Import package for handling command line arguments.
import sys
# end: Import package for handling command line arguments.

# %% Import package providing utility for importing modules by filename.
import imp
# end: Import package providing utility for importing modules by filename.

# %% Importing parameters of the model.
from parameters import Tc, L, sigma, lc, T0, alpha, gamma, lam, Tn
# end: Importing parameters of the model.

# %% Processing file name.
data_file = sys.argv[1]
print("Processing " + data_file + ".")
file_path = os.path.split(data_file)[0]
file_name = os.path.splitext(os.path.split(data_file)[1])[0]
# end: Processing file name.

# %% Importing file with parameters of the model.
parameters_data = imp.load_source('parameters_data', data_file)
from parameters_data import *
# end: Importing file with parameters of the model.

import parameters 
if 'eta' in locals():
    parameters.eta = eta

# %% Import matrices of lattice definition.
from lattice import dr, dt, r, N
# end: Import matrices of lattice definition.

# %% Importing parameters of the model.
from parameters import Tc, L, sigma, lc, T0, alpha, gamma, lam, Tn, eta
# end: Importing parameters of the model.

# %% Importing initial values.
from initialization import phi, T, P, tau, Z, v
# end: Importing initial values.

# %% Import rescaled quantities.
from scaled import GeV, tauZphi2Tpv_mod
from solver import tauZphi2Tpv
# end: Import rescaled quantities.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import sources_matrix as sm
from matrices_high_order import advection_matrix as ho_advection_matrix
from matrices_high_order import mass_matrix
# end: Import matrices of the high order integration scheme.

# %% Import matrices of the low order integration scheme.
from matrices_low_order_dynamically import advection
# end: Import matrices of the low order integration scheme.

# %% Import evolution functions.
from evolution_functions import update_Tpv
from evolution_functions import evaluate_fluxes
from evolution_functions import ho_update_taudot
from evolution_functions import ho_update_Zdot
from evolution_functions import lo_update_taudot
from evolution_functions import lo_update_Zdot
from evolution_functions import update_phidot
# end: Import evolution functions.

# %% Importing FCT variables.
from FCT import antidiffusive_matrix
from FCT import Ubar
from FCT import trans_antidiffusive_matrix as trans_am
from FCT import P_vectors
from FCT import Q_values
from FCT import R_values
# end: Importing FCT variable.

# %% Importing characteristic variables.
from characteristic_variables import T_matrices2
# end: Importing characteristic variables.

# EVOLUTION #

psteps = 10
nsteps = 250

# initial values
phi_H, tau_H, Z_H = cp.deepcopy(phi), cp.deepcopy(tau), cp.deepcopy(Z)
T_H, v_H, P_H = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
phidot_H, taudot_H, Zdot_H = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)

phi_L, tau_L, Z_L = cp.deepcopy(phi), cp.deepcopy(tau), cp.deepcopy(Z)
T_L, v_L, P_L = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
phidot_L, taudot_L, Zdot_L = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)

# loop over time
for k in range(nsteps):
    t = k * dt
    # ############# HIGH ORDER ##################
    # ------ update variables
    T_H, P_H, v_H = update_Tpv(T_H, P_H, v_H, phi_H, tau_H, Z_H)
    Ftau_H, FZ_H = evaluate_fluxes(tau_H, Z_H, P_H, v_H)
    taudot_H, Gtau_H = ho_update_taudot(phi_H, phidot_H, T_H, v_H, Ftau_H)
    Zdot_H, GZ_H = ho_update_Zdot(phi_H, phidot_H, T_H, P_H, v_H, FZ_H)

    # ------tau, Z evolution (temporary values)
    tau_temp = tau_H - 0.5 * dt * taudot_H
    Z_temp = Z_H - 0.5 * dt * Zdot_H

    # ------phi evolution (half step)
    phi_H[1:] += 0.5 * dt * phidot_H[1:]
    phi_H[0] = phi_H[1]

    # ------ update variables
    T_H, P_H, v_H = update_Tpv(T_H, P_H, v_H, phi_H, tau_temp, Z_temp)
    Ftau_H, FZ_H = evaluate_fluxes(tau_H, Z_H, P_H, v_H)

    phidot_H_old = cp.deepcopy(phidot_H)
    phidot_H = update_phidot(phi_H, phidot_H, v_H, T_H)
    phidot_H_mean = (phidot_H + phidot_H_old) / 2

    taudot_H, Gtau_H = ho_update_taudot(phi_H, phidot_H_mean, T_H, v_H, Ftau_H)
    Zdot_H, GZ_H = ho_update_Zdot(phi_H, phidot_H_mean, T_H, P_H, v_H, FZ_H)

    # ------tau, Z evolution (full step)
    tau_H -= dt * taudot_H
    Z_H -= dt * Zdot_H

    # ------phi evolution (half step)
    phi_H[1:] += 0.5 * dt * phidot_H[1:]
    phi_H[0] = phi_H[1]

    # ################LOW ORDER##################
    T_L, P_L, v_L = update_Tpv(T_L, P_L, v_L, phi_L, tau_L, Z_L)
    Ftau_L = tau_L + P_L
    FZ_L = cp.deepcopy(Z_L)

    # ------ update variables
    KL = advection(v_L)
    taudot_L, Gtau_L = lo_update_taudot(phi_L, phidot_L, T_L, v_L, Ftau_L, KL)
    Zdot_L, GZ_L = lo_update_Zdot(phi_L, phidot_L, T_L, P_L, v_L, FZ_L, KL)

    # ------phi evolution (half step)
    phi_L[1:] += 0.5 * dt * phidot_L[1:]
    phi_L[0] = phi_L[1]

    # ------tau, Z evolution (half step)
    tau_L -= 0.5 * dt * taudot_L
    Z_L -= 0.5 * dt * Zdot_L

    T_L, P_L, v_L = update_Tpv(T_L, P_L, v_L, phi_L, tau_L, Z_L)
    phidot_L = update_phidot(phi_L, phidot_L, v_L, T_L)

    # ------phi evolution (half step)
    phi_L[1:] += 0.5 * dt * phidot_L[1:]
    phi_L[0] = phi_L[1]

    # ------tau, Z evolution (half step)
    tau_L -= 0.5 * dt * taudot_L
    Z_L -= 0.5 * dt * Zdot_L

    # ###########FCT###############
    KH = ho_advection_matrix
    tau_bar = Ubar(tau_L)
    Z_bar = Ubar(Z_L)
    antidiffusive_flux_Z = antidiffusive_matrix(KH, KL, FZ_H, FZ_L, dt)
    antidiffusive_flux_tau = antidiffusive_matrix(KH, KL, Ftau_H, Ftau_L, dt)
    A1_prim, A2_prim = \
        trans_am(antidiffusive_flux_tau, antidiffusive_flux_Z,
                 tau_bar, Z_bar, phi_L, T_L, P_L, v_L)
    P1_plus, P1_minus = P_vectors(A1_prim)
    P2_plus, P2_minus = P_vectors(A2_prim)
    R1_plus = np.ndarray((N+1,))
    R1_minus = np.ndarray((N+1,))
    R2_plus = np.ndarray((N+1,))
    R2_minus = np.ndarray((N+1,))
    alpha_limiter1 = np.ndarray((N+1, 3))
    alpha_limiter2 = np.ndarray((N+1, 3))
    limited_flux_tau = np.ndarray((N+1, 3))
    limited_flux_Z = np.ndarray((N+1, 3))

    for i in range(N+1):
        if i == 0:
            indexes = [0, 1]
        elif i == N:
            indexes = [-1, 0]
        else:
            indexes = [-1, 0, 1]
        for j in indexes:
            Ti, Pi, vi = tauZphi2Tpv_mod(tau_bar[i][j], Z_bar[i][j], phi_L[i],
                                         T_L[i], P_L[i], v_L[i])
            T_mat, T_inv = T_matrices2(vi, phi_L[i], Ti)
            tau_prim = T_inv[0][0] * tau_L + T_inv[0][1] * Z_L
            Z_prim = T_inv[1][0] * tau_L + T_inv[1][1] * Z_L
            Q1_plus_i, Q1_minus_i = Q_values(tau_prim, i)
            Q2_plus_i, Q2_minus_i = Q_values(Z_prim, i)
            R1_plus[i], R1_minus[i] = \
                R_values(Q1_plus_i, Q1_minus_i,
                         P1_plus[i], P1_minus[i])
            R2_plus[i], R2_minus[i] = \
                R_values(Q2_plus_i, Q2_minus_i,
                         P2_plus[i], P2_minus[i])
    indexes = [0, 1]
    for j in indexes:
        limited_flux_tau[0][j] = \
                T_mat[0][0] * A1_prim[0][j] + \
                T_mat[0][1] * A2_prim[0][j]
        limited_flux_Z[0][j] = \
                T_mat[1][0] * A1_prim[0][j] + \
                T_mat[1][1] * A2_prim[0][j]
    for i in range(1, N+1):
        if i == 0:
            indexes = [0, 1]
        elif i == N:
            indexes = [-1, 0]
        else:
            indexes = [-1, 0, 1]
        for j in indexes:
            Ti, Pi, vi = tauZphi2Tpv_mod(tau_bar[i][j], Z_bar[i][j], phi_L[i],
                                         T_L[i], P_L[i], v_L[i])
            T_mat, T_inv = T_matrices2(vi, phi_L[i], Ti)
            if A1_prim[i][j] >= 0:
                alpha_limiter1[i][j] = min(R1_plus[i], R1_minus[i+j])
            else:
                alpha_limiter1[i][j] = min(R1_minus[i], R1_plus[i+j])
            if A2_prim[i][j] >= 0:
                alpha_limiter2[i][j] = min(R2_plus[i], R2_minus[i+j])
            else:
                alpha_limiter2[i][j] = min(R2_minus[i], R2_plus[i+j])
            limited_flux_tau[i][j] = \
                T_mat[0][0] * A1_prim[i][j] * alpha_limiter1[i][j] + \
                T_mat[0][1] * A2_prim[i][j] * alpha_limiter2[i][j]
            limited_flux_Z[i][j] = \
                T_mat[1][0] * A1_prim[i][j] * alpha_limiter1[i][j] + \
                T_mat[1][1] * A2_prim[i][j] * alpha_limiter2[i][j]

    tau_H[0] = tau_L[0] - (dt * (Gtau_H[0] - Gtau_L[0]) +
                           limited_flux_tau[0][0] +
                           limited_flux_tau[0][1]) / mass_matrix[0]
    tau_L[0] = tau_H[0]
    Z_H[0] = Z_L[0] - (dt * (GZ_H[0] - GZ_L[0]) +
                       limited_flux_Z[0][0] +
                       limited_flux_Z[0][1]) / mass_matrix[0]
    Z_L[0] = Z_H[0]

    phi_L[0] = phi_H[0]
    phidot_L[0] = phidot_H[0]

    for i in range(1, N):
        tau_H[i] = tau_L[i] - (dt * (Gtau_H[i] - Gtau_L[i]) +
                               limited_flux_tau[i][-1] +
                               limited_flux_tau[i][0] +
                               limited_flux_tau[i][1]) / mass_matrix[i]
        tau_L[i] = tau_H[i]
        Z_H[i] = Z_L[i] - (dt * (GZ_H[i] - GZ_L[i]) +
                           limited_flux_Z[i][-1] +
                           limited_flux_Z[i][0] +
                           limited_flux_Z[i][1]) / mass_matrix[i]
        Z_L[i] = Z_H[i]

        phi_L[i] = phi_H[i]
        phidot_L[i] = phidot_H[i]

    tau_H[N] = tau_L[N] - (dt * (Gtau_H[N] - Gtau_L[N]) +
                           limited_flux_tau[N][-1] +
                           limited_flux_tau[N][0]) / mass_matrix[N]
    tau_L[N] = tau_H[N]
    Z_H[N] = Z_L[N] - (dt * (GZ_H[N] - GZ_L[N]) +
                       limited_flux_Z[N][-1] +
                       limited_flux_Z[N][0]) / mass_matrix[N]
    Z_L[N] = Z_H[N]

    phi_L[N] = phi_H[N]
    phidot_L[N] = phidot_H[N]

    # #########SAVING############
    if((k+1) % psteps == 0):
        np.savez_compressed('profiles_eta_'+str(eta/Tc)+'/t_'+str(t), r=r, xi=r/t, phi=phi_H, v=v_H, T=T_H, P=P_H, tau=tau_H, Z=Z_H)
