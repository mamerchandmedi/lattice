#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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
from parameters import Tc
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
from parameters import Tc, eta, index
# end: Importing parameters of the model.

# %% Importing initial values.
from initialization import phi, T, P, tau, Z, v
# end: Importing initial values.

# %% Import rescaled quantities.
from scaled import GeV, tauZphi2Tpv_mod
# end: Import rescaled quantities.

# # %% Import matrices of the high order integration scheme.
#from matrices_high_order import sources_matrix as sm
from matrices_high_order import advection_matrix as ho_advection_matrix
from matrices_high_order import mass_matrix
# # end: Import matrices of the high order integration scheme.

# # %% Import matrices of the low order integration scheme.
from matrices_low_order_dynamically import advection
# # end: Import matrices of the low order integration scheme.

# %% Turning on html files generation from Cython.
from Cython.Compiler import Options
Options.annotate = True
# end: Turning on html files generation from Cython.

# %% Import package for dynamical compilation of Cython code.
import pyximport
pyximport.install(inplace=True, language_level='3', build_in_temp=False)
# end: Import package for dynamical compilation of Cython code.

# %% Import evolution functions.
from Cevolution_functions import update_Tpv, evaluate_fluxes, evaluate_psi,\
    ho_update_taudot, ho_update_Zdot, update_phidot, lo_update_taudot, \
    lo_update_Zdot, lo_newton
# end: Import evolution functions.

# %% Importing FCT variables.
from Cflux_corrected_transport import antidiffusive_matrix
from flux_corrected_transport import Ubar
from Cflux_corrected_transport import trans_antidiffusive_matrix as trans_am
from flux_corrected_transport import P_vectors
from flux_corrected_transport import Q_values
from flux_corrected_transport import R_values
from Cflux_corrected_transport import limit_fluxes
from Cflux_corrected_transport import prelimiting
# end: Importing FCT variable.

# %% Importing characteristic variables.
from characteristic_variables import T_matrices2
# end: Importing characteristic variables.




# Prepare output folder.
path = 'profiles_standard'
if not os.path.isdir(path):
    os.mkdir(path)
# Prepare output folder.

# EVOLUTION #
psteps = 500
nsteps = 80000

# initial values
phidot = np.zeros((N + 1, ), dtype=float)
phi_H, tau_H, Z_H = cp.deepcopy(phi), cp.deepcopy(tau), cp.deepcopy(Z)
T_H, v_H, P_H = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
phidot_H, taudot_H, Zdot_H = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)
psi_H = np.zeros(r.size)

phi_L, tau_L, Z_L = cp.deepcopy(phi), cp.deepcopy(tau), cp.deepcopy(Z)
T_L, v_L, P_L = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
phidot_L, taudot_L, Zdot_L = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)
psi_L = np.zeros(r.size)

Ftau_H = np.ndarray((N + 1, ), dtype=float)
FZ_H = np.ndarray((N + 1, ), dtype=float)

taudot_H = np.ndarray((N + 1, ), dtype=float)
Gtau_H = np.ndarray((N + 1, ), dtype=float)

Zdot_H = np.ndarray((N + 1, ), dtype=float)
GZ_H = np.ndarray((N + 1, ), dtype=float)

Ftau_L = np.ndarray((N + 1, ), dtype=float)
FZ_L = np.ndarray((N + 1, ), dtype=float)

taudot_L = np.ndarray((N + 1, ), dtype=float)
Gtau_L = np.ndarray((N + 1, ), dtype=float)

Zdot_L = np.ndarray((N + 1, ), dtype=float)
GZ_L = np.ndarray((N + 1, ), dtype=float)

tautemp_L = np.ndarray((N + 1, ), dtype=float)
Ztemp_L = np.ndarray((N + 1, ), dtype=float)
phitemp_L = np.ndarray((N + 1, ), dtype=float)

tau_old = np.ndarray((N + 1, ), dtype=float)
Z_old = np.ndarray((N + 1, ), dtype=float)
v_old = np.ndarray((N + 1, ), dtype=float)

Tf_L = np.zeros((2*N, 2*N), dtype=float)
Tf_L_banded = np.zeros((7, 2*N), dtype=float)

antidiffusive_flux_Z = np.ndarray((N + 1, 3))
antidiffusive_flux_tau = np.ndarray((N + 1, 3))
A1_prim = np.ndarray((N + 1, 3))
A2_prim = np.ndarray((N + 1, 3))
alpha_limiter1 = np.ones((N + 1, 3))
alpha_limiter2 = np.ones((N + 1, 3))
limited_flux_tau = np.ndarray((N + 1, 3))
limited_flux_Z = np.ndarray((N + 1, 3))

# loop over time
for k in range(nsteps):
    t = k * dt
    t = k * dt
    # ############# HIGH ORDER ##################
    # ------ update variables
    evaluate_psi(phi_H, psi_H)
    evaluate_fluxes(tau_H, Z_H, P_H, v_H, Ftau_H, FZ_H)
    ho_update_taudot(phi_H, phidot_H, psi_H, T_H, v_H, Ftau_H, taudot_H, Gtau_H)
    ho_update_Zdot(phi_H, phidot_H, psi_H, T_H, P_H, v_H, FZ_H, Zdot_H, GZ_H)

    # ------tau, Z evolution (temporary values)
    tau_temp = tau_H - 0.5 * dt * taudot_H
    Z_temp = Z_H - 0.5 * dt * Zdot_H

    # ------phi evolution (half step)
    phi_H += 0.5 * dt * phidot_H

    # ------ update variables
    update_Tpv(T_H, P_H, v_H, phi_H, tau_temp, Z_temp)
    evaluate_fluxes(tau_H, Z_H, P_H, v_H, Ftau_H, FZ_H)

    phidot_H_old = cp.deepcopy(phidot_H)
    update_phidot(phi_H, phidot_H, psi_H, v_H, T_H)
    phidot_H_mean = (phidot_H + phidot_H_old) / 2
    evaluate_psi(phi_H, psi_H)
    
    ho_update_taudot(phi_H, phidot_H_mean, psi_H, T_H, v_H, Ftau_H, taudot_H, Gtau_H)
    ho_update_Zdot(phi_H, phidot_H_mean, psi_H, T_H, P_H, v_H, FZ_H, Zdot_H, GZ_H)

    # ------tau, Z evolution (full step)
    #tau_H -= dt * taudot_H # Not needed when FCT used.
    #Z_H -= dt * Zdot_H # Not needed when FCT used.

    # ------phi evolution (half step)
    phi_H += 0.5 * dt * phidot_H

    # ################LOW ORDER##################
    update_Tpv(T_L, P_L, v_L, phi_L, tau_L, Z_L)
    Ftau_L = tau_L + P_L
    FZ_L = cp.deepcopy(Z_L)
    
    v_old = cp.deepcopy(v_L)
    tau_old = cp.deepcopy(tau_L)
    Z_old = cp.deepcopy(Z_L)

    # ------ update variables
    KL = advection(v_L)
    evaluate_psi(phi_L, psi_L)
    lo_update_taudot(phi_L, phidot_L, psi_L, T_L, v_L, Ftau_L, KL, taudot_L, Gtau_L)
    lo_update_Zdot(phi_L, phidot_L, psi_L, T_L, P_L, v_L, FZ_L, KL, Zdot_L, GZ_L)

    # ------phi evolution (half step)
    phi_L += 0.5 * dt * phidot_L

    # ------tau, Z evolution (half step)
    tau_L -= 0.5 * dt * taudot_L
    Z_L -= 0.5 * dt * Zdot_L

    #update_Tpv(T_L, P_L, v_L, phi_L, tau_L, Z_L)
    update_phidot(phi_L, phidot_L, psi_L, v_L, T_L)

    # ------phi evolution (half step)
    phi_L += 0.5 * dt * phidot_L

    # ------tau, Z evolution (half step)
    tau_L -= 0.5 * dt * taudot_L
    Z_L -= 0.5 * dt * Zdot_L
    
    KL_old = advection(v_old)
    rtol = 1.e-5 #1.e-6
    atol = 1.e-5
    while(True):
        tau_Lm = cp.deepcopy(tau_L)
        Z_Lm = cp.deepcopy(Z_L)
        Ftau_L = tau_L + P_L
        FZ_L = cp.deepcopy(Z_L)
        lo_update_taudot(phi_L, phidot_L, psi_L, T_L, v_L, Ftau_L, KL, taudot_L, Gtau_L)
        lo_update_Zdot(phi_L, phidot_L, psi_L, T_L, P_L, v_L, FZ_L, KL, Zdot_L, GZ_L) 
        update_Tpv(T_L, P_L, v_L, phi_L, tau_L, Z_L)
        lo_newton(tau_old, tau_L, taudot_L, Z_old, Z_L, Zdot_L, Tf_L, Tf_L_banded, phi_L, phidot_L, psi_L, T_L, v_L, P_L, KL_old)
        tau_mean = np.mean(tau_L[0:-2]**2 + tau_Lm[0:-2]**2)
        Z_mean = np.mean(Z_L[0:-2]**2 + Z_Lm[0:-2]**2)
        if max(abs((tau_L[0:-2] - tau_Lm[0:-2])/np.sqrt(tau_L[0:-2]**2 + tau_Lm[0:-2]**2 + atol*np.sqrt(tau_mean)))) <=rtol and max(abs((Z_L[0:-2] - Z_Lm[0:-2])/np.sqrt(Z_L[0:-2]**2 + Z_Lm[0:-2]**2 + atol*np.sqrt(Z_mean)))) <= rtol: break

    # ###########FCT###############
    antidiffusive_matrix(KL, FZ_H, FZ_L, dt, antidiffusive_flux_Z)
    antidiffusive_matrix(KL, Ftau_H, Ftau_L, dt, antidiffusive_flux_tau)

    prelimiting(antidiffusive_flux_tau, tau_L)
    prelimiting(antidiffusive_flux_Z, Z_L)
    limit_fluxes(antidiffusive_flux_tau, antidiffusive_flux_Z,
                 tau, Z, phi, v,
                 tau_L, Z_L, phi_L,# tau_bar, Z_bar, phi_bar,
                 alpha_limiter1, alpha_limiter2,
                 limited_flux_tau, limited_flux_Z)


    tau_H[0] = tau_L[0] + (-dt * (Gtau_H[0] - Gtau_L[0]) +
                           limited_flux_tau[0][0] +
                           limited_flux_tau[0][1]) / mass_matrix[0]
    Z_H[0] = Z_L[0] + (-dt * (GZ_H[0] - GZ_L[0]) +
                       limited_flux_Z[0][0] +
                       limited_flux_Z[0][1]) / mass_matrix[0]

    for i in range(1, N-1):
        tau_H[i] = tau_L[i] + (-dt * (Gtau_H[i] - Gtau_L[i]) +
                               limited_flux_tau[i][-1] +
                               limited_flux_tau[i][0] +
                               limited_flux_tau[i][1]) / mass_matrix[i]
        Z_H[i] = Z_L[i] + (-dt * (GZ_H[i] - GZ_L[i]) +
                           limited_flux_Z[i][-1] +
                           limited_flux_Z[i][0] +
                           limited_flux_Z[i][1]) / mass_matrix[i]

    tau_H[N-1] = tau_L[N-1] + (-dt * (Gtau_H[N-1] - Gtau_L[N-1]) +
                           limited_flux_tau[N-1][-1] +
                           limited_flux_tau[N-1][0]) / mass_matrix[N-1]
    Z_H[N-1] = Z_L[N-1] + (-dt * (GZ_H[N-1] - GZ_L[N-1]) +
                       limited_flux_Z[N-1][0] +
                       limited_flux_Z[N-1][-1]) / mass_matrix[N-1]

    update_Tpv(T_H, P_H, v_H, phi_H, tau_H, Z_H)

    tau_L = cp.deepcopy(tau_H)
    tau = cp.deepcopy(tau_H)
    Z_L = cp.deepcopy(Z_H)
    Z = cp.deepcopy(Z_H)
    phi_L = cp.deepcopy(phi_H)
    phi = cp.deepcopy(phi_H)
    phidot_L = cp.deepcopy(phidot_H)
    phidot = cp.deepcopy(phidot_H)
    T_L = cp.deepcopy(T_H)
    P_L = cp.deepcopy(P_H)
    v_L = cp.deepcopy(v_H)

    # #########SAVING############
    if((k+1) % psteps == 0):
        np.savez_compressed('profiles_standard/time_'+str(format(t+dt, ".2f")),
                            r=r, xi=r/t, phi=phi_H, phidot=phidot_H,
                            v=v_H, T=T_H, P=P_H, tau=tau_H, Z=Z_H, 
                            alpha1=alpha_limiter1, alpha2=alpha_limiter2)
