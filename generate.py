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

import pandas as pd
df = pd.read_csv('~/repo_scalar_singlet/points.csv')

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

# %% Importing parameters of the model.
from parameters_friction import eta_h, eta_s
# end: Importing parameters of the model.

import parameters
from parameters import lamh, g1, g, yt
if 'index' in locals():
    parameters.index = index
from parameters import index

parameters.Tc = df["Tc"].loc[index]
parameters.Tn = df["Tnuc"].loc[index]
parameters.alpha = df["alpha"].loc[index]
parameters.ms = df["ms"].loc[index]
parameters.lams = df["lams"].loc[index]
parameters.lamhs = df["lammix"].loc[index]
parameters.cs = 1/12 * (2*lamhs + 3*lams)
parameters.ch = 1/48 * (9*g**2 + 3*g1**2 + 12*yt**2 + 24*lamh + 2*lamhs)

#import parameters_friction
#if 'eta_h' in locals():
#    parameters_friction.eta_h = eta_h
#if 'eta_s' in locals():
#    parameters_friction.eta_s = eta_s
# %% Import matrices of lattice definition.

# %% Importing parameters of the model.
from parameters_friction import eta_h, eta_s
# end: Importing parameters of the model.

from lattice import dr, dt, r, N, GeV
# end: Import matrices of lattice definition.

# %% Importing initial values.
from initialization import h, s, T, P, tau, Z, v
# end: Importing initial values.

# %% Import rescaled quantities.
#from scaled import GeV, tauZphi2Tpv_mod
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
Options.error_on_uninitialized = False
# end: Turning on html files generation from Cython.

# %% Import package for dynamical compilation of Cython code.
import pyximport
pyximport.install(inplace=True, language_level='3', build_in_temp=False)
# end: Import package for dynamical compilation of Cython code.

# %% Import evolution functions.
from Cevolution_functions import update_Tpv, evaluate_fluxes, evaluate_psi,\
    ho_update_taudot, ho_update_Zdot, update_hdot, update_sdot, lo_update_taudot, \
    lo_update_Zdot, lo_newton, lo_newton_fast
# end: Import evolution functions.

# %% Importing FCT variables.
from Cflux_corrected_transport import antidiffusive_matrix
#from flux_corrected_transport import Ubar
from Cflux_corrected_transport import trans_antidiffusive_matrix as trans_am
#from flux_corrected_transport import P_vectors
#from flux_corrected_transport import Q_values
#from flux_corrected_transport import R_values
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
nsteps = 120000

# initial values
hdot = np.zeros((N + 1, ), dtype=float)
sdot = np.zeros((N + 1, ), dtype=float)
h_H, s_H, tau_H, Z_H = cp.deepcopy(h), cp.deepcopy(s), cp.deepcopy(tau), cp.deepcopy(Z)
T_H, v_H, P_H = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
hdot_H, sdot_H, taudot_H, Zdot_H = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)
psi_h_H, psi_s_H = np.zeros(r.size), np.zeros(r.size)

h_L, s_L, tau_L, Z_L = cp.deepcopy(h), cp.deepcopy(s), cp.deepcopy(tau), cp.deepcopy(Z)
T_L, v_L, P_L = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
hdot_L, sdot_L, taudot_L, Zdot_L = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)
psi_h_L, psi_s_L = np.zeros(r.size), np.zeros(r.size)

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
htemp_L = np.ndarray((N + 1, ), dtype=float)
stemp_L = np.ndarray((N + 1, ), dtype=float)

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
    # ############# HIGH ORDER ##################
    # ------ update variables
    evaluate_psi(h_H, psi_h_H)
    evaluate_psi(s_H, psi_s_H)
    evaluate_fluxes(tau_H, Z_H, P_H, v_H, Ftau_H, FZ_H)
    ho_update_taudot(h_H, s_H, hdot_H, sdot_H, psi_h_H, psi_s_H, T_H, v_H, Ftau_H, taudot_H, Gtau_H)
    ho_update_Zdot(h_H, s_H, hdot_H, sdot_H, psi_h_H, psi_s_H, T_H, P_H, v_H, FZ_H, Zdot_H, GZ_H)

    # ------tau, Z evolution (temporary values)
    tau_temp = tau_H - 0.5 * dt * taudot_H
    Z_temp = Z_H - 0.5 * dt * Zdot_H

    # ------phi evolution (half step)
    h_H += 0.5 * dt * hdot_H
    s_H += 0.5 * dt * sdot_H

    # ------ update variables
    update_Tpv(T_H, P_H, v_H, h_H, s_H, tau_temp, Z_temp)
    evaluate_fluxes(tau_H, Z_H, P_H, v_H, Ftau_H, FZ_H)

    hdot_H_old = cp.deepcopy(hdot_H)
    sdot_H_old = cp.deepcopy(sdot_H)
    update_hdot(h_H, s_H, hdot_H, psi_h_H, v_H, T_H)
    update_sdot(h_H, s_H, sdot_H, psi_s_H, v_H, T_H)
    hdot_H_mean = (hdot_H + hdot_H_old) / 2
    sdot_H_mean = (sdot_H + sdot_H_old) / 2
    evaluate_psi(h_H, psi_h_H)
    evaluate_psi(s_H, psi_s_H)
    
    ho_update_taudot(h_H, s_H, hdot_H, sdot_H, psi_h_H, psi_s_H, T_H, v_H, Ftau_H, taudot_H, Gtau_H)
    ho_update_Zdot(h_H, s_H, hdot_H, sdot_H, psi_h_H, psi_s_H, T_H, P_H, v_H, FZ_H, Zdot_H, GZ_H)

    # ------tau, Z evolution (full step)
    #tau_H -= dt * taudot_H # Not needed when FCT used.
    #Z_H -= dt * Zdot_H # Not needed when FCT used.

    # ------phi evolution (half step)
    h_H += 0.5 * dt * hdot_H
    s_H += 0.5 * dt * sdot_H

    # ################LOW ORDER##################
    update_Tpv(T_L, P_L, v_L, h_L, s_L, tau_L, Z_L)
    Ftau_L = tau_L + P_L
    FZ_L = cp.deepcopy(Z_L)
    
    v_old = cp.deepcopy(v_L)
    tau_old = cp.deepcopy(tau_L)
    Z_old = cp.deepcopy(Z_L)

    # ------ update variables
    KL = advection(v_L)
    evaluate_psi(h_L, psi_h_L)
    evaluate_psi(s_L, psi_s_L)
    lo_update_taudot(h_L, s_L, hdot_L, sdot_L, psi_h_L, psi_s_L, T_L, v_L, Ftau_L, KL, taudot_L, Gtau_L)
    lo_update_Zdot(h_L, s_L, hdot_L, sdot_L, psi_h_L, psi_s_L, T_L, P_L, v_L, FZ_L, KL, Zdot_L, GZ_L)

    # ------phi evolution (half step)
    h_L += 0.5 * dt * hdot_L
    s_L += 0.5 * dt * sdot_L

    # ------tau, Z evolution (half step)
    tau_L -= 0.5 * dt * taudot_L
    Z_L -= 0.5 * dt * Zdot_L

    update_Tpv(T_L, P_L, v_L, h_L, s_L, tau_L, Z_L)
    update_hdot(h_L, s_L, hdot_L, psi_h_L, v_L, T_L)
    update_sdot(h_L, s_L, sdot_L, psi_s_L, v_L, T_L)

    # ------phi evolution (half step)
    h_L += 0.5 * dt * hdot_L
    s_L += 0.5 * dt * sdot_L

    # ------tau, Z evolution (half step)
    tau_L -= 0.5 * dt * taudot_L
    Z_L -= 0.5 * dt * Zdot_L
    
    #print(tau_L[0], Z_L[0])
    KL_old = advection(v_old)
    rtol = 1.e-5 #1.e-6
    atol = 1.e-5
    i = 0
    while(True):
        i += 1
        tau_Lm = cp.deepcopy(tau_L)
        Z_Lm = cp.deepcopy(Z_L)
        Ftau_L = tau_L + P_L
        FZ_L = cp.deepcopy(Z_L)
        lo_update_taudot(h_L, s_L, hdot_L, sdot_L, psi_h_L, psi_s_L, T_L, v_L, Ftau_L, KL, taudot_L, Gtau_L)
        lo_update_Zdot(h_L, s_L, hdot_L, sdot_L, psi_h_L, psi_s_L, T_L, P_L, v_L, FZ_L, KL, Zdot_L, GZ_L) 
        update_Tpv(T_L, P_L, v_L, h_L, s_L, tau_L, Z_L)
        lo_newton_fast(tau_old, tau_L, taudot_L, Z_old, Z_L, Zdot_L, Tf_L_banded, h_L, s_L, hdot_L, sdot_L, psi_h_L, psi_s_L, T_L, v_L, P_L, KL_old)
        tau_mean = np.mean(tau_L[0:-2]**2 + tau_Lm[0:-2]**2)
        Z_mean = np.mean(Z_L[0:-2]**2 + Z_Lm[0:-2]**2)
        if max(abs((tau_L[0:-2] - tau_Lm[0:-2])/np.sqrt(tau_L[0:-2]**2 + tau_Lm[0:-2]**2 + atol*np.sqrt(tau_mean)))) <=rtol and max(abs((Z_L[0:-2] - Z_Lm[0:-2])/np.sqrt(Z_L[0:-2]**2 + Z_Lm[0:-2]**2 + atol*np.sqrt(Z_mean)))) <= rtol: break

    # ###########FCT###############

    antidiffusive_matrix(KL, FZ_H, FZ_L, dt, antidiffusive_flux_Z)
    antidiffusive_matrix(KL, Ftau_H, Ftau_L, dt, antidiffusive_flux_tau)
    
    for m in [1,2,3]:
        prelimiting(antidiffusive_flux_tau, tau_L)
        prelimiting(antidiffusive_flux_Z, Z_L)
        tau_H = cp.deepcopy(tau_L)
        Z_H = cp.deepcopy(Z_L)
        limit_fluxes(antidiffusive_flux_tau, antidiffusive_flux_Z,
                 tau, Z, h, s, v,
                 tau_L, Z_L, h_L, s_L,# tau_bar, Z_bar, phi_bar,
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
        antidiffusive_flux_tau -= limited_flux_tau
        antidiffusive_flux_Z -= limited_flux_Z

    update_Tpv(T_H, P_H, v_H, h_H, s_H, tau_H, Z_H)
    tau_L = cp.deepcopy(tau_H)
    tau = cp.deepcopy(tau_H)
    Z_L = cp.deepcopy(Z_H)
    Z = cp.deepcopy(Z_H)
    h_L = cp.deepcopy(h_H)
    s_L = cp.deepcopy(s_H)
    h = cp.deepcopy(h_H)
    s = cp.deepcopy(s_H)
    hdot_L = cp.deepcopy(hdot_H)
    sdot_L = cp.deepcopy(sdot_H)
    hdot = cp.deepcopy(hdot_H)
    sdot = cp.deepcopy(sdot_H)
    T_L = cp.deepcopy(T_H)
    P_L = cp.deepcopy(P_H)
    v_L = cp.deepcopy(v_H)

    # #########SAVING############
    if((k+1) % psteps == 0):
        np.savez_compressed('profiles/time_'+str(format(t+dt, ".2f")),
                            r=r, xi=r/t, h=h_H, s=s_H, hdot=hdot_H, sdot=sdot_H, psi_h = psi_h_H, psi_s = psi_s_H,
                            v=v_H, T=T_H, P=P_H, tau=tau_H, Z=Z_H, 
                            alpha1=alpha_limiter1, alpha2=alpha_limiter2, eta_h = eta_h, eta_s = eta_s)
