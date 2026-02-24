import gc
gc.collect()
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy as cp   
# %% Import matrices of lattice definition.
from lattice import dr, dt, r, N, GeV
# end: Import matrices of lattice definition.

# %% Importing parameters of the model.
from parameters import eta, index
# end: Importing parameters of the model.

# %% Importing initial values.
from initialization import phi, T, P, tau, Z, v
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
    ho_update_taudot, ho_update_Zdot, update_phidot, lo_update_taudot, \
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

# %% Import potential derivatives.
from potential import dVTdT, d2VTdT2
# end: Import potential derivatives.

# %% Importing characteristic variables.
#from characteristic_variables import T_matrices2
# end: Importing characteristic variables.

import csv

# EVOLUTION #
#plt.figure(1)
fig, axs = plt.subplots(4, 2)
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)
# ------plotting
plt.xlim(0, 10)
#plt.rcParams["figure.dpi"] = 400
# plt.ylim(0,0.1)
fig.suptitle("dr = " + str(dr / GeV**(-1)) + ", dt = " +
             str(dt / GeV**(-1)) + r', $\eta=$' + str(eta))
axs[0, 0].set_ylabel(r'$\phi$')
axs[1, 0].set_ylabel(r'$v$')
axs[2, 0].set_ylabel(r'$Z$')
axs[3, 0].set_ylabel(r'$\tau$')
axs[3, 0].set_xlabel(r'$\xi$')

axs[0, 1].set_ylabel(r'$\psi$')
axs[1, 1].set_ylabel(r'$T$')
axs[2, 1].set_ylabel(r'$p$')
axs[3, 1].set_ylabel(r'$c_s$')
axs[3, 1].set_xlabel(r'$\xi$')

axs[0, 0].set_xlim(0, 1)
axs[1, 0].set_xlim(0, 1)
axs[2, 0].set_xlim(0, 1)
axs[3, 0].set_xlim(0, 1)

axs[0, 1].set_xlim(0, 1)
axs[1, 1].set_xlim(0, 1)
axs[2, 1].set_xlim(0, 1)
axs[3, 1].set_xlim(0, 1)

fig_diag, axs_diag = plt.subplots(3, 2)
fig_diag.subplots_adjust(hspace=0.5)
fig_diag.subplots_adjust(wspace=0.5)

#axs_diag[0, 0].set_ylabel(r'$\alpha 1_{-1}$')
#axs_diag[1, 0].set_ylabel(r'$\alpha 1_{0}$')
#axs_diag[2, 0].set_ylabel(r'$\alpha 1_{1}$')
#axs_diag[2, 0].set_xlabel(r'$r$')

#axs_diag[0, 1].set_ylabel(r'$\alpha 2_{-1}$')
#axs_diag[1, 1].set_ylabel(r'$\alpha 2_{0}$')
#axs_diag[2, 1].set_ylabel(r'$\alpha 2_{1}$')
#axs_diag[2, 1].set_xlabel(r'$r$')

axs_diag[0, 0].set_ylabel(r'$F\tau_{-1}$')
axs_diag[1, 0].set_ylabel(r'$F\tau_{0}$')
axs_diag[2, 0].set_ylabel(r'$F\tau 1_{1}$')
axs_diag[2, 0].set_xlabel(r'$r$')

axs_diag[0, 1].set_ylabel(r'$FZ_{-1}$')
axs_diag[1, 1].set_ylabel(r'$FZ_{0}$')
axs_diag[2, 1].set_ylabel(r'$FZ_{1}$')
axs_diag[2, 1].set_xlabel(r'$r$')

axs_diag[0, 0].set_xlim(0, 1)
axs_diag[1, 0].set_xlim(0, 1)
axs_diag[2, 0].set_xlim(0, 1)

axs_diag[0, 1].set_xlim(0, 1)
axs_diag[1, 1].set_xlim(0, 1)
axs_diag[2, 1].set_xlim(0, 1)

#axs_diag[0, 0].set_ylim(0, 1)
#axs_diag[1, 0].set_ylim(0, 1)
#axs_diag[2, 0].set_ylim(0, 1)

#axs_diag[0, 1].set_ylim(0, 1.)
#axs_diag[1, 1].set_ylim(0, 1.)
#axs_diag[2, 1].set_ylim(0, 1.)

psteps = 500
nsteps = 7000

# initial values
phidot = np.zeros((N + 1, ), dtype=float)
phi_H, tau_H, Z_H = cp.deepcopy(phi), cp.deepcopy(tau), cp.deepcopy(Z)
T_H, v_H, P_H = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
q_H = np.zeros(r.size)
phidot_H, taudot_H, Zdot_H = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)
psi_H = np.zeros(r.size)

phi_L, tau_L, Z_L = cp.deepcopy(phi), cp.deepcopy(tau), cp.deepcopy(Z)
T_L, v_L, P_L = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
q_L = np.zeros(r.size)
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

#Tf_L = np.zeros((2*N, 2*N), dtype=float)
Tf_L_banded = np.zeros((7, 2*N), dtype=float)

cs2 = np.ndarray((N + 1, ), dtype=float)

antidiffusive_flux_Z = np.ndarray((N + 1, 3), dtype=float)
antidiffusive_flux_tau = np.ndarray((N + 1, 3), dtype=float)
A1_prim = np.ndarray((N + 1, 3), dtype=float)
A2_prim = np.ndarray((N + 1, 3), dtype=float)
alpha_limiter1 = np.ones((N + 1, 3), dtype=float)
alpha_limiter2 = np.ones((N + 1, 3), dtype=float)
limited_flux_tau = np.zeros((N + 1, 3), dtype=float)
limited_flux_Z = np.zeros((N + 1, 3), dtype=float)
n_run = 1
# loop over time
for k in tqdm(range(nsteps)):
    t = k * dt
    # ############# HIGH ORDER ##################
    # ------ update variables
    evaluate_psi(phi_H, psi_H)
    evaluate_psi(P_H, q_H)
    evaluate_fluxes(tau_H, Z_H, P_H, v_H, Ftau_H, FZ_H)
    ho_update_taudot(phi_H, phidot_H, psi_H, T_H, v_H, Ftau_H, taudot_H, Gtau_H)
    ho_update_Zdot(phi_H, phidot_H, psi_H, T_H, P_H, v_H, q_H, FZ_H, Zdot_H, GZ_H)

    # ------tau, Z evolution (temporary values)
    tau_temp = tau_H - 0.5 * dt * taudot_H
    Z_temp = Z_H - 0.5 * dt * Zdot_H

    # ------phi evolution (half step)
    phi_H += 0.5 * dt * phidot_H

    # ------ update variables
    update_Tpv(T_H, P_H, v_H, phi_H, tau_temp, Z_temp)
    evaluate_psi(P_H, q_H)
    evaluate_fluxes(tau_H, Z_H, P_H, v_H, Ftau_H, FZ_H)

    phidot_H_old = cp.deepcopy(phidot_H)
    update_phidot(phi_H, phidot_H, psi_H, v_H, T_H)
    phidot_H_mean = (phidot_H + phidot_H_old) / 2
    evaluate_psi(phi_H, psi_H)
    
    ho_update_taudot(phi_H, phidot_H_mean, psi_H, T_H, v_H, Ftau_H, taudot_H, Gtau_H)
    ho_update_Zdot(phi_H, phidot_H_mean, psi_H, T_H, P_H, v_H, q_H, FZ_H, Zdot_H, GZ_H)

    # ------tau, Z evolution (full step)
    #tau_H -= dt * taudot_H # Not needed when FCT used.
    #Z_H -= dt * Zdot_H # Not needed when FCT used.

    # ------phi evolution (half step)
    phi_H += 0.5 * dt * phidot_H
    # ################LOW ORDER##################
    update_Tpv(T_L, P_L, v_L, phi_L, tau_L, Z_L)
    evaluate_psi(P_L, q_L)
    Ftau_L = tau_L + P_L
    FZ_L = cp.deepcopy(Z_L)
    
    v_old = cp.deepcopy(v_L)
    tau_old = cp.deepcopy(tau_L)
    Z_old = cp.deepcopy(Z_L)

    # ------ update variables
    KL = advection(v_L)
    #print(sum(abs(KL[:,-1]) + abs(KL[:,-1])) < sum(abs(KL[:,0])))
    evaluate_psi(phi_L, psi_L)
    lo_update_taudot(phi_L, phidot_L, psi_L, T_L, v_L, Ftau_L, KL, taudot_L, Gtau_L)
    lo_update_Zdot(phi_L, phidot_L, psi_L, T_L, P_L, v_L, q_L, FZ_L, KL, Zdot_L, GZ_L)

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
    
    antidiffusive_matrix(KL, FZ_H, FZ_L, dt, antidiffusive_flux_Z)
    antidiffusive_matrix(KL, Ftau_H, Ftau_L, dt, antidiffusive_flux_tau)
    
    for m in [1,2,3]:
        prelimiting(antidiffusive_flux_tau, tau_L)
        prelimiting(antidiffusive_flux_Z, Z_L)
        tau_H = cp.deepcopy(tau_L)
        Z_H = cp.deepcopy(Z_L)
        limit_fluxes(antidiffusive_flux_tau, antidiffusive_flux_Z,
                 tau, Z, phi, v,
                 tau_H, Z_H, phi_L,# tau_bar, Z_bar, phi_bar,
                 alpha_limiter1, alpha_limiter2,
                 limited_flux_tau, limited_flux_Z)

        tau_H[0] = tau_H[0] + (-dt * (Gtau_H[0] - Gtau_L[0]) +
                           limited_flux_tau[0][0] +
                           limited_flux_tau[0][1]) / mass_matrix[0]
        Z_H[0] = Z_H[0] + (-dt * (GZ_H[0] - GZ_L[0]) +
                       limited_flux_Z[0][0] +
                       limited_flux_Z[0][1]) / mass_matrix[0]
                       
        for i in range(1, N-1):
            tau_H[i] = tau_H[i] + (-dt * (Gtau_H[i] - Gtau_L[i]) +
                               limited_flux_tau[i][-1] +
                               limited_flux_tau[i][0] +
                               limited_flux_tau[i][1]) / mass_matrix[i]
            Z_H[i] = Z_H[i] + (-dt * (GZ_H[i] - GZ_L[i]) +
                           limited_flux_Z[i][-1] +
                           limited_flux_Z[i][0] +
                           limited_flux_Z[i][1]) / mass_matrix[i]
        tau_H[N-1] = tau_H[N-1] + (-dt * (Gtau_H[N-1] - Gtau_L[N-1]) +
                           limited_flux_tau[N-1][-1] +
                           limited_flux_tau[N-1][0]) / mass_matrix[N-1]
        Z_H[N-1] = Z_H[N-1] + (-dt * (GZ_H[N-1] - GZ_L[N-1]) +
                       limited_flux_Z[N-1][0] +
                       limited_flux_Z[N-1][-1]) / mass_matrix[N-1]
        antidiffusive_flux_tau -= limited_flux_tau
        antidiffusive_flux_Z -= limited_flux_Z
        #if(np.max(np.abs(limited_flux_tau)) > 2*2.7193944524e-314 or np.max(np.abs(limited_flux_Z)) > 2*2.7193944524e-314):
        	#print(m, np.max(np.abs(limited_flux_tau)), np.max(np.abs(limited_flux_Z)))

    update_Tpv(T_H, P_H, v_H, phi_H, tau_H, Z_H)
    evaluate_psi(P_H, q_H)
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
    
    
    
    
    # #########PLOTTING############
    colormap = plt.get_cmap('Blues')
    colors = colormap(k / nsteps)
    #colors = colormap(0.99)
    #print(k+1, "H: ", phi_H[0:20], v_H[0:20], T_H[0:20], P_H[0:20], Z_H[0:20], tau_H[0:20])
    #print(k+1, "L: ", phi_L[0], v_L[0], T_L[0], P_L[0])
    if((k+1) % psteps == 0):
        cs2 = dVTdT(phi_H, T_H)/(T_H*d2VTdT2(phi_H,T_H))
        xi = r/t
        axs[0, 0].plot(xi / GeV**(-1), phi_H / GeV, c=colors)
        axs[1, 0].plot(xi / GeV**(-1), v_H / (GeV**4), c=colors)
        axs[2, 0].plot(xi / GeV**(-1), Z_H / GeV**4, c=colors)
        axs[3, 0].plot(xi / GeV**(-1), tau_H / GeV**4, c=colors)
        axs[0, 1].plot(xi / GeV**(-1), psi_H, c=colors)
        axs[1, 1].plot(xi / GeV**(-1), T_H / GeV, c=colors)
        axs[2, 1].plot(xi / GeV**(-1), P_H / GeV**4, c=colors)
        axs[3, 1].plot(xi / GeV**(-1), cs2**(0.5), c=colors)
        
        fig.savefig('benchmarks_data/point_'+str(index)+'/plots_new'+ str(k + 1) + '.pdf')
        #f = open('values_plasma/profiles'+str(n_run)+'/profile'+str(k+1)+'.csv', 'w')
        #writer = csv.writer(f)
        #for i_row in range(len(phi_H)):
        	#writer.writerow([r[i_row], phi_H[i_row], psi_H[i_row]])
        #f.close()
        #axs_diag[0, 0].plot(xi, alpha_limiter1[:, -1], c=colors)
        #axs_diag[0, 1].plot(xi, alpha_limiter2[:, -1], c=colors)
        #axs_diag[1, 0].plot(xi, alpha_limiter1[:, 0], c=colors)
        #axs_diag[1, 1].plot(xi, alpha_limiter2[:, 0], c=colors)
        #axs_diag[2, 0].plot(xi, alpha_limiter1[:, 1], c=colors)
        #axs_diag[2, 1].plot(xi, alpha_limiter2[:, 1], c=colors)
        #axs_diag[0, 0].plot(xi, limited_flux_tau[:, -1], c=colors)
        #axs_diag[0, 1].plot(xi, limited_flux_Z[:, -1], c=colors)
        #axs_diag[1, 0].plot(xi, limited_flux_tau[:, 0], c=colors)
        #axs_diag[1, 1].plot(xi, limited_flux_Z[:, 0], c=colors)
        #axs_diag[2, 0].plot(xi, limited_flux_tau[:, 1], c=colors)
        #axs_diag[2, 1].plot(xi, limited_flux_Z[:, 1], c=colors)
        #plt.savefig('plots_plasma1/plots_limiter' + str(k + 1) + '.pdf')
        np.savez_compressed('benchmarks_data/point_'+str(index)+'/profiles_step_new'+str(k+1), r=r, xi=r/t, phi=phi_H, phidot = phidot_H, psi=psi_H, v=v_H, T=T_H, P=P_H, tau=tau_H, Z=Z_H, limiter1_m = alpha_limiter1[:, -1], limiter1_0 = alpha_limiter1[:, 0], limiter1_p = alpha_limiter1[:, 1], limiter2_m = alpha_limiter2[:, -1], limiter2_0 = alpha_limiter2[:, 0], limiter2_p = alpha_limiter2[:, 1])
