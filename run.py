import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve
import matplotlib.pylab as pl
from tqdm import tqdm
import copy as cp

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
plt.figure(1)
fig, axs = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)
# ------plotting
plt.xlim(0, 10)
plt.rcParams["figure.dpi"] = 400
# plt.ylim(0,0.1)
fig.suptitle("dr = " + str(dr / GeV**(-1)) + ", dt = " +
             str(dt / GeV**(-1)) + ', $\eta=$' + str(eta/Tc))
axs[0, 0].set_ylabel(r'$\phi$')
axs[1, 0].set_ylabel(r'$\tau$')
axs[2, 0].set_ylabel(r'$Z$')
axs[2, 0].set_xlabel(r'$\xi$')

axs[0, 1].set_ylabel(r'$v$')
axs[1, 1].set_ylabel(r'$T$')
axs[2, 1].set_ylabel(r'$p$')
axs[2, 1].set_xlabel(r'$\xi$')

axs[0, 0].set_xlim(0, 1)
axs[1, 0].set_xlim(0, 1)
axs[2, 0].set_xlim(0, 1)

axs[0, 1].set_xlim(0, 1)
axs[1, 1].set_xlim(0, 1)
axs[2, 1].set_xlim(0, 1)

#fig_diag, axs_diag = plt.subplots(3, 2)
#fig_diag.subplots_adjust(hspace=0.5)
#fig_diag.subplots_adjust(wspace=0.5)

#axs_diag[0, 0].set_ylabel(r'$\alpha 1_{-1}$')
#axs_diag[1, 0].set_ylabel(r'$\alpha 1_{0}$')
#axs_diag[2, 0].set_ylabel(r'$\alpha 1_{1}$')
#axs_diag[2, 0].set_xlabel(r'$r$')

#axs_diag[0, 1].set_ylabel(r'$\alpha 2_{-1}$')
#axs_diag[1, 1].set_ylabel(r'$\alpha 2_{0}$')
#axs_diag[2, 1].set_ylabel(r'$\alpha 2_{1}$')
#axs_diag[2, 1].set_xlabel(r'$r$')

#axs_diag[0, 0].set_xlim(0, 2.)
#axs_diag[1, 0].set_xlim(0, 2.)
#axs_diag[2, 0].set_xlim(0, 2.)

#axs_diag[0, 1].set_xlim(0, 2.)
#axs_diag[1, 1].set_xlim(0, 2.)
#axs_diag[2, 1].set_xlim(0, 2.)

#axs_diag[0, 0].set_ylim(0, 1.)
#axs_diag[1, 0].set_ylim(0, 1.)
#axs_diag[2, 0].set_ylim(0, 1.)

#axs_diag[0, 1].set_ylim(0, 1.)
#axs_diag[1, 1].set_ylim(0, 1.)
#axs_diag[2, 1].set_ylim(0, 1.)

psteps = 100
nsteps = 30000

# initial values
phi_H, tau_H, Z_H = cp.deepcopy(phi), cp.deepcopy(tau), cp.deepcopy(Z)
T_H, v_H, P_H = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
phidot_H, taudot_H, Zdot_H = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)

phi_L, tau_L, Z_L = cp.deepcopy(phi), cp.deepcopy(tau), cp.deepcopy(Z)
T_L, v_L, P_L = cp.deepcopy(T), cp.deepcopy(v), cp.deepcopy(P)
phidot_L, taudot_L, Zdot_L = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)

# loop over time
for k in tqdm(range(nsteps)):
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

    #T_L, P_L, v_L = update_Tpv(T_L, P_L, v_L, phi_L, tau_L, Z_L)
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
                           limited_flux_tau[0][1]) / mass_matrix[0]
    tau_L[0] = tau_H[0]
    Z_H[0] = Z_L[0] - (dt * (GZ_H[0] - GZ_L[0]) +
                       limited_flux_Z[0][1]) / mass_matrix[0]
    Z_L[0] = Z_H[0]
    
    phi_L[0] = phi_H[0]
    phidot_L[0] = phidot_H[0]

    for i in range(1, N):
        tau_H[i] = tau_L[i] - (dt * (Gtau_H[i] - Gtau_L[i]) +
                               limited_flux_tau[i][-1] +
                               limited_flux_tau[i][1]) / mass_matrix[i]
        tau_L[i] = tau_H[i]
        Z_H[i] = Z_L[i] - (dt * (GZ_H[i] - GZ_L[i]) +
                           limited_flux_Z[i][-1] +
                           limited_flux_Z[i][1]) / mass_matrix[i]
        Z_L[i] = Z_H[i]
        phi_L[i] = phi_H[i]
        phidot_L[i] = phidot_H[i]

    tau_H[N] = tau_L[N] - (dt * (Gtau_H[N] - Gtau_L[N]) +
                           limited_flux_tau[N][-1]) / mass_matrix[N]
    tau_L[N] = tau_H[N]
    Z_H[N] = Z_L[N] - (dt * (GZ_H[N] - GZ_L[N]) +
                       limited_flux_Z[N][-1]) / mass_matrix[N]
    Z_L[N] = Z_H[N]

    phi_L[N] = phi_H[N]
    phidot_L[N] = phidot_H[N]

    # #########PLOTTING############
    colormap = plt.get_cmap('Blues')
    colors = colormap(k / nsteps)
    if((k+1) % psteps == 0):
        xi = r/t
        axs[0, 0].plot(xi / GeV**(-1), phi_H / GeV, c=colors)
        axs[1, 0].plot(xi / GeV**(-1), tau_H / (GeV**4), c=colors)
        axs[2, 0].plot(xi / GeV**(-1), Z_H / GeV**4, c=colors)
        axs[0, 1].plot(xi / GeV**(-1), v_H, c=colors)
        axs[1, 1].plot(xi / GeV**(-1), T_H / GeV, c=colors)
        axs[2, 1].plot(xi / GeV**(-1), P_H / GeV**4, c=colors)
        fig.savefig('plots/eta_0.08/plots' + str(k + 1) + '.png')
        #axs_diag[0, 0].plot(r, alpha_limiter1[:, -1], c=colors)
        #axs_diag[0, 1].plot(r, alpha_limiter2[:, -1], c=colors)
        #axs_diag[1, 0].plot(r, alpha_limiter1[:, 0], c=colors)
        #axs_diag[1, 1].plot(r, alpha_limiter2[:, 0], c=colors)
        #axs_diag[2, 0].plot(r, alpha_limiter1[:, 1], c=colors)
        #axs_diag[2, 1].plot(r, alpha_limiter2[:, 1], c=colors)
        #plt.savefig('plots/eta_0.08/limiter' + str(k + 1) + '.png')
        np.savez_compressed('values/profiles_eta_'+str(eta/Tc)+'_step'+str(k), r=r, xi=r/t, phi=phi_H, v=v_H, T=T_H, P=P_H, tau=tau_H, Z=Z_H)

