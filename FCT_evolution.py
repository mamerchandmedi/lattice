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
# end: Import rescaled quantities.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import sources_matrix as sm
from matrices_high_order import advection_matrix as ho_advection_matrix
from matrices_high_order import mass_matrix
# end: Import matrices of the high order integration scheme.

# %% Import matrices of the low order integration scheme.
from matrices_low_order_dynamically import advection
# end: Import matrices of the low order integration scheme.

# %% Importing evolution steps.
from steps import high_order_step as ho_step
from steps import low_order_step as lo_step
# end: Importing evolution steps.

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
fig, axs = plt.subplots(4, 2)
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
axs[3, 0].set_xlabel(r'$r$')
axs[3, 0].set_ylabel(r'$\alpha_1$')


axs[0, 1].set_ylabel(r'$v$')
axs[1, 1].set_ylabel(r'$T$')
axs[2, 1].set_ylabel(r'$p$')
axs[3, 1].set_ylabel(r'$\alpha_2$')
axs[3, 1].set_xlabel(r'$r$')

axs[0, 0].set_xlim(0, 1)
axs[1, 0].set_xlim(0, 1)
axs[2, 0].set_xlim(0, 1)
axs[3, 0].set_xlim(0, 1)

axs[0, 1].set_xlim(0, 1)
axs[1, 1].set_xlim(0, 1)
axs[2, 1].set_xlim(0, 1)
axs[3, 1].set_xlim(0, 1)


psteps = 1
tmax = 0.01 * GeV**(-1)
counter = 0
KH = ho_advection_matrix
phi_H, tau_H, Z_H, T_H, v_H, P_H, phidot_H, taudot_H = \
    phi, tau, Z, T, v, P, np.zeros(r.size), np.zeros(r.size)
phi_L, tau_L, Z_L, T_L, v_L, P_L, phidot_L = \
    cp.deepcopy(phi_H), cp.deepcopy(tau_H), cp.deepcopy(Z_H),\
    cp.deepcopy(T_H), cp.deepcopy(v_H), cp.deepcopy(P_H), np.zeros(r.size),
for t in tqdm(np.arange(0., tmax, step=dt), leave=True):  # --iterate over time
    phi_L, tau_L, Z_L, T_L, v_L, P_L, phidot_L, Ftau_L, FZ_L, GZ_L, KL =\
        lo_step(phi_L, tau_L, Z_L, T_L, v_L, P_L, phidot_L)
    phi_H, tau_H, Z_H, T_H, v_H, P_H, phidot_H, Ftau_H, FZ_H, GZ_H, taudot_H =\
        ho_step(phi_H, tau_H, Z_H, T_H, v_H, P_H, phidot_H)
    tau_bar = Ubar(tau_L)
    Z_bar = Ubar(Z_L)
    A_tau = antidiffusive_matrix(KH, KL, Ftau_H, Ftau_L, dt)
    A_Z = antidiffusive_matrix(KH, KL, FZ_H, FZ_L, dt)
    A1_prim, A2_prim = \
        trans_am(A_tau, A_Z, tau_bar, Z_bar, phi_L, T_L, P_L, v_L)
    P1_plus, P1_minus = P_vectors(A1_prim)
    P2_plus, P2_minus = P_vectors(A2_prim)
    alpha_limiter1 = np.ndarray((N+1, 3))
    alpha_limiter2 = np.ndarray((N+1, 3))
    aA_tau = np.ndarray((N+1, 3))
    aA_Z = np.ndarray((N+1, 3))
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
            Q1_plus_j, Q1_minus_j = Q_values(tau_prim, i+j)
            Q2_plus_i, Q2_minus_i = Q_values(Z_prim, i)
            Q2_plus_j, Q2_minus_j = Q_values(Z_prim, i+j)
            R1_plus_i, R1_minus_i = \
                R_values(Q1_plus_i, Q1_minus_i,
                         P1_plus[i], P1_minus[i])
            R1_plus_j, R1_minus_j = \
                R_values(Q1_plus_j, Q1_minus_j,
                         P1_plus[i+j], P1_minus[i+j])
            R2_plus_i, R2_minus_i = \
                R_values(Q2_plus_i, Q2_minus_i,
                         P2_plus[i], P2_minus[i])
            R2_plus_j, R2_minus_j = \
                R_values(Q2_plus_j, Q2_minus_j,
                         P2_plus[i+j], P2_minus[i+j])
            if A1_prim[i][j] >= 0:
                alpha_limiter1[i][j] = min(R1_plus_i, R1_minus_j)
            else:
                alpha_limiter1[i][j] = min(R1_minus_i, R1_plus_j)
            if A2_prim[i][j] >= 0:
                alpha_limiter2[i][j] = min(R2_plus_i, R2_minus_j)
            else:
                alpha_limiter2[i][j] = min(R2_minus_i, R2_plus_j)
            aA_tau[i][j] = \
                T_mat[0][0] * A1_prim[i][j] * alpha_limiter1[i][j] \
                + T_mat[0][1] * A2_prim[i][j] * alpha_limiter2[i][j]
            aA_Z[i][j] = \
                T_mat[1][0] * A1_prim[i][j] * alpha_limiter1[i][j] \
                + T_mat[1][1] * A2_prim[i][j] * alpha_limiter2[i][j]
    deltaG = np.zeros(r.size)
    tau_H[0] = tau_L[0] + (aA_tau[0][0] + aA_tau[0][1])/mass_matrix[0]
    tau_L[0] = tau_L[0] + (aA_tau[0][0] + aA_tau[0][1])/mass_matrix[0]
    Z_H[0] = Z_L[0] + (dt * (GZ_H[0] - GZ_L[0]) + aA_Z[0][0] + aA_Z[0][1])\
        / mass_matrix[0]
    Z_L[0] = Z_L[0] + (dt * (GZ_H[0] - GZ_L[0]) + aA_Z[0][0] + aA_Z[0][1])\
        / mass_matrix[0]
    deltaG[0] = GZ_H[0] - GZ_L[0]
    for i in range(1, N):
        tau_H[i] = tau_L[i] + (aA_tau[i][-1] + aA_tau[i][0] + aA_tau[i][1])\
            / mass_matrix[i]
        tau_L[i] = tau_L[i] + (aA_tau[i][-1] + aA_tau[i][0] + aA_tau[i][1])\
            / mass_matrix[i]
        Z_H[i] = Z_L[i] + \
            (dt * (GZ_H[i] - GZ_L[i]) + aA_Z[i][-1] + aA_Z[i][0] + aA_Z[i][1])\
            / mass_matrix[i]
        Z_L[i] = Z_L[i] + \
            (dt * (GZ_H[i] - GZ_L[i]) + aA_Z[i][-1] + aA_Z[i][0] + aA_Z[i][1])\
            / mass_matrix[i]
        deltaG[i] = GZ_H[i] - GZ_L[i]
    tau_H[N] = tau_L[N] + (aA_tau[N][-1] + aA_tau[N][0])\
        / mass_matrix[N]
    tau_L[N] = tau_L[N] + (aA_tau[N][-1] + aA_tau[N][0])\
        / mass_matrix[N]
    Z_H[N] = Z_L[N] + (dt*(GZ_H[N] - GZ_L[N]) + aA_Z[N][-1] + aA_Z[N][0])\
        / mass_matrix[N]
    Z_L[N] = Z_L[N] + (dt*(GZ_H[N] - GZ_L[N]) + aA_Z[N][-1] + aA_Z[N][0])\
        / mass_matrix[N]
    deltaG[N] = GZ_H[N] - GZ_L[N]
    # ------plotting
    colormap = plt.get_cmap('Blues')
    colors = colormap((t + dt) / tmax)
    if((counter+1) % psteps == 0):
        axs[0, 0].plot(r / GeV**(-1), phi_L / GeV,
                       label=str(t / Tc)+"*Tc", c=colors)
        axs[1, 0].plot(r / GeV**(-1), tau_L / (GeV**4),
                       label=str(t / Tc)+"*Tc", c=colors)
        axs[2, 0].plot(r / GeV**(-1), Z_L / GeV**4,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[0, 1].plot(r / GeV**(-1), v_L,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[1, 1].plot(r / GeV**(-1), T_L / GeV,
                       label=str(t/Tc)+"*Tc", c=colors)
        axs[2, 1].plot(r / GeV**(-1), P_L / GeV**4,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[3, 0].plot(r, alpha_limiter1[:, 0], c=colors)
        axs[3, 1].plot(r, alpha_limiter2[:, 0], c=colors)
        plt.savefig('plots/FCT/eta0/plots' + str(counter + 1) + '.png')
        np.savez_compressed('profiles/profiles_eta_'+str(eta/Tc)+'/
        t_'+str(t), r=r, xi=r/t, v=v, T=T, P=P, tau=tau, Z=Z)
    counter += 1

