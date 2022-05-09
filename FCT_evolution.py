import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve
import matplotlib.pylab as pl
from tqdm import tqdm

# %% Import matrices of lattice definition.
from lattice import dr, dt, r, N
# end: Import matrices of lattice definition.

# %% Importing parameters of the model.
from parameters import Tc, L, sigma, lc, T0, alpha, gamma, lam, Tn
# end: Importing parameters of the model.

# %% Importing initial values.
from initialization import phi0, phi, T, P, tau, Z, v, P_off
# end: Importing initial values.

# %% Import rescaled quantities.
from scaled import GeV, tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import sources_matrix
from matrices_high_order import advection_matrix as ho_advection_matrix
from matrices_high_order import mass_matrix
# end: Import matrices of the high order integration scheme.

# %% Import matrices of the low order integration scheme.
from matrices_low_order_dynamically import advection
# end: Import matrices of the low order integration scheme.

# %% Importing evolution steps.
from steps import high_order_step
from steps import low_order_step
# end: Importing evolution steps.

# %% Importing FCT variables.
from FCT import antidiffusive_matrix
from FCT import Ubar
from FCT import trans_antidiffusive_matrix
from FCT import P_vectors
from FCT import Q_values
from FCT import R_values
# end: Importing FCT variable.

# %% Importing characteristic variables.
from characteristic_variables import T_matrices2
# end: Importing characteristic variables.

#EVOLUTION#
fig, axs = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)
#------plotting
plt.xlim(0,10)
plt.rcParams["figure.dpi"] = 400
#plt.ylim(0,0.1)
fig.suptitle("dr = "+ str(dr / GeV**(-1)) + ", dt = "+str(dt / GeV**(-1))
             + ", GeV = "+str(GeV))
axs[0, 0].set_ylabel(r'$\phi$')
axs[1, 0].set_ylabel(r'$\tau$')
axs[2, 0].set_ylabel(r'$Z$')
axs[2, 0].set_xlabel(r'$r$')

axs[0, 1].set_ylabel(r'$v$')
axs[1, 1].set_ylabel(r'$T$')
axs[2, 1].set_ylabel(r'$p$')
axs[2, 1].set_xlabel(r'$r$')

axs[0, 0].set_xlim(0, 1)
axs[1, 0].set_xlim(0, 1)
axs[2, 0].set_xlim(0, 1)
axs[0, 1].set_xlim(0, 1)
axs[1, 1].set_xlim(0, 1)
axs[2, 1].set_xlim(0, 1)

psteps = 10
tmax = 0.5 * GeV**(-1)
counter = 0

phi_H, tau_H, Z_H, T_H, v_H, P_H = phi, tau, Z, T, v, P
phi_L, tau_L, Z_L, T_L, v_L, P_L = phi, tau, Z, T, v, P

for t in tqdm(np.arange(0., tmax, step=dt), leave=True):  # --iterate over time
    phi_H, tau_H, Z_H, T_H, v_H, P_H = low_order_step(phi_H, tau_H, Z_H, T_H, v_H, P_H)
    phi_L, tau_L, Z_L, T_L, v_L, P_L = low_order_step(phi_L, tau_L, Z_L, T_L, v_L, P_L)
    KL = advection(v_L, ho_advection_matrix)
    KH = ho_advection_matrix
    Ftau_H = (tau_H + P_H)*v_H
    Ftau_L = tau_L + P_L
    FZ_H = (Z_H + P_H)*v_H
    FZ_L = Z_L
    tau_bar = Ubar(tau_L)
    Z_bar = Ubar(Z_L)
    A_tau = antidiffusive_matrix(KH, KL, Ftau_H, Ftau_L, dt)
    A_Z = antidiffusive_matrix(KH, KL, FZ_H, FZ_L, dt)
    A1_prim, A2_prim = trans_antidiffusive_matrix(A_tau, A_Z, tau_bar, Z_bar, phi_L, T_L, P_L, v_L)
    P1_plus, P1_minus = P_vectors(A1_prim)
    P2_plus, P2_minus = P_vectors(A2_prim)
    tau_prim = np.ndarray((3,))
    Z_prim = np.ndarray((3,))
    alpha_limiter1 = np.ndarray((N+1, 3))
    alpha_limiter2 = np.ndarray((N+1, 3))
    aA_tau = np.ndarray((N+1, 3))
    aA_Z = np.ndarray((N+1, 3))
    for i in range(N+1):
        for j in [-1, 0, 1]:
            T, P, v = tauZphi2Tpv_mod(tau_bar[i][j], Z_bar[i][j], phi_L[i], T_L[i], P_L[i], v_L[i])
            T_mat, T_inv = T_matrices2(v, phi_L[i], T)
            tau_prim = T_inv[0][0] * tau_L + T_inv[0][1] * Z_L
            Z_prim = T_inv[1][0] * tau_L + T_inv[1][1] * Z_L
            if i+j >= 0 and i+j<=N:
                Q1_plus_i, Q1_minus_i = Q_values(tau_prim, i)
                Q1_plus_j, Q1_minus_j = Q_values(tau_prim, i+j)
                Q2_plus_i, Q2_minus_i = Q_values(Z_prim, i)
                Q2_plus_j, Q2_minus_j = Q_values(Z_prim, i+j)
                R1_plus_i, R1_minus_i = R_values(Q1_plus_i, Q1_minus_i, P1_plus[i], P1_minus[i])
                R1_plus_j, R1_minus_j = R_values(Q1_plus_j, Q1_minus_j, P1_plus[i+j], P1_minus[i+j])
                R2_plus_i, R2_minus_i = R_values(Q2_plus_i, Q2_minus_i, P2_plus[i], P2_minus[i])
                R2_plus_j, R2_minus_j = R_values(Q2_plus_j, Q2_minus_j, P2_plus[i+j], P2_minus[i+j])
                    if A1_prim[i][j] >= 0:
                        alpha_limiter1[i][j] = min(R1_plus_i, R1_minus_j)
                    else:
                        alpha_limiter1[i][j] = min(R1_minus_i, R1_plus_j)
                    if A2_prim[i][j] >= 0:
                        alpha_limiter2[i][j] = min(R2_plus_i, R2_minus_j)
                    else:
                        alpha_limiter2[i][j] = min(R2_minus_i, R2_plus_j)
                aA_tau[i][j] = T_mat[0][0] * A1_prim[i][j] * alpha_limiter1[i][j] + T_mat[0][1] * A2_prim[i][j] * alpha_limiter2[i][j]
                aA_Z[i][j] = T_mat[1][0] * A1_prim[i][j] * alpha_limiter1[i][j] + T_mat[1][1] * A2_prim[i][j] * alpha_limiter2[i][j]
    print(aA_tau)
    tau_L[0] = tau_L[0] + aA_tau[0][0] + aA_tau[0][1]
    Z_L[0] = Z_L[0] + aA_Z[0][0] + aA_Z[0][1]\
                    + sources_matrix[0][0]*(P_H[0] - P_L[0]) \
                    + sources_matrix[0][1]*(P_H[1] - P_L[1])
    for i in range(1, N):
        tau_L[i] = tau_L[i] + aA_tau[i][-1] + aA_tau[i][0] + aA_tau[i][1]
        Z_L[i] = Z_L[i] + aA_Z[i][-1] + aA_Z[i][0] + aA_Z[i][1] \
                        + sources_matrix[i][-1]*(P_H[i-1] - P_L[i-1]) \
                        + sources_matrix[i][0]*(P_H[i] - P_L[i]) \
                        + sources_matrix[i][1]*(P_H[i+1] - P_L[i+1])
    tau_L[N] = tau_L[N] + aA_tau[N][-1] + aA_tau[N][0]
    Z_L[N] = Z_L[N] + aA_Z[N][-1] + aA_Z[N][0]\
                    + sources_matrix[N][-1]*(P_H[N-1] - P_L[N-1]) \
                    + sources_matrix[N][0]*(P_H[N] - P_L[N])

    # ------plotting
    colormap = plt.get_cmap('Blues')
    colors = colormap((t + dt) / tmax)
    if((counter+1) % psteps == 0):
        axs[0, 0].plot(r / GeV**(-1), phi_L / GeV,
                       label=str(t / Tc)+"*Tc", c=colors)
        axs[1, 0].plot(r / GeV**(-1), tau / (GeV**4),
                       label=str(t / Tc)+"*Tc", c=colors)
        axs[2, 0].plot(r / GeV**(-1), Z_L / GeV**4,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[0, 1].plot(r / GeV**(-1), v_L,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[1, 1].plot(r / GeV**(-1), T_L / GeV,
                       label=str(t/Tc)+"*Tc", c=colors)
        axs[2, 1].plot(r / GeV**(-1), P_L / GeV**4,
                       label=str(t / Tc) + "*Tc", c=colors)
        plt.savefig('plots/FCT/plots' + str(counter + 1) + '.png')
    counter += 1
