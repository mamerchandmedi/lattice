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

# %% Import evolution functions.
from evolution_functions import update_Tpv
from evolution_functions import evaluate_fluxes
from evolution_functions import ho_update_taudot
from evolution_functions import ho_update_Zdot
from evolution_functions import update_phidot
# end: Import evolution functions.

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
axs[3, 0].set_ylabel(r'$Gtau$')


axs[0, 1].set_ylabel(r'$v$')
axs[1, 1].set_ylabel(r'$T$')
axs[2, 1].set_ylabel(r'$p$')
axs[3, 1].set_ylabel(r'$GZ$')
axs[3, 1].set_xlabel(r'$r$')

axs[0, 0].set_xlim(0, 1)
axs[1, 0].set_xlim(0, 1)
axs[2, 0].set_xlim(0, 1)
axs[3, 0].set_xlim(0, 1)

axs[0, 1].set_xlim(0, 1)
axs[1, 1].set_xlim(0, 1)
axs[2, 1].set_xlim(0, 1)
axs[3, 1].set_xlim(0, 1)

psteps = 10
nsteps = 1000

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
    print(t)
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
    T_H, P_H, v_H = update_Tpv(T_H, P_H, v_H, phi_H, tau_H, Z_H)
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
    phi_H[0] = phi[1]
    
    colormap = plt.get_cmap('Blues')
    colors = colormap( k / nsteps)
    if((k+1) % psteps == 0):
        axs[0, 0].plot(r / GeV**(-1), phi_H / GeV,
                       label=str(t / Tc)+"*Tc", c=colors)
        axs[1, 0].plot(r / GeV**(-1), tau_H / (GeV**4),
                       label=str(t / Tc)+"*Tc", c=colors)
        axs[2, 0].plot(r / GeV**(-1), Z_H / GeV**4,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[0, 1].plot(r / GeV**(-1), v_H,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[1, 1].plot(r / GeV**(-1), T_H / GeV,
                       label=str(t/Tc)+"*Tc", c=colors)
        axs[2, 1].plot(r / GeV**(-1), P_H / GeV**4,
                       label=str(t / Tc) + "*Tc", c=colors)
        #axs[3, 0].plot(r, alpha_limiter1[:, 0], c=colors)
        axs[3, 0].plot(r, Gtau_H, c=colors)
        #axs[3, 1].plot(r, alpha_limiter2[:, 0], c=colors)
        axs[3, 1].plot(r, GZ_H, c=colors)
        plt.savefig('plots/FCT/eta_0.3/plots' + str(k + 1) + '.png')
        np.savez_compressed('profiles/profiles_eta_'+str(eta/Tc)+'/t_'+str(t), r=r, xi=r/t, phi=phi_H, v=v_H, T=T_H, P=P_H, tau=tau_H, Z=Z_H)
    
