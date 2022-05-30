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
from evolution_functions import evaluate fluxes
from evolution_functions import ho_update_taudot
from evolution_functions import ho_update_Z
# end: Import evolution functions.

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
