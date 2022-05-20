import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve
import matplotlib.pylab as pl
from tqdm import tqdm

# %% Importing parameters of the model.
from parameters import Tc, L, sigma, lc, T0, alpha, gamma, lam, Tn, eta
# end: Importing parameters of the model.

# %% Importing the potetnial of the model.
from potential import V, dV
# end: Importing the potetnial of the model.

# %% Import critical bubble calculated by CosmoTransitions.
from bubble import vt, profile, fphi, phi_tw
# end: Import critical bubble calculated by CosmoTransitions.

# %% Import termodynamically corrected quantities.
from thermo import gst, a, VT, dVTdphi, eps, p, w, gam, Gf_tau, Gf_Z
# end: Import termodynamically corrected quantities.

# %% Import rescaled quantities.
from scaled import GeV, VT_mod, dVTdphi_mod, tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Import matrices of lattice definition.
from lattice import dr, r, N
# end: Import matrices of lattice definition.

# %% Import matrices of wave equation integration scheme.
from matrices_field import laplacian_matrix, kinetic_matrix, gradient_matrix
# end: Import matrices of wave equation integration scheme.

# %% Import mass matrix.
from matrices_low_order import mass_matrix, friction_matrix
# end: Import mass matrix.

# %% Import matrices of the low order integration scheme.
from matrices_low_order_dynamically import advection
# end: Import matrices of the low order integration scheme.

# %% Import matrices of the high order integration scheme.
from matrices_high_order import sources_matrix
from matrices_high_order import advection_matrix as high_order_advection_matrix
# end: Import matrices of the high order integration scheme.

# %% Importing initial values.
from initialization import phi0, phi, T, P, tau, Z, v, P_off
# end: Importing initial values.

# %% Importing procedure for solving algebraic relation
#    T, p, v as function of tau, Z, phi.
from solver import tauZphi2Tpv
# end: Importing procedure for solving algebraic relation
#      T, p, v as function of tau, Z, phi.

dt = 0.001 * GeV**(-1)

phidot = np.zeros(r.size)
taudot = np.zeros(r.size)
Zdot = np.zeros(r.size)

# EVOLUTION #
fig, axs = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)
# ------plotting
plt.xlim(0, 10)
plt.rcParams["figure.dpi"] = 400
# plt.ylim(0,0.1)
fig.suptitle("dr = " + str(dr / GeV**(-1)) + ", dt = " +
             str(dt / GeV**(-1)) + ', $\eta = $' + str(eta/Tc))
axs[0, 0].set_ylabel(r'$\phi$')
axs[1, 0].set_ylabel(r'$\tau$')
axs[2, 0].set_ylabel(r'$Z$')
axs[2, 0].set_xlabel(r'$r$')

axs[0, 1].set_ylabel(r'$v$')
axs[1, 1].set_ylabel(r'$T$')
axs[2, 1].set_ylabel(r'$p$')
axs[2, 1].set_xlabel(r'$r$')

axs[0, 0].set_xlim(0, 2)
axs[1, 0].set_xlim(0, 2)
axs[2, 0].set_xlim(0, 2)
axs[0, 1].set_xlim(0, 2)
axs[1, 1].set_xlim(0, 2)
axs[2, 1].set_xlim(0, 2)

# axs[0,1].set_ylim(-1,1)
# axs[0,0].plot(r,phi,linestyle = 'dashed')
# axs[0, 0].axhline(y = vt, color = 'gray', linestyle = 'dashed')
# axs[1,0].plot(r,tau,linestyle = 'dashed')
# axs[2,0].plot(r,Z,linestyle = 'dashed')
# axs[0,1].plot(r,v,linestyle = 'dashed')
# axs[1,1].plot(r,T,linestyle = 'dashed')
# axs[2,1].plot(r,P,linestyle = 'dashed')
# plt.savefig('plots/low_order/plots_low_order'+str(0))
psteps = 100
tmax = 3 * GeV**(-1)
counter = 0
for t in tqdm(np.arange(0., tmax, step=dt), leave=True):  # --iterate over time
    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])
    Ftau = (tau + P)
    FZ = Z

    # -------evaluation of advection matrix
    advection_matrix = advection(v, high_order_advection_matrix)

    taudot[0] = (advection_matrix[0][0] * Ftau[0] +
                 advection_matrix[0][1] * Ftau[1] +
                 friction_matrix[0][0] *
                 Gf_tau(phi[0], phidot[0], 0, T[0], v[0])) / mass_matrix[0]

    taudot[N] = ((advection_matrix[N][-1] * Ftau[N-1] +
                 advection_matrix[N][0] * Ftau[N]) / mass_matrix[N] +
                 Gf_tau(phi[N], 0, (phi[N] - phi[N-1]) /
                 (r[N] - r[N-1]), T[0], v[0]))

    Zdot[0] = (advection_matrix[0][0] * FZ[0] +
               advection_matrix[0][1] * FZ[1] +
               (sources_matrix[0][0] + high_order_advection_matrix[0][0]) *
               (P[0] - P_off[0]) +
               (sources_matrix[0][1] + high_order_advection_matrix[0][1]) *
               (P[1] - P_off[1]) +
               friction_matrix[0][0] *
               Gf_Z(phi[0], phidot[0], 0, T[0], v[0])) / mass_matrix[0]
    Zdot[N] = ((advection_matrix[N][-1] * FZ[N-1] +
               advection_matrix[N][0] * FZ[N] +
               (sources_matrix[N][-1] + high_order_advection_matrix[N][-1]) *
               (P[N-1] - P_off[N-1]) +
               (sources_matrix[N][0] + high_order_advection_matrix[N][0]) *
               (P[N] - P_off[N])) / mass_matrix[N] -
               Gf_Z(phi[N], 0, (phi[N] - phi[N-1]) /
               (r[N] - r[N-1]), T[N], v[N]))
    for j in range(1, N):
        taudot[j] = (advection_matrix[j][-1] * Ftau[j-1] +
                     advection_matrix[j][0] * Ftau[j] +
                     advection_matrix[j][1] * Ftau[j+1] +
                     friction_matrix[j][-1] *
                     Gf_tau(phi[j], phidot[j],
                     (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                     friction_matrix[j][1] *
                     Gf_tau(phi[j], phidot[j],
                     (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j])) / \
            mass_matrix[j]
        Zdot[j] = (advection_matrix[j][-1] * FZ[j-1] +
                   advection_matrix[j][0] * FZ[j] +
                   advection_matrix[j][1] * FZ[j+1] -
                   friction_matrix[j][-1] *
                   Gf_Z(phi[j], phidot[j],
                   (phi[j] - phi[j-1])/(r[j] - r[j-1]), T[j], v[j]) +
                   friction_matrix[j][1] *
                   Gf_Z(phi[j], phidot[j],
                   (phi[j+1] - phi[j])/(r[j+1] - r[j]), T[j], v[j]) +
                   (sources_matrix[j][-1] +
                    high_order_advection_matrix[j][-1]) * P[j-1] +
                   (sources_matrix[j][0] +
                    high_order_advection_matrix[j][0]) * P[j] +
                   (sources_matrix[j][1] +
                    high_order_advection_matrix[j][1]) * P[j+1]) / \
            mass_matrix[j]
    # ------tau, Z evolution (half step)
    tau -= 0.5 * dt * taudot
    Z -= 0.5 * dt * Zdot

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])

    # phidot[1] += dt * ((laplacian_matrix[1][1] * (phi[2] - phi[1])) /
    #                   kinetic_matrix[1] - dVTdphi_mod(phi[1], T[1]))
    phidot[1] = ((1 + 0.5 * dt * eta * gam(v[1]))**(-1) *
                 ((1 - 0.5 * dt * eta * gam(v[1])) * phidot[1] -
                  dt * (dVTdphi(phi[1], T[1]) -
                  (laplacian_matrix[1][1] * (phi[2] - phi[1]) -
                   eta * gam(v[1]) * v[1] *
                   (gradient_matrix[1][1] * (phi[2] - phi[1]))) /
                  kinetic_matrix[1])))
    for j in range(2, N):
        # phidot[j] += dt * ((laplacian_matrix[j][1] * (phi[j+1] - phi[j]) +
        #                    laplacian_matrix[j][-1] * (phi[j] - phi[j-1])) /
        #                   kinetic_matrix[j] - dVTdphi_mod(phi[j], T[j]))
        phidot[j] = ((1 + 0.5 * dt * eta * gam(v[j]))**(-1) *
                     ((1 - 0.5 * dt * eta * gam(v[j])) * phidot[j] -
                      dt * (dVTdphi(phi[j], T[j]) -
                      (laplacian_matrix[j][1] * (phi[j+1] - phi[j]) +
                       laplacian_matrix[j][-1] * (phi[j] - phi[j-1]) -
                       eta * gam(v[j]) * v[j] *
                       (gradient_matrix[j][-1] * (phi[j] - phi[j-1]) +
                       gradient_matrix[j][1] * (phi[j+1] - phi[j]))) /
                      kinetic_matrix[j])))
    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    # ------tau, Z evolution (half step)
    tau -= 0.5 * dt * taudot
    Z -= 0.5 * dt * Zdot
    # ------plotting
    colormap = plt.get_cmap('Blues')
    colors = colormap((t + dt) / tmax)
    if((counter + 1) % psteps == 0):
        xi = r/t
        axs[0, 0].plot(r, phi / GeV,
                       label=str(t / Tc)+"*Tc", c=colors)
        axs[1, 0].plot(r, tau / (GeV**4),
                       label=str(t / Tc)+"*Tc", c=colors)
        axs[2, 0].plot(r, Z / GeV**4,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[0, 1].plot(r, v,
                       label=str(t / Tc) + "*Tc", c=colors)
        axs[1, 1].plot(r, T / GeV,
                       label=str(t/Tc)+"*Tc", c=colors)
        axs[2, 1].plot(r, P / GeV**4,
                       label=str(t / Tc) + "*Tc", c=colors)
        plt.savefig('plots/Friction/plots_low_order' +
                    str(eta / Tc) + 'step' + str(counter + 1) + '.png')

    counter += 1
