import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve
import matplotlib.pylab as pl
from tqdm import tqdm

# %% Importing parameters of the model.
from parameters import Tc, L, sigma, lc, T0, alpha, gamma, lam, Tn
# end: Importing parameters of the model.

# %% Importing the potetnial of the model.
from potential import V, dV
# end: Importing the potetnial of the model.

# %% Import critical bubble calculated by CosmoTransitions.
from bubble import vt, profile, fphi, phi_tw
# end: Import critical bubble calculated by CosmoTransitions.

# %% Import termodynamically corrected quantities.
from thermo import gst, a, VT, dVT, eps, p, w, gam
# end: Import termodynamically corrected quantities.

# %% Import rescaled quantities.
from scaled import GeV, VT_mod, dVT_mod, tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Import matrices of the integration scheme.
from matrices_low_order import dr, r,  cr1, cr2, cF0, cFm1, \
    cGUp1, cGU0, cGUm1, den, N, advection_matrix, mass_matrix, sources_matrix
# end: Import matrices of the integration scheme.

# %% Importing initial values.
from initialization import phi0, phi, T, P, tau, Z, v, P_off
# end: Importing initial values.

# %% Importing procedure for solving algebraic relation
#    T, p, v as function of tau, Z, phi.
from solver import tauZphi2Tpv
# end: Importing procedure for solving algebraic relation
#      T, p, v as function of tau, Z, phi.

dt = 0.01 * GeV**(-1)

phidot = np.zeros(r.size)
taudot = np.zeros(r.size)
Zdot = np.zeros(r.size)

#EVOLUTION#
fig, axs = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)
#------plotting
plt.xlim(0,10)
#plt.ylim(0,0.1)
fig.suptitle("dr = "+str(dr) + " ,dt = "+str(dt))
axs[0, 0].set_ylabel('phi')
axs[1, 0].set_ylabel('tau')
axs[2, 0].set_ylabel('Z')
axs[2, 0].set_xlabel("r")

axs[0, 1].set_ylabel('v')
axs[1, 1].set_ylabel('T')
axs[2, 1].set_ylabel('P')
axs[2, 1].set_xlabel("r")

axs[0, 0].set_xlim(0, 1)
axs[1, 0].set_xlim(0, 1)
axs[2, 0].set_xlim(0, 1)
axs[0, 1].set_xlim(0, 1)
axs[1, 1].set_xlim(0, 1)
axs[2, 1].set_xlim(0, 1)

#axs[0,1].set_ylim(-1,1)
#axs[0,0].plot(r,phi,linestyle = 'dashed')
axs[0, 0].axhline(y = vt, color = 'gray', linestyle = 'dashed')
#axs[1,0].plot(r,tau,linestyle = 'dashed')
#axs[2,0].plot(r,Z,linestyle = 'dashed')
#axs[0,1].plot(r,v,linestyle = 'dashed')
#axs[1,1].plot(r,T,linestyle = 'dashed')
#axs[2,1].plot(r,P,linestyle = 'dashed')
#plt.savefig('plots/low_order/plots_low_order'+str(0))
nsteps = 20
psteps = 1
tmax = 0.04 * GeV**(-1)
#def Tsolve(t,j):
#	return tau[j] + p(phi[j],t)-0.5*(dr**4*w(phi[j],t)+(w(phi[j],t)**2+4*Z[j]**2)**0.5)
for i in tqdm(np.arange(0., tmax, step=dt), leave=True):  # --iterate over time
    print(i)
    for j in range(N + 1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])

    # ------tau, Z evolution (half step)
    Ftau = (tau + P) * v
    FZ = Z * v + (P - P_off)

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    taudot[0] = advection_matrix[0][0] * Ftau[0] / mass_matrix[0]
    taudot[N] = (advection_matrix[N][-1] * Ftau[N-1] +
                 advection_matrix[N][0] * Ftau[N]) / \
        mass_matrix[N]

    Zdot[0] = (advection_matrix[0][0] * FZ[0] +
               sources_matrix[0][0] * (P[0] - P_off[0]) +
               sources_matrix[0][1] * (P[1] - P_off[1])) / mass_matrix[0]
    Zdot[N] = (advection_matrix[N][-1] * FZ[N-1] +
               advection_matrix[N][0] * FZ[N] +
               sources_matrix[N][-1] * (P[N-1] - P_off[N-1]) +
               sources_matrix[N][0] * (P[N] - P_off[N])) / \
        mass_matrix[N]

    for j in range(1, N):
        taudot[j] = (advection_matrix[j][-1] * Ftau[j-1] +
                     advection_matrix[j][0] * Ftau[j]) / \
            mass_matrix[j]
        Zdot[j] = (advection_matrix[j][-1] * FZ[j-1] +
                   advection_matrix[j][0] * FZ[j] +
                   sources_matrix[j][-1] * (P[j-1] - P_off[j-1]) +
                   sources_matrix[j][0] * (P[j] - P_off[j]) +
                   sources_matrix[j][1] * (P[j+1] - P_off[j+1])) / \
            mass_matrix[j]

    tau = tau - 0.5 * dt * taudot
    Z = Z - 0.5 * dt * Zdot

    for j in range(N+1):
        T[j], P[j], v[j] = tauZphi2Tpv_mod(tau[j], Z[j], phi[j], T[j], P[j],
                                           v[j])

    phidot[1] = phidot[1] + dt*((r[2]**3-r[1]**3)*(phi[2]-phi[1])/dr**2*(r[1]**3 - r[0]**3 + (r[2]-r[1])/4*(3*r[1]**2+2*r[1]*r[2]+r[2]**2))**(-1) - dr*dVT_mod(phi[1],T[1]))
    for j in range(2, N):
    	phidot[j] = phidot[j] + dt*(cr1[j]*(phi[j+1]-phi[j]) - cr2[j]*(phi[j]-phi[j-1]) - dr*dVT_mod(phi[j],T[j]))

    # ------phi evolution (half step)
    phi[1:] += 0.5 * dt * phidot[1:]
    phi[0] = phi[1]

    # ------tau, Z evolution (half step)
    tau = tau - 0.5 * dt * taudot
    Z = Z - 0.5 * dt * Zdot

    # ------plotting
    colormap = plt.get_cmap('Blues')
    colors = colormap((i + dt) / tmax)
    if((i + dt) / tmax % psteps == 0):
        axs[0, 0].plot(r / GeV**(-1), phi / GeV,
                       label=str(i/Tc)+"*Tc", c=colors)
        axs[1, 0].plot(r / GeV**(-1), tau / (GeV**4),
                       label=str(i/Tc)+"*Tc", c=colors)
        axs[2, 0].plot(r / GeV**(-1), Z / GeV**4,
                       label=str(i / Tc) + "*Tc", c=colors)
        axs[0, 1].plot(r / GeV**(-1), v,
                       label=str(i / Tc) + "*Tc", c=colors)
        axs[1, 1].plot(r / GeV**(-1), T / GeV,
                       label=str(i/Tc)+"*Tc", c=colors)
        axs[2, 1].plot(r / GeV**(-1), P / GeV**4,
                       label=str(i / Tc) + "*Tc", c=colors)
        plt.savefig('plots/low_order/plots_low_order' + str(i + dt) + '.png')
