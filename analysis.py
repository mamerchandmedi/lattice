#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:52:02 2024

@author: Tomasz Krajewski
"""

import numpy as np
import matplotlib.pyplot as plt
from scaled import GeV
from parameters import T0, alpha, gamma, lam, Tn
from tqdm import tqdm

psteps = 500
nsteps = 5000

def maximum():
    Delta = (alpha * Tn)**2 - 4. * gamma * (Tn**2 - T0**2) * lam
    return 0.5 * (alpha * Tn  - np.sqrt(Delta)) / lam


def find_wall(r, phi, maximum):
    wall = phi[0]
    for i in range(len(r)):
        if (wall - maximum) * (phi[i] - maximum) >0.:
            wall = phi[i]
        else:
            print(i)
            return (((wall - maximum) * r[i] - \
                    (phi[i] - maximum) * r[i-1]) / (wall - phi[i]) ,
                    ((wall - maximum) * xi[i] - \
                            (phi[i] - maximum) * xi[i-1]) / (wall - phi[i]))

# EVOLUTION #
plt.figure(1)
figs, axs = plt.subplots(3, 2)
figs.subplots_adjust(hspace=0.5)
figs.subplots_adjust(wspace=0.5)
# ------plotting
plt.xlim(0, 10)
plt.rcParams["figure.dpi"] = 300
# plt.ylim(0,0.1)
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

colormap = plt.get_cmap('Blues')

fig, ax = plt.subplots(1,)

for k in tqdm(range(1, nsteps // psteps)):
    file ='values/profiles_FCT_dense_eta_0.1_step' + str(k * psteps - 1) + '.npz'
    print(file)
    loaded = np.load(file)
    r = loaded['r']
    xi = loaded['xi']
    phi = loaded['phi']
    phidot = loaded['phidot']
    v = loaded['v']
    T = loaded['T']
    P = loaded['P']
    tau = loaded['tau']
    Z = loaded['Z']
  
    colors = colormap(k / (nsteps / psteps))  
    
    axs[0, 0].plot(xi / GeV**(-1), phi / GeV, c=colors)
    axs[1, 0].plot(xi / GeV**(-1), tau / (GeV**4), c=colors)
    axs[2, 0].plot(xi / GeV**(-1), Z / GeV**4, c=colors)
    axs[0, 1].plot(xi / GeV**(-1), v, c=colors)
    axs[1, 1].plot(xi / GeV**(-1), T / GeV, c=colors)
    axs[2, 1].plot(xi / GeV**(-1), P / GeV**4, c=colors)
    
    r_wall, xi_wall = find_wall(r, phi, maximum())
    Delta = (alpha * Tn)**2 - 4. * gamma * (Tn**2 - T0**2) * lam
    ax.set_xlim(-10 * lam / (0.5 * (alpha * Tn  + np.sqrt(Delta)) * np.sqrt(lam / 8)),
                10 * lam / (0.5 * (alpha * Tn  + np.sqrt(Delta)) * np.sqrt(lam / 8)))
    ax.plot((r - r_wall)/np.sqrt(1. - xi_wall**2), phi, c=colors)