#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Importing parameters of the model.
from parameters import muh2, mus2, lamh, lamhs, lams, mh, ms, v, g, g1, yt, Tn, Tc, alpha
# end: Importing parameters of the model.
from cosmoTransitions import generic_potential_1
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np


class model1(generic_potential_1.generic_potential):
    def init(self, ms = 50, lams = 1, lammix = 0):
        self.Ndim = 2
        self.ms = ms
        self.lams = lams
        self.lammix = lammix
        self.mus2 = ms**2 - lammix*v**2/2
        self.ch = 1/48 * (9*g**2 + 3*g1**2 + 12*yt**2 + 24*lamh + 2*self.lammix)
        self.cs = 1/12 * (2*self.lammix + 3*self.lams)
       
    def Vtot(self, X, T, include_radiation=True):
        X = np.asanyarray(X)
        h, s = X[...,0], X[...,1]
        pot = 0.5*(muh2 + self.ch*T**2) * h**2 + 0.25*lamh * h**4 + 0.25*self.lammix * h**2 * s**2 + 0.5*(self.mus2 + self.cs*T**2) * s**2 + 0.25*self.lams*s**4
        return pot
        
    def dVdT(self, X, T, include_radiation=True):
        X = np.asanyarray(X)
        h, s = X[...,0], X[...,1]
        pot = self.ch*T * h**2 + self.cs*T * s**2 
        return pot
        
    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        return [np.array([v,0])]
              
m = model1(ms = ms, lams = lams, lammix = lamhs) 


alltrans = m.findAllTransitions()
Tn_prim = m.TnTrans[-1]["Tnuc"]
Tc_prim = m.TnTrans[-1]["crit_trans"]["Tcrit"]
vlow = m.TnTrans[-1]["low_vev"][0]
ulow = m.TnTrans[-1]["low_vev"][1]
vhigh = m.TnTrans[-1]["high_vev"][0]
uhigh = m.TnTrans[-1]["high_vev"][1]
instanton = m.TnTrans[-1]['instanton']
phi = instanton.Phi
R = instanton.profile1D.R

print(Tc, Tc_prim)
print(Tn, Tn_prim)

def get_figsize(columnwidth, wf=0.5, hf=(5.**0.5-1.0) / 2.0):
    r"""Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth * wf
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * hf      # height in inches
    return [fig_width, fig_height]

plt.style.use('jcap.mplstyle')
columnwidth = 420

plt.figure(figsize = get_figsize(columnwidth))
plt.plot(R, phi[:,0], c = "deepskyblue", label = "$h(\\varrho)$")
plt.plot(R, phi[:,1], c = "crimson", label = "$s(\\varrho)$")
plt.title("$\\alpha = $ "+ str(round(alpha, 3))+", $\quad$ $T_n/T_c = $" +str(round(Tn/Tc,2) ))
plt.xlabel("$\\varrho$ [GeV$^{-1}$]")
plt.ylabel("$\\phi$ [GeV]")
plt.grid(lw=0.5)
plt.legend()
plt.savefig("phi0.pdf")

