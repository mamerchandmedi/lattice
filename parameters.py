#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:24:44 2022

@author: Tomasz Krajewski
"""

import os
import pandas as pd

index = 1
#data = pd.read_csv('benchmarks.csv')
#alpha = data['alpha_theta'][index]
#TDM = data['TDM'][index]
#lam = data['lam'][index]
#y = data['y'][index]
#etaV = data['etaV'][index]
#Tn = data['T'][index]
#v = data['v'][index]


#path = 'benchmarks_data/test_'+str(index)
#if not os.path.isdir(path):
#   os.mkdir(path)


#Tc = 100
#lam = 0.1
#v = 100
#etaV = 0.2
#Tn = 43  
#y = 1.5

Tc = 100
lam = 0.146
v = 110
etaV = 0.22
Tn = 51  
y = 1.615

v2 = 246.0**2

m2h=125.0**2
gBL = 0.3
nZp = 57
MBL=10**4
m2t=173.0**2
m2W=80.379**2
m2Z=91.1876**2
g3=1.2279920495357861
nt=12
nZ=3
nW=6
nGs=3
nZp = 3
numBosonDof=28
numFermionDof=90

vphi2 = MBL**2 / (4 * gBL**2)
renormScaleSq = vphi2

c2beta= v2/vphi2

num_fermion_dof = numFermionDof
num_boson_dof = numBosonDof + nZp
g_sstar=numBosonDof+numFermionDof*7/8 + nZp #Number of d.o.f. in the symmetric phase.

g2sq=4*m2W/v2
g1sq=4*m2Z/v2-g2sq
g3=g3
yt2=2*m2t/v2
lHH = 0.5 * m2h / v2
muHsq=-lHH * v2

print(Tn, lam, v, y, etaV)
print("gBL =", gBL, "nZp =", nZp, "MBL =", MBL)
# -----------specify friction here---------------
eta = 0.0 * Tc
