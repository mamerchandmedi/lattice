#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

index = 74

df = pd.read_csv('points.csv')

#STANDARD MODEL VALUES
mh=125.09
v=246.2

lamh = mh**2/(2*v**2)
muh2 = -lamh*v**2
alpha_el=1/137
sinthw=np.sqrt(0.223)
g1=np.sqrt(4*np.pi*alpha_el/(1-sinthw**2))
g=np.sqrt(4*np.pi*alpha_el)/sinthw
yt = 1
Mplanck=2.4*10**18

#SCALAR SINGLET MODEL SPECIFICATION
ms = df["ms"].loc[index]
lams = df["lams"].loc[index]
lamhs = df["lammix"].loc[index]

mus2 = ms**2 - 0.5*lamhs*v**2

cs = 1/12 * (2*lamhs + 3*lams)
ch = 1/48 * (9*g**2 + 3*g1**2 + 12*yt**2 + 24*lamh + 2*lamhs)

Tc = df["Tc"].loc[index]
Tn = df["Tnuc"].loc[index]
alpha = df["alpha"].loc[index]

print(Tc, Tn, ms, lams, lamhs, alpha)
