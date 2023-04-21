#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:24:44 2022

@author: Tomasz Krajewski
"""

import pandas as pd
df = pd.read_csv('points_0.1.csv')
index = 51

#Tc = 100.
Tc = df["Tc"].loc[index]
#L = 0.1 * Tc**4
L = df["L"].loc[index] * Tc**4
#sigma = 0.1 * Tc**3
sigma = df["sigma"].loc[index] * Tc**3
#lc = 6 / Tc
lc = df["lc"].loc[index]


T0 = Tc / (1. + 6. * sigma / (L * lc))**0.5
#T0 = Tc/(2**(0.5))
alpha = 1. / (2. / 3. * sigma * lc**5 * Tc**2)**0.5
#alpha = 10**(0.5)/72.
gamma = (L + 6. * sigma / lc)/(6. * sigma * lc * Tc**2)
#gamma = 2./18.
lam = 1. / (3. * sigma * lc**3)
#lam = 5./648.
Tn = df["Tn"].loc[index]
#Tn = 80.
# -----------specify friction here---------------
eta = 0.1 * Tc
print(Tn, df["alpha"].loc[index])
