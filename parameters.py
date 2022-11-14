#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:24:44 2022

@author: Tomasz Krajewski
"""

Tc = 100.
L = 0.1 * Tc**4
#L = 1.642454973425167 * Tc**4
sigma = 0.1 * Tc**3
#sigma = 1.9792125712418323 * Tc**3
lc = 6 / Tc
T0 = Tc / (1. + 6. * sigma / (L * lc))**0.5
alpha = 1. / (2. / 3. * sigma * lc**5 * Tc**2)**0.5
gamma = (L + 6. * sigma / lc)/(6. * sigma * lc * Tc**2)
lam = 1. / (3. * sigma * lc**3)
#Tn = 78.99484291788242
Tn = 86.
# -----------specify friction here---------------
eta = 0.1 * Tc

#print(gamma, alpha, lam)
