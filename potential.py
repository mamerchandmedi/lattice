#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:26:59 2022

@author: Tomasz Krajewski
"""

# %% Importing parameters of the model.
from parameters import T0, alpha, gamma, lam, Tn
# end: Importing parameters of the model.


# %% Defining potential of the model.
def V(phi):
    return 0.5 * gamma * (Tn**2 - T0**2) * phi**2 - \
        (1 / 3) * alpha * Tn * phi**3 + 0.25 * lam * phi**4
# end: Defining potential of the model.


# %% Defining derivative of the potential of the model.
def dV(phi):
    return gamma * (Tn**2 - T0**2) * phi - alpha * Tn * phi**2 + lam * phi**3
# end: Defining derivative of the potential of the model.
