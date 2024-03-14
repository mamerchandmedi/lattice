#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:07:30 2022

@author: Tomasz Krajewski
"""

# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

GeV = 1

c_alpha = 0.
c_beta = 0.
c_gamma = 0.

rmin = 0. * GeV**(-1)
rmax = 7. * GeV**(-1)
dr = 0.01 * GeV**(-1)
dt = 0.001 * GeV**(-1)
N = int(np.floor((rmax - rmin) / dr))

r = np.array(range(0, N + 1)) * dr
