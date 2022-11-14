#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:24:44 2022

@author: Tomasz Krajewski
"""

cdef double Tc = 100.
cdef double L = 0.1 * Tc**4
cdef double sigma = 0.1 * Tc**3
cdef double lc = 6. / Tc
cdef double T0 = Tc / (1. + 6. * sigma / (L * lc))**0.5
cdef double alpha = 1. / (2. / 3. * sigma * lc**5 * Tc**2)**0.5
cdef double gamma = (L + 6. * sigma / lc)/(6. * sigma * lc * Tc**2)
cdef double lam = 1. / (3. * sigma * lc**3)
cdef double Tn = 86.
# -----------specify friction here---------------
cdef double eta = 0 * Tc
