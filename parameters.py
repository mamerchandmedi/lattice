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

V_parameter = 1.
A_parameter = 30.
Tc = 100
etaV = 0.22
Tn = 1#51  
y = 1.615

# -----------specify friction here---------------
eta = 0.0 * Tc
