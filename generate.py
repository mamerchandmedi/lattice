#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Import system dependent functionality.
import os
# end: Import system dependent functionality.

# %% Import package for handling command line arguments.
import sys
# end: Import package for handling command line arguments.

# %% Import package providing utility for importing modules by filename.
import imp
# end: Import package providing utility for importing modules by filename.

# %% Import numpy.
import numpy as np
# end: Import numpy.

# %% Processing file name.
data_file = sys.argv[1]
print("Processing " + data_file + ".")
file_path = os.path.split(data_file)[0]
file_name = os.path.splitext(os.path.split(data_file)[1])[0]
# end: Processing file name.

# %% Importing file with parameters of the model.
parameters_data = imp.load_source('parameters_data', data_file)
from parameters_data import *
# end: Importing file with parameters of the model.
