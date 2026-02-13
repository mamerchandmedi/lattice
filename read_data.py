#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:41:14 2026

@author: Tomasz Krajewski
"""

import sys
import os
import re
import numpy as np

# Main function.
if __name__ == "__main__":
    data = list()
    with open(sys.argv[1]) as file:
        match_data = None
        while not match_data:
            line = file.readline()
            match = re.search(r'(\s*)#define(\s+)profile_points(\s*)(\d+)', line)
            if match:
                profile_points = int(match.group(4))
            match = re.search(r'(\s*)#define(\s+)profile_step(\s*)([-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?)', line)
            if match:
                profile_step = float(match.group(4))
            match = re.search(r'(\s*)#define(\s+)profile_start(\s*)([-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?)', line)
            if match:
                profile_start = float(match.group(4))
            match_data = re.search(r'(\s*)/*(\s+)!data!', line)
        for i in range(profile_points):
            line = file.readline()
            data.append(float(line))
        data = np.asarray(data)
        with open(sys.argv[1].rsplit('.', maxsplit=1)[0] + '.npy', 'wb') as output:
            np.save(output, data)    
                
