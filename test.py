import numpy as np
data = np.load("critical_bubble_dr_010.npz")
phi0 = data["phi"]
r = data["r"]
print(len(phi0), len(r))

