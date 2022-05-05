# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import matrices.
from matrices_high_order import advection_matrix as high_order_advection_matrix
from matrices_high_order import mass_matrix
from matrices_low_order_dynamically import advection_matrix as low_order_advection_matrix
# end: Import matrices.

# %% Import rescaled quantities.
from lattice import N
# end: Import rescaled quantities.

def antidiffusive_matrix(KH,KL,FH,FL,dt):
	A = np.ndarray((N + 1, 3))
	A[0][0] = dt*(KH[0][0]*FH[0] - KL[0][0]*FL[0])/mass_matrix[0]
	A[0][1] = dt*(KH[0][1]*FH[1] - KL[0][1]*FL[1])/mass_matrix[0]
	for j in range(1,N):
		dt*(A[j][-1] = KH[j][-1]*FH[j-1] - KL[j][-1]*FL[j-1])/mass_matrix[j]
		dt*(A[j][0] = KH[j][0]*FH[j] - KL[j][0]*FL[j])/mass_matrix[j]
		dt*(A[j][1] = KH[j][1]*FH[j+1] - KL[j][1]*FL[j+1])/mass_matrix[j]
	A[N][-1] = dt*(KH[N][-1]*FH[j-1] - KL[N][-1]*FL[j-1])/mass_matrix[N]
	A[N][0] = dt*(KH[N][0]*FH[j] - KL[N][0]*FL[j])/mass_matrix[N]
	return A
