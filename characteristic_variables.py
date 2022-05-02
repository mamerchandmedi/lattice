import numpy as np

# %% Import termodynamically corrected quantities.
from thermo import a, VT, dVT, dVTdT, d2VTdT, w, gam
# end: Import termodynamically corrected quantities.

def dpdT(field,T):
	return 4*a*T**3 - dVTdT(field,T)
	
def dtaudT(v,field,T):
	return gam(v)**2*(16*a*T**3- dVTdT(field,T) - T*d2VTdT(field,T)) - dpdT(field,T)

def dtaudv(v,field,T):
	return 2*w(field,T)*v**2*gam(v)**4

def dZdT(v,field,T):
	return v*gam(v)**2*(16*a*T**3 - dVTdT(field,T) - T*d2VTdT(field,T))

def dZdv(v,field,T):
	return w(field,T)*(1+v**2)/gam(v)**4
	
def determinant(v,field,T):
	return dtaudv(v,field,T)*dZdT(v,field,T) - dtaudT(v,field,T)*dZdv(v,field,T)
	
def dvdtau(v,field,T):
	return dZdT(v,field,T)/determinant(v,field,T)
	
def dvdZ(v,field,T):
	return -dtaudT(v,field,T)/determinant(v,field,T)
	
def dTdtau(v,field,T):
	return -dZdv(v,field,T)/determinant(v,field,T)

def dTdZ(v,field,T):
	return dtaudv(v,field,T)/determinant(v,field,T)

def Jacobian(v,field,T,tau,Z,P):
	J = np.ndarray((2, 2))
	J[0][0] = v + v*dpdT(T,field)*dTdtau(v,T,field) + (tau + P)*dvdtau(v,T,field)
	J[0][1] =  v*dpdT(T,field)*dTdZ(v,T,field) + (tau + P)*dvdZ(v,T,field)
	J[1][0] = Z*dvdtau(v,T,field)
	J[1][1] = v + Z*dvdZ(v,T,field)
	return J
	
def T_matrices(v,field,T,tau,Z,P):
	A = Jacobian(v,field,T,tau,Z,P)
	T = np.ndarray((2, 2))
	T_inv = np.ndarray((2, 2))
	delta = (A[0][0] - A[1][1])**2 - 4*A[0][1]*A[1][0]
	T[0][0] = 1/(2*A[1][0]*delta**0.5)**0.5 * (A[0][0]-A[1][1]+delta**0.5)
	T[0][1] = 1/(2*A[1][0]*delta**0.5)**0.5 * (A[0][0]-A[1][1]-delta**0.5)
	T[1][0] = 1/(2*A[1][0]*delta**0.5)**0.5 * 2*A[1][0]
	T[1][1] = 1/(2*A[1][0]*delta**0.5)**0.5 * 2*A[1][0]
	
	T_inv[0][0] = 1/(2*A[1][0]*delta**0.5)**0.5 * 2*A[1][0]
	T_inv[0][1] = 1/(2*A[1][0]*delta**0.5)**0.5 * (A[1][1]-A[0][0]+delta**0.5)
	T_inv[1][0] = 1/(2*A[1][0]*delta**0.5)**0.5 * 2*A[1][0]
	T_inv[1][1] = T_inv[0][1] = 1/(2*A[1][0]*delta**0.5)**0.5 * (A[0][0]-A[1][1]+delta**0.5)
	
	return T, T_inv
	


