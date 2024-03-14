import numpy as np
from scipy import interpolate

def Jf_low(x):
	return -7/8 * np.pi**4/45 - np.pi**2*x**2/24
def Jb_low(x):
	return -np.pi**4/45 + np.pi**2*x**2/12

def Jf_high(x):
	return np.sqrt(np.pi/2) * x**(3/2)*np.exp(-np.sqrt(x))
def Jb_high(x):
	return - np.sqrt(np.pi/2) * x**(3/2)*np.exp(-np.sqrt(x))

def J_low(x):
	return -2 + x/2
	
def J_high(x):
	return  -np.sqrt(np.pi/2) * x**(3/4)*np.exp(-np.sqrt(x))
# Spline fitting, Jf
xfmin = -6.82200203  # -11.2403168
xfmax = 1.35e3
Jf_dat_path = "thermal_functions/finiteT_f.txt"
xf, yf = np.loadtxt(Jf_dat_path).T
tckf = interpolate.splrep(xf, yf)  
def Jf(x,n=0):
    """Jf interpolated from a saved spline. Input is (m/T)^2."""
    if(x>xfmin and x<xfmax):
    	return interpolate.splev(x, tckf, der=n)
    elif(x<xfmin):
    	return interpolate.splev(xfmin, tckf, der=n)
    else:
    	return Jf_high(x)
    	
# Spline fitting, Jf
xmin = 1.e-3  # -11.2403168
xmax = 1.e3
J_dat_path = "thermal_functions/finiteT.txt"
xi, yi = np.loadtxt(J_dat_path).T
tckf = interpolate.splrep(xi, yi)  
def J(x,n=0):
    """Jf interpolated from a saved spline. Input is (m/T)^2."""
    if(x>xmin and x<xmax):
    	return interpolate.splev(x, tckf, der=n)
    elif(x<xmin):
    	return J_low(x)
    else:
    	return J_high(x)

# Spline fitting, Jb
xbmin = -3.72402637
#xbmin = 0
# We're setting the lower acceptable bound as the point where it's a minimum
# This guarantees that it's a monatonically increasing function, and the first
# deriv is continuous.
xbmax = 1.41e3
Jb_dat_path = "thermal_functions/finiteT_b.txt"
xb, yb = np.loadtxt(Jb_dat_path).T

xbneg = np.loadtxt("thermal_functions/Jb_negative_masses.dat")[:,0]
ybneg = np.loadtxt("thermal_functions/Jb_negative_masses.dat")[:,1]


tckb = interpolate.splrep(xb, yb)
tckbneg = interpolate.splrep(xbneg, ybneg)
def Jb(x,n=0):
    """Jb interpolated from a saved spline. Input is (m/T)^2."""
    if(x>xbmin and x<xbmax):
    	return interpolate.splev(x, tckb, der=n)
    elif(x<xbmin):
    	return interpolate.splev(x, tckbneg, der=n)
    else:
    	return Jb_high(x)
 
Jb = np.vectorize(Jb)
Jf = np.vectorize(Jf)
J = np.vectorize(J)
#print(Jb(100))

#n=10
#p=4
#matrix = np.zeros((len(xi), len(xi)))
#matrix = np.zeros((5*n, 5*n))
#for i in range(n-p):
#	print(i)
#	for j in range(n):
#		if(j - i <= p and j - i >= 0):
#			matrix[i,j] = xi[i]**(j-i)
#for l in [1,2,3]:
#	print("pochodne:", n-p+l-1)
#	for i in range(p-l+1):
#		matrix[n-p+l-1, i+l] = xi[0]**(i)
#for l in [1,2,3]:
#	print("pochodne:", n-p+l+2)
#	for i in range(p-l+1):
#		matrix[n-p+l+2, i+l] = xi[n]**(i)
#print(matrix)
