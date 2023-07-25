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

# Spline fitting, Jb
xbmin = -3.72402637
# We're setting the lower acceptable bound as the point where it's a minimum
# This guarantees that it's a monatonically increasing function, and the first
# deriv is continuous.
xbmax = 1.41e3
Jb_dat_path = "thermal_functions/finiteT_b.txt"
xb, yb = np.loadtxt(Jb_dat_path).T

tckb = interpolate.splrep(xb, yb)
def Jb(x,n=0):
    """Jb interpolated from a saved spline. Input is (m/T)^2."""
    if(x>xbmin and x<xbmax):
    	return interpolate.splev(x, tckb, der=n)
    elif(x<xbmin):
    	return interpolate.splev(xbmin, tckb, der=n)
    else:
    	return Jb_high(x)
 
Jb = np.vectorize(Jb)
