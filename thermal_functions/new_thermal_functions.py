import numpy as np
from scipy import interpolate
from math import factorial
from scipy.integrate import quad
from scipy import linalg
import matplotlib.pyplot as plt
from spline import get_interpolation_coeffiecients, interpolated_function

def I0(k, m):
	return k**2*np.exp(-(m + k**2)**0.5)

def I1(k, m):
	return - k**2*np.exp(-(m + k**2)**0.5) / (2*np.sqrt(k**2 + m))
	
def I2(k, m):
	return  k**2*(np.exp(-(m + k**2)**0.5) / (4*(k**2 + m)**1.5) + np.exp(-(m + k**2)**0.5) / (4*(k**2 + m)))
	
def I3(k, m):
	return  - k**2*np.exp(-(m + k**2)**0.5) *  (3 + k**2 + m + 3 * (k**2 + m)**0.5) / (8*(k**2 + m)**2.5)
    
def J0_full(x2):
	integral_result, _ = quad(I0, 0, np.inf, args=(x2,))
	return -integral_result

def J1_full(x2):
	integral_result, _ = quad(I1, 0, np.inf, args=(x2,))
	return -integral_result

def J2_full(x2):
	integral_result, _ = quad(I2, 0, np.inf, args=(x2,))
	return -integral_result
	
def J3_full(x2):
	integral_result, _ = quad(I3, 0, np.inf, args=(x2,))
	return -integral_result
	
J0_full_vec = np.vectorize(J0_full)
J1_full_vec = np.vectorize(J1_full)
J2_full_vec = np.vectorize(J2_full)
J3_full_vec = np.vectorize(J3_full)


def J_high(x, l):
	if(l==0):
		return -2 + x/2
	elif(l==1):
		return 0.5
	else:
		return 0

def J_low(x, l):
	if(l==0):
		return -np.sqrt(np.pi/2) * x**(0.75)*np.exp(-np.sqrt(x))
	elif(l==1):
		return np.sqrt(np.pi/2)/4 * np.exp(-np.sqrt(x))*(-3+2*np.sqrt(x))/x**(0.25)
	elif(l==2):
		return np.sqrt(np.pi/2)/16 * np.exp(-np.sqrt(x))*(3+8*np.sqrt(x)-4*x)/x**(1.25)
	elif(l==3):
		return np.sqrt(np.pi/2)/64 * np.exp(-np.sqrt(x))*(-15-30*np.sqrt(x)-12*x + 8*x**(1.5))/x**(2.25)
		


function = [lambda x: J0_full_vec(np.exp(x)), lambda x:J1_full_vec(np.exp(x))*np.exp(x), lambda x:J1_full_vec(np.exp(x))*np.exp(x) + J2_full_vec(np.exp(x))*np.exp(2*x), lambda x:J1_full_vec(np.exp(x))*np.exp(x) + J2_full_vec(np.exp(x))*np.exp(2*x) + J2_full_vec(np.exp(x))*2*np.exp(2*x) + J3_full_vec(np.exp(x))*np.exp(3*x)]

function_high = [lambda x: J_low(np.exp(x), 0), lambda x:J_low(np.exp(x), 1)*np.exp(x), lambda x:J_low(np.exp(x), 1)*np.exp(x) + J_low(np.exp(x), 2)*np.exp(2*x), lambda x:J_low(np.exp(x), 1)*np.exp(x) + J_low(np.exp(x), 2)*np.exp(2*x) + J_low(np.exp(x),2)*2*np.exp(2*x) + J_low(np.exp(x), 3)*np.exp(3*x)]
	
function_low = [lambda x: J_high(np.exp(x), 0), lambda x:J_high(np.exp(x), 1)*np.exp(x), lambda x:J_high(np.exp(x), 1)*np.exp(x) + J_high(np.exp(x), 2)*np.exp(2*x) , lambda x:J_high(np.exp(x), 1)*np.exp(x) + J_high(np.exp(x), 2)*np.exp(2*x) + J_high(np.exp(x),2)*2*np.exp(2*x) + J_high(np.exp(x), 3)*np.exp(3*x)]

n = 15
p = 4
k = 3
zmin = 10**(-3)
zmax = 10**(3)
xmin = np.log(zmin)
xmax = np.log(zmax)
intervals = np.linspace(xmin, xmax, n+1)

def weights(x, der):
    return np.exp(-der * x)

coefficients, lambdas = get_interpolation_coeffiecients(n, p, k, xmin, xmax,
                                                        intervals, function,
                                                        function_low,
                                                        function_high,
                                                        1000,
                                                        weights)

def J0_interpolated(z):
    if (z <= zmin):
        return J_high(z, 0)
    elif (z >= zmax):
        return J_low(z, 0)
    else:
        return interpolated_function(np.log(z), 0, intervals,
                                     coefficients, lambdas)

def J1_interpolated(z):
    if (z <= zmin):
        return J_high(z, 1)
    elif (z >= zmax):
        return J_low(z, 1)
    else:
        return interpolated_function(np.log(z), 1, intervals,
                                     coefficients, lambdas) / z


def J2_interpolated(z):
    if (z <= zmin):
        return J_high(z, 2)
    elif (z >= zmax):
        return J_low(z, 2)
    else:
        return (interpolated_function(np.log(z), 2, intervals,
                                     coefficients, lambdas) -
            interpolated_function(np.log(z), 1, intervals,
                                         coefficients, lambdas)) / (z * z) 

def J3_interpolated(z):
    if (z <= zmin):
        return J_high(z, 3)
    elif (z >= zmax):
        return J_low(z, 3)
    else:
        return (interpolated_function(np.log(z), 3, intervals,
                                     coefficients, lambdas) - 
            3 * interpolated_function(np.log(z), 2, intervals,
                                         coefficients, lambdas) +
            2 * interpolated_function(np.log(z), 1, intervals,
                                         coefficients, lambdas)) / (z * z * z)

def J_interpolated(z, der=0):
    if der == 0:
        return J0_interpolated(z)
    elif der == 1:
        return J1_interpolated(z)
    elif der == 2:
        return J2_interpolated(z)
    elif der == 3:
        return J3_interpolated(z)
    else:
        return 0

J0_interpolated = np.vectorize(J0_interpolated)
J1_interpolated = np.vectorize(J1_interpolated)
J2_interpolated = np.vectorize(J2_interpolated)
J3_interpolated = np.vectorize(J3_interpolated)
J_interpolated = np.vectorize(J_interpolated)

if __name__=='__main__':
    xplot = np.linspace(xmin, xmax, 100)
    xplot2 = np.logspace(-3.5,3.5,100) 
    print(xplot[0], xplot[-1])      
    xi = np.linspace(-3, 3, n+1)
    plt.figure()
    plt.plot(xplot2, J0_full_vec(xplot2), "--")
    plt.plot(xplot2, J0_interpolated(xplot2))
    plt.xscale("log")
    plt.savefig("J0.pdf")
    plt.close()
    plt.figure()
    plt.plot(xplot2, J1_full_vec(xplot2), "--")
    plt.plot(xplot2, J1_interpolated(xplot2))
    plt.xscale("log")
    plt.savefig("J1.pdf")
    plt.close()
    plt.figure()
    plt.plot(xplot2, J2_full_vec(xplot2), "--")
    plt.plot(xplot2, J2_interpolated(xplot2))
    plt.xscale("log")
    plt.savefig("J2.pdf")
    plt.close()
    plt.figure()
    plt.plot(xplot2, J3_full_vec(xplot2), "--")
    plt.plot(xplot2, J3_interpolated(xplot2))
    plt.xscale("log")
    plt.savefig("J3.pdf")
    plt.close()











