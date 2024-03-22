import numpy as np
from scipy import interpolate
from math import factorial
from scipy.integrate import quad
from scipy import linalg
import matplotlib.pyplot as plt
from spline import bisection, get_interpolation_coeffiecients

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
p = 5
k = 3
xmin = np.log(10**(-3))
xmax = np.log(10**(3))
intervals = np.linspace(xmin, xmax, n+1)
coefficients, lambdas = get_interpolation_coeffiecients(n, p, k, xmin, xmax, intervals, function, function_low, function_high)

def get_base_function(number=1, der=0):
        return lambdas[der][number]

def J_interpolated(z, der=0):
    if (z <= intervals[0]):
        return function_low[der](z)
    elif (z >= intervals[n]):
        return function_high[der](z)
    else:
        i = bisection(z, n, intervals)
        arg = (z - intervals[i]) / (intervals[i+1] - intervals[i])
        result = 0
        for j in range(p+1):
            result += coefficients[(p+1) * i + j] * \
                get_base_function(number=j, der=der)(arg) / \
                    (intervals[i+1] - intervals[i])**der
        return result
        
J_interpolated = np.vectorize(J_interpolated)
xplot = np.linspace(xmin, xmax, 100)
xplot2 = np.logspace(-3,3,100) 
print(xplot[0], xplot[-1])      
xi = np.linspace(-3, 3, n+1)
plt.figure()
plt.plot(xplot2, J0_full_vec(xplot2), "--")
plt.plot(xplot2, J_interpolated(np.log(xplot2), 0))
plt.xscale("log")
plt.savefig("J0.pdf")
plt.close()
plt.figure()
plt.plot(xplot2, J1_full_vec(xplot2), "--")
plt.plot(xplot2, J_interpolated(np.log(xplot2), 1))
plt.xscale("log")
plt.savefig("J1.pdf")
plt.close()
plt.figure()
plt.plot(xplot2, J2_full_vec(xplot2), "--")
plt.plot(xplot2, J_interpolated(np.log(xplot2), 2))
plt.xscale("log")
plt.savefig("J2.pdf")
plt.close()
plt.figure()
plt.plot(xplot2, J3_full_vec(xplot2), "--")
plt.plot(xplot2, J_interpolated(np.log(xplot2), 3))
plt.xscale("log")
plt.savefig("J3.pdf")
plt.close()











