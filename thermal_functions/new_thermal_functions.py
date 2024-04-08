import numpy as np
from scipy import interpolate
from math import factorial
from scipy.integrate import quad
from scipy import linalg
import matplotlib.pyplot as plt
from sympy import Dummy, diff, lambdify, sinh, asinh, exp, log, Id, Symbol
import os
from sympy.utilities.codegen import default_datatypes

# %% Import utility for automatic generation of C code.
from sympy.utilities.codegen import codegen
from sympy.printing.ccode import C99CodePrinter
from sympy.utilities.codegen import C99CodeGen
# end: Import utility for automatic generation of C code.

# %% Import C code optimization procedures.
from sympy.codegen.rewriting import ReplaceOptim
# end: Import C code optimization procedures.

# %% Import codegen subpackage from sympy package for "monkey patching".
import sympy.utilities.codegen
# end: Import codegen subpackage from sympy package for "monkey patching".

from .spline import interpolated_function, \
 fit_parameters, get_spline_from_parameters, get_base_functions


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


def J_full(x, l):
    if(l==0):
        return J0_full_vec(x)
    elif(l==1):
        return J1_full_vec(x)
    elif(l==2):
        return J2_full_vec(x)
    elif(l==3):
        return J3_full_vec(x)
    else:
        return 0.


def J_high(x, l):
    E_gamma = 0.5772156649
    if(l==0):
        return -2 + x/2 + x**2/32*(-3 + 4*E_gamma - 4*np.log(2) + 4*np.log(np.sqrt(x))) 
    elif(l==1):
        return 0.5 + x/16 + x/16*(-3 + 4*E_gamma - 4*np.log(2) + 4*np.log(np.sqrt(x))) 
    elif(l==2):
        return 3/16 + 1/16*(-3 + 4*E_gamma - 4*np.log(2) + 4*np.log(np.sqrt(x))) 
    elif(l==3):
        return 1/(8*x)
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
    else:
        return 0

n = 10
p = 8
k = 3
zmin = 10**(-3)
zmax = 10**(3)

rescaling = 80 * 10**(-3)
argument_function = lambda x: zmin * np.sinh(x) / rescaling
inverse_argument_function = lambda x: np.arcsinh(rescaling * x / zmin)

xmin = inverse_argument_function(zmin)
xmax = inverse_argument_function(zmax)
intervals = np.linspace(xmin, xmax, n+1)

def symbolic_jacobian(p, k, argument_function = Id):
    x = Symbol('x', real=True)
    jacobian = np.full((k+1, p+1), 0., dtype=object)
    jacobian = 0.*jacobian
    jacobian[0][0] = 1.
    jacobian[1][1] = diff(argument_function(x), x)
    for j in range(2, k+1):
        for l in range(1, p+1):
            jacobian[j][l] = jacobian[j-1][l-1] * diff(argument_function(x), x) + \
                diff(jacobian[j-1][l], x)
    return jacobian, x
    
def generate_jacobian(p, k, argument_function = Id):
    jacobian, x = symbolic_jacobian(p, k, argument_function)
    for l in range(p+1):
        for j in range(k+1):
            jacobian[j][l] = lambdify(x, jacobian[j][l])
    return jacobian


def get_default_datatype(expr, complex_allowed=None):
    return default_datatypes["float"]
    
sympy.utilities.codegen.get_default_datatype = get_default_datatype
   
   
def CythonCodeGen(name_expr, prefix=None):
    with open(prefix+".pxd", 'w') as file:
    	for t in name_expr:
    	    file.write("cdef extern from \""+str(prefix)+".h\":\n") 
    	    file.write("    double "+t[0]+"(double x) \n") 
    file.close()
    #with open(prefix+".pyx", 'w') as file:
    #	for t in name_expr:
    #	    file.write("cdef extern from \""+str(prefix)+".c\":\n") 
    #	    file.write("    double "+t[0]+"(double x) \n") 
    #file.close()
    
    
def code_jacobian(p, k, fun_name, argument_function = Id):
    printer = C99CodePrinter()
    gen = C99CodeGen(printer=printer)
    gen.preprocessor_statements.append('#include <math.h>')
    jacobian, x = symbolic_jacobian(p, k, argument_function)
    code_input = []
    for l in range(p+1):
        for j in range(k+1):
        	#print(j, l, jacobian[j, l])
        	code_input.append((fun_name+"_"+str(j)+"_"+str(l), jacobian[j, l]))
    codegen(code_input, code_gen=gen, argument_sequence=(x, ), to_files=True,
            prefix='thermal_functions/thermal_functions_jacobian')
    CythonCodeGen(code_input, 
                  prefix='thermal_functions/thermal_functions_jacobian')
    
jacobian = generate_jacobian(p, k, lambda x: zmin * sinh(x) / rescaling)
inverse_jacobian = generate_jacobian(p, k, lambda x: asinh(rescaling * x / zmin))
code_jacobian(p, k, "jacobian", lambda x: asinh(rescaling * x / zmin))



function = [lambda x: np.sum([J_full(argument_function(x), l) * jacobian[0][l](x) for l in range(p+1)]),
            lambda x: np.sum([J_full(argument_function(x), l) * jacobian[1][l](x) for l in range(p+1)]),
            lambda x: np.sum([J_full(argument_function(x), l) * jacobian[2][l](x) for l in range(p+1)]),
            lambda x: np.sum([J_full(argument_function(x), l) * jacobian[3][l](x) for l in range(p+1)])]
function_low = [lambda x: np.sum([J_high(argument_function(x), l) * jacobian[0][l](x) for l in range(p+1)]),
                lambda x: np.sum([J_high(argument_function(x), l) * jacobian[1][l](x) for l in range(p+1)]),
                lambda x: np.sum([J_high(argument_function(x), l) * jacobian[2][l](x) for l in range(p+1)]),
                lambda x: np.sum([J_high(argument_function(x), l) * jacobian[3][l](x) for l in range(p+1)])]
function_high = [lambda x: np.sum([J_low(argument_function(x), l) * jacobian[0][l](x) for l in range(p+1)]),
                  lambda x: np.sum([J_low(argument_function(x), l) * jacobian[1][l](x) for l in range(p+1)]),
                  lambda x: np.sum([J_low(argument_function(x), l) * jacobian[2][l](x) for l in range(p+1)]),
                  lambda x: np.sum([J_low(argument_function(x), l) * jacobian[3][l](x) for l in range(p+1)])]


external_bounds = [[[xmin], np.asarray([function_low[j](xmin) for j in range(k+1)])],
                   [[xmax], np.asarray([function_high[j](xmax) for j in range(k+1)])]]
N = 1000
Z = np.linspace(xmin, xmax, N+2)
Z = Z[1:-1]

data = [[[z], [J0_full(argument_function(z)),
               J1_full(argument_function(z)),
               J2_full(argument_function(z)),
               J3_full(argument_function(z))]] for z in Z]

def calculate_jacobian(jacobian):
    return lambda x: [[jacobian[j][l](x) for l in range(p+1)] for j in range(k+1)]

if not os.path.exists("thermal_functions/interpolation_coefficients.npy"):
	dependencies = [[[z], calculate_jacobian(inverse_jacobian)(argument_function(z))] for z in Z]
	coefficients, lambdas = fit_parameters([p], [k], [intervals], external_bounds, data, dependencies)
	coefficients_0 = np.copy(coefficients)
	coefficients_1 = np.copy(coefficients)
	for i in range(len(intervals) - 1):
		for m in range(1, p+1):
			coefficients_1[(p+1) * i + m] = m * coefficients[(p+1) * i + m] / (intervals[i+1] - intervals[i])
	coefficients_2 = np.copy(coefficients)
	for i in range(len(intervals) - 1):
		for m in range(2, p+1):
			coefficients_2[(p+1) * i + m] = m*(m-1) * coefficients[(p+1) * i + m] / (intervals[i+1] - intervals[i])**2
	coefficients_3 = np.copy(coefficients)
	for i in range(len(intervals) - 1):
		for m in range(3, p+1):
			coefficients_3[(p+1) * i + m] = m*(m-1)*(m-2) * coefficients[(p+1) * i + m] / (intervals[i+1] - intervals[i])**3
	np.save('thermal_functions/interpolation_coefficients.npy', [coefficients, coefficients_0, coefficients_1, coefficients_2, coefficients_3])
else:
	lambdas = get_base_functions([p], [k])
	coefficients_data = np.load('thermal_functions/interpolation_coefficients.npy')
	coefficients = coefficients_data[0]
                                     
                                     
def J0_interpolated(z):
    if (z <= zmin):
        return J_high(z, 0)
    elif (z >= zmax):
        return J_low(z, 0)
    else:
        return interpolated_function(inverse_argument_function(z), 0, intervals,
                                     coefficients, lambdas)

def J1_interpolated(z):
    if (z <= zmin):
        return J_high(z, 1)
    elif (z >= zmax):
        return J_low(z, 1)
    else:
        return sum([interpolated_function(inverse_argument_function(z), l,
                                     intervals, coefficients, lambdas) * \
                    inverse_jacobian[1][l](z) for l in range(p+1)])


def J2_interpolated(z):
    if (z <= zmin):
        return J_high(z, 2)
    elif (z >= zmax):
        return J_low(z, 2)
    else:
        return sum([interpolated_function(inverse_argument_function(z), l,
                                     intervals, coefficients, lambdas) * \
                    inverse_jacobian[2][l](z) for l in range(p+1)])

def J3_interpolated(z):
    if (z <= zmin):
        return J_high(z, 3)
    elif (z >= zmax):
        return J_low(z, 3)
    else:
        return sum([interpolated_function(inverse_argument_function(z), l,
                                     intervals, coefficients, lambdas) * \
                    inverse_jacobian[3][l](z) for l in range(p+1)])

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
    xplot2 = argument_function(xplot)
    plt.figure()
    plt.plot(Z, [point[1][0] for point in data])
    plt.show()
    plt.plot(Z, [point[1][1] for point in data])
    plt.show()
    plt.plot(Z, [point[1][2] for point in data])
    plt.show()
    plt.plot(Z, [point[1][3] for point in data])
    plt.show()
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

