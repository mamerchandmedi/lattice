import numpy as np
from scipy import interpolate
from math import factorial
from scipy.integrate import quad
from scipy import linalg
import matplotlib.pyplot as plt
from sympy import Dummy, diff, lambdify, sinh, asinh, exp, log, sqrt, Id, Symbol
import os
from pathlib import Path
from sympy.utilities.codegen import default_datatypes

# %% Import utility for automatic generation of C code.
from sympy.utilities.codegen import codegen
from sympy.printing.c import C99CodePrinter
from sympy.utilities.codegen import C99CodeGen
# end: Import utility for automatic generation of C code.

# %% Import C code optimization procedures.
from sympy.codegen.rewriting import ReplaceOptim
# end: Import C code optimization procedures.

# %% Import codegen subpackage from sympy package for "monkey patching".
import sympy.utilities.codegen
# end: Import codegen subpackage from sympy package for "monkey patching".

from .spline import get_interpolation_coeffiecients, interpolated_function, \
 fit_parameters, get_spline_from_parameters, get_base_functions


file_path = str(Path(__file__).parent)

x = Symbol("x", real=True)
k = Symbol("k", real=True, positive=True)

integrand = -k**2 * log(1 - exp(-sqrt(x + k**2)))

Ib0 = lambdify((k, x), integrand)
Ib1 = lambdify((k, x), diff(integrand, x))
Ib2 = lambdify((k, x), diff(integrand, x, 2))
Ib3 = lambdify((k, x), diff(integrand, x, 3))

def Jb0_full(x2):
	integral_result, _ = quad(Ib0, 0, np.inf, args=(x2,))
	return -integral_result

def Jb1_full(x2):
	integral_result, _ = quad(Ib1, 0, np.inf, args=(x2,))
	return -integral_result

def Jb2_full(x2):
	integral_result, _ = quad(Ib2, 0, np.inf, args=(x2,))
	return -integral_result

def Jb3_full(x2):
	integral_result, _ = quad(Ib3, 0, np.inf, args=(x2,))
	return -integral_result

Jb0_full_vec = np.vectorize(Jb0_full)
Jb1_full_vec = np.vectorize(Jb1_full)
Jb2_full_vec = np.vectorize(Jb2_full)
Jb3_full_vec = np.vectorize(Jb3_full)


def Jb_full(x, l):
    if(l==0):
        return Jb0_full_vec(x)
    elif(l==1):
        return Jb1_full_vec(x)
    elif(l==2):
        return Jb2_full_vec(x)
    elif(l==3):
        return Jb3_full_vec(x)
    else:
        return 0.


def Jb_high(x, l):
    a_b = 16 *  np.pi**2 * np.exp(1.5 - 2 * np.euler_gamma)
    if(l==0):
        return -np.pi**4 / 45 + x * np.pi**2 / 12 - x**(1.5) * np.pi / 6 - x**2 * np.log(x / a_b) / 32
    elif(l==1):
        return np.pi**2 / 12 - np.sqrt(x) * np.pi / 4 - x * np.log(x / a_b) / 16 - x / 32
    elif(l==2):
        return -np.pi / np.sqrt(x) / 8 - np.log(x / a_b) / 16 - 3 / 32
    elif(l==3):
        return np.pi / np.sqrt(x)**3 / 16 - 1 / x / 16
    else:
        return 0
		

def Jb_low(x, l):
    if(l==0):
        return -np.sqrt(np.pi / 2) * x**(0.75) * np.exp(-np.sqrt(x))
    elif(l==1):
        return np.sqrt(np.pi / 2) / 4 * np.exp(-np.sqrt(x)) * (-3 + 2 * np.sqrt(x)) / x**(0.25)
    elif(l==2):
        return np.sqrt(np.pi / 2) / 16 * np.exp(-np.sqrt(x))*(3 + 8 * np.sqrt(x)- 4 * x) / x**(1.25)
    elif(l==3):
        return np.sqrt(np.pi / 2) / 64 * np.exp(-np.sqrt(x))*(-15 - 30 * np.sqrt(x) - 12 * x + 8 * x**(1.5)) / x**(2.25)
    else:
        return 0

n = 12
p = 8
k = 3
zmin = 10**(-3)
zmax = 10**(3)

rescaling = 120 * 10**(-3)
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
            prefix='thermal_functions/bosonic_thermal_functions_jacobian')
    CythonCodeGen(code_input, 
                  prefix='thermal_functions/bosonic_thermal_functions_jacobian')
    
jacobian = generate_jacobian(p, k, lambda x: zmin * sinh(x) / rescaling)
inverse_jacobian = generate_jacobian(p, k, lambda x: asinh(rescaling * x / zmin))
code_jacobian(p, k, "jacobian", lambda x: asinh(rescaling * x / zmin))



function = [lambda x: np.sum([Jb_full(argument_function(x), l) * jacobian[0][l](x) for l in range(p+1)]),
            lambda x: np.sum([Jb_full(argument_function(x), l) * jacobian[1][l](x) for l in range(p+1)]),
            lambda x: np.sum([Jb_full(argument_function(x), l) * jacobian[2][l](x) for l in range(p+1)]),
            lambda x: np.sum([Jb_full(argument_function(x), l) * jacobian[3][l](x) for l in range(p+1)])]
function_low = [lambda x: np.sum([Jb_high(argument_function(x), l) * jacobian[0][l](x) for l in range(p+1)]),
                lambda x: np.sum([Jb_high(argument_function(x), l) * jacobian[1][l](x) for l in range(p+1)]),
                lambda x: np.sum([Jb_high(argument_function(x), l) * jacobian[2][l](x) for l in range(p+1)]),
                lambda x: np.sum([Jb_high(argument_function(x), l) * jacobian[3][l](x) for l in range(p+1)])]
function_high = [lambda x: np.sum([Jb_low(argument_function(x), l) * jacobian[0][l](x) for l in range(p+1)]),
                  lambda x: np.sum([Jb_low(argument_function(x), l) * jacobian[1][l](x) for l in range(p+1)]),
                  lambda x: np.sum([Jb_low(argument_function(x), l) * jacobian[2][l](x) for l in range(p+1)]),
                  lambda x: np.sum([Jb_low(argument_function(x), l) * jacobian[3][l](x) for l in range(p+1)])]


external_bounds = [[xmin, [function_low[j](xmin) for j in range(k+1)]],
                   [xmax, [function_high[j](xmax) for j in range(k+1)]]]
N = 2000
Z = np.linspace(xmin, xmax, N+2)
Z = Z[1:-1]

data = [[z, [Jb0_full(argument_function(z)) / np.fabs(Jb_high(zmin, 0)),
             Jb1_full(argument_function(z)) / np.fabs(Jb_high(zmin, 1)),
             Jb2_full(argument_function(z)) / np.fabs(Jb_high(zmin, 2)),
             Jb3_full(argument_function(z)) / np.fabs(Jb_high(zmin, 3))]] for z in Z]

def calculate_jacobian(jacobian):
    return lambda x: [[jacobian[j][l](x) / np.fabs(Jb_high(zmin, j)) for l in range(p+1)] for j in range(k+1)]

if not os.path.exists(file_path +'/bosonic_interpolation_coefficients.npy'):
	dependencies = [[z, calculate_jacobian(inverse_jacobian)(argument_function(z))] for z in Z]
	coefficients, lambdas = fit_parameters(p, k, intervals, external_bounds, data, dependencies)
	coefficients_0 = np.copy(coefficients)
	coefficients_0 *= np.fabs(Jb_high(zmin, 0))
	coefficients_1 = np.copy(coefficients)
	for i in range(len(intervals) - 1):
		for m in range(1, p+1):
			coefficients_1[(p+1) * i + m] = m * coefficients[(p+1) * i + m] / (intervals[i+1] - intervals[i])
	coefficients_1 *= np.fabs(Jb_high(zmin, 1))    
	coefficients_2 = np.copy(coefficients)
	for i in range(len(intervals) - 1):
		for m in range(2, p+1):
			coefficients_2[(p+1) * i + m] = m*(m-1) * coefficients[(p+1) * i + m] / (intervals[i+1] - intervals[i])**2
	coefficients_2 *= np.fabs(Jb_high(zmin, 2))    
	coefficients_3 = np.copy(coefficients)
	for i in range(len(intervals) - 1):
		for m in range(3, p+1):
			coefficients_3[(p+1) * i + m] = m*(m-1)*(m-2) * coefficients[(p+1) * i + m] / (intervals[i+1] - intervals[i])**3
	coefficients_3 *= np.fabs(Jb_high(zmin, 3))    
	np.save(file_path +'/bosonic_interpolation_coefficients.npy', [coefficients, coefficients_0, coefficients_1, coefficients_2, coefficients_3])

lambdas = get_base_functions(p, k)
coefficients_data = np.load(file_path +'/bosonic_interpolation_coefficients.npy')
coefficients = coefficients_data[0]

print(coefficients_data[1])
print(coefficients_data[2])
print(coefficients_data[3])
print("################")
print(interpolated_function(0.23, 0, intervals, coefficients, lambdas))
print(interpolated_function(0.23, 1, intervals, coefficients, lambdas))
print(interpolated_function(0.23, 2, intervals, coefficients, lambdas))                                     
print(interpolated_function(0.23, 3, intervals, coefficients, lambdas))
                                     
                                     
def Jb0_interpolated(z):
    if (z <= zmin):
        return Jb_high(z, 0)
    elif (z >= zmax):
        return Jb_low(z, 0)
    else:
        return interpolated_function(inverse_argument_function(z), 0, intervals,
                                     coefficients, lambdas)

def Jb1_interpolated(z):
    if (z <= zmin):
        return Jb_high(z, 1)
    elif (z >= zmax):
        return Jb_low(z, 1)
    else:
        return sum([interpolated_function(inverse_argument_function(z), l,
                                     intervals, coefficients, lambdas) * \
                    inverse_jacobian[1][l](z) for l in range(p+1)])


def Jb2_interpolated(z):
    if (z <= zmin):
        return Jb_high(z, 2)
    elif (z >= zmax):
        return Jb_low(z, 2)
    else:
        return sum([interpolated_function(inverse_argument_function(z), l,
                                     intervals, coefficients, lambdas) * \
                    inverse_jacobian[2][l](z) for l in range(p+1)])

def Jb3_interpolated(z):
    if (z <= zmin):
        return Jb_high(z, 3)
    elif (z >= zmax):
        return Jb_low(z, 3)
    else:
        return sum([interpolated_function(inverse_argument_function(z), l,
                                     intervals, coefficients, lambdas) * \
                    inverse_jacobian[3][l](z) for l in range(p+1)])

def Jb_interpolated(z, der=0):
    if der == 0:
        return Jb0_interpolated(z)
    elif der == 1:
        return Jb1_interpolated(z)
    elif der == 2:
        return Jb2_interpolated(z)
    elif der == 3:
        return Jb3_interpolated(z)
    else:
        return 0

Jb0_interpolated = np.vectorize(Jb0_interpolated)
Jb1_interpolated = np.vectorize(Jb1_interpolated)
Jb2_interpolated = np.vectorize(Jb2_interpolated)
Jb3_interpolated = np.vectorize(Jb3_interpolated)
Jb_interpolated = np.vectorize(Jb_interpolated)


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
    plt.plot(xplot2, Jb0_full_vec(xplot2), "--")
    plt.plot(xplot2, Jb0_interpolated(xplot2))
    plt.xscale("log")
    plt.savefig("Jb0.pdf")
    plt.close()
    plt.figure()
    plt.plot(xplot2, Jb1_full_vec(xplot2), "--")
    plt.plot(xplot2, Jb1_interpolated(xplot2))
    plt.xscale("log")
    plt.savefig("Jb1.pdf")
    plt.close()
    plt.figure()
    plt.plot(xplot2, Jb2_full_vec(xplot2), "--")
    plt.plot(xplot2, Jb2_interpolated(xplot2))
    plt.xscale("log")
    plt.savefig("Jb2.pdf")
    plt.close()
    plt.figure()
    plt.plot(xplot2, Jb3_full_vec(xplot2), "--")
    plt.plot(xplot2, Jb3_interpolated(xplot2))
    plt.xscale("log")
    plt.savefig("Jb3.pdf")
    plt.close()

