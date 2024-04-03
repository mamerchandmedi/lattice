#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:46:21 2024

@author: tkrajewski
"""

from sympy import Dummy, diff, lambdify, tanh, series, binomial, \
    cos, symbols, pi, simplify
from copy import deepcopy

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def generate_monomials(order=1):
    x = Dummy('x')
    return [[x**j for j in range(order+1)], x]


def generate_derivatives(polys, der=1):
    derivatives = [polys[0]]
    for l in range(1, der+1):
        derivatives.append([diff(der) for der in derivatives[-1]])
    return [derivatives, *polys[1:]]


def lambdify_poly(polys):
    return [[lambdify(polys[1:], poly) for poly in derivatives] for derivatives in polys[0]]



def bisection(x, intervals):
    i = 0
    j = len(intervals) - 1 
    mid = (i+j) // 2
    while j - i > 1:
        if x >= intervals[mid]:
            i = mid
        else:
            j = mid
        mid = (i+j) // 2
    return i, j

def get_base_functions(p, k):
    monomials = generate_monomials(order=p)
    derivatives = generate_derivatives(monomials, der=p)
    lambdas = lambdify_poly(derivatives)
    return lambdas

def interpolated_function(z, der, intervals, coefficients, lambdas):   
    p = len(lambdas[der]) - 1
    i, j = bisection(z, intervals)
    arg = (z - intervals[i]) / (intervals[j] - intervals[j-1])
    result = 0
    for m in range(p+1):
        result += coefficients[(p+1) * i + m] * lambdas[der][m](arg) / \
                (intervals[j] - intervals[j-1])**der
    return result


def get_spline_from_parameters(p, k, intervals, external_bounds):
    n = len(intervals) - 1
    matrix = np.zeros((n * (p+1), n * (p+1)))
    vector = np.zeros(n * (p+1))
    lambdas = get_base_functions(p, k)
    
    def get_base_function(number, der=0):
        return lambdas[der][number]

    row = 0

    for i in range(1, n):
        for l in range(k+1):
            for j in range(p+1):
                matrix[row][(p+1) * (i-1) + j] = -get_base_function(number=j, der=l)(1.) / \
                     (intervals[i+1] - intervals[i])**l
                matrix[row][(p+1) * i + j] = get_base_function(number=j, der=l)(0.) / \
                     (intervals[i+1] - intervals[i])**l
            row+=1
    
    for z, values in external_bounds:
        i, j = bisection(z, intervals)
        if z - intervals[i] == 0:
            arg = 0.
        else:
            arg = (z - intervals[i]) / (intervals[i+1] - intervals[i])
        for l in range(k+1):
            if values[l] is not None:
                for m in range(p+1):
                    matrix[row][(p+1) * i + m] = get_base_function(number=m, der=l)(arg) / \
                        (intervals[j] - intervals[j-1])**l
                vector[row] = values[l]
                row+=1

    known = row
    y = np.linspace(intervals[0], intervals[-1], (n * (p+1) - known) + 2)
    y = y[1:-1]

    for z in y:
        i, j = bisection(z, intervals)
        arg = (z - intervals[i]) / (intervals[j] - intervals[j-1])
        for m in range(p+1):
            matrix[row][(p+1) * i + m] = get_base_function(number=m, der=0)(arg) 
        row+=1

    inverse_matrix = linalg.inv(matrix)
    
    def coefficients(parameters):
        modified_vector = deepcopy(vector)
        for i in range(n * (p+1) - known):
            modified_vector[known + i] = parameters[i]
        return inverse_matrix.dot(modified_vector)

    return coefficients, lambdas, len(y)


def fit_parameters(p, k, intervals, external_bounds, data, dependencies):
    coefficients, lambdas, n_parameters = \
        get_spline_from_parameters(p, k, intervals, external_bounds)

    Y = []
    X = []
    for function, derivative in list(zip(data, dependencies)):
        #print("function: ",function)
        z, values = function
        zprim, jacobian = derivative
        #print("derivative: ",derivative)
        #print("values: ", values)
        assert(z == zprim)
        for l in range(len(values)):
            if values[l] is not None:
                Y.append(values[l] - np.sum([jacobian[l][j] * interpolated_function(z, j, intervals, coefficients(np.zeros(n_parameters)), lambdas) for j in range(p+1)]))
                parameters = np.identity(n_parameters)
                X.append([np.sum([jacobian[l][j] * (interpolated_function(z, j, intervals, coefficients(parameters[m]), lambdas) - interpolated_function(z, j, intervals, coefficients(np.zeros(n_parameters)), lambdas)) for j in range(p+1)]) for m in range(n_parameters)])

    Y = np.array(Y)
    X = np.array(X)
    #print("Y :", Y)
    #print("X: ", X)
    XT = X.transpose()
    Beta = linalg.solve(np.matmul(XT, X), XT.dot(Y))
    #print(Beta)
    return coefficients(Beta), lambdas


def get_interpolation_coeffiecients(n, p, k, xmin, xmax, intervals, 
                                    function, function_low, function_high,
                                    N=1000, weights=None):
    if weights is None:
        def weights(Z, der=0):
            return 1;

    # Helper wrappers.
    def get_function(der=0):
        return function[der]
    
    def get_function_low(der=0):
        return function_low[der]

    def get_function_high(der=0):
        return function_high[der]
        
    matrix = np.zeros((n * (p+1), n * (p+1)))
    vector = np.zeros(n * (p+1))
    monomials = generate_monomials(order=p)
    derivatives = generate_derivatives(monomials, der=p)
    lambdas = lambdify_poly(derivatives)
    
    def get_base_function(number, der=0):
        return lambdas[der][number]

    row = 0

    for l in range(k+1):
        for j in range(p+1):
            matrix[row][j] = get_base_function(number=j, der=l)(0.) / \
                 (intervals[1] - intervals[0])**l
            vector[row] = get_function_low(der=l)(intervals[0])
        row+=1

    for i in range(1, n):
        for l in range(k+1):
            for j in range(p+1):
                matrix[row][(p+1) * (i-1) + j] = -get_base_function(number=j, der=l)(1.) / \
                     (intervals[i+1] - intervals[i])**l
                matrix[row][(p+1) * i + j] = get_base_function(number=j, der=l)(0.) / \
                     (intervals[i+1] - intervals[i])**l
            row+=1

    for l in range(k+1):
        for j in range(p+1):
            matrix[row][(n-1) * (p+1) + j] = get_base_function(number=j, der=l)(1.) / \
                 (intervals[n] - intervals[n-1])**l
            vector[row] = get_function_high(der=l)(intervals[n])
        row+=1


    y = np.linspace(xmin, xmax, (n * (p+1) - (k+1) * (n+1)) + 2)
    y = y[1:-1]

    for z in y:
        i, j = bisection(z, intervals)
        arg = (z - intervals[i]) / (intervals[j] - intervals[j-1])
        for m in range(p+1):
            matrix[row][(p+1) * i + m] = get_base_function(number=m, der=0)(arg) 
        row+=1

    inverse_matrix = linalg.inv(matrix)

    # Least squares from here
    Z = np.linspace(xmin, xmax, N + 2)
    Z = Z[1:-1]

    Y = weights(Z, der=0) * get_function(der=0)(Z)
    for l in range(1, k+1):
         Y = np.append(Y, weights(Z, der=l) * get_function(der=l)(Z))
    X = np.zeros(((k+1) * N, n * (p+1) - (k+1) * (n+1)))

    for m in range(len(Z)):
        z = Z[m]
        i, j = bisection(z, intervals)
        arg = (z - intervals[i]) / (intervals[j] - intervals[j-1])
        for r in range(n * (p+1) - (k+1) * (n+1)):
            for l in range(k+1):
                result = 0
                for s in range(p+1):
                    result += inverse_matrix[(p+1) * i + s, (k+1) * (n+1) + r] * \
                        get_base_function(number=s, der=l)(arg) / \
                            (intervals[j] - intervals[j-1])**l
                X[l * N + m, r] = weights(z, der=l) * result
        for r in range(p+1):
            for s in range((k+1) * (n+1)):
                for l in range(k+1):
                    Y[l * N + m] -= weights(z, der=l) * \
                        inverse_matrix[(p+1) * i + r, s] * vector[s] * \
                        get_base_function(number=r, der=l)(arg) / \
                            (intervals[j] - intervals[j-1])**l

    XT = X.transpose()
    Beta = linalg.solve(np.matmul(XT, X), XT.dot(Y))

    for row in range(n * (p+1) - (k+1) * (n+1)):
        vector[(k+1) * (n+1) + row] = Beta[row]

    coefficients = inverse_matrix.dot(vector)
    return coefficients, lambdas

