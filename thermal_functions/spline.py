#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:46:21 2024

@author: tkrajewski
"""

from sympy import Dummy, diff, lambdify, tanh, series, binomial, \
    cos, symbols, pi, simplify

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


def bisection(x, n, intervals):
    i = 0
    j = n+1
    mid = (i+j) // 2
    while j - i > 1:
        if x >= intervals[mid]:
            i = mid
        else:
            j = mid
        mid = (i+j) // 2
    return i


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
        i = bisection(z, n, intervals)
        arg = (z - intervals[i]) / (intervals[i+1] - intervals[i])
        for j in range(p+1):
            matrix[row][(p+1) * i + j] = get_base_function(number=j, der=0)(arg) 
        row+=1

    inverse_matrix = linalg.inv(matrix)

    # Least squares from here
    Z = np.linspace(xmin, xmax, N + 2)
    Z = Z[1:-1]

    Y = weights(Z, der=0) * get_function(der=0)(Z)
    for l in range(1, k+1):
         Y = np.append(Y, weights(Z, der=l) * get_function(der=l)(Z))
    X = np.zeros(((k+1) * N, n * (p+1) - (k+1) * (n+1)))

    #weights = np.zeros(k+1)
    #for l in range(k+1):
    #    weights[l] = np.sum(Y**2)**(-1)

    for m in range(len(Z)):
        z = Z[m]
        i = bisection(z, n, intervals)
        arg = (z - intervals[i]) / (intervals[i+1] - intervals[i])
        for r in range(n * (p+1) - (k+1) * (n+1)):
            for l in range(k+1):
                result = 0
                for j in range(p+1):
                    result += inverse_matrix[(p+1) * i + j, (k+1) * (n+1) + r] * \
                        get_base_function(number=j, der=l)(arg) / \
                            (intervals[i+1] - intervals[i])**l
                X[l * N + m, r] = weights(z, der=l) * result
        for j in range(p+1):
            for s in range((k+1) * (n+1)):
                for l in range(k+1):
                    Y[l * N + m] -= weights(z, der=l) * \
                        inverse_matrix[(p+1) * i + j, s] * vector[s] * \
                        get_base_function(number=j, der=l)(arg) / \
                            (intervals[i+1] - intervals[i])**l

    XT = X.transpose()
    Beta = linalg.solve(np.matmul(XT, X), XT.dot(Y))

    for row in range(n * (p+1) - (k+1) * (n+1)):
        vector[(k+1) * (n+1) + row] = Beta[row]

    coefficients = inverse_matrix.dot(vector)
    return coefficients, lambdas

def interpolated_function(z, der, intervals, 
                   coefficients, lambdas):   
    n = len(intervals) - 1
    p = len(lambdas[der]) - 1
    i = bisection(z, n, intervals)
    arg = (z - intervals[i]) / (intervals[i+1] - intervals[i])
    result = 0
    for j in range(p+1):
        result += coefficients[(p+1) * i + j] * lambdas[der][j](arg) / \
                (intervals[i+1] - intervals[i])**der
    return result