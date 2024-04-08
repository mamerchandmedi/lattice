#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:46:21 2024

@author: tkrajewski
"""

from sympy import Dummy, diff, lambdify, binomial, symbols, simplify, factorint
from copy import deepcopy

import numpy as np
from scipy import linalg

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


def generate_base_polynomials(base_polynomials, derivatives, polynomial,
                              orders=[1], ders=[0]):
    assert(len(derivatives) == len(orders))
    if len(orders) == 1:
        for j in range(orders[0]+1):
            for m in range(ders[0]+1):
                new_polynomial = deepcopy(polynomial)
                new_polynomial[0] *= derivatives[0][0][m][j]
                new_polynomial[1].append(derivatives[0][1])
                base_polynomials[m][j] = lambdify(new_polynomial[1],
                                                  new_polynomial[0])
    else:
        for j in range(orders[0]+1):
            for m in range(ders[0]+1):
                new_polynomial = deepcopy(polynomial)
                new_polynomial[0] *= derivatives[0][0][m][j]
                new_polynomial[1].append(derivatives[0][1])
                generate_base_polynomials(base_polynomials[j][m],
                                          derivatives[1:],
                                          new_polynomial,
                                          orders=orders[1:], ders=ders[1:])


def get_base_functions(p, k):
    assert(len(p) == len(k))
    derivatives = []
    for dim in range(len(p)):
        monomials = generate_monomials(order=p[dim])
        derivatives.append(generate_derivatives(monomials,
                                              der=p[dim]))
    lambdas = np.ndarray(tuple(item for order, der in zip(p, k) for item in (der+1, order+1)),
                       dtype=object)
    generate_base_polynomials(lambdas, derivatives, [1., []], orders=p, ders=k)
    return lambdas


def interpolated_function(z, ders, intervals, coefficients, lambdas):
    def call_function(args, i, j, index, ders, coefficients, lambdas, intervals):
        if len(ders) == 1:
            order = len(lambdas[ders[0]]) - 1
            result = 0
            for m in range(order+1):
                result += coefficients[index * (order+1) * \
                                       len(intervals[0]) + (order+1) * i[0] + m] * \
                    lambdas[ders[0]][m](*args) / \
                        (intervals[0][j[0]] - intervals[0][j[0]-1])**ders[0]
            return result
        else:
            order = len(lambdas[ders[0]]) - 1
            result = 0
            for m in range(order+1):
                new_index = index * (order+1) * len(intervals[0]) + (order+1) * i[0] + m
                result += call_function(args, i[1:], j[1:], new_index,
                                       ders[1:], coefficients, lambdas[ders[0]][m],
                                       intervals[1:])
            return result / \
                    (intervals[0][j[0]] - intervals[0][j[0]-1])**ders[0]
    i = np.ndarray((len(z),), dtype=int)
    j = np.ndarray((len(z),), dtype=int)
    args = np.ndarray((len(z),))
    for dim in range(len(z)):
        i[dim], j[dim] = bisection(z[dim], intervals[dim])
        args[dim] = (z[dim] - intervals[dim][i[dim]]) / \
                   (intervals[dim][j[dim]] - intervals[dim][j[dim]-1])
    result = call_function(args, i, j, 0, ders, coefficients, lambdas, intervals)
    return result


def get_spline_from_parameters(p, k, intervals, external_bounds):
    assert(len(p) == len(k) == len(intervals))
    dims = len(intervals)
    ns = [len(interval) - 1 for interval in intervals]
    degrees_of_freedom = np.prod(ns) *  np.prod([order + 1 for order in p])
    matrix = np.zeros((degrees_of_freedom, degrees_of_freedom))
    vector = np.zeros(degrees_of_freedom)
    lambdas = get_base_functions(p, p)

    def _get_base_function(numbers, ders, lambdas):
        if len(numbers) == 1:
            assert(len(numbers) == len(ders))
            return lambdas[ders[0]][numbers[0]]
        else:
            return _get_base_function(numbers[1:], ders[1:],
                                      lambdas[ders[0]][numbers[0]])

    def get_base_function(numbers, ders=0):
        return _get_base_function(numbers, ders, lambdas)

    def get_index(p, point, numbers):
        index = point[0] * (p[0] + 1) + numbers[0]
        for dim in range(dims-1):
            index *= ns[dim]
            index += point[dim+1] * (p[dim + 1] + 1) + numbers[dim]
        return index

    intervals_grid = np.meshgrid(*[range(n - 1) for n in ns])
    points = list(zip(*(x.flat for x in intervals_grid)))
    number_grid = np.meshgrid(*[range(order+1) for order in  p])
    numbers = list(zip(*(x.flat for x in number_grid)))
    derivatives_grid = np.meshgrid(*[range(der+1) for der in  k])
    ders = list(zip(*(x.flat for x in derivatives_grid)))

    row = 0

    for i in range(len(points)):
        for direction in range(dims):
            # Derivatives in direction perpendicular to
            # the face at boundary of elements has to be idencitcal only
            # k[direction] order, but the dependence on the parallel directions
            # need to be identical, i.e. all derivatives need to be identical.
            perpendicular = deepcopy(p)
            perpendicular[direction] = k[direction]
            perpendicular_grid = np.meshgrid(*[range(der+1) for der in  perpendicular])
            pers = list(zip(*(x.flat for x in perpendicular_grid)))
            for l in range(len(pers)):
                for j in range(len(numbers)):
                    args = np.zeros(dims)
                    shifted = list(points[i])
                    shifted[direction] += 1
                    matrix[row][get_index(p, shifted, numbers[j])] =\
                        -get_base_function(numbers=numbers[j], ders=pers[l])(*args) / \
                         np.prod([(intervals[dim][shifted[dim]+1] - intervals[dim][shifted[dim]])**pers[l][dim] for dim in range(dims)])
                    args[direction] = 1.
                    matrix[row][get_index(p, points[i], numbers[j])] = \
                        get_base_function(numbers=numbers[j], ders=pers[l])(*args) / \
                         np.prod([(intervals[dim][points[i][dim]+1] - intervals[dim][points[i][dim]])**pers[l][dim] for dim in range(dims)])
                row+=1

    for z, values in external_bounds:
        i = np.ndarray((len(z),), dtype=int)
        j = np.ndarray((len(z),), dtype=int)
        args = np.ndarray((len(z),))
        for dim in range(len(z)):
            i[dim], j[dim] = bisection(z[dim], intervals[dim])
            if z[dim] - intervals[dim][i[dim]] == 0:
                args[dim] = 0.
            else:
                args[dim] = (z[dim] - intervals[dim][i[dim]]) / \
                           (intervals[dim][j[dim]] - intervals[dim][j[dim]-1])
        for l in range(len(ders)):
            if values.flat[l] is not None:
                for m in range(len(numbers)):
                    matrix[row][get_index(p, i, numbers[m])] = \
                        get_base_function(numbers=numbers[m], ders=ders[l])(*args) / \
                        np.prod([(intervals[dim][i[dim]+1] - intervals[dim][i[dim]])**ders[l][dim] for dim in range(dims)])
                vector[row] = values.flat[l]
                row+=1


    def find_division(ns, total_number):
        if len(ns) == 1:
            return {ns[0]: total_number}
        else:
            max_n = max(ns)
            new_ns = ns
            new_ns.remove(max_n)
            approx_total_number = np.prod(new_ns)
            search_factor = max_n / approx_total_number
            result = 1
            for proposal in range(np.ceil(total_number / 2)):
                if total_number % proposal:
                    continue
                if ((total_number // proposal) * search_factor - 1.) < \
                    ((total_number // result) * search_factor - 1.):
                    result = proposal
            return {**find_division(new_ns, total_number // proposal),
                    **{max_n: proposal}}

    known = row
    division = find_division(ns, degrees_of_freedom - known)
    mesh = [division[n] for n in ns]
    parameters = [np.linspace(intervals[dim][0], intervals[dim][-1], mesh[dim] + 2)[1:-1] for dim in range(dims)]

    parameters_grid = np.meshgrid(*parameters)
    extra_points = list(zip(*(x.flat for x in parameters_grid)))

    for z in extra_points:
        i = np.ndarray((len(z),), dtype=int)
        j = np.ndarray((len(z),), dtype=int)
        args = np.ndarray((len(z),))
        for dim in range(len(z)):
            i[dim], j[dim] = bisection(z[dim], intervals[dim])
            args[dim] = (z[dim] - intervals[dim][i[dim]]) / \
                       (intervals[dim][j[dim]] - intervals[dim][j[dim]-1])
        for m in range(len(numbers)):
            matrix[row][get_index(p, i, numbers[m])] = \
                         get_base_function(numbers=numbers[m],
                                           ders=np.zeros(dims, dtype=int))(*args)
        row+=1

    inverse_matrix = linalg.inv(matrix)

    def coefficients(parameters):
        modified_vector = deepcopy(vector)
        for i in range(degrees_of_freedom - known):
            modified_vector[known + i] = parameters[i]
        return inverse_matrix.dot(modified_vector)

    return coefficients, lambdas, len(extra_points)


def fit_parameters(p, k, intervals, external_bounds, data, dependencies):
    coefficients, lambdas, n_parameters = \
        get_spline_from_parameters(p, k, intervals, external_bounds)

    Y = []
    X = []
    
    number_grid = np.meshgrid(*[range(order+1) for order in  p])
    numbers = list(zip(*(x.flat for x in number_grid)))
    derivatives_grid = np.meshgrid(*[range(der+1) for der in  k])
    ders = list(zip(*(x.flat for x in derivatives_grid)))
    
    for function, derivative in list(zip(data, dependencies)):
        z, values = function
        zprim, jacobian = derivative
        assert(z == zprim)
        flat_values = np.asarray(values).flat
        flat_jacobian = np.reshape(jacobian, (len(ders), len(numbers)))
        for l in range(len(flat_values)):
            if values[l] is not None:
                Y.append(flat_values[l] - np.sum([flat_jacobian[l][j] * interpolated_function(z, numbers[j], intervals, coefficients(np.zeros(n_parameters)), lambdas) for j in range(len(numbers))]))
                parameters = np.identity(n_parameters)
                X.append([np.sum([flat_jacobian[l][j] * (interpolated_function(z, numbers[j], intervals, coefficients(parameters[m]), lambdas) - interpolated_function(z, numbers[j], intervals, coefficients(np.zeros(n_parameters)), lambdas)) for j in range(len(numbers))]) for m in range(n_parameters)])

    Y = np.array(Y)
    X = np.array(X)
    XT = X.transpose()
    Beta = linalg.solve(np.matmul(XT, X), XT.dot(Y))
    return coefficients(Beta), lambdas
