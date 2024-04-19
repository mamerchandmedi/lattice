#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:40:46 2024

@author: Tomasz Krajewski
"""

from sympy import Symbol, symbols, Function, diff, sqrt, simplify, \
    Derivative, Expr, Add, init_printing, series

def _eval_expand_basic(self, **hints):
    if isinstance(self.expr, Expr):
        summand = self.expr.expand(**hints)
        if summand.is_Add and summand.is_commutative:
            return Add(*[self.func(i, *self.variables) for i in summand.args])
        elif summand != self.expr:
            return self.func(summand, *self.variables)
    return self

Derivative._eval_expand_basic = _eval_expand_basic

# init_printing(use_latex='mathjax')

t, r = symbols(r't, r')
eta = Symbol(r'\eta')
phi = Function(r'\phi')
tau = Function(r'\tau')
Z = Function(r'Z')
T = Function(r'T')
p = Function(r'p')
v = Function(r'v')
dVdphi = Function(r'\frac{\partial V}{\partial \phi}')

def gamma(v):
    return 1 / sqrt(1 - v**2)

field_equation = -phi(t, r).diff(t, 2) + \
    (r**2 * phi(t, r).diff(r)).diff(r) / r**2 -\
    dVdphi(phi(t,r), T(t, r))- \
    eta * gamma(v(t, r)) * (phi(t, r).diff(t) + v(t, r) * phi(t, r).diff(r))

conservation_equations = (tau(t, r).diff(t) + 
                          (r**2 * (tau(t, r) + p(t, r)) * v(t, r)).diff(r) / r**2 -
                          dVdphi(phi(t,r), T(t, r)) * phi(t, r).diff(t) -
                          eta * gamma(v(t, r)) * (phi(t, r).diff(t) + v(t, r) * phi(t, r).diff(r)) * phi(t, r).diff(t),
                          Z(t, r).diff(t) + 
                          (r**2 * Z(t, r)).diff(r) / r**2 +
                          p(t, r).diff(r) +
                          dVdphi(phi(t,r), T(t, r)) * phi(t, r).diff(r) +
                          eta * gamma(v(t, r)) * (phi(t, r).diff(t) + v(t, r) * phi(t, r).diff(r)) * phi(t, r).diff(r))

v_w = Symbol(r'v_w')
xi = Symbol(r'\xi')
chi = Symbol(r'\chi')
parameter = Symbol(r'\epsilon')
phi0 = Function(r'phi_0')
phi1 = Function(r'phi_1')
tau0 = Function(r'tau_0')
tau1 = Function(r'tau_1')
Z0 = Function(r'Z_0')
Z1 = Function(r'Z_1')
T0 = Function(r'T_0')
T1 = Function(r'T_1')
p0 = Function(r'p_0')
p1 = Function(r'p_1')
v0 = Function(r'v_0')
v1 = Function(r'v_1')
rules = {phi(t, r): phi0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * phi1((r - v_w * t) * gamma(v_w), parameter * r / t),
         tau(t, r): tau0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * tau1((r - v_w * t) * gamma(v_w), parameter * r / t),
         Z(t, r): Z0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * Z1((r - v_w * t) * gamma(v_w), parameter * r / t),
         T(t, r): T0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * T1((r - v_w * t) * gamma(v_w), parameter * r / t),
         p(t, r): p0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * p1((r - v_w * t) * gamma(v_w), parameter * r / t),
         v(t, r): v0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * v1((r - v_w * t) * gamma(v_w), parameter * r / t)}
result = field_equation.xreplace(rules)
result = result.expand().doit()
#series(result, parameter, x0=0, n=2)