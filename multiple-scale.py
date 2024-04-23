#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:40:46 2024

@author: Tomasz Krajewski
"""

from sympy import Symbol, symbols, Function, diff, sqrt, simplify, \
    Derivative, Expr, Add, init_printing, series, Tuple

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
phidot = Function(r'\dot{\phi}')
psi = Function(r'\psi')
tau = Function(r'\tau')
Z = Function(r'Z')
T = Function(r'T')
p = Function(r'p')
v = Function(r'v')
dVdphi = Function(r'\frac{\partial V}{\partial \phi}')

def gamma(v):
    return 1 / sqrt(1 - v**2)

field_equations = Tuple(phidot(t, r) - phi(t,r).diff(t),
    psi(t,r) - phi(t, r).diff(r),
    -phidot(t, r).diff(t) + \
    (r**2 * psi(t, r)).diff(r) / r**2 -\
    dVdphi(phi(t,r), T(t, r))- \
    eta * gamma(v(t, r)) * (phi(t, r).diff(t) + v(t, r) * phi(t, r).diff(r)))

conservation_equations = Tuple(tau(t, r).diff(t) +
                          (r**2 * (tau(t, r) + p(t, r)) * v(t, r)).diff(r) / r**2 -
                          dVdphi(phi(t,r), T(t, r)) * phidot(t, r) -
                          eta * gamma(v(t, r)) * (phidot(t, r) + v(t, r) * psi(t, r)) * phidot(t, r),
                          Z(t, r).diff(t) +
                          (r**2 * Z(t, r)).diff(r) / r**2 +
                          p(t, r).diff(r) +
                          dVdphi(phi(t,r), T(t, r)) * psi(t, r) +
                          eta * gamma(v(t, r)) * (phidot(t, r) + v(t, r) * psi(t, r)) * psi(t, r))

v_w = Symbol(r'v_w')
parameter = Symbol(r'\epsilon')
phi0 = Function(r'\phi_{(0)}')
phi1 = Function(r'\phi_{(1)}')
phidot0 = Function(r'\dot{\phi}_{(0)}')
phidot1 = Function(r'\dot{\phi}_{(1)}')
psi0 = Function(r'\psi_{(0)}')
psi1 = Function(r'\psi_{(1)}')
tau0 = Symbol(r'\tau_0')
deltatau0 = Function(r'\delta \tau_{(0)}')
deltatau1 = Function(r'\delta \tau_{(1)}')
Z0 = Symbol(r'Z_0')
deltaZ0 = Function(r'\delta Z_{(0)}')
deltaZ1 = Function(r'\delta Z_{(1)}')
T0 = Symbol(r'T_0')
deltaT0 = Function(r'\delta T_{(0)}')
deltaT1 = Function(r'\delta T_{(1)}')
p0 = Symbol(r'p_0')
deltap0 = Function(r'\delta p_{(0)}')
deltap1 = Function(r'\delta p_{(1)}')
v0 = Symbol(r'v_0')
deltav0 = Function(r'\delta v_{(0)}')
deltav1 = Function(r'\delta v_{(1)}')
temp_chi = Function(r'\chi')
temp_xi = Function(r'xi')
expansion_rules = {phi(t, r): phi0(temp_chi(t, r), parameter * temp_xi(t, r)) +
                   parameter * phi1(temp_chi(t, r), parameter * temp_xi(t, r)),
                   phidot(t, r): phidot0(temp_chi(t, r), parameter * temp_xi(t, r)) +
                   parameter * phidot1(temp_chi(t, r), parameter * temp_xi(t, r)),
                   psi(t, r): psi0(temp_chi(t, r), parameter * temp_xi(t, r)) +
                   parameter * psi1(temp_chi(t, r), parameter * temp_xi(t, r)),
                   tau(t, r): tau0 +
                   deltatau0(temp_chi(t, r), parameter * temp_xi(t, r)) +
                   parameter * deltatau1(temp_chi(t, r), parameter * temp_xi(t, r)),
                   Z(t, r): Z0 +
                   deltaZ0(temp_chi(t, r), parameter * temp_xi(t, r)) +
                   parameter * deltaZ1(temp_chi(t, r), parameter * temp_xi(t, r)),
                   T(t, r): T0 +
                   deltaT0(temp_chi(t, r), parameter * temp_xi(t, r)) +
                   parameter * deltaT1(temp_chi(t, r), parameter * temp_xi(t, r)),
                   p(t, r): p0 +
                   deltap0(temp_chi(t, r), parameter * temp_xi(t, r)) +
                   parameter * deltap1(temp_chi(t, r), parameter * temp_xi(t, r)),
                   v(t, r): v0 +
                   deltav0(temp_chi(t, r), parameter * temp_xi(t, r)) +
                   parameter * deltav1(temp_chi(t, r), parameter * temp_xi(t, r))}
derivatives_rules = {temp_chi(t, r).diff(t): ((r - v_w * t) * gamma(v_w)).diff(t),
                     temp_chi(t, r).diff(r): ((r - v_w * t) * gamma(v_w)).diff(r),
                     temp_xi(t, r).diff(t): (r / t).diff(t),
                     temp_xi(t, r).diff(r): (r / t).diff(r)}
xi = Symbol(r'\xi')
chi = Symbol(r'\chi')
result = conservation_equations.xreplace(expansion_rules)
result = Tuple(*[expr.expand().doit() for expr in result])
der = result.xreplace(derivatives_rules)
simplify_rules = {temp_chi(t,r): chi, temp_xi(t,r): xi}
simple = der.xreplace(simplify_rules)
coordinates_rules = {t: chi / ((xi - v_w) * gamma(v_w)),
                    r: xi * chi / ((xi - v_w) * gamma(v_w))}
transformed = simple.xreplace(coordinates_rules)
ser = Tuple(*[series(expr, parameter, x0=0, n=2) for expr in transformed])
coefficients = Tuple(*[Tuple(*[expr.coeff(parameter, i) for i in range(2)]) for expr in ser]) # 0-th order]
