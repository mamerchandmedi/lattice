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
tau0 = Function(r'\tau_{(0)}')
tau1 = Function(r'\tau_{(1)}')
Z0 = Function(r'Z_{(0)}')
Z1 = Function(r'Z_{(1)}')
T0 = Function(r'T_{(0)}')
T1 = Function(r'T_{(1)}')
p0 = Function(r'p_{(0)}')
p1 = Function(r'p_{(1)}')
v0 = Function(r'v_{(0)}')
v1 = Function(r'v_{(1)}')
rules = {phi(t, r): phi0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * phi1((r - v_w * t) * gamma(v_w), parameter * r / t),
         phidot(t, r): phidot0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * phidot1((r - v_w * t) * gamma(v_w), parameter * r / t),
         psi(t, r): psi0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * psi1((r - v_w * t) * gamma(v_w), parameter * r / t),
         tau(t, r): tau0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * tau1((r - v_w * t) * gamma(v_w), parameter * r / t),
         Z(t, r): Z0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * Z1((r - v_w * t) * gamma(v_w), parameter * r / t),
         T(t, r): T0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * T1((r - v_w * t) * gamma(v_w), parameter * r / t),
         p(t, r): p0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * p1((r - v_w * t) * gamma(v_w), parameter * r / t),
         v(t, r): v0((r - v_w * t) * gamma(v_w), parameter * r / t) + parameter * v1((r - v_w * t) * gamma(v_w), parameter * r / t)}
result = field_equations.xreplace(rules)
result = Tuple(*[expr.expand().doit() for expr in result])
ser = Tuple(*[series(expr, parameter, x0=0, n=2) for expr in result])
xi = Symbol(r'\xi')
chi = Symbol(r'\chi')
sub_rules = {r/sqrt(1 - v_w**2) - t*v_w/sqrt(1 - v_w**2): chi}
sub = ser.xreplace(sub_rules).simplify()
sub.coeff(parameter, 0) # 0-th order
