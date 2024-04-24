# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:36:27 2023

@author: User
"""

import numpy as np
from scipy.integrate import solve_ivp , simps
from scipy.optimize import root_scalar

def find_vJ (alN ,cb2 ):
 return np. sqrt (cb2 ) *(1+ np. sqrt (3* alN *(1 - cb2 +3* cb2*alN))) /(1+3* cb2*alN)

def get_vp (vm ,al ,cb2 , branch = -1):
 disc = vm **4 -2* cb2 *vm **2*(1 -6* al)+cb2 **2*(1 -12* vm **2* al *(1 -3* al))
 return 0.5*( cb2+vm **2+ branch *np. sqrt ( disc ))/( vm +3* cb2*vm*al)

def w_from_alpha (al ,alN ,nu ,mu):
  return (abs((1 -3* alN)*mu -nu)+1e-100) /(abs((1 -3* al)*mu-nu)+1e-100)

def eqWall (al ,alN ,vm ,nu ,mu ,psiN , solution = -1):
     vp = get_vp (vm ,al ,1/( nu -1) ,solution )
     ga2m , ga2p = 1/(1 - vm **2) ,1/(1 - vp **2)
     psi = psiN * w_from_alpha (al ,alN ,nu ,mu)**( nu/mu -1)
     return vp*vm*al /(1 -(nu -1) *vp*vm) -(1 -3*al -( ga2p / ga2m )**( nu /2)*psi) /(3* nu)

def solve_alpha (vw ,alN ,cb2 ,cs2 , psiN ):
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vm = min(np. sqrt (cb2),vw)
     vp_max = min (cs2/vw ,vw)
     al_min = max ((vm - vp_max )*( cb2 -vm* vp_max ) /(3* cb2*vm *(1 - vp_max **2) ) ,(mu -nu)/(3* mu))
     al_max = 1/3
     branch = -1
     if eqWall(al_min ,alN ,vm ,nu ,mu , psiN )* eqWall (al_max ,alN ,vm ,nu ,mu , psiN ) >0:
         branch = 1
     sol = root_scalar (eqWall ,( alN ,vm ,nu ,mu ,psiN , branch ),bracket =( al_min , al_max),rtol =1e-10 , xtol =1e-10)
     if not sol.converged:
         print (" WARNING : desired precision not reached in ’solve_alpha ’")
     return sol.root

def dfdv (v,X,cs2 ):
     xi ,w = X
     mu_xiv = (xi -v)/(1 - xi*v)
     dxidv = xi *(1 -v*xi)*( mu_xiv **2/ cs2 -1) /(2* v*(1 -v **2) )
     dwdv = w *(1+1/ cs2 )* mu_xiv /(1 -v **2)
     return [dxidv , dwdv ]

def integrate_plasma (v0 ,vw ,w0 ,c2 , shock_wave = True ):
     def event (v,X,cs2):
         xi ,w = X
         return xi *(xi -v)/(1 - xi*v) - cs2
     event.terminal = True
     sol = None
     if shock_wave :
         sol = solve_ivp (dfdv ,(v0 ,1e-20) ,[vw ,w0], events =event , args =(c2 ,) ,rtol =1e-10 , atol =1e-10)
     else :
         sol = solve_ivp (dfdv ,(v0 ,1e-20) ,[vw ,w0], args =(c2 ,) ,rtol =1e-10 , atol =1e-10)
     if not sol.success :
         print (" WARNING : desired precision not reached in ’ integrate_plasma ’")
     return sol

def shooting (vw ,alN ,cb2 ,cs2 , psiN ):
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vm = min(np. sqrt (cb2),vw)
     al = solve_alpha (vw , alN , cb2 , cs2 , psiN )
     vp = get_vp (vm , al , cb2 )
     wp = w_from_alpha (al , alN , nu , mu)
     sol = integrate_plasma ((vw -vp)/(1 - vw*vp), vw , wp , cs2)
     vp_sw = sol.y[0 , -1]
     vm_sw = (vp_sw -sol.t[ -1]) /(1 - vp_sw * sol.t[ -1])
     wm_sw = sol.y[1 , -1]
     return vp_sw / vm_sw - ((mu -1)* wm_sw +1) /((mu -1)+ wm_sw )


def find_wp (vw ,alN ,cb2 ,cs2 , psiN):
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vm = min(np. sqrt (cb2),vw)
     al = solve_alpha (vw , alN , cb2 , cs2 , psiN )
     vp = get_vp (vm , al , cb2 )
     wp = w_from_alpha (al , alN , nu , mu)
     sol = integrate_plasma ((vw -vp)/(1 - vw*vp),vw ,wp ,cs2)
     v,xi ,w = sol .t,sol .y[0] , sol .y[1]
     return max(w)


def find_vw (alN ,cb2 ,cs2 , psiN ):
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vJ = find_vJ (alN , cb2 )
     if alN < (1- psiN )/3 or alN <= (mu -nu) /(3* mu):
         print ("alN too small")
         return 0
     if alN > max_al (cb2 ,cs2 ,psiN ,100) or shooting (vJ ,alN ,cb2 ,cs2 , psiN ) < 0:
         print ("alN too large")
         return 1
     sol = root_scalar ( shooting ,( alN ,cb2 ,cs2 , psiN ),bracket =[1e-3, vJ], rtol =1e-10 , xtol =1e-10)
     return sol.root

def max_al (cb2 ,cs2 ,psiN , upper_limit =1) :
     nu ,mu = 1+1/ cb2 ,1+1/ cs2
     vm = np. sqrt (cb2 )
     def func (alN ):
         vw = find_vJ (alN , cb2 )
         vp = cs2 /vw
         ga2p , ga2m = 1/(1 - vp **2) ,1/(1 - vm **2)
         wp = (vp+vw -vw*mu)/( vp+vw -vp*mu)
         psi = psiN *wp **( nu/mu -1)
         al = (mu -nu) /(3* mu)+( alN -(mu -nu) /(3* mu))/wp
         return vp*vm*al /(1 -(nu -1) *vp*vm) -(1 -3*al -( ga2p / ga2m )**( nu /2)*psi) /(3*nu)
     if func ( upper_limit ) < 0:
         return upper_limit
     sol = root_scalar (func , bracket =((1 - psiN )/3, upper_limit ),rtol =1e-10 , xtol =1e-10)
     return sol.root

def detonation (alN ,cb2 , psiN ):
     nu = 1+1/ cb2
     vJ = find_vJ (alN , cb2 )
     def matching_eq (vw):
         A = vw **2+ cb2 *(1 -3* alN *(1 - vw **2) )
         vm = (A+np. sqrt (A**2 -4* vw **2* cb2 )) /(2* vw)
         ga2w , ga2m = 1/(1 - vw **2) ,1/(1 - vm **2)
         return vw*vm*alN /(1 -(nu -1) *vw*vm) -(1 -3* alN -( ga2w / ga2m )**( nu /2)* psiN )/(3* nu)
     if matching_eq (vJ +1e-10) * matching_eq (1 -1e-10) > 0:
         print ("No detonation solution ")
         return 0
     sol = root_scalar ( matching_eq , bracket =( vJ +1e-10 ,1 -1e-10) ,rtol =1e-10 , xtol =1e-10)
     return sol.root

def find_kappa (alN ,cb2 ,cs2 ,psiN ,vw= None ):
     if vw is None :
         vw = find_vw (alN ,cb2 ,cs2 , psiN )
         nu ,mu = 1+1/ cb2 ,1+1/ cs2
         kappa ,wp ,vm ,vp = 0 ,1 ,0 ,0
     if vw < 1:
         vm = min (np. sqrt (cb2),vw)
         al = solve_alpha (vw ,alN ,cb2 ,cs2 , psiN )
         vp = get_vp (vm ,al ,cb2 )
         wp = w_from_alpha (al ,alN ,nu ,mu)
         sol = integrate_plasma ((vw -vp)/(1 - vw*vp),vw ,wp ,cs2)
         v,xi ,w = sol .t,sol .y[0] , sol .y[1]
         kappa += 4* simps (( xi*v) **2* w/(1 -v **2) ,xi)/( vw **3* alN)
     if vw **2 > cb2 :
         w0 = psiN *wp **( nu/mu)*((1 - vm **2) /(1 - vp **2) )**( nu /2) if vw < 1 else 1+6* alN /(nu -2)
         v0 = (vw -vm)/(1 - vw*vm) if vw < 1 else 3* alN /(nu -2+3* alN )
         sol = integrate_plasma (v0 ,vw ,w0 ,cb2 , False )
         v,xi ,w = np. flip (sol .t),np. flip (sol.y [0]) ,np. flip (sol.y [1])
         mask = np. append (xi [1:] > xi [: -1] , True )
         kappa += 4* simps ((( xi*v) **2* w/(1 -v **2) )[ mask ],xi[ mask ]) /( vw **3* alN )
     return kappa

