import numpy as np
from cosmoTransitions.tunneling1D import SingleFieldInstanton
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import optimize as optimize
import scipy.integrate as integrate
from scipy import interpolate, special
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pylab as pl

#Definition of the potential#
Tc = 100.
L = 0.1*Tc**4
sigma = 0.1*Tc**3
lc = 6/Tc
T0 = Tc/(1+6*sigma/(L*lc))**0.5
alpha = 1/(2/3*sigma*lc**5*Tc**2)**0.5
gamma = (L+6*sigma/lc)/(6*sigma*lc*Tc**2)
lam = 1/(3*sigma*lc**3)
Tn = 93.
def V(phi): return 0.5*gamma*(Tn**2-T0**2)*phi**2-1/3*alpha*Tn*phi**3+0.25*lam*phi**4
def dV(phi): return gamma*(Tn**2-T0**2)*phi-alpha*Tn*phi**2+lam*phi**3

#critical bubble#
xi = np.array(range(2500000))*0.0001
Vmin = np.amin(V(xi))
vt = xi[np.where(V(xi) == Vmin)[0]][0] ##find location of the global minimum
profile = SingleFieldInstanton(vt, 0.0, V, dV).findProfile(npoints=1000)
fphi = interpolate.interp1d(profile.R, profile.Phi, fill_value=(0), bounds_error=False)

#Thermodynamical quantities#
gst = 106.75
a = gst*(math.pi**2)/90
def VT(field,temp): return 0.5*gamma*(temp**2-T0**2)*field**2-1/3*alpha*temp*field**3+0.25*lam*field**4
def dVT(field,temp): return gamma*(temp**2-T0**2)*field-alpha*temp*field**2+lam*field**3
def eps(field, temp): return 3*a*temp**4+ VT(field,temp)- temp*dVT(field,temp)
def p(field,temp): return a*temp**4-VT(field,temp)
def w(field,temp): return eps(field,temp)+p(field,temp)
def gam(vel): return 1/(1-vel**2)**0.5



def cr1(j):
	return 2*(r[j+1]**3 - r[j]**3)*(r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1]*r[j-1] + 2*r[j]*r[j+1])**(-1)/dr**3
def cr2(j):
	return 2*(r[j]**3 - r[j-1]**3)*(r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1]*r[j-1] + 2*r[j]*r[j+1])**(-1)/dr**3

def cFp1(j):
	return 4*(r[j+1]**2 + r[j+1]*r[j] + r[j]**2) 
def cF0n(j):
	return 4*(r[j-1]**2 + r[j-1]*r[j] + r[j]**2)
def cF0p(j):
	return 4*(r[j+1]**2 + r[j+1]*r[j] + r[j]**2)
def cFm1(j):
	return 4*(r[j]**2 + r[j]*r[j-1] + r[j-1]**2)
def cGU0(j):
	return 2*dr*(r[j+1]+6*r[j]+r[j-1]) - ((3*r[j]**2 + 2*r[j]*r[j+1] + r[j+1]**2) - (r[j-1]**2 + 2*r[j-1]*r[j] + 3*r[j]**2))
def cGUp1(j):
	return 2*dr*(r[j+1]+r[j]) - (r[j]**2 + 2*r[j]*r[j+1] + 3*r[j+1]**2)
def cGUm1(j):
	return 2*dr*(r[j]+r[j-1]) + (3*r[j-1]**2 + 2*r[j-1]*r[j] + r[j]**2)
def den(j):
	return (r[j+1]**2 + r[j]**2 + r[j-1]**2 + r[j+1]*r[j-1] + r[j+1]*r[j] + r[j]*r[j-1])
def func(x, t, z, f):
    return [t - w(f,x[0])*gam(x[1])**2 + x[2],
            z - w(f,x[0])*gam(x[1])**2*x[1],
            x[2] - a*x[0]**4+VT(f,x[0])]
def thin_wall(r,v0,L,r0):
    return v0/2*(-np.tanh((r-r0)/L) + 1.)
   
def poly(r,a,b,c,e,f,g):
    return a + b*r + c*r**2 + e*r**3 +f*r**4 +g*r**5
fit = curve_fit(thin_wall, profile.R, profile.Phi, bounds=(0, [1.5*vt, 1, 1]))[0]

dr = 0.01
dt = 0.002
r  = np.array(range(0, int(np.floor(10/dr))))*dr
plt.plot(r, thin_wall(r, fit[0], fit[1],fit[2]),label="thin-wall approximation")
plt.scatter(profile.R,profile.Phi,c = "red", s = 1, label="Numerical profile")
plt.xlim(0,0.2)
plt.legend()
plt.savefig('phi0')
plt.close()
phi_tw = []
for i in range(len(r)):
    #if r[i]/1.01 > profile.R[-1]:
    phi_tw.append(thin_wall(r[i]/1.01,fit[0], fit[1], fit[2]))
    #else:
    #phi_tw.append(fphi(r[i]/1.01))
phi_tw = np.array(phi_tw)
#EVOLUTION#
phi0 = phi_tw
phi = phi0
phi2 = np.zeros(r.size)
N = len(phi)-1
T = Tn*np.ones(r.size)
v = 0*np.ones(r.size)
P = p(phi0,T)
tau = w(phi0,T)*gam(v)**2 - P
#tau = tau[0]*np.exp(-2*(r-5)**2)

Z = w(phi0,T)*gam(v)**2*v
tau2 = np.zeros(r.size)
Z2= np.zeros(r.size)
P_off = 0*p(phi0[N],Tn)*np.ones(r.size)
	
	
fig, axs = plt.subplots(4, 2)
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)
#------plotting
plt.xlim(0,10)
#plt.ylim(0,0.1)
fig.suptitle("dr = "+str(dr) + " ,dt = "+str(dt))
axs[0,0].set_ylabel('phi')
axs[1,0].set_ylabel('tau')
axs[2,0].set_ylabel('Z')
axs[2,0].set_xlabel("r")

axs[0,1].set_ylabel('v')
axs[1,1].set_ylabel('T')
axs[2,1].set_ylabel('P')
axs[2,1].set_xlabel("r")

axs[0,0].set_xlim(0,1)
axs[1,0].set_xlim(0,1)
axs[2,0].set_xlim(0,1)
axs[0,1].set_xlim(0,1)
axs[1,1].set_xlim(0,1)
axs[2,1].set_xlim(0,1)


#axs[0,1].set_ylim(-1,1)
#axs[0,0].plot(r,phi,linestyle = 'dashed')
axs[0,0].axhline(y = vt, color = 'gray', linestyle = 'dashed')
#axs[1,0].plot(r,tau,linestyle = 'dashed')
#axs[2,0].plot(r,Z,linestyle = 'dashed')
#axs[0,1].plot(r,v,linestyle = 'dashed')
#axs[1,1].plot(r,T,linestyle = 'dashed')
#axs[2,1].plot(r,P,linestyle = 'dashed')
plt.savefig('plots/low_order/plots_low_order'+str(0))
nsteps = 100
psteps = 5
#def Tsolve(t,j):
#	return tau[j] + p(phi[j],t)-0.5*(dr**4*w(phi[j],t)+(w(phi[j],t)**2+4*Z[j]**2)**0.5)

def Tsolve(x,t,f,z):
	return t + p(f,x[0])-0.5*(w(f,x[0])+(w(f,x[0])**2+4*z**2)**0.5)

for i in range(nsteps):#------------------------------------iterate over time
    print(i)
    for j in range(N+1):
    	#root = fsolve(lambda x: func(x, tau[j], Z[j], phi[j]), [T[j], v[j], P[j]])
    	#T[j] = root[0]
    	#v[j] = root[1]
    	#P[j] = root[2]
    	solT = fsolve(lambda x: Tsolve(x, tau[j], phi[j], Z[j]), T[j], xtol=10**(-8))
    	T[j] = solT[0]
    	P[j] = p(phi[j], T[j])
    	v[j] = Z[j]/(tau[j]+P[j])

    #------tau, Z evolution (half step)
    Ftau = (tau + P)*v
    FZ = Z*v #+ (P - P_off)

    #------phi evolution (half step)
    phi[1:-1] = 0.5*dt*phi2[1:-1] + phi[1:-1]
    phi[0] = phi[1]
    
    if v[0]>=0:
    	tau2[0] = (4*(r[0]**2 + r[0]*r[1] + r[1]**2)*Ftau[0])/(3*r[0]**2+2*r[0]*r[1]+r[1]**2)/dr
    	Z2[0] = (4*(r[0]**2 + r[0]*r[1] + r[1]**2)*FZ[0] - 2*dr*(3*r[0]+r[1])*P[0] - 2*dr*(r[0]+r[1])*P[1]+ (3*r[0]**2 + 2*r[0]*r[1] + r[1]**2)*P[0] + (r[0]**2 + 2*r[0]*r[1] + 3*r[1]**2)*P[1])/(3*r[0]**2+2*r[0]*r[1]+r[1]**2)/dr 
    else:
    	tau[0] = (4*(r[0]**2 + r[0]*r[1] + r[1]**2)*Ftau[1])/(3*r[0]**2+2*r[0]*r[1]+r[1]**2)/dr
    	Z2[0] = (4*(r[0]**2 + r[0]*r[1] + r[1]**2)*FZ[1] - 2*dr*(3*r[0]+r[1])*P[0] - 2*dr*(r[0]+r[1])*P[1]+ (3*r[0]**2 + 2*r[0]*r[1] + r[1]**2)*P[0] + (r[0]**2 + 2*r[0]*r[1] + 3*r[1]**2)*P[1])/(3*r[0]**2+2*r[0]*r[1]+r[1]**2)/dr 
    
    if v[N]>=0:
    	tau2[N] = (-4*(r[N-1]**2 + r[N-1]*r[N] + r[N]**2)*Ftau[N-1] + 12*r[N]**2*Ftau[N])/(r[N-1]**2+2*r[N-1]*r[N]+3*r[N]**2)/dr
    	Z2[N] = (-4*(r[N-1]**2 + r[N-1]*r[N] + r[N]**2)*FZ[N-1] + 12*r[N]**2*FZ[N] - 2*dr*(r[N-1] + r[N])*P[N-1] - 2*dr*(r[N-1] + 3*r[N])*P[N] - (3*r[N-1]**2 + 2*r[N-1]*r[N] + r[N]**2)*P[N-1] - (r[N-1]**2 + 2*r[N-1]*r[N] + 3*r[N]**2)*P[N] + 12*r[N]**2*P[N])/(r[N-1]**2+2*r[N-1]*r[N]+3*r[N]**2)/dr 
    else:
    	tau2[N] = (-4*(r[N-1]**2 + r[N-1]*r[N] + r[N]**2)*Ftau[N] + 12*r[N]**2*Ftau[N])/(r[N-1]**2+2*r[N-1]*r[N]+3*r[N]**2)/dr
    	Z2[N] = (-4*(r[N-1]**2 + r[N-1]*r[N] + r[N]**2)*FZ[N] + 12*r[N]**2*FZ[N] - 2*dr*(r[N-1] + r[N])*P[N-1] - 2*dr*(r[N-1] + 3*r[N])*P[N] - (3*r[N-1]**2 + 2*r[N-1]*r[N] + r[N]**2)*P[N-1] - (r[N-1]**2 + 2*r[N-1]*r[N] + 3*r[N]**2)*P[N] + 12*r[N]**2*P[N])/(r[N-1]**2+2*r[N-1]*r[N]+3*r[N]**2)/dr 
    
  
    for j in range(1, N): 
        if v[j]>=0:
           tau2[j] = (-cFm1(j)*Ftau[j-1] + cF0p(j)*Ftau[j])/(2*dr)/den(j)
           Z2[j] = (-cFm1(j)*FZ[j-1] + cF0p(j)*FZ[j] - cGU0(j)*(P[j]-P_off[j]) - cGUp1(j)*(P[j+1]-P_off[j+1]) - cGUm1(j)*(P[j-1]-P_off[j-1]))/(2*dr)/den(j)
        else:
           tau2[j] = (cFp1(j)*Ftau[j+1] - cF0n(j)*Ftau[j])/(2*dr)/den(j)
           Z2[j] = (cFp1(j)*FZ[j+1] - cF0n(j)*FZ[j] - cGU0(j)*(P[j]-P_off[j]) - cGUp1(j)*(P[j+1]-P_off[j+1]) - cGUm1(j)*(P[j-1]-P_off[j-1]))/(2*dr)/den(j)
    tau = tau - 0.5*dt*tau2
    Z = Z - 0.5*dt*Z2
    for j in range(N+1):
    	#root = fsolve(lambda x: func(x, tau[j], Z[j], phi[j]), [T[j], v[j], P[j]])
    	#T[j] = root[0]
    	#v[j] = root[1]
    	#P[j] = root[2]
    	#v[j] = Z[j]/(Z[j]**2 + w(phi0[j],T[j])**2)**0.5
    	#v[j] = (P[j]**2+2*P[j]*tau[j]+tau[j]**2-w(phi0[j],T[j]))**0.5/(P[j]+tau[j])
    	solT = fsolve(lambda x: Tsolve(x, tau[j], phi[j], Z[j]), T[j], xtol=10**(-8))
    	T[j] = solT[0]
    	P[j] = p(phi[j], T[j])
    	v[j] = Z[j]/(tau[j]+P[j])
    phi2[1] = phi2[1] + dt*((r[2]**3-r[1]**3)*(phi[2]-phi[1])/dr**2*(r[1]**3 - r[0]**3 + (r[2]-r[1])/4*(3*r[1]**2+2*r[1]*r[2]+r[2]**2))**(-1) - dVT(phi[1],T[1]))
    for j in range(2,N):
    	phi2[j] = phi2[j] + dt*(cr1(j)*(phi[j+1]-phi[j]) - cr2(j)*(phi[j]-phi[j-1]) - dVT(phi[j],T[j]))


    #print(dVT(phi[1],T[1]))
    print((r[2]**3-r[1]**3)*(phi[2]-phi[1])/dr**2*(r[1]**3 - r[0]**3 + (r[2]-r[1])/4*(3*r[1]**2+2*r[1]*r[2]+r[2]**2))**(-1),dVT(phi[1],T[1]))
    #------phi evolution (half step)
    phi[1:-1] = 0.5*dt*phi2[1:-1] + phi[1:-1]
    phi[0] = phi[1]
    
    #------tau, Z evolution (half step)
    tau = tau - 0.5*dt*tau2
    Z = Z - 0.5*dt*Z2
    #------plotting
    colormap = plt.get_cmap('Blues')
    colors = colormap((i+1)/nsteps)
    if((i+1)%psteps==0):
    	
    	axs[0,0].plot(r,phi,label= str(i/Tc)+"*Tc",c=colors)
    	axs[1,0].plot(r,tau,label= str(i/Tc)+"*Tc",c=colors)
    	axs[2,0].plot(r,Z,label= str(i/Tc)+"*Tc",c=colors)
    	axs[0,1].plot(r,v,label= str(i/Tc)+"*Tc",c=colors)
    	axs[1,1].plot(r,T,label= str(i/Tc)+"*Tc",c=colors)
    	axs[2,1].plot(r,P,label= str(i/Tc)+"*Tc",c=colors)
    	#axs[3,0].plot(r[2:],r[2:]**(-2),c=colors,lw=0.2)
    	plt.savefig('plots/low_order/plots_low_order'+str(i+1))
        


#plt.savefig('plots_low_order')
