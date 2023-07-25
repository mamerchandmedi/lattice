from cosmoTransitions import generic_potential_1
from cosmoTransitions import transitionFinder
from cosmoTransitions.tunneling1D import SingleFieldInstanton
import matplotlib.pyplot as plt
import numpy as np
import random
import functools
import sys, os

class model1(generic_potential_1.generic_potential):
    def init(self, m0 = 100*np.sqrt(2), lam = 1, q = 1):
        self.Ndim = 1
        self.m0 = m0
        self.lam = lam 
        self.mu2 = 0.5*m0**2
        self.q = q
        self.v = np.sqrt(m0**2/(2*lam))
       
    def V0(self, X):
        X = np.asanyarray(X)
        phi = X[...,0]
        pot =  - 0.5 * self.mu2 * phi**2 + 0.25*self.lam * phi**4
        return pot
        
    def boson_massSq(self, X, T):
        
        X = np.array(X)
        phi = X[...,0]

        #####Scalar thermal masses##
        Pi_phi = T**2*(self.lam/3 + self.q**2/4)
     
        ##Scalar mass matrix##
        m = -self.mu2 + 3*self.lam*phi**2 + Pi_phi
        
        ####Gauge boson masses
        mAl = self.q**2*phi**2 + 1/3*self.q**2*T**2
        mAt = self.q**2*phi**2

        M = np.array([m, mAl, mAt])
        Mphys = np.array([self.m0**2, self.q**2*self.v**2, self.q**2*self.v**2])

        # At this point, we have an array of boson masses, but each entry might
        # be an array itself. This happens if the input X is an array of points.
        # The generic_potential class requires that the output of this function
        # have the different masses lie along the last axis, just like the
        # different fields lie along the last axis of X, so we need to reorder
        # the axes. The next line does this, and should probably be included in
        # all subclasses.
        M = np.rollaxis(M, 0, len(M.shape))
        Mphys = np.rollaxis(Mphys, 0, len(Mphys.shape))

        # The number of degrees of freedom for the masses. This should be a
        # one-dimensional array with the same number of entries as there are
        # masses.

        dof = np.array([1,1,2])
        #dof = np.array([1,0,0])


        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        #c = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        c = np.array([1.5,1.5,1.5])
        
        #return M, dof, c
        return M, dof, c, Mphys
      
    def fermion_massSq(self, X):
        X = np.array(X)
        phi = X[...,0]
        mt = self.q**2*phi**2
        M = np.array([mt])
        Mphys = np.array([self.q**2*self.v**2])

        # At this point, we have an array of boson masses, but each entry might
        # be an array itself. This happens if the input X is an array of points.
        # The generic_potential class requires that the output of this function
        # have the different masses lie along the last axis, just like the
        # different fields lie along the last axis of X, so we need to reorder
        # the axes. The next line does this, and should probably be included in
        # all subclasses.
        M = np.rollaxis(M, 0, len(M.shape))
        Mphys = np.rollaxis(Mphys, 0, len(Mphys.shape))
        
        dof = np.array([0])
        #return M, dof
        return M, dof, Mphys
        
    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        return np.array([self.v])
        
    #def V1(self, bosons, fermions):
    	#return 0
    	
    def Vtot(self, X, T, include_radiation=True):
        """
        The total finite temperature effective potential.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        return y
        
    def Vtot_pure(self, X, T, include_radiation=True):
        """
        The total finite temperature effective potential.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        #y += self.V1(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        return y
        
    def DVtot(self, X, T):
        """
        The finite temperature effective potential, but offset
        such that V(0, T) = 0.
        """
        X0 = np.zeros(self.Ndim)
        return self.Vtot(X,T,False) - self.Vtot(X0,T,False)
"""      
v_T = []
v_ST = []
for T in np.arange(0, 150):
	m=model1(m0 = 100, lam = 1, q = 3)
	vev = m.findMinimum(X=[m.v], T=T)[0]
	def V(phi):
		return m.Vtot([phi],T) - m.Vtot([0],T)
	def V_pure(phi):
		return m.Vtot_pure([phi],T) - m.Vtot_pure([0],T)
	def V_CW(phi):
		return m.Vtot([phi],T) - m.Vtot_pure([phi],T)
	def dV(phi):
		eps = 0.0001
		return (V(phi+eps) - V(phi))/eps
	print(T, vev)
	V = np.vectorize(V)
	try:
		#profile = SingleFieldInstanton(vev, 0, V).findProfile(npoints=1000)
		#action = SingleFieldInstanton(vev, 0, V).findAction(profile)
		#print(T, profile.Phi[0], profile.Phi[-1], action/T)
		#v_T.append(T)
		#v_ST.append(action/T)
		Vvec = np.vectorize(V)
		Vvec2 = np.vectorize(V_pure)
		VvecCW = np.vectorize(V_CW)
		phi = np.arange(100)
		plt.close()
		plt.figure()
		plt.plot(phi, Vvec(phi)) 
		plt.plot(phi, Vvec2(phi), '--') 
		plt.grid()
		plt.savefig("testplots/potential_T"+str(T)+".pdf")
		plt.close()
		#plt.figure()
		#plt.plot(profile.R, profile.Phi) 
		#plt.grid()
		#plt.savefig("testplots/profile_T"+str(T)+".pdf")
		#plt.close()
		
	except:
		continue
plt.figure()
plt.plot(v_T, v_ST) 
plt.grid()
plt.savefig("testplots/action.pdf")
"""
m=model1(m0 = 100, lam = 1, q = 3)
print(m.Vtot([70], 50))
calc_Tc = m.calcTcTrans()
alltrans = m.findAllTransitions()
print(alltrans[0]["trantype"], alltrans[0]["Tnuc"], calc_Tc[0]["Tcrit"])

quit()

#profile = SingleFieldInstanton(m.v, 0, V).findProfile()
#plt.figure()
#plt.plot(profile.R, profile.Phi) 
#plt.grid()
#plt.savefig("testplots/critical_bubble.pdf")    
#print(profile) 

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


#alltrans=m.findAllTransitions()
#print(m.TnTrans)
v_deltaV = []
v_Tc = []
v_vlow = []
v_vhigh = []
v_m = []
v_lam = []
v_q = []
v_Tn = []
v_TnTc = []
for i in range(5000):
	m0i = random.uniform(0,500)
	#m0i = 100
	gi = random.uniform(0,5)
	try:
		m=model1(m0 = m0i, lam = 1, q = gi)
		print(i)
		blockPrint()
		calc_Tc = m.calcTcTrans()
		enablePrint()
		if (len(calc_Tc) != 0):
			Tc = calc_Tc[0]["Tcrit"]
			blockPrint()
			alltrans = m.findAllTransitions()
			enablePrint()
			if(len(alltrans)==1):
				Tn = alltrans[0]["Tnuc"]
				vlow = alltrans[0]["low_vev"][0]
				vhigh = alltrans[0]["high_vev"][0]
				if (vlow > 10):
					deltaV = abs(m.Vtot([vlow], Tn) - m.Vtot([vhigh], Tn))
					v_Tc.append(Tc)
					v_Tn.append(Tn)
					v_m.append(m.m0)
					v_q.append(m.q)
					v_lam.append(m.lam)
					v_TnTc.append(Tn/Tc)
					v_deltaV.append(deltaV)
					v_vlow.append(vlow)
					v_vhigh.append(vhigh)
					print(i, Tc, Tn)
			
	except:
		continue
		
		
import csv
with open('results/points.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["#", "m", "lam", "q", "Tc", "Tn","v low", "v high", "Delta V"])
	for j in range(len(v_m)):
    		writer.writerow([j, v_m[j], v_lam[j], v_q[j], v_Tc[j], v_Tn[j], v_vlow[j], v_vhigh[j], v_deltaV[j]])       
	file.close()
plt.figure()
plt.scatter(v_m, v_q, c=v_TnTc) 
plt.xlabel("$m$")
plt.ylabel("$q$")
plt.colorbar(label = "$T_n/T_c$")
plt.grid()
plt.savefig("testplots/scan.pdf")
