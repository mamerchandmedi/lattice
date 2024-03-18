import numpy as np
from scipy import interpolate
from math import factorial


def J_high(x, l):
	if(l==0):
		return -2 + x/2
	elif(l==1):
		return 0.5
	else:
		return 0

def J_low(x, l):
	if(l==0):
		return -np.sqrt(np.pi/2) * x**(0.75)*np.exp(-np.sqrt(x))
	elif(l==1):
		return np.sqrt(np.pi/2)/4 * np.exp(-np.sqrt(x))*(-3+2*np.sqrt(x))/x**(0.25)
	elif(l==2):
		return np.sqrt(np.pi/2)/16 * np.exp(-np.sqrt(x))*(3+8*np.sqrt(x)-4*x)/x**(1.25)
	elif(l==3):
		return np.sqrt(np.pi/2)/64 * np.exp(-np.sqrt(x))*(-15-30*np.sqrt(x)-12*x + 8*x**(1.5))/x**(2.25)

J_dat_path = "finiteT.txt"
xi, yi = np.loadtxt(J_dat_path).T
chii, etai = np.loadtxt(J_dat_path).T



n=len(xi)-1
p=4 
k=3

def monomial(x, x0, power):
	if(power==0):
		return 1
	else:
		return (x-x0)**power
	

#matrix = np.zeros((len(xi), len(xi)))
matrix = np.zeros(((p+1)*n, (p+1)*n))

vec_J = np.zeros((p+1)*n)

############  J_0 + derivatives: k+1 equations ##############
print("J0:")
for l in range(k+1):
	for j in range(l,p+1):
		print(l, j)
		matrix[l, j] = factorial(j)/factorial(j - l)*monomial(xi[0], xi[0], j-l)
		vec_J[l] = J_high(xi[0], l)	
########### continuity of J_i and derivatives: (k+1)*(n-1) equations ##########
for i in range(1, n):
	print("J",i,":")
	for l in range(k+1):
		for j in range(l,p+1):
			print((k+1)*i+l, (p+1)*(i-1)+j, (p+1)*i+j) 
			matrix[(k+1)*i+l, (p+1)*(i-1)+j] = -factorial(j)/factorial(j - l)*monomial(xi[i], xi[i-1], j-l)
			matrix[(k+1)*i+l, (p+1)*i+j] = factorial(j)/factorial(j - l)*monomial(xi[i], xi[i], j-l)
			

############  J_n + derivatives: k+1 equations ##############
print("Jn:")
for l in range(k+1):
	for j in range(l,p+1):
		print((k+1)*n+l, (p+1)*(n-1)+j)
		matrix[(k+1)*n+l, (p+1)*(n-1)+j] = factorial(j)/factorial(j - l)*monomial(xi[n], xi[n-1], j-l)
	vec_J[(k+1)*n+l] = J_low(xi[n], l)		
		
############  J_i: n - (k+1) equations #############

xmin = 1.e-3
xmax = 1.e3
for m in range(len(chii)):
	i = 0
	j = len(xi) - 1
	if xmin < chii[m] < xmax:
        	while i < j:
            		mid = (i + j) // 2
            		if xi[mid] < chii[m]:
                		i = mid + 1
            		else:
                		j = mid
	i = max(i - 1, 0)
	print("J",m,":")
	for j in range(p+1):
		print(p*(n+1)+m, (p+1)*i+j) 
		matrix[(k+1)*(n+1)+m, (p+1)*i+j] = factorial(j)/factorial(j)*monomial(chii[m], xi[i], j)
	vec_J[(k+1)*(n+1)+m] = etai[m]


