import numpy as np
from scipy.integrate import ode
import base_with_bb_fixed

##################################################################
#FIXED PARAMETERS
##################################################################


#constants

m_0 = 5.68e21  ## meV / c^2, where c is in m / fs
a = 10**(-10) #lattice spacing
e = 1.6021766208 *10**(-19) #electric charge
c = 3.*10**(-7) #speed of light
meV = 10**(-3) * e
hbar = 658 # meV fs
#fs=10**(-15)

#superconductor parameters
m = 1.9*m_0 #SC mass
E_F = 9470 # fermi energy
E_D = 8.3# debye energy
Delta_0 = 1.35 # initial gap (real)
T = 0 #temperature

#Pump-Pulse parameters
A_0 = 7.8 *10**(-8) # pump-pulse amplitude
E_omega0  = 3. #pump-pulse energy
tau = 1400 # pumpe-pulse decay constant 
t_c = 631 #critical time
omega_0 = E_omega0 / (hbar ) #pump-pulse frequency
q_0 = omega_0 / c
Q = 1     # momentum kick in units of q_0

#Discretization
dt = 1 #time-step
N_c = 4 #order in perturbation theory
k_y = 1

##################################################################
#1D Discretiztion
##################################################################

#For a fixed value of k_y, the max and min values of k_x are deetermined by
#the Fermi and Debye energies:

kmin = (2 * m * a**2 * (E_F - E_D)/(hbar**2) - (k_y)**2)**(.5)
kmax = (2 * m * a**2 * (E_F + E_D)/(hbar**2)  - (k_y)**2)**(.5)        

dk = omega_0 /c *a  #momentum discretization
N = int((kmax - kmin) / dk /100)  #size of momentum array

##################################################################
#       Time-Indepdent Functions
##################################################################

#We can calculate the functions epsilon_k,E_k,v_k,u_k since they are time-indepdent:

def epsilon_f(k):      #the single-particle energy
    return (hbar**2 /(2*m)) * (k**2 + k_y**2)**(0.5) / a - E_F

def E_f(k,epsilon_k):          #The hamiltonian energy
    return (epsilon_k**2 + (abs(Delta_0))**2)**(0.5)

#bogoliubov parameters:
def u_f(k,epsilon_k,E_k):
    return ( 0.5 * (1 + epsilon_k/E_k))**(0.5)

def v_f(k,epsilon_k,E_k):
    return ( 0.5 * (1 - epsilon_k / E_k))**(0.5)


#initial lists to store values of epsilon, E_k, u_k and v_k. The subscript
#l stands for list.
k = []
epsilon = []
E= []
u = []
v = []

# we create a list of N values of k.
#For each value, we calculate the four functions:
for i in range(0, N):
    k_i = kmin  + i *dk
    epsilon_ki =epsilon_f(k_i)
    E_ki = E_f(k_i, epsilon_ki)
    u_ki = u_f(k_i,epsilon_ki,E_ki)
    v_ki = v_f(k_i,epsilon_ki,E_ki)
    k.append(k_i)
    epsilon.append(epsilon_ki)
    E.append(E_ki)
    u.append(u_ki)
    v.append(v_ki)

#It will be easier to work with the (i,j) = (k,k') indices of the NxN
#array instead of the (m,n) indices of the NxP array, so define an
#index function:

def ind_f(i,j):
    #insert function m,n = f(i,j)
    m= int(i) 
    n= int(j-i + 4)
    return m,n 

def ind_j(i,j):
    #insert function m,n = f(i,j)
    if i in range(0,N) and j in range(0,N) and (j-i+4) in range(0,P):
        m= i 
        n= j-i + 4
    else:
        m= N
        n= 0
    return m,n 


a = { }
a[(0,-q_0)] = (0,1)


def ind_q_f(i,q):
    #this function will deal with the index `k+q', for q \propto q_0:
    #it will also 
    if q == q_0:
        return i+1
    elif q ==-q_0:
        return i-1
    elif q ==0:
        return i
    elif q== 2*q_0:
        return i+2
    elif q ==- 2*q_0:
        return i-2
    else:
        print 'fix index_q argument'
        return 1

ind = {}
for i in range(0,N):
    for j in range(0,N):
        ind[(i,j)] = ind_f(i,j)

ind_q ={}
for i in range(0,N):
    ind_q[(i,q_0)] = i+1
    ind_q[(i,2*q_0)] = i+2
    ind_q[(i,0)] = i
    ind_q[(i,-q_0)] = i-1
    ind_q[(i,-2*q_0)] = i-2


##################################################################
#       L and M functions 
##################################################################


### There are additional `abreviation' functions:
#They are: Lp, Ln, Mp, Mn.
#Need only Lp[k,\pm q_0] and Mp[k, \pm q_0] so we store these as Nx2 arrays.
#Need only Ln[k,x] and Mp[k, x] for x= 0, \pm2q_0, so we store these as Nx3 arrays.
#We set these functions to zero for `edge cases', such as (kmin, kmin-q_0).

def Lp_f(i,a):
    #a = 0,1. Lp(i,0) = Lp(k,q_0); Lp(i,1) = Lp(k,-q_0)
    #Set Lp to zero for the edge cases.
    if i==0 or i== N-1 :
        return 0
    else:
        j = i - 1 + 2*a
        return u[i]*np.conj(u[j]) + np.conj(v[i])*v[j] 

def Mn_f(i,a):
    #a = 0,1. Mn(i,0) = Mn(k,q_0); Mn(i,1) = Mn(k,-q_0)
    #Set Mn to zero for edge cases:
    if i==0 or i== N-1:
        return 0
    else:
        j = i - 1 + 2*a 
        return np.conj(v[i])*u[j] - u[i]*np.conj(v[j])

def Ln_f(i,b):
    #b = 0,1,2 Lp(i,0) = Ln(k,-2q_0); Ln(i,1) = Ln(k,0), Ln(i,2) = Ln(k,2q_0)
    #Set Ln to zero for the edge cases.
    if i==0 or i== N-1 :
        return 0
    else:
        j = i - 2 + 2*a
        return u[i]*np.conj(u[j]) - np.conj(v[i])*v[j] 

def Mp_f(i,b):
    #b = 0,1,2 Mp(i,0) = Mp(k,-2q_0); Mp(i,1) = Mp(k,0), Mp(i,2) = Mp(k,2q_0)
    #Set Mp to zero for the edge cases.
    if i==0 or i== N-1:
        return 0
    else:
        j = i - 2 + 2*a 
        return np.conj(v[i])*u[j] + u[i]*np.conj(v[j])


#Now we calculate and store the values of the abreviation functions:
#we also calculate the interaction strength W:

Lp = np.zeros((int(N),2))
Ln = np.zeros((int(N),3))
Mp = np.zeros((int(N),3))
Mn = np.zeros((int(N),2))
part_W = 0 #this stores the sum in the equation for W.


for i in range(0,N):
    part_W += 1/abs(E[i]) #this stores the interaction strength summand.
    for a in range(0,2):
        Lp[i,a] = Lp_f(i,a)
        Mn[i,a] = Mn_f(i,a)
    for b in range(0,3):
        Ln[i,b] = Ln_f(i,b)
        Mp[i,b] = Mp_f(i,b)
    
W = -2 *(part_W)**(-1) * N

def kron(i,j,x,y):
    if i == x and j==y:
        return 1
    else:
        return 0

def kron1(k,q):
	if k == q:
		return 1.0
	else:
		return 0.0	


##################################################################
#     R and C functions
##################################################################

#The R functions are real, while the C functions are complex.
#They depend on a complex Delta:

def R_f(Delta_r):
    Rx = np.zeros(N)
    for i in range(0,N):
        Rx[i] =  epsilon[i]*(1.0-2.0*v[i]**2.0)+2.0*Delta_r*u[i]*v[i]
		
    return Rx
                   
def C_r_f(Delta_r):
    C_rx = np.zeros(N)
    
    for i in range(0, N):
        C_rx[i] = -2.0*epsilon[i]*u[i]*v[i] + Delta_r*(u[i]**2.0-v[i]**2.0)
    
    return C_rx

def C_i_f(Delta_i):
    C_ix = np.zeros(N)
    
    for i in range(0, N):
        C_ix[i] = Delta_i*(u[i]**2.0+v[i]**2.0)
		
    return C_ix



##################################################################
#     Initial Conditions
##################################################################

#We have 3 complex N by N matrices to calculate. We will store these as
#N by P arrays, with indices (m,n).
#Here P = 2N_c+1:
P = 2*N_c+1

aDa_r = np.zeros((N,P))
aDa_i = np.zeros((N,P))
bDb_r = np.zeros((N,P))
bDb_i = np.zeros((N,P))
ab_r = np.zeros((N,P))
ab_i = np.zeros((N,P))


#### We also produce initial values of R, C_r, C_i:

R = R_f(Delta_0)
C_r = C_r_f(Delta_0)
C_i = C_i_f(Delta_0)



#In order to pass these arrays to various functions, we will define
#an array G of dimension 6 x N x P of the unknown exp. values:

G = np.concatenate((aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i))

#We define an another array of size 3xN to store the R,C_r,C_i values:

H = np.concatenate((R,C_r,C_i))

##################################################################
#     Gap Function
##################################################################

#We define real and imaginary gap functions:

def Delta_r_f(G):
    #unpack the function G: do  we need to pass W? I think so.
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)

    outDelta_r  = 0.0

    for i in range(0,N):
            m,n = ind[(i,i)]
            outDelta_r += u[i]*v[i]*(aDa_r[m,n] + bDb_r[m,n] - 1.0) \
                            - u[i]**2.0*ab_r[m,n] + v[i]**2.0*ab_r[m,n]

    outDelta_r = W*outDelta_r
	
    return outDelta_r / N

def Delta_i_f(G):
    #unpack the function G:
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)

    outDelta_i  = 0.0

    for i in range(0,N):
            m,n = ind[(i,i)]
            outDelta_i += u[i]*v[i]*(aDa_i[m,n] + bDb_i[m,n]) \
                            - u[i]**2.0*ab_i[m,n] - v[i]**2.0*ab_i[m,n]

    outDelta_i = W*outDelta_i 
	
    return outDelta_i / N

##################################################################
#    Vector Potential:
##################################################################
# We define the time-dependent vector potential according to (61):

def A_r(t,q):
    x= A_0 * (np.e)**(-4*np.log(2) * (t/tau)**2) #the prefactor in 61
    y= 0 #the other factor of 61.
    if q == q_0:
        y = np.e**(-1j*omega_0*t)
    elif q== -q_0:
        y = np.e**(+1j*omega_0*t)
    else:
        y=0
    return np.real(x*y)

def A_i(t,q):
    x= A_0 * (np.e)**(-4*np.log(2) * (t/tau)**2) #the prefactor in 61
    y= 0 #the other factor of 61.
    if q == q_0:
        y = np.e**(-1j*omega_0*t)
    elif q== -q_0:
        y = np.e**(+1j*omega_0*t)
    else:
        y=0
    return np.imag(x*y)


##################################################################
#   RHS functions:
##################################################################

# I love it when you call me
def big_Poppa(t,G):
    #Throw your hands in the air, if you'se a true player
    #Takes in vector G of length 6NP
    # returns dG/dt given G, t

    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6) #split into 6 vectors
    aDa_r = np.reshape(aDa_r, (N,P)) #turn these vectors into Nxp arrays:
    aDa_i = np.reshape(aDa_i, (N,P))	
    bDb_r = np.reshape(bDb_r, (N,P))
    bDb_i = np.reshape(bDb_i, (N,P))
    ab_r = np.reshape(ab_r, (N,P))
    ab_i = np.reshape(ab_i, (N,P))	

    # Initialize _ddt arrays
    aDa_r_ddt = np.zeros((N,P))
    aDa_i_ddt = np.zeros((N,P))
    bDb_r_ddt = np.zeros((N,P))
    bDb_i_ddt = np.zeros((N,P))
    ab_r_ddt = np.zeros((N,P))
    ab_i_ddt = np.zeros((N,P))

    # calculate Delta(t)
    Delta_r = Delta_r_f(G)
    Delta_i = Delta_i_f(G)

    # calcualte R(t), C(t)
    R = R_f(Delta_r)
    C_r = C_r_f(Delta_r)
    C_i = C_i_f(Delta_i)

    H = np.concatenate((R,C_r,C_i))

    # calculate dG/dt (i,j)
    for i in range(0,N):
            for j in range (i-4,i+5):
            
                    # check if j is in (0,N)
                    if 0 <= j < N:
                            m,n = ind[(i,j)] #d/dt <...>_(k,k')
                            
                            aDa_r_ddt[m,n] = F_aDa_r(i, j, G, H, t)
                            aDa_i_ddt[m,n] = F_aDa_i(i, j, G, H, t)
                            
                            bDb_r_ddt[m,n] = F_bDb_r(i, j, G, H, t)
                            bDb_i_ddt[m,n] = F_bDb_i(i, j, G, H, t)
                            
                            ab_r_ddt[m,n] = F_ab_r(i, j, G, H, t)
                            ab_i_ddt[m,n] = F_ab_i(i, j, G, H, t)

    #now flatten these arrays:			
    aDa_r_ddt = aDa_r_ddt.flatten()
    aDa_i_ddt = aDa_i_ddt.flatten()
    bDb_r_ddt = bDb_r_ddt.flatten()
    bDb_i_ddt = bDb_i_ddt.flatten()
    ab_r_ddt = ab_r_ddt.flatten()
    ab_i_ddt = ab_i_ddt.flatten()
    

    #now we concatenate :
    G_ddt = np.concatenate((aDa_r_ddt, aDa_i_ddt, bDb_r_ddt, bDb_i_ddt, ab_r_ddt, ab_i_ddt))
    return  G_ddt




#### Now define the following six functions:

#write your functions in terms of the index function above (to be defined)

##################### STEPAN ########################
def F_aDa_r(i,j,G, H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    term1 = 0
    if abs( i - j ) <= N_c:
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = ( R[j] - R[i] ) * aDa_i[ind[(i,j)]] - C_r[j] * ab_i[ind[(i,j)]] + C_i[j] * ab_r[ind[(i,j)]] + \
                C_r[i] * ab_i[ind[(j,i)]] - C_i[i] * ab_r[ind[(j,i)]]
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, q_0) * \
                                                                         ( - Lp[i,1] * aDa_i[ind[(ind_q[(i,q_0)],j)] ]  + \
                                                                           Mn[i,1] * ab_i[ ind[( j, ind_q[(i,q_0)] )] ] 
                                                                            ) \
                            + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, q_0) * \
                                                                         ( - Lp[i,1] * aDa_r[ind[(ind_q[(i,q_0)],j)] ]  + \
                                                                           Mn[i,1] * ab_r[ ind[( j, ind_q[(i,q_0)] )] ]
                                                                            )
        if ind_q[(i,-q_0)] in range(0,N) and abs( ind_q[( i,-q_0 )] -j ) <= N_c: # this checks if q displacement pushes us 
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, -q_0) * \
                                                                         ( - Lp[i,0] * aDa_i[ind[(ind_q[(i,-q_0)],j)] ] + \
                                                                           Mn[i,0] * ab_i[ ind[( j, ind_q[(i,-q_0)] )] ]
                                                                            ) \
                            + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, -q_0) * \
                                                                         ( - Lp[i,0] * aDa_r[ind[(ind_q[(i,-q_0)],j)] ]  + \
                                                                           Mn[i,0] * ab_r[ ind[( j, ind_q[(i,-q_0)] )] ]
                                                                            )
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, -q_0) * \
                                                                         ( Lp[j,1] * aDa_i[ ind[( i,ind_q[(j,-(-q_0))] )] ]  - \
                                                                           Mn[j,1] * ab_i[ ind[( i, ind_q[( j,-(-q_0) )] )] ]
                                                                            ) \
                            + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, -q_0) * \
                                                                         ( Lp[j,1] * aDa_r[ ind[( i,ind_q[(j,-(-q_0))] )] ]  + \
                                                                           Mn[j,1] * ab_r[ ind[( i, ind_q[( j,-(-q_0) )] )] ]
                                                                            )
        if ind_q[(j,-q_0)] in range(0,N) and abs( ind_q[( j,-q_0 )] -i ) <= N_c: # this checks if q displacement pushes us outside of
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, q_0) * \
                                                                         (Lp[j,0] * aDa_i[ ind[( i,ind_q[(j,-q_0)] )] ]  - \
                                                                           Mn[j,0] * ab_i[ ind[( i, ind_q[( j,-q_0 )] )] ]
                                                                            ) \
                            + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, q_0) * \
                                                                         ( Lp[j,0] * aDa_r[ ind[( i,ind_q[(j,-q_0)] )] ] + \
                                                                           Mn[j,0] * ab_r[ ind[( i, ind_q[( j,-q_0 )] )] ]
                                                                            )
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,2] * aDa_i[ind[( ind_q[(i,2.*q_0)],j )] ]  - \
                                                            Mp[i,2] * ab_i[ind[( j,ind_q[( i,2.*q_0 )] )] ] ) \
                                                      + A_factor_i * (
                                                            - Ln[i,2] * aDa_r[ind[( ind_q[(i,2.*q_0)],j )] ] - \
                                                            Mp[i,2] * ab_r[ind[( j,ind_q[( i,2.*q_0 )] )] ]  )
                                                                           ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,0] * aDa_i[ind[( i, ind_q[( j, -(-2.*q_0) )] )] ] + \
                                                            Mp[j,0] * ab_i[ ind[( i , ind_q[( j, -(-2.*q_0) )] )] ] ) \
                                                      + A_factor_i * (
                                                            Ln[j,0] * aDa_r[ind[( i, ind_q[( j, -(-2.*q_0) )] )] ] - \
                                                            Mp[j,0] * ab_r[ ind[( i , ind_q[( j, -(-2.*q_0) )] )] ] )
                                                                           )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,1] * aDa_i[ind[( ind_q[(i,0*q_0)],j )] ] - \
                                                            Mp[i,1] * ab_i[ind[( j,ind_q[( i,0*q_0 )] )] ] ) \
                                                      + A_factor_i * (
                                                            - Ln[i,1] * aDa_r[ind[( ind_q[(i,0*q_0)],j )] ]  - \
                                                            Mp[i,1] * ab_r[ind[( j,ind_q[( i,0*q_0 )] )] ]  )
                                                                           )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,1] * aDa_i[ind[( i, ind_q[( j, 0*q_0 )] )] ] + \
                                                            Mp[j,1] * ab_i[ ind[( i , ind_q[( j, 0*q_0 )] )] ] ) \
                                                      + A_factor_i * (
                                                            Ln[j,1] * aDa_r[ind[( i, ind_q[( j, 0*q_0 )] )] ] - \
                                                            Mp[j,1] * ab_r[ ind[( i , ind_q[( j, 0*q_0 )] )] ] )
                                                                           )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,0] * aDa_i[ind[( ind_q[(i,-2.*q_0)],j )] ]  - \
                                                            Mp[i,0] * ab_i[ind[( j,ind_q[( i,-2.*q_0 )] )] ] ) \
                                                      + A_factor_i * (
                                                            - Ln[i,0] * aDa_r[ind[( ind_q[(i,-2.*q_0)],j )] ]  - \
                                                            Mp[i,0] * ab_r[ind[( j,ind_q[( i,-2.*q_0 )] )] ]  )
                                                                           )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,2] * aDa_i[ind[( i, ind_q[( j, -2.*q_0 )] )] ] + \
                                                            Mp[j,2] * ab_i[ ind[( i , ind_q[( j, -2.*q_0 )] )] ] ) \
                                                      + A_factor_i * (
                                                            Ln[j,2] * aDa_r[ind[( i, ind_q[( j, -2.*q_0 )] )] ] - \
                                                            Mp[j,2] * ab_r[ ind[( i , ind_q[( j, -2.*q_0 )] )] ] )
                                                                           ) 
       
     
    return (term1 + term2 + term3)/hbar

######################################################################
#                JACOBIANS
######################################################################                   



def J_aDa_r_aDa_r(i,j,x,y,G, H, t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    term1 = 0
    if abs( i - j ) <= N_c:
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = 0
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, q_0) * \
                                                                         ( - Lp[i,1] * kron( ind_q[i,q_0],j,x,y )
                                                                            )
        if ind_q[(i,-q_0)] in range(0,N) and abs( ind_q[( i,-q_0 )] -j ) <= N_c: # this checks if q displacement pushes us 
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, -q_0) * \
                                                                         ( - Lp[i,0] * kron(ind_q[i,-q_0],j,x,y)
                                                                            )
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, -q_0) * \
                                                                         ( Lp[j,1] * kron(i, ind_q[j,-(-q_0)],x,y)
                                                                            )
        if ind_q[(j,-q_0)] in range(0,N) and abs( ind_q[( j,-q_0 )] -i ) <= N_c: # this checks if q displacement pushes us outside of
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, q_0) * \
                                                                         ( Lp[j,0] * kron(i, ind_q[j,-q_0],x,y)
                                                                            )
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * ( A_factor_i * (
                                                            - Ln[i,2] * kron( ind_q[i,2.*q_0],j,x,y ) )
                                                                           ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * (
                                                            Ln[j,0] * kron( i,ind_q[j,-(-2.*q_0)],x,y ) )
                                                                           )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * (
                                                            - Ln[i,1] * kron( ind_q[i,0*q_0],j,x,y ) )
                                                                           )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * (
                                                            Ln[j,1] * kron(i, ind_q[j,0*q_0],x,y ) )
                                                                           )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * (
                                                            - Ln[i,0] * kron( ind_q[i,-2.*q_0],j,x,y ) )
                                                                           )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * (
                                                            Ln[j,2] * kron( i, ind_q[j,-2.*q_0],x,y ) )
                                                                           ) 
       
     
    return (term1 + term2 + term3)/hbar


def J_aDa_r_aDa_i(i,j,x,y,G,H,t):

    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    term1 = 0
    if abs( i - j ) <= N_c:
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = ( R[j] - R[i] ) * kron(i,j,x,y)
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, q_0) * \
                                                                         ( - Lp[i,1] * kron(ind_q[i,q_0],j,x,y) 
                                                                            ) 
        if ind_q[(i,-q_0)] in range(0,N) and abs( ind_q[( i,-q_0 )] -j ) <= N_c: # this checks if q displacement pushes us 
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, -q_0) * \
                                                                         ( - Lp[i,0] * kron(ind_q[i,-q_0],j,x,y)
                                                                            ) 
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, -q_0) * \
                                                                         ( Lp[j,1] * kron(i, ind_q[j,-(-q_0)],x,y)
                                                                            ) 
        if ind_q[(j,-q_0)] in range(0,N) and abs( ind_q[( j,-q_0 )] -i ) <= N_c: # this checks if q displacement pushes us outside of
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, q_0) * \
                                                                         (Lp[j,0] * kron(i, ind_q[j,-q_0],x,y)
                                                                            ) 
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,2] * kron(ind_q[i,2.*q_0],j,x,y)) \
                                                      ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,0] * kron(i, ind_q[j,-(-2.*q_0)],x,y) ) \
                                                      )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,1] * kron(ind_q[i,0.*q_0],j,x,y) ) \
                                                                           )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,1] * kron(i, ind_q[j,0.*q_0],x,y) ) \
                                                                           )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,0] * kron(ind_q[i,2.*q_0],j,x,y) ) \
                                                      )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,2] * kron(i, ind_q[j,-2.*q_0],x,y) ) \
                                                      ) 
       
     
    return (term1 + term2 + term3)/hbar
    
def J_aDa_r_bDb_r(i,j,x,y,G,H,t):

    return 0

def J_aDa_r_bDb_i(i,j,x,y,G,H,t):

    return 0

def J_aDa_i_bDb_r(i,j,x,y,G,H,t):

    return 0

def J_aDa_i_bDb_i(i,j,x,y,G,H,t):

    return 0

def J_aDa_r_ab_r(i,j,x,y,G,H,t):

    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    term1 = 0
    if abs( i - j ) <= N_c:
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = C_i[j] * kron(i,j,x,y) - C_i[i] * kron(j,i,x,y)
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, q_0) * \
                                                                         ( Mn[i,1] * kron(j,ind_q[i,q_0],x,y)
                                                                            )
        if ind_q[(i,-q_0)] in range(0,N) and abs( ind_q[( i,-q_0 )] -j ) <= N_c: # this checks if q displacement pushes us 
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, -q_0) * \
                                                                         ( Mn[i,0] * kron(j,ind_q[i,-q_0],x,y)
                                                                            )
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, -q_0) * \
                                                                         ( Mn[j,1] * kron(i,ind_q[j,-(-q_0)],x,y)
                                                                            )
        if ind_q[(j,-q_0)] in range(0,N) and abs( ind_q[( j,-q_0 )] -i ) <= N_c: # this checks if q displacement pushes us outside of
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_i(t, q_0) * \
                                                                         ( Mn[j,0] * kron(i,ind_q[j,-q_0],x,y)
                                                                            )
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * ( - \
                                                            Mp[i,2] * kron(j,ind_q[i,2.*q_0],x,y) )
                                                                           ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * ( - \
                                                            Mp[j,0] * kron(i,ind_q[j,-(-2.*q_0)],x,y) )
                                                                           )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * ( - \
                                                            Mp[i,1] * kron(j,ind_q[i,0.*q_0],x,y) )
                                                                           )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * ( - \
                                                            Mp[j,1] * kron(i,ind_q[j,0.*q_0],x,y) )
                                                                           )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * (  - \
                                                            Mp[i,0] * kron(j,ind_q[i,-2.*q_0],x,y)  )
                                                                           )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_i * ( - \
                                                            Mp[j,2] * kron(i,ind_q[j,-2.*q_0],x,y) )
                                                                           ) 
       
     
    return (term1 + term2 + term3)/hbar

def J_aDa_r_ab_i(i,j,x,y,G,H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    term1 = 0
    if abs( i - j ) <= N_c:
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 =  - C_r[j] * kron(i,j,x,y)  + C_r[i] * kron(j,i,x,y)
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, q_0) * \
                                                                         ( Mn[i,1] * kron(j,ind_q[i,q_0],x,y)
                                                                            ) 
        if ind_q[(i,-q_0)] in range(0,N) and abs( ind_q[( i,-q_0 )] -j ) <= N_c: # this checks if q displacement pushes us 
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, -q_0) * \
                                                                         ( Mn[i,0] * kron(j,ind_q[i,-q_0],x,y)
                                                                            ) 
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, -q_0) * \
                                                                         (   - \
                                                                           Mn[j,1] * kron(i,ind_q[j,-(-q_0)],x,y) 
                                                                            ) 
        if ind_q[(j,-q_0)] in range(0,N) and abs( ind_q[( j,-q_0 )] -i ) <= N_c: # this checks if q displacement pushes us outside of
                                                # available matrix space. We apply it particularly to q_0
                                                # because in second sum we only have \pm q_0 summands
            term2 += + ( (1.0e18*hbar) / ( 2. * m ) ) * 2. * k_y * A_r(t, q_0) * \
                                                                         (- Mn[j,0] * kron(i,ind_q[j,-q_0],x,y) # MINUS BC CONJ
                                                                            ) 
         ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * ( -Mp[i,2] * kron(j,ind_q[i,2.*q_0],x,y)) \
                                                      ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * ( Mp[j,0] * kron(i,ind_q[j,-(-2.*q_0)],x,y) ) \
                                                      )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * (- Mp[i,1] * kron(j,ind_q[i,0.*q_0],x,y) ) \
                                                      )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * ( Mp[j,1] * kron(i,ind_q[j,0*q_0],x,y) ) \
                                                      )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * ( - Mp[i,0] * kron(j,ind_q[i,-2.*q_0],x,y) ) \
                                                      )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            term3 += (1.0e18**2 / 2. / m) * (A_factor_r * ( Mp[j,2] * kron(i,ind_q[j,-2.*q_0],x,y) ) \
                                                      ) 
       
     
    return (term1 + term2 + term3)/hbar


def J_aDa_i_aDa_r(i,j,x,y,G,H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    if abs(i-j) <= N_c: ## this checks, just like in Kyle's case, that we are only N_c
                                                        # diags away
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = ( R[j] - R[i] ) * kron(i,j,x,y)
        term1 = -1 * term1 ## accounts for -i from LHS
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, q_0) * \
                                                          ( - Lp[i,0] * kron(ind_q[i,q_0],j,x,y) )  \
                                                           )
        
        if ind_q[(i,-q_0)] in range(0,N) and abs(ind_q[(i,-q_0)] -j) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, -q_0) * \
                                                          ( - Lp[i,1] * kron(ind_q[i,-q_0],j,x,y) )
                                                           )
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, -q_0) * \
                                                          (  Lp[j,1] * kron(i, ind_q[j,-(-q_0)],x,y) )
                                                           ) 
        
        if ind_q[(j,-q_0)] in range(0,N) and abs(ind_q[(j,-q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, q_0) * \
                                                          ( Lp[j,0] * kron(i, ind_q[j,-q_0],x,y) )
                                                            )
 
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,2] * kron(ind_q[i,2.*q_0],j,x,y) ) \
                                                      ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,0] * kron(i, ind_q[j,-(-2.*q_0)],j,x,y) ) \
                                                      )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,1] * kron(ind_q[i,0*q_0],j,x,y) ) \
                                                      )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,1] * kron(i, ind_q[j,0*q_0],x,y) ) \
                                                      )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Ln[i,0] * kron(ind_q[i,-2.*q_0],j,x,y) ) \
                                                      )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            Ln[j,2] * kron(i,ind_q[j,-2.*q_0],x,y) ) \
                                                     ) 
    

    return (term1 + term2 + term3)/hbar

def J_aDa_i_aDa_i(i,j,x,y,G,H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    if abs(i-j) <= N_c: ## this checks, just like in Kyle's case, that we are only N_c
                                                        # diags away
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = 0
        term1 = -1 * term1 ## accounts for -i from LHS
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( -A_i(t,q_0) * \
                                                          ( - Lp[i,0] * kron(ind_q[i,q_0],j,x,y) ) \
                                                           )
        
        if ind_q[(i,-q_0)] in range(0,N) and abs(ind_q[(i,-q_0)] -j) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( \
                                                           - A_i(t, -q_0) * \
                                                          ( - Lp[i,1] * kron(ind_q[i,-q_0],j,x,y) )
                                                           )
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( \
                                                           - A_i(t, -q_0) * \
                                                          ( Lp[j,1] * kron(i, ind_q[j,-(-q_0)],x,y)  )
                                                           ) 
        
        if ind_q[(j,-q_0)] in range(0,N) and abs(ind_q[(j,-q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( \
                                                            - A_i(t, q_0) * \
                                                          ( Lp[j,0] * kron(i, ind_q[j,-q_0],x,y)  )
                                                           )
 
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * ( - A_factor_i * (
                                                            - Ln[i,2] * kron(ind_q[i,2.*q_0],j,x,y)  )
                                                                           ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * (
                                                            Ln[j,0] * kron(i, ind_q[j,-(-2.*q_0)],x,y)  )
                                                                           )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * (
                                                            - Ln[i,1] * kron(ind_q[i,0.*q_0],j,x,y) )
                                                                           )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * (
                                                            Ln[j,1] * kron(i, ind_q[j,0*q_0],x,y)  )
                                                                           )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * (
                                                            - Ln[i,0] * kron(ind_q[i,-2.*q_0],j,x,y) )
                                                                           )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * (
                                                            Ln[j,2] * kron(i, ind_q[j,-2.*q_0],x,y)  )
                                                                           ) 
    

    return (term1 + term2 + term3)/hbar
    
def J_aDa_i_ab_r(i,j,x,y,G,H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    if abs(i-j) <= N_c: ## this checks, just like in Kyle's case, that we are only N_c
                                                        # diags away
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = C_r[j] * kron(i,j,x,y)  + C_r[i] * kron(j,i,x,y)
        term1 = -1 * term1 ## accounts for -i from LHS
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, q_0) * \
                                                          ( Mn[i,0] * kron(j,ind_q[i,q_0],x,y) )  \
                                                            )
        
        if ind_q[(i,-q_0)] in range(0,N) and abs(ind_q[(i,-q_0)] -j) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, -q_0) * \
                                                          ( Mn[i,1] * kron(j,ind_q[i,-q_0],x,y) )
                                                           )
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, -q_0) * \
                                                          ( Mn[j,1] * kron(i,ind_q[j,-(-q_0)],x,y) )
                                                           ) 
        
        if ind_q[(j,-q_0)] in range(0,N) and abs(ind_q[(j,-q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, q_0) * \
                                                          ( Mn[j,0] * kron(i,ind_q[j,-q_0],x,y) )
                                                            )
 
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Mp[i,2] * kron(j,ind_q[i,2.*q_0],x,y) ) \
                                                      ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                        - Mp[j,0] * kron(i,ind_q[j,-(-2.*q_0)],x,y) ) \
                                                      )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Mp[i,1] * kron(j,ind_q[i,0*q_0],x,y) ) \
                                                      )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Mp[j,1] * kron(i,ind_q[j,0*q_0],x,y) ) \
                                                      )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                            - Mp[i,0] *  kron(j,ind_q[i,-2.*q_0],x,y) ) \
                                                      )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * (
                                                                - Mp[j,2] * kron(i,ind_q[j,-2.*q_0],x,y) ) \
                                                      ) 
    

    return (term1 + term2 + term3)/hbar


def J_aDa_i_ab_i(i,j,x,y,G,H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    if abs(i-j) <= N_c: ## this checks, just like in Kyle's case, that we are only N_c
                                                        # diags away
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = C_i[j] * kron(i,j,x,y) + C_i[i] * kron(j,i,x,y)
        term1 = -1 * term1 ## accounts for -i from LHS
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( \
                                                            - A_i(t, q_0) * \
                                                          ( Mn[i,0] * kron(j,ind_q[i,q_0],x,y) ) \
                                                           )
        
        if ind_q[(i,-q_0)] in range(0,N) and abs(ind_q[(i,-q_0)] -j) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( \
                                                           - A_i(t, -q_0) * \
                                                          ( Mn[i,1] * kron(j,ind_q[i,-q_0],x,y) ) \
                                                           )
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( \
                                                           - A_i(t, -q_0) * \
                                                          ( - Mn[j,1] * kron(i,ind_q[j,-(-q_0)],x,y) ) \
                                                           ) 
        
        if ind_q[(j,-q_0)] in range(0,N) and abs(ind_q[(j,-q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( \
                                                            - A_i(t, q_0) * \
                                                          ( - Mn[j,0] * kron(i,ind_q[j,-q_0],x,y) ) \
                                                           )
 
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * (- Mp[i,2] * kron(j,ind_q[i,2.*q_0],x,y)  )
                                                                           ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * ( Mp[j,0] * kron(i,ind_q[j,-(-2.*q_0)],x,y) ) \
                                                                           )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * (- Mp[i,1] * kron(j,ind_q[i,0*q_0],x,y)  )
                                                                           )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * ( Mp[j,1] * kron(i,ind_q[j,0*q_0],x,y) )
                                                                           )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * (- Mp[i,0] * kron(j,ind_q[i,-2.*q_0],x,y) )
                                                                           )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (- A_factor_i * ( Mp[j,2] * kron(i,ind_q[j,-2.*q_0],x,y) )
                                                                           ) 
    

    return (term1 + term2 + term3)/hbar
    



def F_aDa_i(i,j,G, H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR STEPAN
    if abs(i-j) <= N_c: ## this checks, just like in Kyle's case, that we are only N_c
                                                        # diags away
        ######################################################################
        #                TERM 1
        ######################################################################                   
        term1 = ( R[j] - R[i] ) * aDa_r[ind[(i,j)]] + C_r[j] * ab_r[ind[(i,j)]] + C_i[j] * ab_i[ind[(i,j)]] + \
                C_r[i] * ab_r[ind[(j,i)]] + C_i[i] * ab_i[ind[(j,i)]]
        term1 = -1 * term1 ## accounts for -i from LHS
        ######################################################################
        #                TERM 2
        ######################################################################                   
        term2 = 0
        if ind_q[(i,q_0)] in range(0,N) and abs(ind_q[(i,q_0)] -j) <= N_c: # this checks if q displacement pushes us outside of
                                                    # available matrix space. We apply it particularly to q_0
                                                    # because in second sum we only have \pm q_0 summands
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, q_0) * \
                                                          ( - Lp[i,0] * aDa_r[ind[(ind_q[(i,q_0)],j)]] + \
                                                               Mn[i,0] * ab_r[ ind[(j,ind_q[( i,q_0 )] )] ])  \
                                                            - A_i(t, q_0) * \
                                                          ( - Lp[i,0] * aDa_i[ind[(ind_q[(i,q_0)],j)]] + \
                                                               Mn[i,0] * ab_i[ ind[(j,ind_q[( i,q_0 )]) ] ] ) \
                                                           )
        
        if ind_q[(i,-q_0)] in range(0,N) and abs(ind_q[(i,-q_0)] -j) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, -q_0) * \
                                                          ( - Lp[i,1] * aDa_r[ind[(ind_q[(i,-q_0)],j)]] + \
                                                              Mn[i,1] * ab_r[ ind[( j,ind_q[( i,-q_0 )] )] ] ) \
                                                           - A_i(t, -q_0) * \
                                                          ( - Lp[i,1] * aDa_i[ind[(ind_q[(i,-q_0)],j)] ] + \
                                                              Mn[i,1] * ab_i[ ind[(j,ind_q[( i,-q_0) ]) ] ] ) \
                                                           )
        if ind_q[(j,q_0)] in range(0,N) and abs(ind_q[(j,q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, -q_0) * \
                                                          (  Lp[j,1] * aDa_r[ ind[( i, ind_q[(j,-(-q_0))] )] ] + \
                                                              Mn[j,1] * ab_r[ ind[( i, ind_q[( j,-(-q_0) )] )] ] ) \
                                                           - A_i(t, -q_0) * \
                                                          ( Lp[j,1] * aDa_i[ ind[( i, ind_q[(j,-(-q_0))] )] ] - \
                                                              Mn[j,1] * ab_i[ ind[( i, ind_q[( j,-(-q_0) )] )] ] ) \
                                                           ) 
        
        if ind_q[(j,-q_0)] in range(0,N) and abs(ind_q[(j,-q_0)] -i) <= N_c:
            
            term2 += - (1.0e18*hbar/ 2 / m ) * 2. * k_y * ( A_r(t, q_0) * \
                                                          ( Lp[j,0] * aDa_r[ ind[( i, ind_q[(j,-q_0)] )] ]  + \
                                                              Mn[j,0] * ab_r[ ind[( i, ind_q[( j,-q_0 )] )] ] ) \
                                                            - A_i(t, q_0) * \
                                                          ( Lp[j,0] * aDa_i[ ind[( i, ind_q[(j,-q_0)] )] ]  - \
                                                              Mn[j,0] * ab_i[ ind[( i, ind_q[( j,-q_0 )] )] ] ) \
                                                           )
 
        ######################################################################
        #                TERM 3
        ######################################################################                   
        term3 = 0
        if ind_q[(i,2.*q_0)] in range(0,N) and abs(ind_q[(i,2.*q_0)] -j) <= N_c:
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * ( \
                                                            - Ln[i,2] * aDa_r[ind[( ind_q[(i,2.*q_0)],j )] ]  - \
                                                            Mp[i,2] * ab_r[ind[( j,ind_q[( i,2.*q_0 )] ) ] ] ) \
                                                      - A_factor_i * ( \
                                                            - Ln[i,2] * aDa_i[ind[( ind_q[(i,2.*q_0)],j )] ] - \
                                                            Mp[i,2] * ab_i[ind[( j,ind_q[( i,2.*q_0 )] )] ]  ) \
                                                                           ) 
        if ind_q[(j,2.*q_0)] in range(0,N) and abs(ind_q[(j,2.*q_0)] -i) <= N_c:
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * ( \
                                                            Ln[j,0] * aDa_r[ind[( i, ind_q[( j, -(-2.*q_0) )] )] ] - \
                                                            Mp[j,0] * ab_r[ ind[( i , ind_q[( j, -(-2.*q_0) )] )] ] ) \
                                                      - A_factor_i * ( \
                                                            Ln[j,0] * aDa_i[ind[( i, ind_q[( j, -(-2.*q_0) )] )] ] + \
                                                            Mp[j,0] * ab_i[ ind[( i , ind_q[( j, -(-2.*q_0) )] )] ] ) \
                                                                           )
        if ind_q[(i,0*q_0)] in range(0,N) and abs(ind_q[(i,0*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * ( \
                                                            - Ln[i,1] * aDa_r[ind[( ind_q[(i,0*q_0)],j )] ] - \
                                                            Mp[i,1] * ab_r[ind[( j,ind_q[( i,0*q_0 )] )] ] ) \
                                                      - A_factor_i * ( \
                                                            - Ln[i,1] * aDa_i[ind[( ind_q[(i,0*q_0)],j )] ]  - \
                                                            Mp[i,1] * ab_i[ind[( j,ind_q[( i,0*q_0 )] )] ]  ) \
                                                                           )
        if ind_q[(j,0*q_0)] in range(0,N) and abs(ind_q[(j,0.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,-q_0) * A_r(t, q_0) + A_r(t,q_0) * A_r(t, -q_0) - \
                               A_i(t,-q_0) * A_i(t, q_0) - A_i(t,q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0) + \
                               A_r(t,-q_0) * A_i(t, q_0) + A_i(t,q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * ( \
                                                            Ln[j,1] * aDa_r[ind[( i, ind_q[( j, 0*q_0 )] )] ] - \
                                                            Mp[j,1] * ab_r[ ind[( i , ind_q[( j, 0*q_0 )] )] ] ) \
                                                      - A_factor_i * ( \
                                                            Ln[j,1] * aDa_i[ind[( i, ind_q[( j, 0*q_0 )] )] ] + \
                                                            Mp[j,1] * ab_i[ ind[( i , ind_q[( j, 0*q_0 )] )] ] ) \
                                                                           )

        if ind_q[(i,-2.*q_0)] in range(0,N) and abs(ind_q[(i,-2.*q_0)] -j) <= N_c: 
            A_factor_r = A_r(t,-3.*q_0) * A_r(t, q_0) + A_r(t,-q_0) * A_r(t, -q_0) - \
                               A_i(t,-3.*q_0) * A_i(t, q_0) - A_i(t,-q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0) + \
                               A_r(t,-3.*q_0) * A_i(t, q_0) + A_i(t,-q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * ( \
                                                            - Ln[i,0] * aDa_r[ind[( ind_q[(i,-2.*q_0)],j )] ]  - \
                                                            Mp[i,0] * ab_r[ind[( j,ind_q[( i,-2.*q_0 )] )] ] ) \
                                                      - A_factor_i * ( \
                                                            - Ln[i,0] * aDa_i[ind[( ind_q[(i,-2.*q_0)],j )] ]  - \
                                                            Mp[i,0] * ab_i[ind[( j,ind_q[( i,-2.*q_0 )] )] ]  ) \
                                                                           )
        if ind_q[(j,-2.*q_0)] in range(0,N) and abs(ind_q[(j,-2.*q_0)] -i) <= N_c: 
            A_factor_r = A_r(t,q_0) * A_r(t, q_0) + A_r(t,3.*q_0) * A_r(t, -q_0) - \
                               A_i(t,q_0) * A_i(t, q_0) - A_i(t,3.*q_0) * A_i(t, -q_0)
            A_factor_i = A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0) + \
                               A_r(t,q_0) * A_i(t, q_0) + A_i(t,3.*q_0) * A_r(t, -q_0)
            term3 += - (1.0e18**2 / 2. / m) * (A_factor_r * ( \
                                                            Ln[j,2] * aDa_r[ind[( i, ind_q[( j, -2.*q_0 )] )] ] - \
                                                            Mp[j,2] * ab_r[ ind[( i , ind_q[( j, -2.*q_0 )] )] ] ) \
                                                      - A_factor_i * ( \
                                                            Ln[j,2] * aDa_i[ind[( i, ind_q[( j, -2.*q_0 )] )] ] + \
                                                            Mp[j,2] * ab_i[ ind[( i , ind_q[( j, -2.*q_0 )] )] ] ) \
                                                                           ) 
    

    return (term1 + term2 + term3)/hbar



##################### KYLE ##########################

def F_bDb_r(i,j,G, H,t):
    ##########################
    #THESE FUNCTIONS WORK WITHOUT EDGE CASES. I ALSO NEED NON-EDGE-CASE FUNCTIONS.
    #i,j are the indices of the NxN array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #
    #
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    #multiply through by hbar at the end
    ######################################################################
    #                TERM 1
    ######################################################################               
    term1 = (R[j] - R[i])*bDb_i[ind[i,j]] +C_r[j]*ab_i[ind[j,i]]
    + C_r[i]*ab_i[ind[i,j]]
    -( C_i[j]*ab_r[ind[j,i]] +
       C_i[i]*ab_r[ind[i,j]])
    ###
    ######################################################################
    #                TERM 2
    ######################################################################
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc.
        if abs(i-j) +1 <= N_c:
        ###################################################################
        #We first check if it possible to go beyond N_c. If not, we skip any
        #worrying and evaluate the function.
        #
        #If it is possible, we must only keep terms that keep us in band.
        ###################################################################
            if ind_q[i,-q] in range(0,N):
                ##################################################################
                #
                #Additionally, we must worry about edge cases. We always check
                #that are momentum shifts don't take us out of the band.
                ################################################################
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                A_r(t,q) *(
                Lp[i,a_p]*bDb_i[ind[ind_q[i,-q],j]]
                   -Mn[i,a_p]*ab_i[ind[ind_q[i,-q],j]]) 
                +A_i(t,q) *(
                 Lp[i,a_p]*bDb_r[ind[ind_q[i,-q],j]]
                  -Mn[i,a_p]*ab_r[ind[ind_q[i,-q],j]]))
            if ind_q[j,q] in range(0,N):
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                A_r(t,q) *(
                 - Lp[j,a]*bDb_i[ind[i,ind_q[j,q]]] 
                  -Mn[j,a]*ab_i[ind[ind_q[j,q],i]])
                +A_i(t,q) *(
                 - Lp[j,a]*bDb_r[ind[i,ind_q[j,q]]]
                  +Mn[j,a]*ab_r[ind[ind_q[j,q],i]]))
        else: 
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(
                    Lp[i,a_p]*bDb_i[ind[ind_q[i,-q],j]]
                       -Mn[i,a_p]*ab_i[ind[ind_q[i,-q],j]]) 
                    +A_i(t,q) *(
                     - Lp[j,a]*bDb_r[ind[i,ind_q[j,q]]]
                      +Mn[j,a]*ab_r[ind[ind_q[j,q],i]]))
                if ind_q[j,q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(
                     - Lp[j,a]*bDb_i[ind[i,ind_q[j,q]]] 
                      -Mn[j,a]*ab_i[ind[ind_q[j,q],i]])
                    +A_i(t,q) *(
                     - Lp[j,a]*bDb_r[ind[i,ind_q[j,q]]]
                      +Mn[j,a]*ab_r[ind[ind_q[j,q],i]]))
            else:
                term2 +=0
    #####
    ######################################################################
    #                TERM 3
    ######################################################################
    #Term 3 becomes problematic if abs(i-j) +2 >N_c:
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                   -Ln[i,b_p]*bDb_i[ind[ind_q[i,-q],j]] -Mp[i,b_p]*ab_i[ind[ind_q[i,-q],j]]) 
                  +( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*
                   ( - Ln[i,b_p]*bDb_r[ind[ind_q[i,-q],j]]
                                    -Mp[i,b_p]*ab_r[ind[ind_q[i,-q],j]]))
                if ind_q[j,q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                   + Ln[j,b]*bDb_i[ind[i,ind_q[j,q]]]
                   -Mp[j,b]*ab_i[ind[ind_q[j,q],i]] )
                  +( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                   + Ln[j,b]*bDb_r[ind[i,ind_q[j,q]]] 
                    +Mp[j,b]*ab_r[ind[ind_q[j,q],i]] ))                  
            else:
                if abs(ind_q[i,-q] -j) +2 <= N_c:
                    if ind_q[i,-q] in range(0,N):
                        term3 += 1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                       -Ln[i,b_p]*bDb_i[ind[ind_q[i,-q],j]] -Mp[i,b_p]*ab_i[ind[ind_q[i,-q],j]]) 
                      +( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*
                       ( - Ln[i,b_p]*bDb_r[ind[ind_q[i,-q],j]]
                                        -Mp[i,b_p]*ab_r[ind[ind_q[i,-q],j]]))
                    if ind_q[j,q] in range(0,N):
                        term3 += 1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                       + Ln[j,b]*bDb_i[ind[i,ind_q[j,q]]]
                       -Mp[j,b]*ab_i[ind[ind_q[j,q],i]] )
                      +( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                       + Ln[j,b]*bDb_r[ind[i,ind_q[j,q]]] 
                        +Mp[j,b]*ab_r[ind[ind_q[j,q],i]] )) 
                else:
                    term3 +=0
    return (term1 + term2 + term3)/hbar

def F_bDb_i(i,j,G, H,t):
#i,j are the indices of the NxN array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #
    #
    x,y = ind[i,j]
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    #multiply through by hbar at the end
    ######################################################################
    #                TERM 1
    ###################################################################### 
    term1 = -(R[j]-R[i])*bDb_r[ind[i,j]] + C_r[j]*ab_r[ind[j,i]] + C_i[j]*ab_i[ind[j,i]]
    -( C_r[i]*ab_r[ind[i,j]] + C_i[j]*ab_i[ind[i,j]])
    ###
    ######################################################################
    #                TERM 2
    ######################################################################
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc
        #
        ###################################################################
        #We first check if it possible to go beyond N_c. If not, we skip any
        #worrying and evaluate the function.
        #
        #If it is possible, we must only keep terms that keep us in band.
        ###################################################################
        if abs(i-j) +1 <= N_c:       
            if ind_q[i,-q] in range(0,N):
                ##################################################################
                #
                #Additionally, we must worry about edge cases. We always check
                #that are momentum shifts don't take us out of the band.
                #################################################################
                term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *( A_r(t,q) *(
                 Lp[i,a_p]*bDb_r[ind[ind_q[i,-q],j]] -Mn[i,a_p]*ab_r[ind[ind_q[i,-q],j]])
                 +A_i(t,q) *(
                 - Lp[i,a_p]*bDb_i[ind[ind_q[i,-q],j]]
                      + Mn[i,a_p]*ab_i[ind[ind_q[i,-q],j]]))
            if ind_q[j,q] in range(0,N):
                 term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(
                 - Lp[j,a]*bDb_r[ind[i,ind_q[j,q]]]
                 + Mn[j,a]*ab_r[ind[ind_q[j,q],i]] )
                 +A_i(t,q) *(
                 + Lp[j,a]*bDb_i[ind[i,ind_q[j,q]]]
                  +Mn[j,a]*ab_i[ind[ind_q[j,q],i]]))
        else:
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):    
                    term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                        A_r(t,q) *(
                     Lp[i,a_p]*bDb_r[ind[ind_q[i,-q],j]] -Mn[i,a_p]*ab_r[ind[ind_q[i,-q],j]])
                     +A_i(t,q) *(
                     - Lp[i,a_p]*bDb_i[ind[ind_q[i,-q],j]]
                          + Mn[i,a_p]*ab_i[ind[ind_q[i,-q],j]]))
                if ind_q[j,q] in range(0,N):
                     term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                        A_r(t,q) *(
                     - Lp[j,a]*bDb_r[ind[i,ind_q[j,q]]]
                     + Mn[j,a]*ab_r[ind[ind_q[j,q],i]] )
                     +A_i(t,q) *(
                     + Lp[j,a]*bDb_i[ind[i,ind_q[j,q]]]
                      +Mn[j,a]*ab_i[ind[ind_q[j,q],i]]))
            else:
                term2 += 0
    ######################################################################
    #                TERM 3
    ######################################################################
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                # This checks if leaving the band is possible.
                if ind_q[i,-q] in range(0,N):
                    #This rules out edge cases.
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                     - Ln[i,b_p]*bDb_r[ind[ind_q[i,-q],j]]
                      -Mp[i,b_p]*ab_r[ind[ind_q[i,-q],j]])
                     + ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                     + Ln[i,b_p]*bDb_i[ind[ind_q[i,-q],j]]
                                 + Mp[i,b_p]*ab_i[ind[ind_q[i,-q],j]]))
                if ind_q[j,q] in range(0,N):
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                     + Ln[j,b]*bDb_r[ind[i,ind_q[j,q]]]
                     + Mp[j,b]*ab_r[ind[ind_q[j,q],i]] )
                     + ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                     - Ln[j,b]*bDb_i[ind[i,ind_q[j,q]]] 
                      +Mp[j,b]*ab_i[ind[ind_q[j,q],i]] ))
            else:
                if abs(ind_q[i,-q] - j) +2 <= N_c:
                    if ind_q[i,-q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                         - Ln[i,b_p]*bDb_r[ind[ind_q[i,-q],j]]
                          -Mp[i,b_p]*ab_r[ind[ind_q[i,-q],j]])
                         + ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                         + Ln[i,b_p]*bDb_i[ind[ind_q[i,-q],j]]
                                     + Mp[i,b_p]*ab_i[ind[ind_q[i,-q],j]]))
                    if ind_q[j,q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                         + Ln[j,b]*bDb_r[ind[i,ind_q[j,q]]]
                         + Mp[j,b]*ab_r[ind[ind_q[j,q],i]] )
                         + ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                         - Ln[j,b]*bDb_i[ind[i,ind_q[j,q]]] 
                          +Mp[j,b]*ab_i[ind[ind_q[j,q],i]] ))
                else:
                    term3 +=0
    return (term1 + term2 + term3)/hbar



def F_ab_r(m,n,G, H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR Jordan
    return 1

def F_ab_i(m,n,G, H,t):
    #m,n are the indices of the NxP array
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR Jordan
    return 1


####################################################################
#          JACOBIANS
####################################################################    
######################################################################
#       JACOBIAN for F_bDb_r
# following Jordan, all partials wrt    f(p,p') for f=ab,ab_r,ab_i,aDa_r, etc.
# I wil use the indices x,y for p,p'
# use the notation J_f_g(i,j,x,y) for Jacobian of f(i,j) wrt g(x,y)
###################################################################### 

def J_bDb_r_aDa_r(i,j,x,y,G,H,t):
    return 0

def J_bDb_r_aDa_i(i,j,x,y,G,H,t):
    return 0

def J_bDb_r_bDb_r(i,j,x,y,G,H,t):
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    # the first term is offdiagonal, and vanishes:
    term1 =0
    #
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc.
        if abs(i-j) +1 <= N_c:
        ###################################################################
        #We first check if it possible to go beyond N_c. If not, we skip any
        #worrying and evaluate the function.
        #
        #If it is possible, we must only keep terms that keep us in band.
        ###################################################################
            if ind_q[i,-q] in range(0,N):
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                +A_i(t,q) *
                 Lp[i,a_p]*kron(ind_q[i,-q],j,x,y))
            if ind_q[j,q] in range(0,N):
                term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *A_i(t,q) *Lp[j,a]*kron(i,ind_q[j,q],x,y)
        else: 
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    +A_i(t,q) *
                     Lp[i,a_p]*kron(ind_q[i,-q],j,x,y))
                if ind_q[j,q] in range(0,N):
                    term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *A_i(t,q) *Lp[j,a]*kron(i,ind_q[j,q],x,y)
            else:
                term2 +=0
    #####
    ######################################################################
    #                TERM 3
    ######################################################################
    #Term 3 becomes problematic if abs(i-j) +2 >N_c:
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                   A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                   ( - Ln[i,b_p]*kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                   A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                   + Ln[j,b]*kron(i,ind_q[j,q],x,y))                
            else:
                if abs(ind_q[i,-q] -j) +2 <= N_c:
                        if ind_q[i,-q] in range(0,N):
                            term3 += 1.0e-18**2/(2*m) *2*k_y *(
                           A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                           ( - Ln[i,b_p]*kron(ind_q[i,-q],j,x,y)))
                        if ind_q[j,q] in range(0,N):
                            term3 += 1.0e-18**2/(2*m) *2*k_y *(
                           A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                           + Ln[j,b]*kron(i,ind_q[j,q],x,y)) 
                else:
                    term3 +=0
    return  (term1 + term2 + term3)/hbar

def J_bDb_r_bDb_i(i,j,x,yG, H,t):
    k_i = k[i]
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    #               
    term1 = (R[j] - R[i]) *kron(i,j,x,y)
    #term2
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc.
        if abs(i-j) +1 <= N_c:
        ###################################################################
        #We first check if it possible to go beyond N_c. If not, we skip any
        #worrying and evaluate the function.
        #
        #If it is possible, we must only keep terms that keep us in band.
        ###################################################################
            if ind_q[i,-q] in range(0,N):
                ##################################################################
                #
                #Additionally, we must worry about edge cases. We always check
                #that are momentum shifts don't take us out of the band.
                ################################################################
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                A_r(t,q) *(
                Lp[i,a_p] * kron(ind_q[i,-q],j,x,y)))
            if ind_q[j,q] in range(0,N):
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                A_r(t,q) *(-1)*(
                  Lp[j,a]*kron(i,ind_q[j,q],x,y)))
        else: 
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(
                    Lp[i,a_p] * kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(-1)*(
                      Lp[j,a]*kron(i,ind_q[j,q],x,y)))
            else:
                term2 +=0
    #term3
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                   -1)*Ln[i,b_p] *kron(ind_q[i,-q],j,x,y))
                if ind_q[j,q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *
                   Ln[j,b] *kron(i,ind_q[j,q],x,y) )                
            else:
                if abs(ind_q[i,-q] -j) +2 <= N_c:
                    if ind_q[i,-q] in range(0,N):
                        term3 += 1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                       -1)*Ln[i,b_p] *kron(ind_q[i,-q],j,x,y))
                    if ind_q[j,q] in range(0,N):
                        term3 += 1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *
                       Ln[j,b] *kron(i,ind_q[j,q],x,y) )
                else:
                    term3 +=0
    return  (term1 + term2 + term3)/hbar


def J_bDb_r_ab_r(i,j,x,y,G,H,t):
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)              
    term1 =      -( C_i[j]*kron(j,i,x,y) + C_i[i]*kron(i,j,x,y))
    #term2:
    term2 = 0
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc.
        if abs(i-j) +1 <= N_c:
        ###################################################################
        #We first check if it possible to go beyond N_c. If not, we skip any
        #worrying and evaluate the function.
        #
        #If it is possible, we must only keep terms that keep us in band.
        ###################################################################
            if ind_q[i,-q] in range(0,N):
                ##################################################################
                #
                #Additionally, we must worry about edge cases. We always check
                #that are momentum shifts don't take us out of the band.
                ################################################################
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                +A_i(t,q) *(-1) *Mn[i,a_p] *kron(ind_q[i,q],j,x,y))
            if ind_q[j,q] in range(0,N):
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                A_i(t,q) *Mn[j,a] * kron(ind_q[j,q],i,x,y))
        else: 
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    +A_i(t,q) *(-1) *Mn[i,a_p] *kron(ind_q[i,q],j,x,y))
                if ind_q[j,q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    A_i(t,q) *Mn[j,a] * kron(ind_q[j,q],i,x,y))
            else:
                term2 +=0
    #term 3:
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                  ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                    (-1)*Mp[i,b_p]*kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                   A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                    Mp[j,b]*kron(ind_q[j,q],i,x,y) )                
            else:
                if abs(ind_q[i,-q] -j) +2 <= N_c:
                    if ind_q[i,-q] in range(0,N):
                        term3 += 1.0e-18**2/(2*m) *2*k_y *(
                      ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                        (-1)*Mp[i,b_p]*kron(ind_q[i,-q],j,x,y)))
                    if ind_q[j,q] in range(0,N):
                        term3 += 1.0e-18**2/(2*m) *2*k_y *(
                       A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                        Mp[j,b]*kron(ind_q[j,q],i,x,y)  )
                else:
                    term3 +=0
    return  (term1 + term2 + term3)/hbar


def J_bDb_r_ab_i(i,j,x,y,G,H,t):
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    #term1:              
    term1 = C_r[j] *kron(j,i,x,y) + C_r[i]*kron(i,j,x,y)
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc.
        if abs(i-j) +1 <= N_c:
        ###################################################################
        #We first check if it possible to go beyond N_c. If not, we skip any
        #worrying and evaluate the function.
        #
        #If it is possible, we must only keep terms that keep us in band.
        ###################################################################
            if ind_q[i,-q] in range(0,N):
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                A_r(t,q) *(-1)* Mn[i,a_p] *kron(ind_q[i,-q],j,x,y))
            if ind_q[j,q] in range(0,N):
                term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                A_r(t,q) *(-Mn[j,a]* kron(ind_q[j,q],i,x,y)))
        else: 
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(-1)* Mn[i,a_p] *kron(ind_q[i,-q],j,x,y))
                if ind_q[j,q] in range(0,N):
                    term2 += 1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(-Mn[j,a]* kron(ind_q[j,q],i,x,y)))
            else:
                term2 +=0
    #term3
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                    (-Mp[i,b_p]*kron(ind_q[i,-q],j,x,y))))
                if ind_q[j,q] in range(0,N):
                    term3 += 1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                   -Mp[j,b]*kron(ind_q[j,q],i,x,y) ) )              
            else:
                if abs(ind_q[i,-q] -j) +2 <= N_c:
                   if ind_q[i,-q] in range(0,N):
                        term3 += 1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                        (-Mp[i,b_p]*kron(ind_q[i,-q],j,x,y))))
                   if  ind_q[j,q] in range(0,N):
                        term3 += 1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                       -Mp[j,b]*kron(ind_q[j,q],i,x,y) ) )
                else:
                    term3 +=0
    return (term1 + term2 + term3)/hbar
    


######################################################################
#       JACOBIAN for F_bDb_i
######################################################################


def J_bDb_i_aDa_r(i,j,x,y,G,H,t):
    return 0

def J_bDb_i_aDa_i(i,j,x,y,G,H,t):
    return 0

def J_bDb_i_bDb_r(i,j,x,y,G,H,t):
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    #term1 
    term1 = -(R[j]-R[i])*kron(i,j,x,y)
    #term2:
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc
        if abs(i-j) +1 <= N_c:       
            if ind_q[i,-q] in range(0,N):
                term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *Lp[i,a_p] *kron(ind_q[i,-q],j,x,y))
            if ind_q[j,q] in range(0,N):
                 term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(
                 - Lp[j,a]* kron(i,ind_q[j,q],x,y)))
        else:
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                        A_r(t,q) *Lp[i,a_p] *kron(ind_q[i,-q],j,x,y))
                if ind_q[j,q] in range(0,N):
                     term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                        A_r(t,q) *(
                     - Lp[j,a]* kron(i,ind_q[j,q],x,y)))
            else:
                term2 += 0
    #term3:
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                # This checks if leaving the band is possible.
                if ind_q[i,-q] in range(0,N):
                    #This rules out edge cases.
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                     - Ln[i,b_p] * kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                     Ln[j,b]* kron(i,ind_q[j,q],x,y)))
            else:
                if abs(ind_q[i,-q] - j) +2 <= N_c:
                    if ind_q[i,-q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                         - Ln[i,b_p] * kron(ind_q[i,-q],j,x,y)))
                    if ind_q[j,q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                         Ln[j,b]* kron(i,ind_q[j,q],x,y)))
                else:
                    term3 +=0
    return (term1 + term2 + term3)/hbar


def J_bDb_i_bDb_i(i,j,x,y,G,H,t):
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    term1 = 0
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc
        if abs(i-j) +1 <= N_c:       
            if ind_q[i,-q] in range(0,N):
                term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                 +A_i(t,q) *(
                 - Lp[i,a_p]*kron(ind_q[i,-q],j,x,y)))
            if ind_q[j,q] in range(0,N):
                 term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                 A_i(t,q) *(
                 Lp[j,a]*kron(i,ind_q[j,q],x,y)))
        else:
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                     +A_i(t,q) *(
                     - Lp[i,a_p]*kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                     term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                     A_i(t,q) *(
                     Lp[j,a]*kron(i,ind_q[j,q],x,y)))
            else:
                term2 += 0
    ######################################################################
    #                TERM 3
    ######################################################################
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                # This checks if leaving the band is possible.
                if ind_q[i,-q] in range(0,N):
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                      ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                      Ln[i,b_p]*kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                     (A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                     - Ln[j,b] *kron(i,ind_q[j,q],x,y)))
            else:
                if abs(ind_q[i,-q] - j) +2 <= N_c:
                    if ind_q[i,-q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                          ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                          Ln[i,b_p]*kron(ind_q[i,-q],j,x,y)))
                    if ind_q[j,q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                         (A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                         - Ln[j,b] *kron(i,ind_q[j,q],x,y)))
                else:
                    term3 +=0
    return (term1 + term2 + term3)/hbar

def J_bDb_i_ab_r(i,j,x,y,G,H,t):
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    #term1
    term1 =  C_r[j] *kron(j,i,x,y)   -C_r[i]*kron(i,j,x,y)
    #term2
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc
        if abs(i-j) +1 <= N_c:       
            if ind_q[i,-q] in range(0,N):
                term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *( -Mn[i,a_p]*  kron(ind_q[i,-q],j,x,y)))
            if ind_q[j,q] in range(0,N):
                 term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                    A_r(t,q) *(
                 Mn[j,a] *kron(ind_q[j,q],i,x,y)))
        else:
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                        A_r(t,q) *( -Mn[i,a_p]*  kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                     term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                        A_r(t,q) *(
                     Mn[j,a] *kron(ind_q[j,q],i,x,y)))
            else:
                term2 += 0
    ######################################################################
    #                TERM 3
    ######################################################################
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                # This checks if leaving the band is possible.
                if ind_q[i,-q] in range(0,N):
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                      -Mp[i,b_p]* kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                    ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                     Mp[j,b]*kron(ind_q[j,q],i,x,y)))
            else:
                if abs(ind_q[i,-q] - j) +2 <= N_c:
                    if ind_q[i,-q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                          -Mp[i,b_p]* kron(ind_q[i,-q],j,x,y)))
                    if ind_q[j,q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                        ( A_r(t,q-q_p)*A_r(t,q_p) - A_i(t,q-q_p)*A_i(t,q_p)) *(
                         Mp[j,b]*kron(ind_q[j,q],i,x,y)))
                else:
                    term3 +=0
    return (term1 + term2 + term3)/hbar

def J_bDb_i_ab_i(i,j,x,y,G,H,t):
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6)
    R, C_r,C_i = np.split(H,3)
    #term1
    term1 =  C_i[j]* kron(j,i,x,y)  - C_i[j]*kron(i,j,x,y)
    #term2
    term2 = 0   #term2 has a sum over q=\pm q_0, which we implement using a loop
    for a in range(0,2):
        q = (-1 + 2*a)*q_0
        a_p = (a +1) % 2 ### this is `aprime'. Allows me to call Lp[i,-q_0], etc
        if abs(i-j) +1 <= N_c:       
            if ind_q[i,-q] in range(0,N):
                term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                 A_i(t,q) * Mn[i,a_p]*kron(ind_q[i,-q],j,x,y))
            if ind_q[j,q] in range(0,N):
                 term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                 A_i(t,q) * Mn[j,a]* kron(ind_q[j,q],i,x,y))
        else:
            if abs(ind_q[i,-q] - j) +1 <= N_c:
                if ind_q[i,-q] in range(0,N):
                    term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                     A_i(t,q) * Mn[i,a_p]*kron(ind_q[i,-q],j,x,y))
                if ind_q[j,q] in range(0,N):
                     term2 += (-1)*1.0e-18*hbar/(2*m) *2*k_y *(
                     A_i(t,q) * Mn[j,a]* kron(ind_q[j,q],i,x,y))
            else:
                term2 += 0
    ######################################################################
    #                TERM 3
    ######################################################################
    term3 = 0 #term3 has sum over q=\pm2q_0,0, which we implement using a loop
    for b in range(0,3):
        q = (-2 + 2*b)*q_0
        #now we define a `bprime' index to allow for calling `-q' index:
        if b ==1:
            b_p = 1
        elif b==0:
            b_p = 2
        else:
            b_p = 0
        for a in range(0,2):    #term3 also has a sum over q' = \pmq_0: 
            q_p = (-1 + 2*a)*q_0 #this is q'
            if abs(i-j) +2 <= N_c:
                # This checks if leaving the band is possible.
                if ind_q[i,-q] in range(0,N):
                    #This rules out edge cases.
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                      ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                                 Mp[i,b_p]* kron(ind_q[i,-q],j,x,y)))
                if ind_q[j,q] in range(0,N):
                    term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                     ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                    Mp[j,b]*kron(ind_q[j,q],i,x,y)))
            else:
                if abs(ind_q[i,-q] - j) +2 <= N_c:
                    if ind_q[i,-q] in range(0,N):
                        #This rules out edge cases.
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                          ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                                     Mp[i,b_p]* kron(ind_q[i,-q],j,x,y)))
                    if ind_q[j,q] in range(0,N):
                        term3 += (-1)*1.0e-18**2/(2*m) *2*k_y *(
                         ( A_r(t,q-q_p)*A_i(t,q_p) + A_i(t,q-q_p)*A_r(t,q_p))*(
                        Mp[j,b]*kron(ind_q[j,q],i,x,y)))
                else:
                    term3 +=0
    return (term1 + term2 + term3)/hbar



######################## JORDAN #######################
#######################################################
#######################################################

def F_ab_r(k,kp,G, H,t):
    #k,kp are the indices of the NxN array NOT MOMENTA
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR Jordan
    
	#####
	# I modified ind() to check if its arguements are in the band, if not return (N,0).
	# Below I set the [N,0] arguement of all correlators to zero. This bypasses separating
	#  the function into if-else chunks.
	#####
	
	
    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6) #unpack correlators and time dependent functions  
    R, C_r,C_i = np.split(H,3)

    extraZeros = np.zeros((1,P)) #Create (1xP) array of zeros to append to the aDa_r,...,ab_i matrices
    aDa_r = np.concatenate((aDa_r,extraZeros),axis=0)
    aDa_i = np.concatenate((aDa_i,extraZeros),axis=0)
    bDb_r = np.concatenate((bDb_r,extraZeros),axis=0)
    bDb_i = np.concatenate((bDb_i,extraZeros),axis=0)
    ab_r  = np.concatenate((ab_r,extraZeros),axis=0)
    ab_i  = np.concatenate((ab_i,extraZeros),axis=0)

    if k in range(0,N) and kp in range(0,N) and (kp-k+4) in range(0,P):
            RHS = ((1/hbar)*(R[k]+R[kp])*ab_i[ind_j(k,kp)]
                    + (1/hbar)*(C_r[kp]*aDa_i[ind_j(kp,k)] + C_i[kp]*aDa_r[ind_j(kp,k)])
                    + (1/hbar)*(C_r[kp]*bDb_i[ind_j(k,kp)] + C_i[kp]*(bDb_r[ind_j(k,kp)]-kron1(k,kp)))
                    +    (1.0e18/(2*m))*2*k_y*A_r(t,q_0)*( Lp[k,1]  * ab_i[ind_j(k+Q,kp)]                    
                                                                                     -Lp[kp,0] * ab_i[ind_j(k,kp-Q)]
                                                                                     -Mn[kp,0] * aDa_i[ind_j(kp-Q,k)]
                                                                                     +Mn[k,1]  * bDb_i[ind_j(k+Q,kp)]
                                                                                     +Lp[k,0]  * ab_i[ind_j(k-Q,kp)]
                                                                                     -Lp[kp,1] * ab_i[ind_j(k,kp+Q)]
                                                                                     -Mn[kp,1] * aDa_i[ind_j(kp+Q,k)]
                                                                                     +Mn[k,0]  * bDb_i[ind_j(k-Q,kp)])
                    +    (1.0e18/(2*m))*2*k_y*A_i(t,q_0)*( Lp[k,1]  * ab_r[ind_j(k+Q,kp)]                   
                                                                                     -Lp[kp,0] * ab_r[ind_j(k,kp-Q)]
                                                                                     -Mn[kp,0] * aDa_r[ind_j(kp-Q,k)]
                                                                                     +Mn[k,1]  *(bDb_r[ind_j(k+Q,kp)]-kron1(kp-k,+Q))
                                                                                     -Lp[k,0]  * ab_r[ind_j(k-Q,kp)]
                                                                                     +Lp[kp,1] * ab_r[ind_j(k,kp+Q)]
                                                                                     +Mn[kp,1] * aDa_r[ind_j(kp+Q,k)]
                                                                                     -Mn[k,0]  *(bDb_r[ind_j(k-Q,kp)]-kron1(kp-k,-Q)))
                    +    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)
                                     *( np.cos(2*omega_0*t)*( Ln[k,2]  * ab_i[ind_j(k+2*Q,kp)]              
                                                                                     +Ln[kp,0] * ab_i[ind_j(k,kp-2*Q)]
                                                                                     -Mp[kp,0] * aDa_i[ind_j(kp-2*Q,k)]
                                                                                     -Mp[k,2]  * bDb_i[ind_j(k+2*Q,kp)]
                                                                                     +Ln[k,0]  * ab_i[ind_j(k-2*Q,kp)]
                                                                                     +Ln[kp,2] * ab_i[ind_j(k,kp+2*Q)]
                                                                                     -Mp[kp,2] * aDa_i[ind_j(kp+2*Q,k)]
                                                                                     -Mp[k,0]  * bDb_i[ind_j(k-2*Q,kp)])
                                       +                   2*(Ln[k,1]  * ab_i[ind_j(k,kp)]
                                                                                     +Ln[kp,1] * ab_i[ind_j(k,kp)]
                                                                                     -Mp[kp,1] * aDa_i[ind_j(kp,k)]
                                                                                     -Mp[k,1]  * bDb_i[ind_j(k,kp)])
                                       +np.sin(2*omega_0*t)*( Ln[k,2]  * ab_r[ind_j(k+2*Q,kp)]             
                                                                                     +Ln[kp,0] * ab_r[ind_j(k,kp-2*Q)]
                                                                                     -Mp[kp,0] * aDa_r[ind_j(kp-2*Q,k)]
                                                                                     -Mp[k,2]  *(bDb_r[ind_j(k+2*Q,kp)]-kron1(kp-k,2*Q))
                                                                                     -Ln[k,0]  * ab_r[ind_j(k-2*Q,kp)]
                                                                                     -Ln[kp,2] * ab_r[ind_j(k,kp+2*Q)]
                                                                                     +Mp[kp,2] * aDa_r[ind_j(kp+2*Q,k)]
                                                                                     +Mp[k,0]  *(bDb_r[ind_j(k-2*Q,kp)]-kron1(kp-k,-2*Q)))))
    else: 
            RHS = 0
    return RHS


 # RHS for the ODE of for imaginary part of the ab_i 
def F_ab_i(k,kp,G, H,t):
    #k,kp are the indices of the NxN array NOT MOMENTA
    #G is the 6 x NxP array of unknown functions
    #H are time-dependent coefficent functions (R,C_r,C_i)
    # t is the time.
    #FOR Jordan

    #####
    # I modified ind_j() to check if its arguements are in the band, if not return (N,0).
    # Below I set the [N,0] arguement of all correlators to zero. This bypasses separating
    #  the function into if-else chunks.
    #####

    aDa_r,aDa_i,bDb_r,bDb_i,ab_r,ab_i = np.split(G,6) #unpack correlators and time dependent functions
    R, C_r,C_i = np.split(H,3)

    extraZeros = np.zeros((1,P)) #Create (1xP) array of zeroes to append to the aDa_r,...,ab_i matrices
    aDa_r = np.concatenate((aDa_r,extraZeros),axis=0)
    aDa_i = np.concatenate((aDa_i,extraZeros),axis=0)
    bDb_r = np.concatenate((bDb_r,extraZeros),axis=0)
    bDb_i = np.concatenate((bDb_i,extraZeros),axis=0)
    ab_r  = np.concatenate((ab_r,extraZeros),axis=0)
    ab_i  = np.concatenate((ab_i,extraZeros),axis=0)

    if k in range(0,N) and kp in range(0,N) and (kp-k+4) in range(0,P):
            RHS = (-(1/hbar)*(R[k]+R[kp])*ab_r[ind_j(k,kp)]
                    - (1/hbar)*(C_r[kp]*aDa_r[ind_j(kp,k)] - C_i[kp]*aDa_i[ind_j(kp,k)])
                    - (1/hbar)*(C_r[kp]*(bDb_r[ind_j(k,kp)]-kron1(k,kp)) - C_i[kp]*bDb_i[ind_j(k,kp)])
                    -    (1.0e18/(2*m))*2*k_y*A_r(t,q_0) *(Lp[k,1]  * ab_r[ind_j(k+Q,kp)]                      
                                                                                     -Lp[kp,0] * ab_r[ind_j(k,kp-Q)]
                                                                                     -Mn[kp,0] * aDa_r[ind_j(kp-Q,k)]
                                                                                     +Mn[k,1]  *(bDb_r[ind_j(k+Q,kp)]-kron1(kp-k,+Q))
                                                                                     +Lp[k,0]  * ab_r[ind_j(k-Q,kp)]
                                                                                     -Lp[kp,1] * ab_r[ind_j(k,kp+Q)]
                                                                                     -Mn[kp,1] * aDa_r[ind_j(kp+Q,k)]
                                                                                     +Mn[k,0]  *(bDb_r[ind_j(k-Q,kp)]-kron1(kp-k,-Q)))  
                    +    (1.0e18/(2*m))*2*k_y*A_i(t,q_0) *(Lp[k,1]  * ab_i[ind_j(k+Q,kp)]
                                                                                     -Lp[kp,0] * ab_i[ind_j(k,kp-Q)]
                                                                                     -Mn[kp,0] * aDa_i[ind_j(kp-Q,k)]
                                                                                     +Mn[k,1]  * bDb_i[ind_j(k+Q,kp)]
                                                                                     -Lp[k,0]  * ab_i[ind_j(k-Q,kp)]
                                                                                     +Lp[kp,1] * ab_i[ind_j(k,kp+Q)]
                                                                                     +Mn[kp,1] * aDa_i[ind_j(kp+Q,k)]
                                                                                     -Mn[k,0]  * bDb_i[ind_j(k-Q,kp)])
                    -    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)                      
                                     *( np.cos(2*omega_0*t)*( Ln[k,2]  * ab_r[ind_j(k+2*Q,kp)]                 
                                                                                     +Ln[kp,0] * ab_r[ind_j(k,kp-2*q_0)]
                                                                                     -Mp[kp,0] * aDa_r[ind_j(kp-2*Q,k)]
                                                                                     -Mp[k,2]  *(bDb_r[ind_j(k+2*Q,kp)]-kron1(kp-k,2*Q))
                                                                                     +Ln[k,0]  * ab_r[ind_j(k-2*Q,kp)]
                                                                                     +Ln[kp,2] * ab_r[ind_j(k,kp+2*q_0)]
                                                                                     -Mp[kp,2] * aDa_r[ind_j(kp+2*Q,k)]
                                                                                     -Mp[k,0]  *(bDb_r[ind_j(k-2*Q,kp)]-kron1(kp-k,-2*Q)))
                                       +                   2*(Ln[k,1]  * ab_r[ind_j(k,kp)]
                                                                                     +Ln[kp,1] * ab_r[ind_j(k,kp)]
                                                                                     -Mp[kp,1] * aDa_r[ind_j(kp,k)]
                                                                                     -Mp[k,1]  *(bDb_r[ind_j(k,kp)]-kron1(kp-k,0)))
                                       -np.sin(2*omega_0*t)*( Ln[k,2]  * ab_i[ind_j(k+2*Q,kp)]                
                                                                                     +Ln[kp,0] * ab_i[ind_j(k,kp-2*Q)]
                                                                                     -Mp[kp,0] * aDa_i[ind_j(kp-2*Q,k)]
                                                                                     -Mp[k,2]  * bDb_i[ind_j(k+2*Q,kp)]
                                                                                     -Ln[k,0]  * ab_i[ind_j(k-2*Q,kp)]
                                                                                     -Ln[kp,2] * ab_i[ind_j(k,kp+2*Q)]
                                                                                     +Mp[kp,2] * aDa_i[ind_j(kp+2*Q,k)]
                                                                                     +Mp[k,0]  * bDb_i[ind_j(k-2*Q,kp)])))
    else:
            RHS= 0
    return RHS

#########################################
# Contributions to jacobian from derivatives of F_ab_r 
#########################################


# wrt aDa_r(p,pp)
def F_ab_r_aDa_r(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
			
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement = ((1/hbar)*C_i[kp]*kron1(kp,p)*kron1(k,pp)
			+    (1.0e18/(2*m))*2*k_y*A_i(t,q_0)*(-Mn[kp,0] * kron1(kp-Q,p)*kron1(k,pp)
											 +Mn[kp,1] * kron1(kp_Q,p)*kron1(k,pp))
			+    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)
					 *  np.sin(2*omega_0*t)*(-Mp[kp,0] * kron1(kp-2*Q,p)*kron1(k,pp)
											 +Mp[kp,2] * kron1(kp+2*Q,p)*kron1(k,pp)))
	else:
		matrixElement = 0
	return matrixElement

	
# wrt aDa_i(p,pp)
def F_ab_r_aDa_i(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
		
		R, C_r, C_i = np.split(H,3)
			
		matrixElement = ((1/hbar)*C_r[kp]*kron1(kp,p)*kron1(k,pp)
			+    (1.0e18/(2*m))*2*k_y*A_r(t,q_0)*(-Mn[kp,0] * kron1(kp-Q,p)*kron1(k,pp)
											-Mn[kp,1] * kron1(kp+Q,p)*kron1(k,pp))
			+    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)
					*( np.cos(2*omega_0*t)*(-Mp[kp,0] * kron1(kp-2*Q,p)*kron1(k,pp)
											-Mp[kp,2] * kron1(kp+2*Q,p)*kron1(k,pp))
					  +                   -2*Mp[kp,1] * kron1(kp,p)*kron1(k,pp)))
	else:
		matrixElement = 0
	return matrixElement

	
# wrt bDb_r(p,pp)
def F_ab_r_bDb_r(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement = ((1/hbar)*C_i[kp]*kron1(k,p)*kron1(kp,pp)
			+    (1.0e18/(2*m))*2*k_y*A_i(t,q_0)*( Mn[k,1]  * kron1(k+Q,p)*kron1(kp,pp)
											 -Mn[k,0]  * kron1(k-Q,p)*kron1(kp,pp))
			+    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)
					 *np.sin(2*omega_0*t) * (-Mp[k,2]  * kron1(k+2*Q,p)*kron1(kp,pp)
											 +Mp[k,0]  * kron1(k-2*Q,p)*kron1(kp,pp)))
	else:
		matrixElement = 0
	return matrixElement
	

# wrt bDb_i(p,pp)
def F_ab_r_bDb_i(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement = ((1/hbar)*C_r[kp]*kron1(k,p)*kron1(kp,pp)
		    +  (1.0e18/(2*m))*2*k_y*A_r(t,q_0) * ( Mn[k,1]  * kron1(k+Q,p)*kron1(kp,pp)
											 +Mn[k,0]  * kron1(k-Q,p)*kron1(kp,pp))
		 	+  (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)
					*  (np.cos(2*omega_0*t)*(-Mp[k,2]  * kron1(k+2*Q,p)*kron1(kp,pp)
											 -Mp[k,0]  * kron1(k-2*Q,p)*kron1(kp,pp))
						-				    2*Mp[k,1]  * kron1(k,p)*kron1(kp,pp)))
	else:
		matrixElement = 0
	return matrixElement

	
# wrt ab_r(p,pp)
def F_ab_r_ab_r(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement = (
			   (1.0e18/(2*m))*2*k_y*A_i(t,q_0) * ( Lp[k,1]  * kron1(k+Q,p)*kron1(kp,pp)                   
											 -Lp[kp,0] * kron1(k,p)*kron1(kp-Q,pp)
											 -Lp[k,0]  * kron1(k-Q,p)*kron1(kp,pp)
											 +Lp[kp,1] * kron1(k,p)*kron1(kp+Q,pp))
		    +  (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)
					*   np.sin(2*omega_0*t)*( Ln[k,2]  * kron1(k+2*Q,p)*kron1(kp,pp)            
											 +Ln[kp,0] * kron1(k,p)*kron1(kp-2*Q,pp)
											 -Ln[k,0]  * kron1(k-2*Q,p)*kron1(kp,pp)
											 -Ln[kp,2] * kron1(k,p)*kron1(kp+2*Q,pp)))
	else:
		matrixElement = 0
	return matrixElement

	
# wrt ab_i(p,pp)
def F_ab_r_ab_i(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement = ((1/hbar)*(R[k]+R[kp])* kron1(k,p)*kron1(kp,pp)
			+  (1.0e18/(2*m))*2*k_y*A_r(t,q_0) * ( Lp[k,1]  * kron1(k+Q,p)*kron1(kp,pp)                   
											 -Lp[kp,0] * kron1(k,p)*kron1(kp-Q,pp)
											 +Lp[k,0]  * kron1(k-Q,p)*kron1(kp,pp)
											 -Lp[kp,1] * kron1(k,p)*kron1(kp+Q,pp))
			+  (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)
					*  (np.cos(2*omega_0*t)*( Ln[k,2]  * kron1(k+2*Q,p)*kron1(kp,pp)  
											 +Ln[kp,0] * kron1(k,p)*kron1(kp-2*Q,pp)
											 +Ln[k,0]  * kron1(k-2*Q,p)*kron1(kp,pp)
											 +Ln[kp,2] * kron1(k,p)*kron1(kp+2*Q,pp))
					    +                   2*Ln[k,1]  * kron1(k,p)*kron1(kp,pp)
						    			   +2*Ln[kp,1] * kron1(k,p)*kron1(kp,p)))
	else:
		matrixElement = 0
	return matrixElement

										 
#########################################
# Contributions to jacobian from derivatives of F_ab_i (in schematic notation on the LHS)
#########################################
# wrt aDa_r(p,pp)
def F_ab_i_aDa_r(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement = (-(1/hbar)*C_r[kp]*kron1(kp,p)*kron1(k,pp)
			-    (1.0e18/(2*m))*2*k_y*A_r(t,q_0)*(-Mn[kp,0] * kron1(kp-Q,p)*kron1(k,pp)
											 -Mn[kp,1] * kron1(kp+Q,p)*kron1(k,pp))  
			-    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)                      
					 *( np.cos(2*omega_0*t)*(-Mp[kp,0] * kron1(kp-2*Q,p)*kron1(k,pp)
											 -Mp[kp,2] * kron1(kp+2*Q,p)*kron1(k,pp))
					   -                    2*Mp[kp,1] * kron1(kp,p)*kron1(k,pp)))
	else:
		matrixElement = 0
	return matrixElement

				   
# wrt aDa_i(p,pp)
def F_ab_i_aDa_i(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement = (-(1/hbar)*(-C_i[kp]*kron1(kp,p)*kron1(k,pp))
			+    (1.0e18/(2*m))*2*k_y*A_i(t,q_0)*(-Mn[kp,0] * kron1(kp-Q,p)*kron1(k,pp)
											 +Mn[kp,1] * kron1(kp+Q,p)*kron1(k,pp))
			-    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)                      
					 *(-np.sin(2*omega_0*t)*(-Mp[kp,0] * kron1(kp-2*Q,p)*kron1(k,pp)
											 +Mp[kp,2] * kron1(kp+2*Q,p)*kron1(k,pp))))
	else:
		matrixElement = 0
	return matrixElement


# wrt bDb_r(p,pp)
def F_ab_i_bDb_r(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement = (- (1/hbar)*C_r[kp]*kron1(k,p)*kron1(kp,pp)
			-    (1.0e18/(2*m))*2*k_y*A_r(t,q_0)*( Mn[k,1]  * kron1(k+Q,p)*kron1(kp,pp)
											 +Mn[k,0]  * kron1(k-Q,p)*kron1(kp,pp))
			-    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)                      
					 *( np.cos(2*omega_0*t)*(-Mp[k,2]  * kron1(k+2*Q,p)*kron1(kp,pp)
											 -Mp[k,0]  * kron1(k-2*Q,p)*kron1(kp,pp)) 
					   -				    2*Mp[k,1]  * kron1(k,p)*kron1(kp,pp)))		
	else:
		matrixElement = 0
	return matrixElement	

	
# wrt bDb_i(p,pp)
def F_ab_i_bDb_i(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement =	(-(1/hbar)*(-C_i[kp]*kron1(k,p)*kron1(kp,pp))
			+    (1.0e18/(2*m))*2*k_y*A_i(t,q_0)*( Mn[k,1]  * kron1(k+Q,p)*kron1(kp,pp)
											 -Mn[k,0]  * kron1(k-Q,p)*kron1(kp,pp))
			-    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)                      
					 *(-np.sin(2*omega_0*t)*(-Mp[k,2]  * kron1(k+2*Q,p)*kron1(kp,pp)
											 +Mp[k,0]  * kron1(k-2*Q,p)*kron1(kp,pp))))	
	else:
		matrixElement = 0
	return matrixElement	


# wrt ab_r(p,pp)
def F_ab_i_ab_r(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement =	(-(1/hbar)*(R[k]+R[kp])*kron1(k,p)*kron1(kp,pp)
			-    (1.0e18/(2*m))*2*k_y*A_r(t,q_0) *(Lp[k,1]  * kron1(k+Q,p)*kron1(kp,pp)                  
											 -Lp[kp,0] * kron1(k,p)*kron1(kp-Q,pp)
											 +Lp[k,0]  * kron1(k-Q,p)*kron1(kp,pp)
											 -Lp[kp,1] * kron1(k,p)*kron1(kp+Q,pp)) 
			-    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)                      
					 *( np.cos(2*omega_0*t)*( Ln[k,2]  * kron1(k+2*Q,p)*kron1(kp,pp)             
											 +Ln[kp,0] * kron1(k,p)*kron1(kp-2*Q,pp)
											 +Ln[k,0]  * kron1(k-2*Q,p)*kron1(kp,pp)
											 +Ln[kp,2] * kron1(k,p)*kron1(kp+2*Q,pp))
					   +                    2*Ln[k,1]  * kron1(k,p)*kron1(kp,pp)
										   +2*Ln[kp,1] * kron1(k,p)*kron1(kp,pp)))
	else:
		matrixElement = 0
	return matrixElement	


# wrt ab_i(p,pp)
def F_ab_i_ab_i(k,kp,p,pp,G,H,t):
	if k in range(0,N) and kp in range(0,N) and p in range(0,N) and pp in range(0,N) and (kp-k+4) in range(0,P) and (pp-p+4) in range(0,P):
	
		R, C_r,C_i = np.split(H,3) # Unpack time dependent functions
			
		matrixElement =	(
				 (1.0e18/(2*m))*2*k_y*A_i(t,q_0) *(Lp[k,1]  * kron1(k+Q,p)*kron1(kp,pp)
											 -Lp[kp,0] * kron1(k,p)*kron1(kp-Q,pp)
											 -Lp[k,0]  * kron1(k-Q,p)*kron1(kp,pp)
											 +Lp[kp,1] * kron1(k,p)*kron1(kp+Q,pp))
			-    (1.0e18**2/(2*m*hbar))*A_0**2*(np.e)**(-8*np.log(2)*(t/tau)**2)                      
					 *(-np.sin(2*omega_0*t)*( Ln[k,2]  * kron1(k+2*Q,p)*kron1(kp,pp)            
											 +Ln[kp,0] * kron1(k,p)*kron1(kp-2*Q,pp)
											 -Ln[k,0]  * kron1(k-2*Q,p)*kron1(kp,pp)
											 -Ln[kp,2] * kron1(k,p)*kron1(kp+2*Q,pp))))
	else:
		matrixElement = 0
	return matrixElement
