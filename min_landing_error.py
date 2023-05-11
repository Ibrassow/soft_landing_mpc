import numpy as np
import matplotlib.pyplot as plt

import cvxpy as cp

from rocket_dynamics import point_rocket_relaxed, discrete_point_rocket_relaxed
from utils import rk4



# Mars -- From their 2011 paper "Lossless convexification of Powered-Descent Guidance with Non-Convex Thrust Bound and Pointing Constraints"
w = np.array([2.53e-5, 0, 6.62e-5])
g = np.array([-3.71,0,0]) 


## Number of knot points
N = 50
# Timestep
dt = 1

## Getting discretized matrices
Ad, Bd, Dd = discrete_point_rocket_relaxed(dt)


m0 = 2000
mf = 300
Vmax = 90
alpha = 5e-4


## Thrust variables
Tmax = 24000 # in N
Tmin = 0

# For constraints on slack variable
rho1 = 0.2 * Tmax
rho2 = 0.8 * Tmax

# Thust pointing angle
theta = 90 * np.pi / 180

# Direction vector in Tc-Tau space
n = np.array([1,0,0]) 

e1 = np.array([1,0,0])
e2 = np.array([0,1,0])
e3 = np.array([0,0,1])
E = np.array([e2,e3])

gamma_glideslope = 70 * np.pi / 180 # between 0 and pi/2
c = e1 / np.tan(gamma_glideslope)


#X0 = np.array([2400,3400,-554,-40,45,16,m0])
X0 = np.array([2400,450,-330,-10,-40,10, np.log(m0)])
X0 = np.array([1840,1020,-650,-48,-11,7, np.log(m0)])
#X0 = np.array([500,0,0,0,0,0, m0])


# Goal state
Xg = np.zeros(7)
# Landing target on the y-z surface
q = np.array([0,0]) 


X = cp.Variable(shape=(7,N))
U = cp.Variable(shape=(4,N-1)) # Slack variable Tau is the fourth input control variable


## Objective function

objective = cp.norm2(E @ X[:3,N-1] - q) 

objective = cp.Minimize(objective)



## Constraints
constraints = []


## Initial state constraint
constraints += [X[:, 0] == X0]

## Final state constraints

# Terminal mass
constraints += [np.log(m0/mf) <= X[6, N-1]] # do operation on ln since change of variables
constraints += [0 <= X[6, N-1]] 

# X position
constraints += [e1.T @ X[:3,N-1] == 0] ## basically x = 0

# Zero velocity at touchdown
constraints += [X[3:6, N-1] == np.zeros(3)]

# Needed for the convexification of the control inequalities following change of variables (see below)
z0 = np.zeros(N)
for i in range(len(z0)):
    z0[i] = np.log(m0 - alpha*rho2*(i)*dt)



for k in range(N-1):
    # Dynamics constraint
    constraints += [X[:, k+1] == rk4(point_rocket_relaxed, X[:, k], U[:, k], dt)]

    # Velocity constraint
    constraints += [ cp.norm2(X[3:6,k]) <= Vmax]

    # Glideslope constraint
    constraints += [cp.norm2(E @ (X[:3,k] - X[:3,N-1])) - c.T @ (X[:3,k] - X[:3,N-1]) <= 0]


    # Thrust and Slack variable constraints 
    """
    Context: the input Tc/m leads to nonlinearities in the dynamics so a change of variables is applied on the thrust vector and mass
    sigma = gamma / m, u = Tc / m, z = ln(m) 
    See paper for more details (A. Change of variables)
    - Sigma is U[3, k]
    """
    # Convex upper bound on thrust
    constraints += [cp.norm2(U[:3, k]) <= U[3, k]]

    # Convex lower bound on slack variable
    #constraints += [rho1 <= U[3, k]] #-- rho*e^(-z) approximated through a taylor series
    #constraints += [rho1 * cp.exp(-X[6,k]) <= U[3, k]]
    constraints += [rho1 * np.exp(-z0[k]) * (1 - (X[6,k] - z0[k])) + 0.5 * ((X[6,k] - z0[k]))**2 <= U[3, k]]

    # Convex upper bound on slack variable 
    #constraints += [U[3, k] <= rho2 ] 
    #constraints += [U[3, k] <= rho2 * cp.exp(-X[6,k])] #-- rho*e^(-z) approximated through a taylor series
    constraints += [U[3, k] <= rho2 * np.exp(-z0[k]) * (1 - (X[6,k] - z0[k]))]

    # Convex thrust pointing constraint
    constraints += [U[3, k] * np.cos(theta) <= n.T @ U[:3, k]]
    


cp.Problem(objective, constraints).solve(solver='ECOS', verbose=True)




Xcp = X.value
Ucp = U.value



## Simulation
Xs = np.zeros((7, N)) 
Xs[:,0] = X0
zeroU = np.zeros((4)) 



for step in range(N-1):
    #Xs[:, step+1] = rk4(point_rocket_relaxed, Xs[:, step], Ucp[:, step], dt)
    #Xs[:, step+1] = rk4(point_rocket_relaxed, Xs[:, step], zeroU, dt)
    Xs[:, step+1] = Ad @ Xs[:, step] + Bd @ Ucp[:, step] + Dd

print("Final time: ", N*dt)

print("Final location: ", Xs[:, -1])

## Plotting

"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Pay attention to mismatched order
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

mx = max(X0[2], X0[1])
ax.set_xlim(-mx-5, mx+5)
ax.set_ylim(-mx-5, mx+5)
ax.set_zlim(-5, X0[0])

ax.scatter(0,0,0,c='g',label="Target", marker='o', s=20)
ax.scatter(X0[1],X0[2],X0[0],c='r',label="Initial pos", marker='o', s=20)
ax.plot3D(Xs[1,:].flatten(), Xs[2,:].flatten(), Xs[0,:].flatten(), c='r', label="Rocket trajectory")
ax.set_title("Minimum Landing Error problem")"""


"""fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlabel('y')
ax2.set_ylabel('z')
ax2.set_zlabel('x') 
mx = max(X0[4], X0[5])
ax2.set_xlim(-mx-5, mx+5)
ax2.set_ylim(-mx-5, mx+5)
ax2.set_zlim(-X0[3]+2, X0[3]+2)
ax2.scatter(0,0,0,c='g',label="Target", marker='o', s=20)
ax2.scatter(X0[4],X0[5],X0[3],c='r',label="Initial vel", marker='o', s=20)
ax2.plot3D(Xs[4,:].flatten(), Xs[5,:].flatten(), Xs[6,:].flatten(), c='r', label="Rocket trajectory")"""

"""fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(np.exp(Xs[6,:]).flatten(), label="Rocket mass")"""

# Control inputs
"""fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.plot(Ucp[0,:].flatten(), c='b', label="z")
ax4.plot(Ucp[1,:].flatten(), c='r', label="y")
ax4.plot(Ucp[2,:].flatten(), c='g', label="x")
ax4.set_title("Minimum Landing Error problem - Thrust")"""


"""fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
ax6.plot(np.linalg.norm(Ucp, axis=0), c='b')
ax6.set_title("Minimum Landing Error problem - Thrust")"""



#Thrust pointing TODO 
"""fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.plot()"""


plt.legend()
plt.show()