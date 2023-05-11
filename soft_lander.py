
import numpy as np
import matplotlib.pyplot as plt

import cvxpy as cp

import time  
from rocket_dynamics import point_rocket_relaxed, discrete_point_rocket_relaxed
from utils import rk4


class SoftLander:


    def __init__(self):
        # Mars -- From their 2011 paper "Lossless convexification of Powered-Descent Guidance with Non-Convex Thrust Bound and Pointing Constraints"
        self.w = np.array([2.53e-5, 0, 6.62e-5])
        self.g = np.array([-3.71,0,0]) 

        ## Default values
        self.m0 = 2000
        self.mf = 300
        self.Vmax = 90
        self.alpha = 5e-4


        ## Thrust variables
        self.Tmax = 24000 # in N
        self.Tmin = 0

        # For constraints on slack variable
        self.rho1 = 0.2 * self.Tmax
        self.rho2 = 0.8 * self.Tmax

        # Thust pointing angle
        self.theta = 120 * np.pi / 180

        # Direction vector in Tc-Tau space
        self.n = np.array([1,0,0]) 

        self.e1 = np.array([1,0,0])
        self.e2 = np.array([0,1,0])
        self.e3 = np.array([0,0,1])
        self.E = np.array([self.e2,self.e3])

        self.gamma_glideslope = 45 * np.pi / 180 
        self.c = self.e1 / np.tan(self.gamma_glideslope)

        self.Ad, self.Bd, self.Dd, self.dt = None, None, None, None


        ## CUrrent solution 
        # min landing error
        self.Xle = None
        self.Ule = None
        self.current_target = None
        self.X0_le = None
        # fuel optimal
        self.Xfo = None
        self.Ufo = None

    def set_vmax(self, vmax):
        self.Vmax = vmax


    def set_thrust_limits(self, tmin, tmax):
        ## Thrust variables
        self.Tmax = tmax # in N
        self.Tmin = tmin

        ## TODO remove that just Tmax

        # For constraints on slack variable
        self.rho1 = 0.2 * self.Tmax
        self.rho2 = 0.8 * self.Tmax


    def set_masses(self, m0, mf):
        self.m0 = m0
        self.mf = mf


    def set_dynamics(self, dt, w=np.array([2.53e-5, 0, 6.62e-5]), g = np.array([-3.71,0,0]), alpha = 5e-4):
        self.w = w
        self.g = g
        self.dt = dt
        ## Getting discretized matrices
        self.Ad, self.Bd, self.Dd = discrete_point_rocket_relaxed(dt)
        return self.Ad, self.Bd, self.Dd


    def set_glideslope(self, deg_angle):
        self.gamma_glideslope = deg_angle * np.pi / 180 # between 0 and pi/2
        self.c = self.e1 / np.tan(self.gamma_glideslope)

    def set_thrust_pointing(self, deg_angle):
        # Thust pointing angle
        self.theta = deg_angle * np.pi / 180

    def solve_min_landing_error(self, X0, landing_target, N, verbose=True, solver='MOSEK', H = None):
        """
        landing_target: (x,y,z) with y,z being the plane coordinate
        H: Horizon (must be < N-1)
        """
        
        if H == None:
            H = N - 1 
            
        X = cp.Variable(shape=(7,N))
        U = cp.Variable(shape=(4,N-1)) # Slack variable Tau is the fourth input control variable
        ## Objective function
        objective = cp.norm2(self.E @ X[:3,N-1] - landing_target[1:]) 
        objective = cp.Minimize(objective)
        ## Constraints
        constraints = []
        ## Initial state constraint
        constraints += [X[:, 0] == X0]
        # Terminal mass
        constraints += [np.log(self.m0/self.mf) <= X[6, N-1]] # do operation on ln since change of variables
        constraints += [0 <= X[6, N-1]] 
        # X position
        constraints += [self.e1.T @ X[:3,N-1] == 0] ## basically x = 0
        # Zero velocity at touchdown
        constraints += [X[3:6, N-1] == np.zeros(3)]
        # Needed for the convexification of the control inequalities following change of variables (see below)
        z0 = np.zeros(N)
        for i in range(len(z0)):
            z0[i] = np.log(np.exp(X0[6]) - self.alpha*self.rho2*(i)*self.dt) ## exp because X0 contains the mass in log form

        for k in range(H):
            # Dynamics constraint
            #constraints += [X[:, k+1] == rk4(point_rocket_relaxed, X[:, k], U[:, k], self.dt)]
            constraints += [X[:, k+1] == self.Ad @ X[:, k] + self.Bd @ U[:, k] + self.Dd]
            # Velocity constraint
            constraints += [ cp.norm2(X[3:6,k]) <= self.Vmax]
            # Glideslope constraint
            constraints += [cp.norm2(self.E @ (X[:3,k] - X[:3,N-1])) - self.c.T @ (X[:3,k] - X[:3,N-1]) <= 0]
            # Convex upper bound on thrust
            constraints += [cp.norm2(U[:3, k]) <= U[3, k]]
            # Convex lower bound on slack variable
            constraints += [self.rho1 * np.exp(-z0[k]) * (1 - (X[6,k] - z0[k])) + 0.5 * ((X[6,k] - z0[k]))**2 <= U[3, k]]
            # Convex upper bound on slack variable 
            constraints += [U[3, k] <= self.rho2 * np.exp(-z0[k]) * (1 - (X[6,k] - z0[k]))]
            # Convex thrust pointing constraint
            constraints += [U[3, k] * np.cos(self.theta) <= self.n.T @ U[:3, k]]


            ## Me adding this 
            #constraints += [self.e1.T @ X[:3,k] >= 0]
        
        cp.Problem(objective, constraints).solve(solver=solver, verbose=verbose)
        Xcp = X.value
        Ucp = U.value
        return Xcp, Ucp





    def solve_fuel_optimal(self, X0, d_opt, landing_target, N, verbose=True, solver='MOSEK', H = None):
        """
        d_opt: finak optimal position from landing_error - x,y,z (y,z is the ground plane)
        H: Horizon (must be < N-1)
        """

        if H == None:
            H = N - 1 
        

        X = cp.Variable(shape=(7,N))
        U = cp.Variable(shape=(4,N-1)) # Slack variable Tau is the fourth input control variable
        ## Objective function
        objective = 0
        for k in range(H):
            objective += U[3, k]
        objective = cp.Minimize(objective)
        ## Constraints
        constraints = []
        constraints += [cp.norm2(self.E @ X[:3,N-1] - landing_target[1:]) <= cp.norm2(np.array(d_opt[1:]) - landing_target[1:])]
        ## Initial state constraint
        constraints += [X[:, 0] == X0]
        # Terminal mass
        constraints += [np.log(self.m0/self.mf) <= X[6, N-1]] # do operation on ln since change of variables
        constraints += [0 <= X[6, N-1]] 
        # X position
        constraints += [self.e1.T @ X[:3,N-1] == 0] ## basically x = 0
        # Zero velocity at touchdown
        constraints += [X[3:6, N-1] == np.zeros(3)]
        # Needed for the convexification of the control inequalities following change of variables (see below)
        z0 = np.zeros(N)
        for i in range(len(z0)):
            z0[i] = np.log(np.exp(X0[6]) - self.alpha*self.rho2*(i)*self.dt)

        for k in range(H):
            # Dynamics constraint
            #constraints += [X[:, k+1] == rk4(point_rocket_relaxed, X[:, k], U[:, k], self.dt)]
            constraints += [X[:, k+1] == self.Ad @ X[:, k] + self.Bd @ U[:, k] + self.Dd]
            # Velocity constraint
            constraints += [ cp.norm2(X[3:6,k]) <= self.Vmax]
            # Glideslope constraint
            constraints += [cp.norm2(self.E @ (X[:3,k] - X[:3,N-1])) - self.c.T @ (X[:3,k] - X[:3,N-1]) <= 0]
            # Convex upper bound on thrust
            constraints += [cp.norm2(U[:3, k]) <= U[3, k]]
            # Convex lower bound on slack variable
            constraints += [self.rho1 * np.exp(-z0[k]) * (1 - (X[6,k] - z0[k])) + 0.5 * ((X[6,k] - z0[k]))**2 <= U[3, k]]
            # Convex upper bound on slack variable 
            constraints += [U[3, k] <= self.rho2 * np.exp(-z0[k]) * (1 - (X[6,k] - z0[k]))]
            # Convex thrust pointing constraint
            constraints += [U[3, k] * np.cos(self.theta) <= self.n.T @ U[:3, k]]
            
            ## Me adding this 
            #constraints += [self.e1.T @ X[:3,k] >= 0]


        cp.Problem(objective, constraints).solve(solver=solver, verbose=verbose)
        Xcp = X.value
        Ucp = U.value
        return Xcp, Ucp
    


    def solve_quad_tracking(self, X0, Xref, Uref, N, Q = None, R = None, Qf = None, verbose=False, solver='MOSEK'):
        """
        N is the horizon of the MPC 
        """

        # Cost matrices 
        if Q == None:
            Q = np.eye(7) 
        if R == None:
            R = np.eye(4)
        if Qf == None:
            Qf = Q 
        
        X = cp.Variable(shape=(7,N))
        U = cp.Variable(shape=(4,N-1)) # Slack variable Tau is the fourth input control variable

        ## Objective function - Quadratic tracking cost
        objective = 0
        for k in range(N-1):
            objective += 0.5 * cp.quad_form(X[:, k] - Xref[:, k], Q) + 0.5 * cp.quad_form(U[:, k] - Uref[:, k], R)
        objective += 0.5 * cp.quad_form(X[:, N-1] - Xref[:, N-1], Qf)
        objective = cp.Minimize(objective)

        ## Constraints
        constraints = []

        ## Initial state constraint
        constraints += [X[:, 0] == X0]
        ## Terminal state of the horizon
        constraints += [X[:, N-1] == Xref[:, N-1]]

        # Terminal mass
        constraints += [np.log(self.m0/self.mf) <= X[6, N-1]] # do operation on ln since change of variables
        constraints += [0 <= X[6, N-1]] 

        # Needed for the convexification of the control inequalities following change of variables (see below)
        z0 = np.zeros(N)
        for i in range(len(z0)):
            z0[i] = np.log(np.exp(X0[6]) - self.alpha*self.rho2*(i)*self.dt)


        for k in range(N-1):
            # Dynamics constraint
            #constraints += [X[:, k+1] == rk4(point_rocket_relaxed, X[:, k], U[:, k], self.dt)]
            constraints += [X[:, k+1] == self.Ad @ X[:, k] + self.Bd @ U[:, k] + self.Dd]
            # Velocity constraint
            constraints += [ cp.norm2(X[3:6,k]) <= self.Vmax]
            # Convex upper bound on thrust
            #constraints += [cp.norm2(U[:3, k]) <= U[3, k]]
            # Convex lower bound on slack variable
            #constraints += [self.rho1 * np.exp(-z0[k]) * (1 - (X[6,k] - z0[k])) + 0.5 * ((X[6,k] - z0[k]))**2 <= U[3, k]]
            # Convex upper bound on slack variable 
            #constraints += [U[3, k] <= self.rho2 * np.exp(-z0[k]) * (1 - (X[6,k] - z0[k]))]
            # Convex thrust pointing constraint
            constraints += [U[3, k] * np.cos(self.theta) <= self.n.T @ U[:3, k]]

            ## Me
            #constraints += [X[0,k+1] >= 0]


        cp.Problem(objective, constraints).solve(solver=solver, verbose=verbose)
        Xcp = X.value
        Ucp = U.value
        return Xcp, Ucp
    
