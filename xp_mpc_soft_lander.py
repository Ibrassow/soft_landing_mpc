import numpy as np
import matplotlib.pyplot as plt


from soft_lander import SoftLander
from rocket_dynamics import point_rocket_relaxed, discrete_point_rocket_relaxed
from utils import rk4, save_data_to_csv, read_data_from_csv


# Timestep
dt = 1
m0 = 2000
mf = 300


sl = SoftLander()
sl.set_thrust_limits(0, 24000)
sl.set_masses(m0, mf)
sl.set_glideslope(70)
sl.set_thrust_pointing(90)
sl.set_vmax(90)
Ad, Bd, Dd = sl.set_dynamics(dt)

# Landing target on the y-z surface
landing_target = np.array([0,0,0]) 
#X0 = np.array([2400,450,-330,-10,-40,10, np.log(m0)])
#X0 = np.array([2400,220,-47,-20,-18,24, np.log(m0)])
X0 = np.array([1840,1020,-650,-48,-11,7, np.log(m0)])


"""N = 48
Xcp, Ucp = sl.solve_min_landing_error(X0, landing_target, N, verbose=True, solver='MOSEK')
Xcp, Ucp = sl.solve_fuel_optimal(X0, Xcp[:3, -1], landing_target, N, verbose=False, solver='MOSEK')
save_data_to_csv(Xcp, Ucp, dt, "mpc-reference_tracking_dt1.csv")
from time import sleep
sleep(500)"""


"""
## Simulation
Xs = np.zeros((7, N)) 
Xs[:,0] = X0
for step in range(N-1):
    Xs[:, step+1] = rk4(point_rocket_relaxed, Xs[:, step], Ucp[:, step], dt)
"""


## Number of knot points
N = 48


Xs = []
Us = []

curr_X = X0

Xs.append(X0)


Ad, Bd, Dd = discrete_point_rocket_relaxed(dt)


rest_N = 16

def add_state_disturbances():
    w = np.zeros(7)
    # position less than 1m
    w[:3] = np.random.normal(0, 1, 3)
    # velocity - less than 1cm/sec 
    w[3:6] = np.random.normal(0, 1, 3)
    # mass
    unc_mass = 1
    unc_val = np.log( 1 + ( unc_mass / np.exp(X0[6]) )) ##Special case for mass
    w[6] = np.random.normal(0, unc_val, 1)
    return w


horizon = 8

while curr_X[0] > 0: #While we haven't reached the ground

    psolved = False
    cps = 0
    while not psolved:
        Xcp, Ucp = sl.solve_min_landing_error(curr_X, landing_target, N, verbose=True, solver='MOSEK', H=horizon)
        if Xcp is not None:
            psolved = True
            break
        elif cps == 5:
            break
        else:
            print("Resolving min landing")
            cps += 1

    #print("OPTIMAL MIN ERR:", Xcp[:3, -1])

    fsolved = False
    while not fsolved:
        if psolved:
            Xcp, Ucp = sl.solve_fuel_optimal(curr_X, Xcp[:3, -1], landing_target, N, verbose=False, solver='MOSEK', H=horizon)
        else:
            Xcp, Ucp = sl.solve_fuel_optimal(curr_X, landing_target, landing_target, N, verbose=False, solver='MOSEK', H=horizon)

        if Xcp is not None:
            fsolved = True
            break
        else:
            print("Resolving min fuel")

    w = add_state_disturbances()
    print(w)
    #curr_X = rk4(point_rocket_relaxed, curr_X, Ucp[:,0], dt) + w
    curr_X = Ad @ curr_X + Bd @ Ucp[:,0] + Dd + w
    Xs.append(curr_X)
    Us.append(Ucp[:,0])
    print("CONTROL INPUT", Ucp[:,0])
    print("NEW STATE", curr_X)
    print("-----")

    if (N > rest_N):
        N = N - 1
        print(N)
    else:
        #Solve once without the horizon 
        print("SOlving once again")
        Xcp, Ucp = sl.solve_min_landing_error(curr_X, landing_target, N, verbose=True, solver='MOSEK')
        Xcp, Ucp = sl.solve_fuel_optimal(curr_X, Xcp[:3, -1], landing_target, N, verbose=False, solver='MOSEK')


        #for i in range(1,rest_N-1):
        for i in range(len(Ucp.T)):

            #curr_X = rk4(point_rocket_relaxed, curr_X, Ucp[:,i], dt) #+ add_state_disturbances()
            curr_X = Ad @ curr_X + Bd @ Ucp[:,i] + Dd
            Xs.append(curr_X)
            Us.append(Ucp[:,i])

            print("CONTROL INPUT", Ucp[:,i])
            print("NEW STATE", curr_X)
            print("-----")
        
        break



print(Xcp.T)
print(Ucp.T)


Xs = np.array(Xs).T
Us = np.array(Us).T


#save_data_to_csv(Xs, Us, dt, "mpc-N48-dt1_disturb-Pres2.csv")


## Plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Pay attention to mismatched order
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

mx = X0[0]
ax.set_xlim(-mx-5, mx+5)
ax.set_ylim(-mx-5, mx+5)
ax.set_zlim(-5, mx)

ax.scatter(0,0,0,c='g',label="Target", marker='o', s=20)
ax.scatter(X0[1],X0[2],X0[0],c='r',label="Initial pos", marker='o', s=20)
ax.plot3D(Xs[1,:].flatten(), Xs[2,:].flatten(), Xs[0,:].flatten(), c='r', label="Rocket trajectory")
ax.set_title("Soft Landing problem")




plt.legend()
plt.show()