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
sl.set_thrust_pointing(50)
sl.set_vmax(90)
Ad, Bd, Dd = sl.set_dynamics(dt)


# Landing target on the y-z surface
landing_target = np.array([0,0,0]) 
#X0 = np.array([2400, 450,-330,-10,-40,10, np.log(m0)])
X0 = np.array([1840,1020,-650,-48,-11,7, np.log(m0)])



## Number of knot points
N = 48

Xs = []
Us = []

curr_X = X0

Xs.append(X0)


Ad, Bd, Dd = discrete_point_rocket_relaxed(dt)




def add_control_disturbances():
    w = np.zeros(4)
    # Thursters
    w[:3] = np.random.normal(0, 0.5, 3)
    return w



while N != 1:


    Xcp, Ucp = sl.solve_min_landing_error(curr_X, landing_target, N, verbose=False, solver='ECOS')
    Xcp, Ucp = sl.solve_fuel_optimal(curr_X, Xcp[:3, -1], landing_target, N, verbose=False, solver='ECOS')


    if N > 8:
        wu = add_control_disturbances()
        #w = add_state_disturbances()
    else:
        wu = np.zeros(4)
        #w = np.zeros(7)

    curr_X = Ad @ curr_X + Bd @ (Ucp[:,0] + wu) + Dd #+ w
    Xs.append(curr_X)
    Us.append(Ucp[:,0])
    print("CONTROL INPUT", Ucp[:,0])
    print("NEW STATE", curr_X)
    print("-----")

    N = N - 1
    print(N)




Xs = np.array(Xs).T
Us = np.array(Us).T


#save_data_to_csv(Xs, Us, dt, "data/RP/prioritized/RP_mpc-prioritized_N48-dt1_disturb88.csv")