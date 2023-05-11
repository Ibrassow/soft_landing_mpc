from utils import *
import matplotlib.pyplot as plt

import numpy as np


filename = "data/RP/prioritized/RP_mpc-prioritized_N48-dt1_disturb.csv"
# Timestep
dt = 1
m0 = 2000
mf = 300

#X0 = np.array([1840,1020,-650,-48,-11,7, np.log(m0)])

[Xs, Us, time] = read_data_from_csv(filename)
print(Xs.shape)
print(Us.shape)
print(time.shape)
print(Xs[:,-1])

X0 = Xs[:,0]

## Plotting

## Precomputations


## Mass
mass_landing = np.exp(Xs[6, :])
## Control norm
norm_ctrl_landing = np.linalg.norm(Us[:3, :]*mass_landing, axis=0)
## Thrust pointing
n = np.array([1, 0, 0])
theta_landing = np.rad2deg(np.arccos(Us[0,:]*mass_landing / norm_ctrl_landing))





print(theta_landing)

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex="col", sharey="row")

Tmax = 24 #kN
rho2 = 0.8*Tmax
rho1 = 0.2*Tmax

axs[0].plot(norm_ctrl_landing/1000, label='_')
axs[0].set_ylabel('Thrust [kN]')

axs[0].axhline(y=rho1, color='r', label="Minimum Thrust")
axs[0].axhline(y=rho2, color='orange', label="Maximum Thrust")

axs[0].legend(loc='center')




k = ["x", "y", "z"]
for i in range(3):
    axs[1].plot(Xs[i, :], label=k[i])
axs[1].set_ylabel('Position [m]')
axs[1].legend(loc='upper right')


axs[2].plot(theta_landing, label='_', c="b")
axs[2].set_ylabel('Angle from vertical [°]')
axs[2].axhline(y=50, linestyle='dotted', color='grey', label="Pointing angle limit 50°")
axs[2].legend(loc='upper right')



axs[3].plot(mass_landing, c='black', label='_')
axs[3].set_ylabel('Mass [kg]')
# Add x-axis labels to the bottom row of subplots
axs[3].set_xlabel('Time [s]')


# Adjust the spacing between subplots
fig.tight_layout()
#plt.legend()
plt.show()