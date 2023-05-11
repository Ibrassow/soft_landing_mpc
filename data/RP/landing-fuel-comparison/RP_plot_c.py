from utils import *
import matplotlib.pyplot as plt

import numpy as np



filename_landing = "data/RP/landing-fuel-comparison/RP_mpc-landing_error_N48-dt1_disturb.csv"
filename_fuel = "data/RP/landing-fuel-comparison/RP_mpc-fuel_optimal_N48-dt1_disturb.csv"

[Xs, Us, time] = read_data_from_csv(filename_landing)
[Xs2, Us2, time] = read_data_from_csv(filename_fuel)

print(Xs.shape)
print(Us.shape)
print(time.shape)
print(Xs[:,-1])

X0 = Xs[:,0]

## Plotting

## Precomputations


## Mass
mass_landing = np.exp(Xs[6, :])
mass_fuel = np.exp(Xs2[6, :])
## Control norm
norm_ctrl_landing = np.linalg.norm(Us[:3, :]*mass_landing, axis=0)
norm_ctrl_fuel = np.linalg.norm(Us2[:3, :]*mass_fuel, axis=0)
## Thrust pointing
n = np.array([1, 0, 0])
theta_landing = np.rad2deg(np.arccos(Us[0,:]*mass_landing / norm_ctrl_landing))
theta_fuel = np.rad2deg(np.arccos(Us2[0,:]*mass_fuel / norm_ctrl_fuel))




print(theta_landing)


fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 10), sharex="col", sharey="row")

Tmax = 24 #kN
rho2 = 0.8*Tmax
rho1 = 0.2*Tmax

axs[0, 0].plot(norm_ctrl_landing/1000, label='_')
axs[0, 1].plot(norm_ctrl_fuel/1000, label='_')
axs[0, 0].set_ylabel('Thrust [kN]')

axs[0, 0].axhline(y=rho1, color='r', label="Minimum Thrust")
axs[0, 1].axhline(y=rho1, color='r', label="Minimum Thrust")
axs[0, 0].axhline(y=rho2, color='orange', label="Maximum Thrust")
axs[0, 1].axhline(y=rho2, color='orange', label="Maximum Thrust")
axs[0, 0].legend(loc='center left')
axs[0, 1].legend(loc='center left')



k = ["x", "y", "z"]
for i in range(3):
    axs[1, 0].plot(Xs[i, :], label=k[i])
    axs[1, 1].plot(Xs2[i, :], label=k[i])
axs[1, 0].set_ylabel('Position [m]')
axs[1, 0].legend(loc='upper right')
axs[1, 1].legend(loc='upper right')


axs[2, 0].plot(theta_landing, label='_', c="b")
axs[2, 1].plot(theta_fuel, label='_',c="b")
axs[2, 0].set_ylabel('Angle from vertical [°]')
axs[2, 0].axhline(y=70, linestyle='dotted', color='grey', label="Pointing angle limit 70°")
axs[2, 1].axhline(y=70, linestyle='dotted', color='grey', label="Pointing angle limit 70°")
axs[2, 0].legend(loc='center left')
axs[2, 1].legend(loc='center left')




axs[3, 0].plot(mass_landing, c='black', label='_')
axs[3, 1].plot(mass_fuel, c='black', label='_')
axs[3, 0].set_ylabel('Mass [kg]')
# Add x-axis labels to the bottom row of subplots
axs[3, 0].set_xlabel('Time [s]')
axs[3, 1].set_xlabel('Time [s]')



# Adjust the spacing between subplots
fig.tight_layout()

# Create a separate legend only for the second row
"""handles, labels = axs[1, 0].get_legend_handles_labels()
handles, labels = axs[2, 0].get_legend_handles_labels()
axs[1, 0].legend(handles, labels, loc='upper right')
axs[2, 0].legend(handles, labels, loc='upper right')"""

#plt.legend()
plt.show()



"""Assume you have two numpy array Xs and Xs2, both of shapes (7,48). As you might guess, each column of plot corresponds to each of them. Associated with them, you have Us and Us2 of shape (4, 48)

So now, on the first row of plots, I want their respective np.linalg.norm(Us[:3, :]) to be plotted. Same for Us2 on the right. 
On the second row of plots, I want their respective Xs[0,:], Xs[1,:], Xs[2,:] to be plotted. Same for Xs2. So I should see three curves in each of these two plots of the same row. Label each components with x,y, and z 
On the third row of plots, I want Xs[6,:] to be plotted on the left and Xs2[:6, :] on the right
All elements of column 1 share the same x axis that you will name "Time [s]". Same for the second column.

All elements in a same row share the same y axis 
1 st row axis label: "Thrust [N/kg]"
2nd row axis label: "State position [m]"
3rd row axis label: "Mass [kg]"""