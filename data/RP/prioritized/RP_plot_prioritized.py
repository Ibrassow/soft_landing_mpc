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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Pay attention to mismatched order
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

mx = X0[0]

m = max(X0[1], X0[2])
d = 5
ax.set_xlim(-m-d+200, m+d)
ax.set_ylim(-m-d, m+d-200)
ax.set_zlim(-5, mx)

ax.scatter(0,0,0,c='g',label="Target", marker='o', s=20)
ax.scatter(X0[1],X0[2],X0[0],c='r',label="Initial pos", marker='o', s=20)
ax.plot3D(Xs[1,:].flatten(), Xs[2,:].flatten(), Xs[0,:].flatten(), c='r', label="Minimum landing error")
ax.set_title("Soft Landing problem")


plt.legend(loc='upper left')
plt.show()