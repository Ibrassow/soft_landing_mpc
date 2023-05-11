import numpy as np
import os
from time import sleep
import meshcat

import meshcat.geometry as g
import meshcat.transformations as tf

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import quaternion

from utils import *

urdf_path = os.getcwd() + "/F5_.urdf"
mesh_dir = [os.path.dirname(urdf_path)]

# Load the urdf model
model, geom_model, visual_model = pin.buildModelsFromUrdf(urdf_path, package_dirs=mesh_dir)
scale = 0.01



data = model.createData()

body_name = "rocket_body"




print('model name: ' + model.name)

viz = MeshcatVisualizer(model, geom_model, visual_model)
viz.initViewer()
viz.loadViewerModel()



viz.viewer['/Cameras/default'].set_property('zoom', 0.5)
viz.viewer['/Cameras/default/rotated_top_down'].set_transform(tf.translation_matrix([0., 0., 2000.]))
# Set the grid size
# Set the grid visibility
viz.viewer["/Grid"].set_transform(tf.translation_matrix([0, 0, -0.33]))
viz.viewer["/Background"].set_property("grid", True)
viz.viewer["/Background"].set_property("grid_size", 2000)
viz.viewer["/Background"].set_property("top_color", [1, 0, 0])



body_name = "rocket_body"
body_id = model.getFrameId(body_name)


nv = model.nv
nq = model.nq

print("nv", nv)
print("nq", nq)



#Xs, Us, time = read_data_from_csv("mpc-N60-dt1.csv")
Xs, Us, time = read_data_from_csv("mpc-N48-dt1_disturb-Pres.csv")
# Permute columns Xs and Us for proper usage
Xs = Xs[[1, 2, 0, 4, 5, 3], :]*scale
Us = Us[[1, 2, 0, 3], :]*scale



from scipy.linalg import expm, norm
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

def thrust_to_quat(vec_3d):
    # Convert to axis-angle
    angle = np.linalg.norm(vec_3d)
    axis = vec_3d / angle
    ## NOTE TODO -  I'm making a big approximation here anyway
    ## Thrust vector ~≃ orientation is false obviously but ok for visu 
    axis[:2] = axis[:2]
    #Convert to quaternion (w, x,y,z) 
    q = np.zeros(4)
    q[1:] = axis * np.sin(angle * 0.5)
    q[0] = np.cos(angle * 0.5)
    q = q / np.linalg.norm(q)
    return q


q = np.array([0,0,0,0,0,0,1])
q[:3] = Xs[:3, 0]
print(q)
## Init 
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
#pin.updateGeometryPlacements(model,data,geom_model,geom_data,q)



dq = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
idx = 0


viz.display(q)


print("---")
print(model.frames[body_id].inertia)
sleep(5000)



### Interpolation 

"""k = interpolate_xyz_traj(Xs[:3, :])
print(k.shape)
print(k[:, -1])
print(Xs[:3, -1])
import matplotlib.pyplot as plt
#plt.plot(Xs[3:,:].T)
plt.plot(k.T)
plt.show()
sleep(500)"""



def interpolate_thrust_to_quat(Us, dt=0.1):
    qout = []
    print(Us.shape)
    for i in range(1, Us.shape[1]):
        prev_qs = thrust_to_quat(Us[:,i-1])
        qs = thrust_to_quat(Us[:,i])
        
        s = 0
        for h in range(int(1/dt)):
            s , _, _ = cubic_time_scaling(1, h*dt)
            qq = quaternion.slerp_evaluate(quaternion.as_quat_array(prev_qs), quaternion.as_quat_array(qs), s)
            qout.append(np.array([qq.x, qq.y, qq.z, qq.w])) ## for pin

        if (i == Us.shape[1]-1):
            prev_qs = qs
            qs = np.array([1.0,0.0,0.0,0.0])
            for h in range(int(1/dt)):
                s , _, _ = cubic_time_scaling(1, h*dt)
                qq = quaternion.slerp_evaluate(quaternion.as_quat_array(prev_qs), quaternion.as_quat_array(qs), s)
                qout.append(np.array([qq.x, qq.y, qq.z, qq.w])) ## for pin
    return np.array(qout)


"""k = interpolate_thrust_to_quat(Us[:3, :])
print(k.shape)
print(k)
import matplotlib.pyplot as plt
plt.plot(np.linalg.norm(k, axis=1))
plt.show()
sleep(500)"""

dt = 0.001
dt_interp = 0.01


XX = interpolate_xyz_traj(Xs[:3, :], dt=dt_interp)
QQ = interpolate_thrust_to_quat(Us[:3, :], dt=dt_interp)
N = int(XX.shape[0]/dt_interp)
print(N)
print(XX.shape)
print(QQ.shape)
print(XX)
print(QQ)






sleep(7)

while MeshcatVisualizer:

    #q[:3] = Xs[3:6, idx]
    #q = pin.integrate(model, q, 1 * dq)
    #print(XX[idx,:])
    q[:3] = XX[idx,:]
    q[3:] = QQ[idx, :]
    print(q[3:])
    #print(q)
    q = pin.integrate(model, q, dt * dq)
    # Update the robot model and visualize it
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)




    viz.display(q)
    sleep(dt)
    idx += 1

    if (idx == N):
        break

