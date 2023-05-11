import numpy as np
import scipy
import csv

def skew_symmetric(w):
    """
    Returns the skew symmetric form of a numpy array.
    w --> [w]
    """
    
    return np.array([[0, -w[2], w[1]], 
                     [w[2], 0, -w[0]], 
                     [-w[1], w[0], 0]])



def rk4(dynamics, x, u, dt):
    k1 = dt * dynamics(x, u)
    k2 = dt * dynamics(x + k1 * 0.5, u)
    k3 = dt * dynamics(x + k2 * 0.5, u)
    k4 = dt * dynamics(x + k3, u)
    x = x + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x


def dlqr(A, B, Q, R, N=0):
    """
    Computes the optimal gain matrix K such that the state-feedback law u[k] = - K x[k] minimizes
    the cost function J(u) = SUM(k=1:N)[ x[k]' Q x[k] + u[k]' R u[k] + 2 x[k] N x[k] ]
    """
    if N == 0:
        N = np.zeros((A.shape[0],B.shape[1]))
        
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    K = np.matrix(scipy.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A + N.T))
    return K



def save_data_to_csv(Xs, Us, dt, filename):
    """
    Given two numpy arrays Xs (size (6,N)) and Us (size (4,N-1))
    and timestamp dt as arguments saves them in a .csv file where
    the columns corresponds to each variable (time ,
    X[0], X[1], ... , U[0], .. , U[3])
    """
    # Get the number of rows in the arrays
    N = Xs.shape[1]
    # Create an array for the time column
    time = np.arange(N) * dt
    # Pad Us with zeros to match the number of rows in Xs
    Us = np.pad(Us, ((0, 0), (0, 1)), 'constant', constant_values=0)
    # Stack the arrays horizontally
    data = np.hstack((time.reshape(-1, 1), Xs.T, Us.T))
    # Create a CSV file and write the data to it
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Time'] + ['X{}'.format(i) for i in range(Xs.shape[0])] + ['U{}'.format(i) for i in range(Us.shape[0])])
        for row in data:
            writer.writerow(row)


def read_data_from_csv(filename):
    """
    Reads a CSV file with data in the format produced by the save_data_to_csv function,
    and returns a numpy array of numpy arrays of the data.
    """
    data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        num_cols = len(header)
        for row in reader:
            data_row = []
            for i in range(num_cols):
                if i == 0:
                    data_row.append(float(row[i]))
                else:
                    data_row.append(float(row[i]))
            data.append(data_row)
    data = np.array(data)
    Xs = data[:, 1:8].T
    Us = data[:, 8:].T
    return Xs, Us, data[:, 0]

"""Xs = np.random.rand(6, 100)
Us = np.random.rand(4, 99)
dt = 0.1
filename = 'data.csv'
save_data_to_csv(Xs, Us, dt, filename)
[Xs, Us, time] = read_data_from_csv(filename)
print(Xs.shape)
print(Us.shape)
print(time.shape)"""

def interpolate_xyz_traj(traj, dt=0.1):
    """
    Interpolates to a new time step.
    """
    from scipy.interpolate import interp1d
    # Compute the original time vector
    t_orig = np.linspace(0, traj.shape[1]-1, traj.shape[1])
    # Create a new time vector with the desired time step
    t_interp = np.linspace(0, traj.shape[1]-1, int(traj.shape[1]/dt))
    # Create an interpolation function for each component of the trajectory
    fx = interp1d(t_orig, traj[0,:], kind='linear')
    fy = interp1d(t_orig, traj[1,:], kind='linear')
    fz = interp1d(t_orig, traj[2,:], kind='linear')
    # Evaluate the interpolation functions at the new time vector
    traj_interp = np.vstack([fx(t_interp), fy(t_interp), fz(t_interp)])
    return traj_interp.T

def cubic_time_scaling(Tf, t):
    """
    a0 = a1 = a2 = 0, a3 = 10/T**3, a4 = 15/T**4, a5 = 6/T**5
    """
    s = 10 * (t/Tf)**3 - 15 * (t/Tf)**4 + 6 * (t/Tf)**5
    sdot = (30/Tf**3)*(t**2) - (60/Tf**4)*(t**3) + (30/Tf**5)*(t**4)
    sdotdot = (60/Tf**3) * t - (180/Tf**4)*(t**2) + (120/Tf**5)*(t**3)
    return s,sdot,sdotdot
