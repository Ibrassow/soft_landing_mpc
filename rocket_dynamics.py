import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import expm
import cvxpy as cp

from utils import skew_symmetric

def point_rocket(x, u, w=[1,1,1]):
    """
    w: constant angular velocity of the planet (w1, w2, w3)
    g: uniform gravity vector (g1, g2, g3)
    
    """
    A = np.zeros((6,6))
    A[:3,3:] = np.eye(3)
    A[3:,:3] = - skew_symmetric(w) @ skew_symmetric(w)
    A[3:,3:] = - 2 * skew_symmetric(w)

    B = np.zeros((6,3))
    B[3:,:] = np.eye(3)

    xdot = A @ x + B @ u


    return xdot



def point_rocket_dynamics(w, alpha=5e-4):
    """
    w: constant angular velocity of the planet (w1, w2, w3)
    g: uniform gravity vector (g1, g2, g3)
    
    """
    A = np.zeros((6,6))
    A[:3,3:] = np.eye(3)
    A[3:6,:3] = - skew_symmetric(w) @ skew_symmetric(w)
    A[3:6,3:] = - 2 * skew_symmetric(w)

    B = np.zeros((6,3))
    B[3,:] = 1
    B[4,:] = 1
    B[5,:] = 1


    return A, B

   
def get_discrete_dynamics(Ac, Bc, dt):
   
    an = Ac.shape[1]
    bn = Bc.shape[1]
   
    dd = np.zeros((an+bn,an+bn))
    dd[0:an,0:an] = Ac
    dd[0:an,an:an+bn] = Bc
    exp_mx = expm(dd*dt)

    Ad = exp_mx[0:an,0:an]
    Bd = exp_mx[0:an,an:an+bn]

    return Ad, Bd




def point_rocket_relaxed(x, u, w=np.array([2.53e-5, 0, 6.62e-5]), g = np.array([-3.71,0,0]), alpha = 5e-4):
    """
    w: constant angular velocity of the planet (w1, w2, w3)
    g: uniform gravity vector (g1, g2, g3)
    alpha: mass depletion constant
    """
    A = np.zeros((7,7))
    A[:3,3:6] = np.eye(3)
    A[3:6,:3] = - skew_symmetric(w) @ skew_symmetric(w)
    A[3:6,3:6] = - 2 * skew_symmetric(w)

    mB = np.zeros((6,3))
    mB[3:,:] = np.eye(3)

    B = np.zeros((7,4))
    B[:6,:3] = mB
    B[6,3] = -alpha

    D = np.zeros((7,3))
    D[:6,:3] = mB


    #m = x[-1]

    xdot = A @ x + B @ u + D @ g

    return xdot
    


def discrete_point_rocket_relaxed(dt, w=np.array([2.53e-5, 0, 6.62e-5]), g = np.array([-3.71,0,0]), alpha = 5e-4):
    """
    w: constant angular velocity of the planet (w1, w2, w3)
    g: uniform gravity vector (g1, g2, g3)
    alpha: mass depletion constant
    """
    ## Continuous dynamics

    A = np.zeros((7,7))
    A[:3,3:6] = np.eye(3)
    A[3:6,:3] = - skew_symmetric(w) @ skew_symmetric(w)
    A[3:6,3:6] = - 2 * skew_symmetric(w)
    
    mB = np.zeros((6,3))
    mB[3:,:] = np.eye(3)

    B = np.zeros((7,4))
    B[:6,:3] = mB
    B[6,3] = -alpha

    D = np.zeros((7,3))
    D[:6,:3] = mB
    
    
    ## Getting discretized matrices through the matrix exponential

    an = A.shape[1]
    bn = B.shape[1]
    dn = D.shape[1]
   
    M = np.zeros((an+bn+dn,an+bn+dn))

    M[0:an,0:an] = A
    M[0:an,an:an+bn] = B
    M[0:an, an+bn:an+bn+dn] = D
    exp_mx = expm(M*dt)

    Ad = exp_mx[0:an,0:an]
    Bd = exp_mx[0:an,an:an+bn]
    Dd = exp_mx[0:an,an+bn:an+bn+dn] @ g

    return Ad, Bd, Dd


