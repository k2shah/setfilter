import numpy as np
import numpy.linalg as la

# a collection of utility functions

def inside(r1, r2):
    # returns True is r1 \in r2
    # assumes r1, r2 are 1D array-like
    # print r1, r2
    return r1[0] > r2[0] and r1[-1] < r2[-1]


def normalize(V):
    # normalizes a vector
    return V/la.norm(V)


def quadRoot(a, b, c):
    # solve for the set of x's | ax^2 + bx + c = 0
    if (b**2-4*a*c) < 0:
        raise ValueError("no real solution to quadratic")
    return [(-b+np.sqrt(b**2-4*a*c))/(2*a), (-b-np.sqrt(b**2-4*a*c))/(2*a)]


def trigRoot(a, b, c):
    # solves for t in [-pi, pi] | asin(t) + bcos(t) = c
    dem = (a**2 + b**2)**.5
    A = a/dem
    B = b/dem
    beta = np.arctan2(B, A)
    t = np.arcsin(c/dem) - beta

    # trig wrapping
    if t > np.pi/2:
        return np.pi - t
    if t < -np.pi/2:
        return -np.pi - t
    return t

def Rot2d(alpha):
    # returns a 2D rotation matrix where alpha is the angle in rad from the x-axis
    return np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha), np.cos(alpha)]])


def Rot3d(theta):
    # returns a 3D rotation matrix where theta is [roll, pitch, yaw] in rad
    R_x = np.array([[1,         0,                  0],
                    [0,         np.cos(theta[0]), -np.sin(theta[0])],
                    [0,         np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])],
                    [0,                   1,      0],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                   0,                    1]])
    return np.dot(R_x, R_y.dot(R_z))

