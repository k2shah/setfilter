import numpy as np
import numpy.linalg as la
import numpy.random as rand
import cvxpy as cvx
from warnings import warn
from .utils import normalize


class Ellipsoid(object):
    """container for a generic ellipsoid object in any dimensions"""
    def __init__(self, center, body):
        # center of the ellipsoid
        self.center = np.array(center)  # np array of size n
        if not self.isPSD(body):
            warn("body matrix is not PSD, projecting to PSD cone")

        self.body = np.array(body)  # body matrix n by n assumes PSD
        # cache some values matrices
        self.dim = len(self.center)
        self.G = la.cholesky(self.body)  # cholesky deomp of the body matrix
        self.bodyInv = la.inv(self.body)  # save the inverse of the body matrix\\

    def isPSD(self, A):
        if np.array_equal(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def getAxis(self):
        # return the ellipsoid semi-axis
        L, T = la.eig(self.body)
        # print T
        semiAxis = np.array([np.power(li, .5)*ti for li, ti in zip(L, T)])
        return semiAxis

    def surfaceMap(self, d):
        # maps a point d on the unit sphere to the ellipsoid defined with self
        return self.G @ d + self.center

    def normalMap(self, d, normalized=True):
        # maps a point d on the unit sphere to it's normal on the ellipsoid
        normal = la.inv(self.G.T) @ d
        return normalize(normal) if normalized else normal

    def project(self, pt):
        # projects the point pt onto the ellipsoid defined by  x | (x-u)^T S^-1(x-u)=1
        assert len(pt) ** 2 == len(self.body.flatten())  # assert the dimensions are compatible
        cvx_x = cvx.Variable(self.dim)
        # objective
        obj = cvx.Minimize(cvx.norm(cvx_x - pt, 2))
        # constraints
        cnts = [cvx.quad_form(cvx_x - self.center, self.bodyInv) <= 1]
        # solve problem.
        prob = cvx.Problem(obj, cnts)
        prob.solve()
        # return projection
        return np.array(cvx_x.value).flatten()

    def dist(self, pt):
        # finds the distance between the point pt and the ellipsoid
        return la.norm(pt - self.project(pt))

    def inside(self, pt):
        # checks if the pt is inside the ellipsoid
        return (pt-self.center).T @ self.bodyInv @ (pt-self.center) < 1

    def sample(self):
        # return a random point inside the ellipsoid
        # https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        pt = rand.multivariate_normal(np.zeros((self.dim)), np.eye(self.dim))  # sample from n-dim normal
        pt *= (rand.random() ** (1/self.dim)) / la.norm(pt)
        return self.G @ pt + self.center
