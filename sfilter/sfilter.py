"""Main module."""
# math
import numpy as np
import numpy.linalg as la
import numpy.random as rand
from scipy.linalg import qr
import cvxpy as cvx

# lib
from .ellipsoid import Ellipsoid
from .utils import *


class LinearModel(object):
    """basic container for a linear model"""
    def __init__(self, A, B, C, D=0):
        self.A = A  # state
        self.B = B  # control
        self.C = C  # measurement
        self.D = D  # feed forward


class Sfilter(object):
    def __init__(self, center, body,
                 processNoise, measurementNoise,
                 model=None, initNoise=True):
        # unpack dimension and noises
        self.dim = len(center)
        self.pNoise = processNoise
        self.mNoise = measurementNoise
        if model is None:
            self.model = LinearModel(
                np.eye(self.dim), np.eye(self.dim), np.eye(self.dim))
        else:
            self.model = model
        centerInit = self.measure(center) if initNoise else center  # mean
        self.estimate= Ellipsoid(centerInit, body)
        self.scale = 1.0  # scalar for the body metric, used by the update algo

        # cache
        self.G = la.cholesky(self.body)

    def measure(self, truePosition):
        # simulates a measurement, requires the true value of the item being estimated
        # returns a sample drawn from the bounded set v'Sm^-1v<1 where Sm is self.mNoise
        # hypersphere sample https://mathworld.wolfram.com/HyperspherePointPicking.html
        pt = rand.multivariate_normal(self.dim, np.eye(self.dim)) # sample from n-dim normal
        pt = normalize(pt)
        F = la.cholesky(self.mNoise)
        return F.dot(pt)+truePosition

    def projectEstimate(self):
        # returns a projected estimate object that takes into future size

        A = self.model.A
        B = self.model.B
        C = self.model.C

        center_hat = self.elip.center
        body_hat = self.elip.body

        # minimal trace method
        p = np.sqrt(np.trace(A.dot(body_hat.dot(A.T)))
                    ) + np.sqrt(np.trace(self.pNoise))
        p = np.sqrt(np.trace(self.pNoise))/p

        center_hat = np.dot(A, center_hat)
        # intermediate body matrix
        return  np.power(1-p, -1)*A.dot(body_hat.dot(A.T)) + np.power(p, -1) * self.pNoise

    def project(self, pt):
        # projects a point onto the ellipsoid
        return self.elip.project(pt)

    def addMargin(self, r, typ='ball'):
        ''' returns an the body of an ellipsoid that contains the estimate
        with a margin of at least r '''
        # outer bounding ellipsoid of the minsowski sum of epi and radius r ball
        margin = 0
        if typ == 'ball':
            if not isinstance(r, float):
                raise RuntimeError("margin type doesn't match argument")
            margin = np.eye(self.dim)
            margin *= r**2
        if typ == 'elip':
            margin = r

        p = np.sqrt(np.trace(self.scale*self.body)) + np.sqrt(np.trace(margin))
        p = np.sqrt(np.trace(margin))/p

        # intermediate body matrix
        return np.power(1-p, -1)*self.body + np.power(self.scale*p, -1)*margin

    def dist(self, pt):
        # finds the distance between the point pt and the estimate ellipsoid
        return la.norm(pt - self.project(pt))

    def elipMap(self, d):
        # maps a point d on the unit sphere to the ellipsoid defined with self
        return self.G.dot(d)+self.center

    def normalMap(self, d):
        # maps a point d on the unit sphere to it's normal on the ellipsoid
        return np.dot(la.inv(self.G.T), d)

    def update(self, measurement):
        # implements some sort of bounded set measurement update
        # method from Yushuang Liu ,Yan Zhao, Falin Wu
        # unpack
        self.scale = 1
        # model
        A = self.model.A
        B = self.model.B
        C = self.model.C
        # ellipsoid
        center = self.elip.center
        body = self.elip.body

        # minimal trace method
        p = np.sqrt(np.trace(self.scale*A.dot(self.body.dot(A.T)))) + \
            np.sqrt(np.trace(self.pNoise))
        p = np.sqrt(np.trace(self.pNoise))/p

        center_hat = np.dot(A, self.center)
        body_hat = np.power(1-p, -1)*A.dot(self.body.dot(A.T)) + \
            np.power(self.scale*p, -1)*self.pNoise  # intermediate body matrix

        # measurement update
        # upper bound method
        delta = measurement-C.dot(center_hat)  # innovation
        CP_quad = C.dot(body_hat.dot(C.T))  # C*P_hat*C.T
        V_bar = la.cholesky(la.inv(self.mNoise)).T
        delta_bar = V_bar.dot(delta)
        G = V_bar.dot(CP_quad.dot(V_bar.T))
        val, vec = la.eig(G)
        g = max(val)

        beta = (1.0-self.scale)/la.norm(delta_bar)
        # mixing parameter for measurement
        if self.scale+la.norm(delta_bar) <= 1.0:
            L = 0.0
        elif g == 1:
            L = (1.0-beta)/2.0
        else:
            L = 1.0/(1.0-g) * (1.-np.sqrt(g/(1.+beta*(g-1.))))
        # print l
        if L == 0:
            # handle the edge case
            Q_inv = np.zeros((self.dim, self.dim))
        else:
            Q = np.power(L, -1)*self.mNoise+np.power(1-L, -1) * \
                CP_quad  # intermediate thing
            Q_inv = la.inv(Q)

        K = np.power(1-L, -1)*body_hat.dot(np.dot(C.T, Q_inv))  # "kalman" gain
        self.scale = (1-L)*self.scale+L-np.dot(delta.T,
                                               Q_inv.dot(delta))  # scale
        # print("scale dec: ", self.scale)
        self.body = self.scale * np.power(1-L, -1)*np.dot(np.eye(self.dim)-K.dot(C), body_hat)
        self.center = center_hat+K.dot(delta)
        self.G = la.cholesky(self.body)
