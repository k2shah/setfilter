"""Main module."""
# math
import numpy as np
import numpy.linalg as la
import numpy.random as rand

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


class Setfilter(object):
    def __init__(self, center, body,
                 processNoise, measurementNoise,
                 model=None, initNoise=True):
        # unpack dimension and noises
        self.dim = len(center)
        # 0 centered ellipsoids for the noise objects
        self.processNoise = Ellipsoid(np.zeros(self.dim), processNoise)
        self.measureNoise = Ellipsoid(np.zeros(self.dim), measurementNoise)
        # build model, default is the single integrator
        if model is None:
            self.model = LinearModel(
                np.eye(self.dim), np.eye(self.dim), np.eye(self.dim))
        else:
            self.model = model
        # initialize estimate object
        centerInit = self.measure(center) if initNoise else center  # mean
        self.estimate = Ellipsoid(centerInit, body)
        self.scale = 1.0  # scalar for the body metric, used by the update algo

    def measure(self, trueValue):
        """
        simulates a measurement

        produces a noise corrupted measurement by a sample drawn from the noise set
        :param trueValue: the true value of the item being estimated
        :return: trueValue plus the sampled noise
        """
        print()
        return self.measureNoise.sample() + trueValue

    def projectEstimate(self):
        """
        open loop projection of the estimate

        :return: projected estimate object Ellipsoid that takes into future size
        """

        A = self.model.A
        B = self.model.B
        C = self.model.C

        center_hat = self.estimate.center
        body_hat = self.estimate.body

        # minimal trace method
        p = np.sqrt(np.trace(A @ body_hat @ A.T)) + np.sqrt(np.trace(self.processNoise.body))
        p = np.sqrt(np.trace(self.processNoise.body))/p

        center_hat = A @ center_hat
        # intermediate body matrix
        return Ellipsoid(center_hat,
                         np.power(1-p, -1)*A @ body_hat @ A.T + np.power(p, -1) * self.processNoise.body)

    def addMargin(self, r, typ='ball'):
        """
        inflate the estimate with a user defined margin (for physical size)

        produces a outer bounding ellipsoid of the minsowski sum of epi and radius r ball
        :param r:
        :param typ: "ball": add's spherical margin of radius r
                    "elips: add an ellipsoidal margin
        :return: an inflated estimate Ellipsoid object with the added margin
        """
        margin = 0
        if typ == 'ball':
            if not isinstance(r, float):
                raise RuntimeError("margin type doesn't match argument")
            margin = np.eye(self.dim)
            margin *= r**2
        if typ == 'elip':
            margin = r

        p = np.sqrt(np.trace(self.scale*self.estimate.body)) + np.sqrt(np.trace(margin))
        p = np.sqrt(np.trace(margin))/p

        # expanded body matrix
        return Ellipsoid(self.estimate.center, np.power(1-p, -1)*self.estimate.body + np.power(self.scale*p, -1)*margin)

    def project(self, pt):
        """
        projects a point onto the estimate ellipsoid
        """
        return self.estimate.project(pt)

    def dist(self, pt):
        """
        finds the distance between the point pt and the estimate ellipsoid in the L2 Hausdorff sense
        """
        return self.estimate.dist(pt)

    def update(self, measurement):
        """
        implements a bounded set measurement update

        minimal trace method from Yushuang Liu ,Yan Zhao, Falin Wu
        :param measurement:
        :return: None. updates the internal estimate object
        """
        # reset scale
        self.scale = 1
        # unpack
        # model
        A = self.model.A
        B = self.model.B
        C = self.model.C
        # ellipsoid
        center = self.estimate.center
        body = self.estimate.body

        # minimal trace method
        p = np.sqrt(np.trace(self.scale*A @ body @ A.T)) + np.sqrt(np.trace(self.processNoise.body))
        p = np.sqrt(np.trace(self.processNoise.body))/p

        center_hat = A @ center
        # intermediate body matrix
        body_hat = np.power(1-p, -1)* A @ body @ A.T + np.power(self.scale*p, -1) * self.processNoise.body

        # measurement update
        # upper bound method
        delta = measurement-C.dot(center_hat)  # innovation
        CP_quad = C @ body_hat @ C.T  # C*P_hat*C.T
        V_bar = la.cholesky(self.measureNoise.bodyInv).T
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
            Q = np.power(L, -1)*self.measureNoise.body + np.power(1-L, -1) * CP_quad
            Q_inv = la.inv(Q)

        K = np.power(1-L, -1)*body_hat @ C.T @ Q_inv  # "kalman" gain
        self.scale = (1-L)*self.scale+L - delta.T @ Q_inv @ delta  # scale
        # update estimate
        self.estimate = Ellipsoid(center_hat + K @ delta,
                                  self.scale * np.power(1-L, -1)* (np.eye(self.dim)-K @ C) @ body_hat)
