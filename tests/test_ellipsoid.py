#!/usr/bin/env python
"""Tests for Ellipsoid class in the 'setfilter` package."""
import pytest
# math
import numpy as np
import numpy.linalg as la
import numpy.random as rand

@pytest.fixture
def base():
    """test Ellipsoid class """
    # build fixture
    from setfilter.ellipsoid import Ellipsoid
    center = [1., 2., 3.]
    body = np.array([[2, .50, .50], [.50, 1.0, .2], [.5, .2, 1.0]])
    return Ellipsoid(center, body)

def test_inside(base):
    center = np.array([1., 2., 3.])
    assert base.inside(center) , "center is not inside ellipsoid"

def test_sample(base):
    nTests = 100
    for i in range(nTests):
        sample = base.sample()
        assert base.inside(sample), "sample not inside ellipsoid"

def test_project(base):
    point = np.array((10, 10 ,10))
    proj = base.project(point)
    assert abs(base.dist(proj)) < 1e-6

def test_surfaceMap(base):
    nTests = 100
    # sample a bunch of points on the unit sphere to check if they are are on ellipsoid after surfaceMap
    for i in range(nTests):
        spherePt = rand.multivariate_normal(np.zeros((base.dim)), np.eye(base.dim))
        spherePt /= la.norm(spherePt)
        elipsPt = base.surfaceMap(spherePt)
        assert abs(base.dist(elipsPt)) < 1e-6

def test_normalMap(base):
    nTests = 100
    # sample a bunch of points on the unit sphere, get the normal, and move some small dist along and check distance
    for i in range(nTests):
        spherePt = rand.multivariate_normal(np.zeros((base.dim)), np.eye(base.dim))
        spherePt /= la.norm(spherePt)
        elipsPt = base.surfaceMap(spherePt)
        elipsNormal = base.normalMap((spherePt))
        step = rand.random()
        assert abs(base.dist(elipsPt+step*elipsNormal)-step) < 1e-6, i


@pytest.fixture
def sphereR2():
    """test Ellipsoid class """
    # build fixture
    # sphere centered at (1, 0, 0) with radius 2
    from setfilter.ellipsoid import Ellipsoid
    center = [1., 0., 0.]
    body = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]])
    return Ellipsoid(center, body)

def test_getAxis(sphereR2):
    semiAxis = sphereR2.getAxis()
    assert np.array_equal(semiAxis[0], [2., 0., 0.])
    assert np.array_equal(semiAxis[1], [0., 2., 0.])
    assert np.array_equal(semiAxis[2], [0., 0., 2.])

