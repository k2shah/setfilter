#!/usr/bin/env python
"""Tests for Ellipsoid class in the 'sfilter` package."""
import pytest
# math
import numpy as np


@pytest.fixture
def base():
    """test Ellipsoid class """
    # build fixture
    from sfilter.ellipsoid import Ellipsoid
    center = [1., 2., 3.]
    body = np.array([[2, .50, .50], [.50, 1.0, .2], [.5, .2, 1.0]])
    return Ellipsoid(center, body)

def test_inside(base):
    import numpy as np
    center = np.array([1., 2., 3.])
    print((center-base.center).T @ base.bodyInv @ (center-base.center))
    assert base.inside(center), "center is not inside ellipsoid"

def test_sample(base):
    nTests = 100
    for i in range(nTests):
        sample = base.sample()
        assert base.inside(sample), "sample not inside ellipsoid"

def test_project(base):
    point = np.array((10, 10 ,10))
    proj = base.project(point)
    assert abs(base.dist(proj)) < 1e-6

@pytest.fixture
def sphereR2():
    """test Ellipsoid class """
    # build fixture
    from sfilter.ellipsoid import Ellipsoid
    center = [0., 0., 0.]
    body = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]])
    return Ellipsoid(center, body)

def test_getAxis(sphereR2):
    semiAxis = sphereR2.getAxis()
    assert np.array_equal(semiAxis[0], [2., 0., 0.])
    assert np.array_equal(semiAxis[1], [0., 2., 0.])
    assert np.array_equal(semiAxis[2], [0., 0., 2.])
