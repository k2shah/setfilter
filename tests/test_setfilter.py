#!/usr/bin/env python
"""Tests for `sfilter` package."""
import pytest
import numpy as np
import numpy.random as rand



@pytest.fixture
def base():
    """test Setfilter """
    # build fixture
    from setfilter.setfilter import Setfilter

    # initialization
    center = np.array([0, 0, 0, 0.])
    body = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
    noise = np.eye(len(center))
    tracker = Setfilter(center, body,
                        noise*.2, noise*.1,
                        initNoise=False)

    return tracker


def test_update(base):
    nTests = 50
    for i in range(nTests):
        delta = base.processNoise.sample()
        print(delta)
        trueValue = np.array([0, 1, 2, 3.]) + delta
        measurement = base.measure(trueValue)
        base.update(measurement)
        print(base.estimate.dist((measurement)))
        assert base.estimate.inside(trueValue), i


