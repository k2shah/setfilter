#!/usr/bin/env python
"""Tests for `sfilter` package."""
import pytest

from sfilter import sfilter


@pytest.fixture
def base():
    """test Sfitler """
    # build fixture
    import numpy as np
    import numpy.random as rand
    from sfilter.utils import normalize
    dim = 3



    # S=np.array([[1, 0, 0],[0, 2, .0 ], [0, 0, 2.0]])
    S = np.array([[2, .50, .50], [.50, 1.0, .2], [.5, .2, 1.0]])
    # S=np.eye(3)*4
    # S=np.eye(3)
    noise = np.eye(3)
    pts = 20
    estimate = EstimateBounded3D(u, S, noise, noise*.5)

    sfilter.Sfilter


def test_measure(base):
    self.fail()


def test_projectEstimate(base):
    self.fail()


def test_project(base):
    self.fail()


def test_addMargin(base):
    self.fail()


def test_dist(base):
    self.fail()


def test_elipMap(base):
    self.fail()


def test_normalMap(base):
    self.fail()


def test_update(base):
    self.fail()
