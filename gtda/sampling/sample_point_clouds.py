"""Point-cloud–related plotting functions and classes."""
# License: GNU AGPLv3

import numpy as np
from scipy.optimize import root_scalar

from gtda.utils import validate_params
from gtda.utils.intervals import Interval


def sphere_sampling(n=1000, r=1, noise=0):
    """Uniformly and randomly samples points from a round
    sphere centered at the origin of 3-space.
    
    Parameters
    ----------
    n : int or None, optional, default: 1000
        The number of points to be sampled.

    r : float or None, optional, default: 1
        The radius of the sphere to be sampled from.
        Must be a positive number.

    noise : float or None, optional, default: 0
        The noise of the sampling, which is introduced by
        adding Gaussian noise around each data point.
        Must be a non-negative number.

    Returns
    -------
    points : ndarray of shape (n, 3).
        NumPy-array containing the sampled points.
    """

    validate_params({"n": n},
                    {"n": {"type": (int,),
                           "in": Interval(0, np.inf, closed="neither")}})
    validate_params({"r": r},
                    {"r": {"type": (int, float),
                           "in": Interval(0, np.inf, closed="neither")}})
    validate_params({"noise": noise},
                    {"noise": {"type": (int, float),
                               "in": Interval(0, np.inf, closed="left")}})

    def parametrization(theta, phi):
        x = r*np.cos(theta)*np.sin(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(phi)
        return np.array([x, y, z])
    U_phi_inverse = (lambda y: np.arccos(1-2*y))
    points = np.zeros(shape=(n, 3))
    for i in range(n):
        theta = np.random.uniform(low=0, high=2*np.pi)
        phi = U_phi_inverse(np.random.uniform())
        pt = parametrization(theta, phi)
        if noise:
            pt = pt + noise * np.random.randn(3)
        points[i] = pt
    return points


def torus_sampling(n=1000, R=3, r=1, noise=None):
    """Uniformly and randomly samples points from a torus
    centered at the origin of 3-space and lying in it
    horizontally.

    Parameters
    ----------
    n : int or None, optional, default: 1000
        The number of points to be sampled.

    R : float or None, optional, default: 3
        The inner radius of the torus to be sampled from,
        that is, the radius of the circle along which the
        ``tube`` follows.
        Must be a positive number.

    r : float or None, optional, default: 1
        The outer radius of the torus to be sampled from,
        that is, the radius of the ``tube``.
        Must be a positive number.

    noise : float or None, optional, default: 0
        The noise of the sampling, which is introduced by
        adding Gaussian noise around each data point.
        Must be a non-negative number.

    Returns
    -------
    points : ndarray of shape (n, 3).
        NumPy-array containing the sampled points.
    """

    validate_params({"n": n},
                    {"n": {"type": (int,),
                           "in": Interval(0, np.inf, closed="neither")}})
    validate_params({"R": R},
                    {"R": {"type": (int, float),
                           "in": Interval(0, np.inf, closed="neither")}})
    validate_params({"r": r},
                    {"r": {"type": (int, float),
                           "in": Interval(0, np.inf, closed="neither")}})
    validate_params({"noise": noise},
                    {"noise": {"type": (int, float),
                               "in": Interval(0, np.inf, closed="left")}})

    def parametrization(theta, phi):
        x = np.cos(theta)*(R + r*np.cos(phi))
        y = np.sin(theta)*(R + r*np.cos(phi))
        z = r*np.sin(phi)
        return np.array([x, y, z])
    U_phi = (lambda x: (0.5/np.pi)*(x + r*np.sin(x)/R))

    def U_phi_inverse(y):
        U_phi_shifted = (lambda x: U_phi(x) - y)
        sol = root_scalar(U_phi_shifted, bracket=[0, 2*np.pi])
        return sol.root
    points = np.zeros(shape=(n, 3))
    for i in range(n):
        theta = np.random.uniform(low=0, high=2*np.pi)
        phi = U_phi_inverse(np.random.uniform())
        pt = parametrization(theta, phi)
        if noise:
            pt = pt + noise * np.random.randn(3)
        points[i] = pt
    return points


def circle_sampling(n=1000, r=1, noise=None):
    """Uniformly and randomly samples points from a circle
    centered at the origin of 3-space and lying in it
    horizontally.

    Parameters
    ----------
    n : int or None, optional, default: 1000
        The number of points to be sampled.

    r : float or None, optional, default: 1
        The radius of the circle to be sampled from.
        Must be a positive number.

    noise : float or None, optional, default: 0
        The noise of the sampling, which is introduced by
        adding Gaussian noise around each data point.
        Must be a non-negative number.

    Returns
    -------
    points : ndarray of shape (n, 3).
        NumPy-array containing the sampled points.
    """

    validate_params({"n": n},
                    {"n": {"type": (int,),
                           "in": Interval(0, np.inf, closed="neither")}})
    validate_params({"r": r},
                    {"r": {"type": (int, float),
                           "in": Interval(0, np.inf, closed="neither")}})
    validate_params({"noise": noise},
                    {"noise": {"type": (int, float),
                               "in": Interval(0, np.inf, closed="left")}})

    def parametrization(theta):
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = 0
        return np.array([x, y, z])
    points = np.zeros(shape=(n, 3))
    for i in range(n):
        theta = np.random.uniform(low=0, high=2*np.pi)
        pt = parametrization(theta)
        if noise:
            pt = pt + noise * np.random.randn(3)
        points[i] = pt
    return points

