import numpy as np


def euclid_dist(x, mu) -> list:
    r = [np.sqrt(np.sum(np.square(c - x))) for c in mu]
    return r


def euclid_dist_square(x, mu) -> list:
    r = [np.sum(np.square(c - x)) for c in mu]
    return r


def manhattan_dist(x, mu) -> list:
    r = [np.sum(np.abs(c - x)) for c in mu]
    return r


def chebyshev_dist(x, mu) -> list:
    r = [np.max(np.abs(c - x)) for c in mu]
    return r


def power_dist(x, mu, r, p):
    d = [np.power(np.sum(np.power(c - x, p)), 1 / r) for c in mu]
    return d


dist_metrics = {
    'euclid': euclid_dist,
    'euclid_square': euclid_dist_square,
    'manhattan': manhattan_dist,
    'chebyshev': chebyshev_dist,
    'power': power_dist
}