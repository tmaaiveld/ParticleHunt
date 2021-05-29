import numpy as np


def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.array([x, y])


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return np.array([r, theta])


def sum_polars(*args):
    return cart2pol(*sum([pol2cart(*v) for v in args]))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
