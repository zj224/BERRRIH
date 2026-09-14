import numpy as np


def get_miki(rho, l, E, r):
    """Port of getmiki.m

    Individual mass (mi) and stiffness (ki) matrices for a 2D beam
    finite element model (see Crawley, Campbell).
    """
    A = np.pi * r ** 2          # m^2  cross sectional area for circular rod
    I = 0.25 * np.pi * r ** 4    # m^4  area moment of inertia for rectangle

    mi = (rho * A * l / 420) * np.array([
        [140,    0,       0,    70,    0,       0],
        [0,    156,   22 * l,     0,   54,  -13 * l],
        [0, 22 * l, 4 * l ** 2,     0, 13 * l, -3 * l ** 2],
        [70,     0,       0,   140,    0,       0],
        [0,     54,   13 * l,     0,  156,  -22 * l],
        [0, -13 * l, -3 * l ** 2,     0, -22 * l, 4 * l ** 2],
    ], dtype=float)

    ki = (E * I / l ** 3) * np.array([
        [l ** 2 * A / I,   0,     0, -l ** 2 * A / I,    0,     0],
        [0,               12,   6 * l,              0,  -12,   6 * l],
        [0,             6 * l, 4 * l ** 2,           0, -6 * l, 2 * l ** 2],
        [-l ** 2 * A / I,  0,     0,  l ** 2 * A / I,    0,     0],
        [0,              -12,  -6 * l,              0,   12,  -6 * l],
        [0,             6 * l, 2 * l ** 2,           0, -6 * l, 4 * l ** 2],
    ], dtype=float)

    return mi, ki
