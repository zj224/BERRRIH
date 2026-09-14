import numpy as np


def fem_to_inertial(p, Xstates):
    """Port of femToInertial.m"""
    Xfem = Xstates[: p.ns, :]
    Xrigid = Xstates[p.ns: p.ns + 3, :]

    # fish model: first add the x-length of each finite element
    Xfish = Xfem.copy()
    Xfish[p.ix, :] = Xfish[p.ix, :] + p.xnode.reshape(-1, 1)

    # pull off the inertial rigid body coordinates (for simplicity)
    xkk = Xrigid[0, :]  # inertial frame
    ykk = Xrigid[1, :]  # inertial frame
    thk = Xrigid[2, :]

    # Find the rotated fish
    Xinertial = np.zeros_like(Xfish)
    Xinertial[p.ix, :] = Xfish[p.ix, :] * np.cos(thk) - Xfish[p.iy, :] * np.sin(thk) + xkk
    Xinertial[p.iy, :] = Xfish[p.ix, :] * np.sin(thk) + Xfish[p.iy, :] * np.cos(thk) + ykk
    Xinertial[p.ith, :] = Xfish[p.ith, :] + thk
    return Xinertial
