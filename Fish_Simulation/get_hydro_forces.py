import numpy as np


def get_hydro_forces(th, Vx, Vy, p):
    """Port of getHydroForces.m, specialized to the shape it's actually
    called with (a single time-step column of nn values). The original
    MATLAB version supported an arbitrary (sx,sy) shape via reshape/repmat
    tricks — a vectorized-over-time-columns calling convention that this
    codebase never uses (simulate_sys only ever calls it one step at a
    time), so that machinery is dropped in favor of plain 1-D array math.

    th, Vx, Vy: 1-D or (nn,1) arrays of node angle / velocities.
    Assumes hydro forces only act perpendicular to the link,
    F = -uw*sign(vperp)*(vperp)^2, plus an extra force at the head tip.

    Returns an (2*nn, 1) array: stacked [Fx; Fy], matching Ghd's expected
    input layout.
    """
    th = th.flatten()
    Vx = Vx.flatten()
    Vy = Vy.flatten()

    # side forces (perpendicular to each link)
    eth = np.column_stack([-np.sin(th), np.cos(th)])
    Vmagperp = Vx * eth[:, 0] + Vy * eth[:, 1]
    Fxy = -np.sign(Vmagperp)[:, None] * (Vmagperp ** 2)[:, None] * eth
    Fx = p.uwvec * Fxy[:, 0]
    Fy = p.uwvec * Fxy[:, 1]

    # add head tip forces
    er = np.array([np.cos(th[-1]), np.sin(th[-1])])
    VmagperpHead = Vx[-1] * er[0] + Vy[-1] * er[1]
    Ahead = 0.25 * 3.14 * p.rvec[-1] ** 2
    Fhead = 0.5 * 0.167 * 1000 * Ahead * (-np.sign(VmagperpHead) * VmagperpHead ** 2) * er

    Fx[-1] += Fhead[0]
    Fy[-1] += Fhead[1]

    return np.concatenate([Fx, Fy]).reshape(-1, 1)


def get_hydro_forces_batch(th, Vx, Vy, p):
    """Batched get_hydro_forces: th, Vx, Vy are (nn, n_batch).
    Returns (2*nn, n_batch): stacked [Fx; Fy], one column per batch element.
    """
    eth_x = -np.sin(th)   # (nn, n_batch)
    eth_y = np.cos(th)

    Vmagperp = Vx * eth_x + Vy * eth_y                     # (nn, n_batch)
    coef     = -np.sign(Vmagperp) * (Vmagperp ** 2)        # (nn, n_batch)
    Fx = p.uwvec[:, None] * coef * eth_x
    Fy = p.uwvec[:, None] * coef * eth_y

    # head tip forces
    er_x = np.cos(th[-1, :])   # (n_batch,)
    er_y = np.sin(th[-1, :])
    VmagperpHead = Vx[-1, :] * er_x + Vy[-1, :] * er_y   # (n_batch,)
    Ahead = 0.25 * 3.14 * p.rvec[-1] ** 2
    coefHead = 0.5 * 0.167 * 1000 * Ahead * (-np.sign(VmagperpHead) * VmagperpHead ** 2)
    Fx[-1, :] += coefHead * er_x
    Fy[-1, :] += coefHead * er_y

    return np.concatenate([Fx, Fy], axis=0)   # (2*nn, n_batch)
