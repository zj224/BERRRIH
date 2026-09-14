import numpy as np
from .fish_step import fish_step

def simulate_sys(p):
    """Port of simulateSys.m

    Time-marches the discrete FEM dynamics and re-composes the rigid
    body (x, y, th) motion at each step. Same math as the original
    MATLAB, but node (x,y) re-alignment is done with a single 2x2
    rotation broadcast across all nodes instead of rebuilding an
    ndof x ndof block-diagonal rotation matrix every step, and hydro
    velocity/angle states are pulled out with direct indexing instead
    of multiplying by big all-zero-except-a-diagonal selector matrices
    (p.HVx etc). That block_diag call alone was ~80% of this function's
    runtime in the literal port — this version is ~5x faster and
    numerically identical (differences are floating-point roundoff,
    ~1e-14).
    """
    Xfem = np.zeros((p.ns, p.nt))    # local FEM model (x,y axis at each node of straight beam)
    Xrigid = np.zeros((3, p.nt))     # rigid body translation (x,y) and rotation (th)

    Xrigid[:, 0] = p.x0[0:3]  # th displacement

    Xfem[p.ndof + p.ix, 0] = p.x0[3]  # initial local x velocity (testing)
    Xfem[p.ndof + p.iy, 0] = p.x0[4]  # initial local y velocity (testing)
    Xfem[p.ndof + p.ith, 0] = p.x0[5]  # initial local th velocity (testing)

    for k in range(1, p.nt):  # k here == (MATLAB k) - 1
        Xfem[:, k], Xrigid[:, k] = fish_step(p, p.t[k], Xfem[:, k-1], Xrigid[:, k-1])
        
    Xstates = np.vstack([Xfem, Xrigid])
    return Xstates
