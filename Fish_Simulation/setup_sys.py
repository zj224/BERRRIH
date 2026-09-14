import numpy as np
from scipy.linalg import eigh
from scipy.signal import cont2discrete
from scipy.sparse import lil_matrix

from .get_miki import get_miki


def setup_sys(p):
    """Port of setupSys.m

    Builds the FEM mass/stiffness matrices, forms a modal state-space
    model (x, y, th rigid body modes + flexible modes), and discretizes
    it with zero-order hold (matching MATLAB's c2d default).

    M/K assembly uses a sparse matrix while accumulating element
    contributions (each element only touches a small overlapping block),
    then densifies once for the eigendecomposition and modal transform,
    which need a dense solve regardless.

    Note: unlike the original MATLAB port, this no longer builds the
    dense Hx/Hy/Hth/HVx/HVy/HOm selector matrices — downstream code
    (simulate_sys, fem_to_inertial, fish_animation) indexes p.ix/p.iy/p.ith
    directly instead, which is both faster and avoids ~600x600 all-zero
    matrices sitting around for no reason.

    Returns
    -------
    p : Params
        Updated with p.ns, p.ix/p.iy/p.ith, p.xnode.
    sysOL_D : dict
        Discrete state-space system with keys 'A', 'B', 'C', 'D', 'dt'.
    """
    # Material parameters for an approximate oar fish
    rho = 1200.0   # kg/m^3
    E = 1e6        # N/m^2

    # Size parameters for an approximate oar fish
    L = p.L        # m length

    # Finite Element Model (FEM)
    p.nn = p.n + 1          # number of nodes
    l = L / p.n               # length of an element
    ni = 3                    # number of DOF in each node

    # Form Finite Element Model from individual elements (sparse accumulation)
    p.ndof = p.nn * ni
    M_sp = lil_matrix((p.ndof, p.ndof))
    K_sp = lil_matrix((p.ndof, p.ndof))
    for i0 in range(p.n):  # i0 = MATLAB i - 1
        r = p.rvec[i0]
        mi, ki = get_miki(rho, l, E, r)
        ii_start = ni * i0          # 0-based inclusive start
        ii_end = ni * (i0 + 2)      # 0-based exclusive end
        K_sp[ii_start:ii_end, ii_start:ii_end] += ki
        M_sp[ii_start:ii_end, ii_start:ii_end] += mi
    # note: we do not apply any boundary conditions, as we want a free-free beam

    # Densify: the eigendecomposition and modal transform below need every
    # mode (not just a few), so a dense solve is the right tool from here on.
    M = M_sp.toarray()
    K = K_sp.toarray()

    # Solve the eigenvalue problem to find a state space model
    # symmetric-definite generalized eigenproblem (K,M symmetric, M pos. def.)
    Lam_diag, Phi = eigh(K, M)  # ascending eigenvalues, like MATLAB eig(K,M)
    Lam_diag = Lam_diag.copy()
    Lam_diag[0:3] = 0.0  # small numerical imaginary components due to zero eigenvalues
    ww = np.sqrt(Lam_diag)  # system natural frequencies

    # Need to fix the rigid body modes so that they align with x, y, th
    ix = np.arange(p.nn) * ni       # 0-based x dof indices
    iy = ix + 1                      # 0-based y dof indices
    ith = ix + 2                     # 0-based th dof indices

    Phi2 = Phi.copy()
    Phi2[:, 0:3] = 0.0
    Phi2[ix, 0] = 1.0  # set all x components to 1, indicating only x motion
    Phi2[iy, 1] = 1.0  # set all y components to 1, indicating only y motion

    # mode shape for rigid body rotation (theta - rotation about the center node)
    # x,y,th all change in this case - solve for a small theta
    th3 = 0.01
    p.xnode = np.linspace(-L / 2, L / 2, p.nn)
    x3 = p.xnode * (np.cos(th3) - 1)
    y3 = p.xnode * np.sin(th3)
    Phi2[ix, 2] = x3    # changes in x for small rotation theta
    Phi2[iy, 2] = y3    # changes in y for small rotation theta
    Phi2[ith, 2] = th3  # changes in th for small rotation theta

    # normalize modes via mass matrix
    Mrigid = Phi2.T @ M @ Phi2
    Phi2[:, 0] = Phi2[:, 0] / np.sqrt(Mrigid[0, 0])
    Phi2[:, 1] = Phi2[:, 1] / np.sqrt(Mrigid[1, 1])
    Phi2[:, 2] = Phi2[:, 2] / np.sqrt(Mrigid[2, 2])

    # Create new/approximate M,K,C - typically better to use modal coordinates,
    # but keeping x,y,th for the state space model for ease in understanding
    Mmodal = np.eye(p.ndof)          # normalized mass matrix is identity
    Kmodal = np.diag(Lam_diag)        # normalized stiffness matrix is the frequencies squared
    Cmodal = 2 * 0.7 * np.diag(ww)    # normalized damping matrix based on zeta=0.7

    Phi2_inv = np.linalg.inv(Phi2)
    Phi2T_inv = np.linalg.inv(Phi2.T)
    M = Phi2T_inv @ Mmodal @ Phi2_inv  # transform back to find M
    K = Phi2T_inv @ Kmodal @ Phi2_inv  # transform back to find K
    C = Phi2T_inv @ Cmodal @ Phi2_inv  # transform back to find C

    # actuator inputs: p.nin torques over p.nin sections of the body
    nsec = p.n // p.nin  # number of nodes in each section
    Bin = np.zeros((p.ndof, p.nin))
    for i0 in range(p.nin):  # i0 = MATLAB i - 1
        section_nodes = i0 * nsec + np.arange(nsec)  # 0-based node indices
        th_dof = ith[section_nodes]
        if nsec % 2 == 0:  # even number of nodes in the section
            Bin[th_dof[: nsec // 2], i0] = 1.0
            Bin[th_dof[nsec // 2:], i0] = -1.0
        else:  # odd number of nodes in the section
            half = (nsec - 1) // 2
            Bin[th_dof[:half], i0] = 1.0
            Bin[th_dof[half + 1:], i0] = -1.0

    # hydrodynamic forces inputs (x,y at each node, based on velocity)
    Bh = np.zeros((p.ndof, p.nn * 2))
    for i0 in range(p.nn):  # for each node
        Bh[ix[i0], i0] = 1.0
        Bh[iy[i0], p.nn + i0] = 1.0

    # state space model: system matrices
    F = np.block([
        [np.zeros((p.ndof, p.ndof)), np.eye(p.ndof)],
        [-np.linalg.solve(M, K), -np.linalg.solve(M, C)],
    ])
    p.ns = 2 * p.ndof

    # nin body section control inputs
    G = np.vstack([np.zeros((p.ndof, p.nin)), np.linalg.solve(M, Bin)])
    # hydrodynamic force inputs at x,y of each node
    Gh = np.vstack([np.zeros((p.ndof, p.nn * 2)), np.linalg.solve(M, Bh)])

    # general output matrix - all states
    H = np.eye(p.ns)
    noout = p.ns  # all states currently
    D = np.zeros((noout, p.nin + 2 * p.nn))

    # formulate index vectors used downstream (fem_to_inertial, simulate_sys,
    # fish_animation all index directly with these instead of matmul-ing
    # a dense selector matrix)
    p.ix = ix    # all x dof indices
    p.iy = iy    # all y dof indices
    p.ith = ith  # all th dof indices

    # form state space system, then discretize (zero-order hold, like c2d)
    Bfull = np.hstack([G, Gh])
    Ad, Bd, Cd, Dd, _ = cont2discrete((F, Bfull, H, D), p.dt, method="zoh")

    p.sysOL_D = {"A": Ad, "B": Bd, "C": Cd, "D": Dd, "dt": p.dt}

    return p
