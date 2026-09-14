import numpy as np
from .get_hydro_forces import get_hydro_forces, get_hydro_forces_batch



def fish_step(p, t_curr, Xfemi, Xrigidi, tau = [0.9,0.2]):
    Fd = p.sysOL_D["A"]                       # discrete system matrix
    Gd = p.sysOL_D["B"][:, : p.nin]            # discrete control input matrix
    Ghd = p.sysOL_D["B"][:, p.nin:]            # discrete hydrodynamic force input matrix

    ix, iy, ith, ndof, nn = p.ix, p.iy, p.ith, p.ndof, p.nn

    # Get Hydro Forces (direct indexing instead of Hx/Hy/Hth matmuls)
    Vx = Xfemi[ndof + ix]
    Vy = Xfemi[ndof + iy]
    th = Xfemi[ith]
    F = get_hydro_forces(th, Vx, Vy, p)

    # Get Controls
    uk = p.controller(p, t_curr, tau)

    # propagate local FEM dynamics
    Xfemf = Fd @ Xfemi + Gd @ uk.flatten() + Ghd @ F.flatten()

    # find change in translation, rotation - approximate rigid body
    # change in the local frame
    dx = np.mean(Xfemf[ix])
    dy = np.mean(Xfemf[iy])
    dth = np.mean(Xfemf[ith])

    # convert the local rigid body motion into the inertial coordinates
    thk = Xrigidi[2] + dth
    c, s = np.cos(thk), np.sin(thk)
    Xrigidf = 0*Xrigidi
    Xrigidf[0] = Xrigidi[0] + c * dx - s * dy
    Xrigidf[1] = Xrigidi[1] + s * dx + c * dy
    Xrigidf[2] = Xrigidi[2] + dth

    # Need to convert the new local axis of fish by removing rigid
    # body rotation/translation
    xtmp = Xfemf.copy()
    xtmp[:ndof] -= np.tile([dx, dy, dth], nn)  # subtract rigid dx,dy,dth
    xtmp[ix] += p.xnode  # add xnode locations back on

    # rotate all nodes back to 0 (i.e. th via rot^T), applied directly
    # to the (x,y) pair at every node at once instead of via a big
    # block-diagonal matrix; th is unaffected by construction
    cd, sd = np.cos(dth), np.sin(dth)
    x_nodes = xtmp[ix]
    y_nodes = xtmp[iy]
    xtmp[ix] = cd * x_nodes + sd * y_nodes
    xtmp[iy] = -sd * x_nodes + cd * y_nodes

    xtmp[ix] -= p.xnode  # subtract xnode locations back off

    # updated fem states based on new local axis
    Xfemf = xtmp

    # Xstates = np.vstack([Xfemf, Xrigidf])
    return Xfemf, Xrigidf


def fish_step_batch(p, t_curr, Xfemi, Xrigidi, tau_batch):
    """Batched fish_step: advances many independent simulations one p.dt
    step at once via a single matmul instead of one fish_step() call per
    simulation. Much better use of BLAS than many small (ns,)-shaped GEMVs
    spread across a multiprocessing pool.

    Xfemi:     (p.ns, n_batch)
    Xrigidi:   (3, n_batch)
    tau_batch: (n_batch, 2)
    t_curr:    scalar, or (n_batch,) if each column is at a different point
               in time (only affects the controller's phase term)

    Returns (Xfemf, Xrigidf) with the same shapes as (Xfemi, Xrigidi).
    """
    Fd = p.sysOL_D["A"]
    Gd = p.sysOL_D["B"][:, : p.nin]
    Ghd = p.sysOL_D["B"][:, p.nin:]

    ix, iy, ith, ndof = p.ix, p.iy, p.ith, p.ndof
    n_batch = Xfemi.shape[1]

    Vx = Xfemi[ndof + ix, :]
    Vy = Xfemi[ndof + iy, :]
    th = Xfemi[ith, :]
    F = get_hydro_forces_batch(th, Vx, Vy, p)

    uk = p.controller_batch(p, t_curr, tau_batch)

    Xfemf = Fd @ Xfemi + Gd @ uk + Ghd @ F

    dx  = Xfemf[ix, :].mean(axis=0)
    dy  = Xfemf[iy, :].mean(axis=0)
    dth = Xfemf[ith, :].mean(axis=0)

    thk = Xrigidi[2, :] + dth
    c, s = np.cos(thk), np.sin(thk)
    Xrigidf = np.zeros_like(Xrigidi)
    Xrigidf[0, :] = Xrigidi[0, :] + c * dx - s * dy
    Xrigidf[1, :] = Xrigidi[1, :] + s * dx + c * dy
    Xrigidf[2, :] = Xrigidi[2, :] + dth

    rigid_disp = np.zeros((ndof, n_batch), dtype=Xfemi.dtype)
    rigid_disp[ix, :]  = dx
    rigid_disp[iy, :]  = dy
    rigid_disp[ith, :] = dth

    xtmp = Xfemf.copy()
    xtmp[:ndof, :] -= rigid_disp
    xtmp[ix, :] += p.xnode[:, None]

    cd, sd = np.cos(dth), np.sin(dth)
    x_nodes = xtmp[ix, :]
    y_nodes = xtmp[iy, :]
    xtmp[ix, :] = cd * x_nodes + sd * y_nodes
    xtmp[iy, :] = -sd * x_nodes + cd * y_nodes

    xtmp[ix, :] -= p.xnode[:, None]

    Xfemf = xtmp
    return Xfemf, Xrigidf