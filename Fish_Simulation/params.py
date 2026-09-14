"""
Typed container mirroring the MATLAB struct `p` used throughout the
original code. A dataclass instead of a bare object gives autocomplete
and catches attribute typos (assigning to an undeclared field still
works like a normal object, but every field the pipeline actually uses
is documented here in one place).
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class Params:
    # --- time (get_time_vars) ---
    dt: Optional[float] = None
    tf: Optional[float] = None
    nt: Optional[int] = None
    t: Optional[np.ndarray] = None

    # --- system size / initial conditions (get_sys_vars) ---
    n: Optional[int] = None
    nn: Optional[int] = None
    nin: Optional[int] = None
    ndof: Optional[int] = None
    ns: Optional[int] = None
    xInit: Optional[float] = None
    yInit: Optional[float] = None
    thInit: Optional[float] = None
    vxInit: Optional[float] = None
    vyInit: Optional[float] = None
    wInit: Optional[float] = None
    x0: Optional[np.ndarray] = None

    # --- fish shape / hydro constants (get_fish_vars) ---
    rvec: Optional[np.ndarray] = None
    L: Optional[float] = None
    uwvec: Optional[np.ndarray] = None

    # --- control (get_control_vars) ---
    A: Optional[float] = None
    w: Optional[float] = None
    AmpCo: Optional[np.ndarray] = None
    offsetCo: Optional[np.ndarray] = None
    controller: Optional[Callable] = None

    # --- plotting ---
    t_pause: Optional[float] = None

    # --- FEM / state-space (setup_sys) ---
    xnode: Optional[np.ndarray] = None
    ix: Optional[np.ndarray] = None   # 0-based x dof index per node
    iy: Optional[np.ndarray] = None   # 0-based y dof index per node
    ith: Optional[np.ndarray] = None  # 0-based th dof index per node

    def __repr__(self):
        set_fields = [k for k, v in self.__dict__.items() if v is not None]
        return f"Params({', '.join(set_fields)})"


def cast_dtype(p, dtype):
    """Cast every float-valued array field on p (and in p.sysOL_D) to dtype,
    in place. Leaves integer/bool arrays (e.g. p.ix/p.iy/p.ith, used for
    indexing) untouched. Useful for trading precision for speed/memory by
    running the discretized system in float32 instead of float64.
    """
    for k, v in vars(p).items():
        if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
            setattr(p, k, v.astype(dtype))
    sys_ol_d = getattr(p, 'sysOL_D', None)
    if sys_ol_d is not None:
        for k, v in sys_ol_d.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
                sys_ol_d[k] = v.astype(dtype)
    return p
