import numpy as np
import os
import sys
import re
import glob
import time

# Project root: Simulation_Data_Generation/ -> up one level
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from Fish_Simulation import (
    setup_var, get_controls, get_controls_batch_cacc, setup_sys, fish_step_batch, cast_dtype,
)
from paths import DATA_POOL_DIR as OUT_DIR

# ── Parameters ────────────────────────────────────────────────────────────────
# Grid resolution — total sims = len(CA_VALUES) × len(COFFSET_VALUES) × len(CACC_VALUES).
# Cacc is a third grid axis (continuous acceleration coefficient, same role as
# C_A/C_offset — see get_controls: it scales the traveling-wave phase offset,
# +1 = full acceleration-direction wave, -1 = full deceleration-direction,
# continuous in between). Per-axis resolution is kept lower than the old
# (C_A, C_offset)-only grid's 50×50 specifically *because* adding this third
# axis multiplies total sim count (and therefore the in-memory pool_data
# array below) by len(CACC_VALUES) — 50×50×50 would be a ~50x memory/runtime
# blowup over the old grid. Raise these if you have the RAM/time budget.
CA_VALUES      = np.round(np.linspace(0.0, 1.0, 20), 3)
COFFSET_VALUES = np.round(np.linspace(-1.0, 1.0, 20), 3)
CACC_VALUES    = np.round(np.linspace(-1.0, 1.0, 9), 3)


SIM_DURATION = 10.0    # total simulation time (s) per tau
# Sample spacing is p.dt (set in Fish_Simulation.setup_var), since that is
# the step size baked into the discretized system fish_step advances by.

# All rows at every dt timestep are saved to a single run file. This
# preserves the full trajectory while avoiding one file per timestep.
# Downstream code can still sample period boundaries from this finer-grained
# pool data if desired.
PERIOD_S = 1.0

# Set to np.float32 to roughly halve memory/bandwidth for the dominant Fd @ Xfem
# matmul, at the cost of precision. Left at float64 by default so this doesn't
# silently change existing simulation behavior — try float32 and compare if
# you want the speed.
DTYPE = np.float64


def _next_run_number(pattern):
    """max(existing run numbers matched by glob pattern) + 1 — scans every
    match rather than walking forward from 1, so a gap from a deleted run
    (e.g. run 2 removed, run 3 still present) can't cause a new run to reuse
    a lower number than one that already exists."""
    nums = []
    for path in glob.glob(pattern):
        m = re.search(r'(\d+)', os.path.basename(path))
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


if __name__ == '__main__':
    # Setup Variables
    tf = 30.0        # final time
    x_init = -0.5     # initial x position
    y_init = 0.0       # initial y position
    t_pause = 0.01     # time to pause animation for

    # Setup Variables
    p = setup_var(tf, x_init, y_init, t_pause)
    p.controller = get_controls
    # tau_matrix below carries [C_A, C_offset, Cacc] per simulation; split it
    # back into (tau, Cacc) for get_controls_batch_cacc, which now accepts a
    # per-sample Cacc array (see Fish_Simulation/get_controls.py).
    p.controller_batch = lambda p_local, t_local, tau_batch: get_controls_batch_cacc(
        p_local, t_local, tau_batch[:, :2], tau_batch[:, 2]
    )

    # Setup System
    p = setup_sys(p)

    if DTYPE != np.float64:
        p = cast_dtype(p, DTYPE)

    Xfem0 = np.zeros(p.ns, dtype=DTYPE)   # local FEM model initial state (x,y axis at each node of straight beam)
    Xrigid0 = np.zeros(3, dtype=DTYPE)    # rigid body translation (x,y) and rotation (th)

    Xrigid0[0:2] = p.x0[0:2]  # x,y displacement
    Xrigid0[2] = 0.0          # placeholder for per-simulation theta below

    Xfem0[p.ndof + p.ix] = p.x0[3]  # initial local x velocity (testing)
    Xfem0[p.ndof + p.iy] = p.x0[4]  # initial local y velocity (testing)
    Xfem0[p.ndof + p.ith] = p.x0[5]  # initial local th velocity (testing)

    CA_g, CO_g, CACC_g = np.meshgrid(CA_VALUES, COFFSET_VALUES, CACC_VALUES, indexing='ij')
    tau_matrix = np.column_stack([CA_g.ravel(), CO_g.ravel(), CACC_g.ravel()]).astype(DTYPE)
    n_tau = len(tau_matrix)

    # All n_tau simulations are advanced together, one dt at a time, as a
    # single batched matmul (Fd @ Xfem_batch) instead of n_tau independent
    # per-simulation loops spread across a multiprocessing pool. Same total
    # FLOPs, but BLAS handles one big GEMM far more efficiently than many
    # small GEMVs plus process/IPC overhead.
    Xfemi   = np.tile(Xfem0[:, None],   (1, n_tau))   # (ns, n_tau)
    Xrigidi = np.tile(Xrigid0[:, None], (1, n_tau))   # (3, n_tau)

    # Randomize the initial heading for each rollout independently.
    rng = np.random.default_rng()
    Xrigidi[2, :] = rng.uniform(0.0, 2.0 * np.pi, size=n_tau).astype(DTYPE)

    t_points = np.arange(0.0, SIM_DURATION, p.dt)
    n_total_steps = len(t_points)

    print(f"Total simulations: {n_tau}  "
          f"({len(CA_VALUES)} C_A × {len(COFFSET_VALUES)} C_offset × {len(CACC_VALUES)} Cacc)")
    print(f"Each: {SIM_DURATION}s  {n_total_steps} steps  dt={p.dt}  "
          f"period boundaries every {PERIOD_S}s  randomized th0  dtype={DTYPE.__name__}")

    # ── Pick a run number: this run's files are fish_simulation_data_pool_{n}_b*.npy
    # Existence of tau_matrix_data_pool_{n}.npy marks that run n has (at least)
    # started, since it's written first, immediately.
    os.makedirs(OUT_DIR, exist_ok=True)
    n = _next_run_number(f'{OUT_DIR}/tau_matrix_data_pool_*.npy')
    tau_path = f'{OUT_DIR}/tau_matrix_data_pool_{n}.npy'
    np.save(tau_path, tau_matrix)

    t0 = time.time()
    prog = max(1, n_total_steps // 10)   # progress-print cadence only
    pool_data = np.empty((n_tau, n_total_steps, 1 + p.ns + 3), dtype=DTYPE)

    for step_i, t_curr in enumerate(t_points):
        t_row = np.full((1, n_tau), t_curr, dtype=DTYPE)
        step_data = np.concatenate([t_row, Xfemi, Xrigidi], axis=0).T   # (n_tau, 1+ns+3)
        pool_data[:, step_i, :] = step_data

        Xfemi, Xrigidi = fish_step_batch(p, t_curr, Xfemi, Xrigidi, tau_matrix)

        if (step_i + 1) % prog == 0 or step_i == n_total_steps - 1:
            pct = int((step_i + 1) / n_total_steps * 100)
            print(f"  {pct}%  ({step_i+1}/{n_total_steps})  t={time.time()-t0:.1f}s")

    pool_path = f'{OUT_DIR}/fish_simulation_data_pool_{n}.npy'
    np.save(pool_path, pool_data)

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Saved (pool #{n}):")
    print(f"  {pool_path}   shape={pool_data.shape}")
    print(f"  {tau_path}    shape={tau_matrix.shape}")
    print(f"  C_A range:      [{tau_matrix[:,0].min():.2f}, {tau_matrix[:,0].max():.2f}]")
    print(f"  C_offset range: [{tau_matrix[:,1].min():.2f}, {tau_matrix[:,1].max():.2f}]")
    print(f"  Cacc range:     [{tau_matrix[:,2].min():.2f}, {tau_matrix[:,2].max():.2f}]")
