"""
Noise-experiment pipeline — same RUN/REGISTRY pattern as main_MLP_Pipeline.py,
scoped to the PCA-vs-autoencoder-under-noise experiments: PCA_noise.py /
Autoencoder_noise.py (fit/train on a noisy random sample of the data pool)
and their matching single-trajectory reconstruction plots.

All settings for those four scripts — noise levels, sample fraction, seed,
autoencoder architecture, and the reconstruction-plot simulation params —
live here in one place. PCA_noise.py, Autoencoder_noise.py, and the two
reconstruction plot scripts all `import` these constants from this file
instead of defining their own copies, so changing a value here changes it
everywhere at once. Only the `if __name__ == '__main__':` block below
actually runs anything — importing this file elsewhere for its settings has
no side effects beyond the (cheap) setup_var() call used to build
NOISE_INDEX_GROUPS.

Add or remove step names in RUN to select which steps to run.
"""

import subprocess
import sys
import time
import os
# Project root: PCA_AutoEncoder/ -> up one level
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
from Fish_Simulation import setup_var
from paths import (
    PROJECT_ROOT as ROOT, MLP_TRAINING_DIR, SCRIPT_PCA_TRAIN, SCRIPT_PCA_MODEL_COMPARISON,
    SCRIPT_AUTOENCODER_TRAIN, SCRIPT_PCA_RECONSTRUCTION_PLOT, SCRIPT_AUTOENCODER_RECONSTRUCTION_PLOT,
    SCRIPT_PCA_VS_AUTOENCODER_PLOT,
)

# ── Noise / sampling settings (PCA_noise.py, Autoencoder_noise.py) ────────────
RAND_SEED = 42

# Random fraction of data-pool rows to fit/train on (mmap-sampled, not a full
# load) — shared by PCA_noise.py and Autoencoder_noise.py.
SAMPLE_FRACTION = 0.1

# Which data-pool run(s) to sample from — None = every fish_simulation_data_pool_*.npy
# file found in DATA_POOL_DIR (all runs, pooled together). Set a list of run
# numbers (the trailing int in the filename, e.g. [1, 3] for
# fish_simulation_data_pool_1.npy and _3.npy) to restrict training to
# specific run(s) instead — shared by PCA_noise.py and Autoencoder_noise.py.
DATA_POOL_RUNS = None

# Gaussian noise, one (mean, variance) pair per FEM state component type.
# (0.0, 0.0) = no noise for that component (skipped entirely, not just a
## zero-width draw).
#NOISE = {
#    'x':  (0.0, 0.002),   # position, x (m)
#    'y':  (0.0, 0.002),   # position, y (m)
#    'th': (0.0, 0.002),   # position, theta (rad)
#    'vx': (0.0, 0.002),   # velocity, x (m/s)
#    'vy': (0.0, 0.002),   # velocity, y (m/s)
#    'w':  (0.0, 0.002),   # velocity, omega (rad/s)
#}
NOISE = {
    'x':  (0.0, 0.00),   # position, x (m)
    'y':  (0.0, 0.00),   # position, y (m)
    'th': (0.0, 0.00),   # position, theta (rad)
    'vx': (0.0, 0.00),   # velocity, x (m/s)
    'vy': (0.0, 0.00),   # velocity, y (m/s)
    'w':  (0.0, 0.00),   # velocity, omega (rad/s)
}

# ── Autoencoder architecture/training settings (Autoencoder_noise.py) ─────────
LATENT_DIM  = 9
HIDDEN_DIMS = [128, 64]
DROPOUT     = 0.2

# ── Reconstruction-plot settings (PCA and autoencoder variants) ───────────────
TAU            = np.array([0.93, 0.32])   # [C_A, C_offset], held constant for the whole run
SIM_DURATION   = 3.0                      # seconds
X_INIT, Y_INIT = -0.5, 0.0

# Which saved fit each reconstruction plot should load. None = auto-use the
# most recent (highest-numbered) fit of the matching kind.
PCA_PATH         = None
#AUTOENCODER_PATH = "Data_Collection/Autoencoder_Models/autoencoder_xfem_3.pth"
AUTOENCODER_PATH = None

# Which saved PCA fit(s) PCA_Model_Comparison.py compares. None = every
# pca_xfem_noisy_*.joblib fit in PCA_MODELS_DIR; else a list of run numbers
# (the trailing int in the filename, e.g. [1, 3]) to compare a subset.
PCA_MODEL_RUNS = [4, 5, 6]

# setup_var() alone gives us p.ndof/p.nn without the costly full setup_sys()
# FEM eigendecomposition — the x/y/th dof-index arithmetic below is copied
# from Fish_Simulation/setup_sys.py's construction of p.ix/p.iy/p.ith, which
# only ever depends on p.nn.
_p = setup_var(20.0, -0.5, 0.0, 0.01)
NDOF = _p.ndof
P_NS = 2 * NDOF

_ix  = np.arange(_p.nn) * 3   # 0-based x  dof indices (position block)
_iy  = _ix + 1                 # 0-based y  dof indices (position block)
_ith = _ix + 2                 # 0-based th dof indices (position block)

# Index (within the P_NS-dim Xfem block) of every node's value of each
# component type — velocities live in the second half of the block, offset
# by NDOF from the matching position index.
NOISE_INDEX_GROUPS = {
    'x':  _ix,
    'y':  _iy,
    'th': _ith,
    'vx': NDOF + _ix,
    'vy': NDOF + _iy,
    'w':  NDOF + _ith,
}

# ── Select steps — add/remove names to run ────────────────────────────────────
RUN = [
    # ── Train Models (noisy) ───────────────────────────────────────────────
    #"PCA Train",                  # sample+noise Xfem from the data pool, fit+save PCA
    "Autoencoder Train",          # sample+noise Xfem from the data pool, train+save autoencoder

    # ── Plotting/Analysis ───────────────────────────────────────────────────
    #"PCA Reconstruction",          # true vs PCA round-trip of noisy Xfem vs true+noise, single trajectory
    #"PCA Model Comparison",        # compare saved PCA fits: MSE + variance, and round-trip of one trajectory
    #"Autoencoder Reconstruction", # true vs autoencoder round-trip of noisy Xfem vs true+noise, single trajectory
    #"PCA vs Autoencoder",          # both models round-tripped on the same noisy Xfem, one plot + MAE table
]

# ── Step registry — maps each name to (script_path, description) ──────────────
REGISTRY = {
    "PCA Train": (
        SCRIPT_PCA_TRAIN,
        'Fit PCA on a noisy random sample of the data pool',
    ),
    "Autoencoder Train": (
        SCRIPT_AUTOENCODER_TRAIN,
        'Train an autoencoder on a noisy random sample of the data pool',
    ),
    "PCA Reconstruction": (
        SCRIPT_PCA_RECONSTRUCTION_PLOT,
        'Plot true vs PCA round-trip of noisy Xfem vs true+noise, single simulated trajectory',
    ),
    "PCA Model Comparison": (
        SCRIPT_PCA_MODEL_COMPARISON,
        'Compare saved PCA fits — reconstruction MSE + retained variance, and round-trip of one shared trajectory',
    ),
    "Autoencoder Reconstruction": (
        SCRIPT_AUTOENCODER_RECONSTRUCTION_PLOT,
        'Plot true vs autoencoder round-trip of noisy Xfem vs true+noise, single simulated trajectory',
    ),
    "PCA vs Autoencoder": (
        SCRIPT_PCA_VS_AUTOENCODER_PLOT,
        'Round-trip the same noisy Xfem through both PCA and the autoencoder — one plot + MAE table',
    ),
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run(script, description):
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  Script: {script}")
    print(f"{'='*70}")
    t0  = time.time()
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        str(ROOT) + os.pathsep +
        str(MLP_TRAINING_DIR) + os.pathsep +
        env.get('PYTHONPATH', '')
    )
    result = subprocess.run([sys.executable, str(script)], env=env, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ERROR: {script} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)
    print(f"\n  Done in {elapsed / 60:.1f} min")


def main():
    if not RUN:
        print("No steps selected — add step names to the RUN list.")
        return

    unknown = [name for name in RUN if name not in REGISTRY]
    if unknown:
        print(f"Unknown step name(s): {unknown}")
        print(f"Valid names: {list(REGISTRY)}")
        sys.exit(1)

    steps = [(REGISTRY[name][0], REGISTRY[name][1]) for name in RUN]

    print(f"\nRunning {len(steps)} step(s):")
    for i, (_, desc) in enumerate(steps, 1):
        print(f"  {i}. {desc}")

    total_start = time.time()
    for script, description in steps:
        run(script, description)

    total_min = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"  All steps complete.  Total time: {total_min:.1f} min")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
