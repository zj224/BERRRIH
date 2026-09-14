"""
Single fixed-tau simulation (SIM_DURATION seconds at tau=[C_A, C_offset]) used
to visually check how much the saved Xfem PCA fit (MLP_PCA_Analysis_FEM.py)
loses on reconstruction: the true physics trajectory is compared against the
same trajectory with Xfem run through pca.transform() -> pca.inverse_transform()
at every timestep. This is purely a post-hoc, open-loop comparison of the
recorded state — the reconstructed Xfem is never fed back into the
simulation, so it can't affect the true trajectory.

Xrigid is identical in both cases (the PCA never touches it), so the two
trajectories only diverge through the head node's *world-frame* position,
which is Xrigid rotated/translated plus the local Xfem displacement — i.e.
any visible gap between the two curves is exactly the PCA's reconstruction
error on Xfem.
"""

import os
import sys
# Project root: PCA_AutoEncoder/PCA/ → up two levels
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import joblib
from Fish_Simulation import setup_var, setup_sys, get_controls, fish_step, fem_to_inertial
from PCA_AutoEncoder.PCA.PCA_Train import add_noise_to_xfem
from PCA_AutoEncoder.main_PCA_AE import (
    TAU, SIM_DURATION, X_INIT, Y_INIT, PCA_PATH, NOISE, NOISE_INDEX_GROUPS, RAND_SEED,
)
from paths import PCA_MODELS_DIR, IMAGES_DIR

# ── Parameters ────────────────────────────────────────────────────────────────
# TAU / SIM_DURATION / X_INIT / Y_INIT / PCA_PATH / NOISE all come from
# main_PCA_AE.py — edit them there, not here.


def _most_recent_numbered(pattern):
    nums = []
    for path in glob.glob(pattern):
        m = re.search(r'(\d+)', os.path.basename(path))
        if m:
            nums.append((int(m.group(1)), path))
    if not nums:
        raise FileNotFoundError(f"No files matching {pattern}")
    return max(nums, key=lambda t: t[0])[1]


# ── Setup ─────────────────────────────────────────────────────────────────────
p = setup_var(SIM_DURATION, X_INIT, Y_INIT, 0.01)
p.controller = get_controls
p = setup_sys(p)
n_steps = round(SIM_DURATION / p.dt)

print(f"Simulating {SIM_DURATION}s at tau={TAU.tolist()}  ({n_steps} steps, dt={p.dt})")

# ── Initial state (zero local FEM state, rigid body at (x_init, y_init, 0)) ──
Xfemi   = np.zeros(p.ns, dtype=np.float64)
Xrigidi = np.zeros(3, dtype=np.float64)
Xrigidi[:] = p.x0[0:3]

# ── Run true physics, recording every step ────────────────────────────────────
Xfem_hist, Xrigid_hist = [Xfemi.copy()], [Xrigidi.copy()]
for s in range(n_steps):
    Xfemi, Xrigidi = fish_step(p, s * p.dt, Xfemi, Xrigidi, TAU)
    Xfem_hist.append(Xfemi.copy())
    Xrigid_hist.append(Xrigidi.copy())
Xfem_hist   = np.array(Xfem_hist)     # (n_steps+1, p.ns)
Xrigid_hist = np.array(Xrigid_hist)   # (n_steps+1, 3)

# ── Add Gaussian noise to Xfem before anything else touches it ────────────────
# Same NOISE definition as main_PCA_AE.py. The PCA round-trip below
# runs on this noisy Xfem, not the clean Xfem_hist — this is a "how well does
# PCA denoise a noisy reading" test, not a clean-data reconstruction test.
Xfem_noisy = add_noise_to_xfem(Xfem_hist.copy(), NOISE, NOISE_INDEX_GROUPS,
                                rng=np.random.default_rng(RAND_SEED))

# ── PCA round-trip on noisy Xfem only (Xrigid is untouched) ───────────────────
pca_path = PCA_PATH or _most_recent_numbered(f'{PCA_MODELS_DIR}/pca_xfem_*.joblib')
fit = joblib.load(pca_path)
scaler, pca, k, n_dims = fit['scaler'], fit['pca'], fit['k'], fit['n_dims']
print(f"Loaded PCA fit: {pca_path}  (k={k}, n_dims={n_dims})")

Xfem_pca = pca.transform(scaler.transform(Xfem_noisy))[:, :k]

# Zero-pad from k back up to n_dims before inverse_transform — only the
# first k components of the fitted n_dims-component PCA were ever kept.
Xfem_pca_full = np.zeros((len(Xfem_pca), n_dims))
Xfem_pca_full[:, :k] = Xfem_pca
Xfem_recon = scaler.inverse_transform(pca.inverse_transform(Xfem_pca_full))

# ── World-frame head-node position, true vs reconstructed vs noisy ────────────
def head_xy(Xfem_arr, Xrigid_arr):
    """(x, y) of the head node (last node) at every timestep, batched over
    all timesteps in one fem_to_inertial() call."""
    Xstates = np.vstack([Xfem_arr.T, Xrigid_arr.T])   # (p.ns+3, N)
    Xinertial = fem_to_inertial(p, Xstates)
    return Xinertial[p.ix[-1], :], Xinertial[p.iy[-1], :]

true_x, true_y   = head_xy(Xfem_hist, Xrigid_hist)
recon_x, recon_y = head_xy(Xfem_recon, Xrigid_hist)
noisy_x, noisy_y = head_xy(Xfem_noisy, Xrigid_hist)

mae = np.abs(np.concatenate([true_x - recon_x, true_y - recon_y])).mean()
print(f"Head-node position MAE (true vs PCA round-trip of noisy Xfem): {mae:.6f} m")

mae_noise = np.abs(np.concatenate([true_x - noisy_x, true_y - noisy_y])).mean()
print(f"Head-node position MAE (true vs true+noise, no PCA): {mae_noise:.6f} m")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal', adjustable='box')
ax.plot(true_x, true_y, '-', lw=2, color='steelblue', label='true')
ax.plot(recon_x, recon_y, ':', lw=2, color='tomato', label='PCA round-trip (noisy input)')
ax.plot(noisy_x, noisy_y, '.', lw=1.5, color='seagreen', label='true + noise (no PCA)')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'Head-node trajectory — true vs PCA round-trip of noisy Xfem\n'
             f'tau={TAU.tolist()}  {SIM_DURATION}s  (k={k})')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(IMAGES_DIR, exist_ok=True)
out_path = f'{IMAGES_DIR}/pca_reconstruction_trajectory.png'
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
plt.show()
