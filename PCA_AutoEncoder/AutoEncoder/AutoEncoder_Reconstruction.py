"""
Round-trip test for a saved AutoEncoder_Train.py fit — same structure as
PCA_AutoEncoder/PCA/PCA_Reconstruction.py: run a single fixed-tau simulation,
encode then decode the recorded Xfem trajectory through a saved autoencoder,
and plot the reconstructed head-node trajectory over the true one. This is
purely a post-hoc, open-loop comparison of the recorded state — the
reconstructed Xfem is never fed back into the simulation, so it can't affect
the true trajectory.

Xrigid is identical in both cases (the autoencoder never touches it), so the
two trajectories only diverge through the head node's *world-frame*
position, which is Xrigid rotated/translated plus the local Xfem
displacement — i.e. any visible gap between the two curves is exactly the
autoencoder's reconstruction error on Xfem.

Standalone: not wired into main_MLP_Pipeline.py. Run directly for the demo
round trip + plot; import load_autoencoder / encode / decode / reconstruct
to use elsewhere.
"""

import os
import sys
# Project root: PCA_AutoEncoder/AutoEncoder/ -> up two levels
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

from Fish_Simulation import setup_var, setup_sys, get_controls, fish_step, fem_to_inertial
from PCA_AutoEncoder.AutoEncoder.AutoEncoder_Train import Autoencoder
from PCA_AutoEncoder.PCA.PCA_Train import add_noise_to_xfem
from PCA_AutoEncoder.main_PCA_AE import (
    TAU, SIM_DURATION, X_INIT, Y_INIT, AUTOENCODER_PATH, NOISE, NOISE_INDEX_GROUPS, RAND_SEED,
)
from paths import AUTOENCODER_MODELS_DIR, IMAGES_DIR

# ── Parameters ────────────────────────────────────────────────────────────────
# TAU / SIM_DURATION / X_INIT / Y_INIT / AUTOENCODER_PATH / NOISE all come
# from main_PCA_AE.py — edit them there, not here.


def _most_recent_numbered(pattern):
    nums = []
    for path in glob.glob(pattern):
        m = re.search(r'(\d+)', os.path.basename(path))
        if m:
            nums.append((int(m.group(1)), path))
    if not nums:
        raise FileNotFoundError(f"No files matching {pattern}")
    return max(nums, key=lambda t: t[0])[1]


def load_autoencoder(path=None):
    """Load a saved autoencoder fit. Returns (model, scaler_mean, scaler_scale, meta)
    where model is an eval()-mode Autoencoder on CPU, and meta has n_dims/
    latent_dim/hidden_dims/dropout/label.
    """
    path = path or _most_recent_numbered(f'{AUTOENCODER_MODELS_DIR}/autoencoder_xfem_*.pth')
    ckpt = torch.load(path, weights_only=False, map_location='cpu')

    model = Autoencoder(ckpt['n_dims'], ckpt['latent_dim'], ckpt['hidden_dims'], dropout=ckpt['dropout'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    scaler_mean = np.asarray(ckpt['scaler_mean'])
    scaler_scale = np.asarray(ckpt['scaler_scale'])
    meta = {k: ckpt[k] for k in ('n_dims', 'latent_dim', 'hidden_dims', 'dropout', 'label')}
    print(f"Loaded autoencoder fit: {path}  (latent_dim={meta['latent_dim']}, n_dims={meta['n_dims']})")
    return model, scaler_mean, scaler_scale, meta


def encode(model, scaler_mean, scaler_scale, X):
    """X: (N, n_dims) array in original (unscaled) units -> (N, latent_dim) latent array."""
    X = np.asarray(X, dtype=np.float64)
    X_scaled = (X - scaler_mean) / scaler_scale
    with torch.no_grad():
        z = model.encode(torch.tensor(X_scaled, dtype=torch.float32))
    return z.numpy()


def decode(model, scaler_mean, scaler_scale, Z):
    """Z: (N, latent_dim) latent array -> (N, n_dims) reconstruction in original units."""
    Z = np.asarray(Z, dtype=np.float64)
    with torch.no_grad():
        x_scaled = model.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    return x_scaled * scaler_scale + scaler_mean


def reconstruct(model, scaler_mean, scaler_scale, X):
    """X: (N, n_dims) -> (N, n_dims) round-tripped through encode then decode."""
    z = encode(model, scaler_mean, scaler_scale, X)
    return decode(model, scaler_mean, scaler_scale, z)


if __name__ == '__main__':
    model, scaler_mean, scaler_scale, meta = load_autoencoder(AUTOENCODER_PATH)

    # ── Setup ─────────────────────────────────────────────────────────────────
    p = setup_var(SIM_DURATION, X_INIT, Y_INIT, 0.01)
    p.controller = get_controls
    p = setup_sys(p)
    n_steps = round(SIM_DURATION / p.dt)

    print(f"Simulating {SIM_DURATION}s at tau={TAU.tolist()}  ({n_steps} steps, dt={p.dt})")

    # ── Initial state (zero local FEM state, rigid body at (x_init, y_init, 0)) ──
    Xfemi   = np.zeros(p.ns, dtype=np.float64)
    Xrigidi = np.zeros(3, dtype=np.float64)
    Xrigidi[:] = p.x0[0:3]

    # ── Run true physics, recording every step ────────────────────────────────
    Xfem_hist, Xrigid_hist = [Xfemi.copy()], [Xrigidi.copy()]
    for s in range(n_steps):
        Xfemi, Xrigidi = fish_step(p, s * p.dt, Xfemi, Xrigidi, TAU)
        Xfem_hist.append(Xfemi.copy())
        Xrigid_hist.append(Xrigidi.copy())
    Xfem_hist   = np.array(Xfem_hist)     # (n_steps+1, p.ns)
    Xrigid_hist = np.array(Xrigid_hist)   # (n_steps+1, 3)

    # ── Add Gaussian noise to Xfem before anything else touches it ─────────────
    # Same NOISE definition as main_Noise_Pipeline.py. The autoencoder
    # round-trip below runs on this noisy Xfem, not the clean Xfem_hist —
    # this is a "how well does the autoencoder denoise a noisy reading" test,
    # not a clean-data reconstruction test.
    Xfem_noisy = add_noise_to_xfem(Xfem_hist.copy(), NOISE, NOISE_INDEX_GROUPS,
                                    rng=np.random.default_rng(RAND_SEED))

    # ── Autoencoder round-trip on noisy Xfem only (Xrigid is untouched) ───────
    Xfem_recon = reconstruct(model, scaler_mean, scaler_scale, Xfem_noisy)

    # ── World-frame head-node position, true vs reconstructed vs noisy ────────
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
    print(f"Head-node position MAE (true vs autoencoder round-trip of noisy Xfem): {mae:.6f} m")

    mae_noise = np.abs(np.concatenate([true_x - noisy_x, true_y - noisy_y])).mean()
    print(f"Head-node position MAE (true vs true+noise, no autoencoder): {mae_noise:.6f} m")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal', adjustable='box')
    ax.plot(true_x, true_y, '-', lw=2, color='steelblue', label='true')
    ax.plot(recon_x, recon_y, ':', lw=2, color='tomato', label='autoencoder round-trip (noisy input)')
    ax.plot(noisy_x, noisy_y, ':', lw=1.5, color='seagreen', label='true + noise (no autoencoder)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Head-node trajectory — true vs autoencoder round-trip of noisy Xfem\n'
                 f'tau={TAU.tolist()}  {SIM_DURATION}s  (latent_dim={meta["latent_dim"]})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out_path = f'{IMAGES_DIR}/autoencoder_reconstruction_trajectory.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()
