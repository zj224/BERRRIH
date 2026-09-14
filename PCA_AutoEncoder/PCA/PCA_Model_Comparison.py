"""
Compare every saved PCA fit (PCA_Models/pca_xfem_noisy_*.joblib) against each
other: reconstruction MSE and retained variance side by side, plus a single
simulated (noisy) trajectory round-tripped through every model and overlaid
on one head-node position plot — the multi-model counterpart to
PCA_Reconstruction.py's single-fit version.

MSE is computed the same way as PCA_Reconstruction.py's head-node MAE: each
model's own scaler.inverse_transform(pca.inverse_transform(...)) of a shared
noisy Xfem sample, compared against that same noisy sample (not the clean
ground truth) — so this measures how well each model's round-trip preserves
a noisy reading, not how well it denoises. Like the trajectory plot (one
simulated run), the MSE/variance evaluation is judged on a single data-pool
run too, picked at random by default — a shared sample within that one run
keeps the comparison apples-to-apples across models even when they were fit
on different DATA_POOL_RUNS.

Run directly to compare + report + plot every saved fit (or the
PCA_MODEL_RUNS subset set in main_PCA_AE.py); import compare_pca_models to
use elsewhere.
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
import matplotlib.cm as cm
import joblib
from Fish_Simulation import setup_var, setup_sys, get_controls, fish_step, fem_to_inertial
from PCA_AutoEncoder.PCA.PCA_Train import add_noise_to_xfem, load_noisy_xfem_sample, _pool_file_paths
from PCA_AutoEncoder.main_PCA_AE import (
    TAU, SIM_DURATION, X_INIT, Y_INIT, NOISE, NOISE_INDEX_GROUPS, RAND_SEED,
    SAMPLE_FRACTION, DATA_POOL_RUNS, PCA_MODEL_RUNS,
)
from paths import PCA_MODELS_DIR, DATA_POOL_DIR, IMAGES_DIR

# ── Parameters ────────────────────────────────────────────────────────────────
# TAU / SIM_DURATION / X_INIT / Y_INIT / NOISE / DATA_POOL_RUNS / PCA_MODEL_RUNS
# all come from main_PCA_AE.py — edit them there, not here.

OUT_IMG_METRICS    = IMAGES_DIR / 'pca_model_comparison_metrics.png'
OUT_IMG_TRAJECTORY = IMAGES_DIR / 'pca_model_comparison_trajectory.png'


def _run_number(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _model_paths(models_dir=PCA_MODELS_DIR, runs=PCA_MODEL_RUNS):
    """Every saved pca_xfem_noisy_*.joblib fit, sorted by run number and
    optionally restricted to specific run number(s) (the trailing int in the
    filename, e.g. pca_xfem_noisy_3.joblib -> 3)."""
    paths = sorted(glob.glob(f'{models_dir}/pca_xfem_noisy_*.joblib'), key=_run_number)
    if runs is not None:
        wanted = set(runs)
        paths = [p for p in paths if _run_number(p) in wanted]
    if not paths:
        raise FileNotFoundError(f"No PCA fits found in {models_dir}/ (runs={runs})")
    return paths


def _round_trip(fit, X):
    """scaler.transform -> pca.transform -> keep k comps -> zero-pad back to
    n_dims -> pca.inverse_transform -> scaler.inverse_transform. Same
    round-trip as PCA_Reconstruction.py, factored out since every model here
    needs it."""
    scaler, pca, k, n_dims = fit['scaler'], fit['pca'], fit['k'], fit['n_dims']
    Xp = pca.transform(scaler.transform(X))[:, :k]
    Xp_full = np.zeros((len(Xp), n_dims))
    Xp_full[:, :k] = Xp
    return scaler.inverse_transform(pca.inverse_transform(Xp_full))


# ── Metrics: reconstruction MSE + retained variance, all models on the same
#    shared eval sample ───────────────────────────────────────────────────────

def compare_model_metrics(models_dir=PCA_MODELS_DIR, runs=PCA_MODEL_RUNS,
                           data_dir=DATA_POOL_DIR, data_pool_runs=DATA_POOL_RUNS,
                           sample_fraction=SAMPLE_FRACTION, seed=RAND_SEED,
                           out_img=OUT_IMG_METRICS):
    """Load every selected PCA fit, round-trip the same noisy Xfem sample
    through each, and report/plot reconstruction MSE (against the clean,
    pre-noise ground truth — not the noisy input) + cumulative explained
    variance at each model's saved k.

    data_pool_runs: which data-pool run(s) to evaluate on. None (default)
    picks a single run at random (seeded by `seed`) — same spirit as
    compare_model_trajectories' single simulated trajectory: models are
    judged on one run, not everything pooled together. Pass a list of run
    numbers to evaluate on specific run(s) instead.

    Returns a list of {'path', 'label', 'k', 'n_dims', 'variance', 'mse',
    'cumvar'} dicts, one per model, in the same order as _model_paths.
    """
    model_paths = _model_paths(models_dir, runs)
    fits = [joblib.load(p) for p in model_paths]

    eval_runs = data_pool_runs
    if eval_runs is None:
        available = sorted({_run_number(p) for p in _pool_file_paths(data_dir, runs=None)})
        eval_runs = [int(np.random.default_rng(seed).choice(available))]

    X_noisy, X_clean = load_noisy_xfem_sample(data_dir, sample_fraction, NOISE, seed,
                                               eval_runs, return_clean=True)
    print(f"Evaluating {len(fits)} model(s) on {X_noisy.shape[0]} shared noisy Xfem rows "
          f"from data-pool run(s) {eval_runs} ({sample_fraction*100:.0f}% sample, seed={seed}), "
          f"MSE measured against the clean (pre-noise) rows")

    results = []
    for path, fit in zip(model_paths, fits):
        k, n_dims = fit['k'], fit['n_dims']
        label = f"run {_run_number(path)} (k={k})"
        cumvar = np.cumsum(fit['pca'].explained_variance_ratio_)
        variance = cumvar[k - 1]

        X_recon = _round_trip(fit, X_noisy)
        mse = float(np.mean((X_clean - X_recon) ** 2))

        results.append({
            'path': path, 'label': label, 'k': k, 'n_dims': n_dims,
            'variance': variance, 'mse': mse, 'cumvar': cumvar,
        })

    print(f"\n{'model':<20} {'k':>5} {'variance':>10} {'MSE':>12}")
    for r in results:
        print(f"{r['label']:<20} {r['k']:>5} {r['variance']*100:>9.3f}% {r['mse']:>12.6g}")

    # ── Plot: cumulative variance curves (left) + MSE bar chart (right) ──────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = cm.tab10(np.linspace(0, 1, max(len(results), 1)))

    for r, color in zip(results, colors):
        n_comp = len(r['cumvar'])
        axes[0].plot(range(1, n_comp + 1), r['cumvar'] * 100, color=color, label=r['label'])
        axes[0].axvline(r['k'], color=color, linestyle=':', linewidth=1)
    axes[0].set_xlabel('number of components')
    axes[0].set_ylabel('cumulative explained variance (%)')
    axes[0].set_title("Retained variance (dotted line = each model's saved k)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(results)), [r['mse'] for r in results], color=colors)
    axes[1].set_xticks(range(len(results)))
    axes[1].set_xticklabels([r['label'] for r in results], rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('reconstruction MSE (round-trip vs. shared noisy sample)')
    axes[1].set_title('Reconstruction MSE')
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('PCA model comparison — variance vs. reconstruction error')
    plt.tight_layout()

    if out_img:
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        plt.savefig(out_img, dpi=150)
        print(f"\nSaved plot -> {out_img}")

    return results


# ── Round-trip trajectory: every model on the same single simulated run ──────

def compare_model_trajectories(models_dir=PCA_MODELS_DIR, runs=PCA_MODEL_RUNS,
                                tau=TAU, sim_duration=SIM_DURATION,
                                x_init=X_INIT, y_init=Y_INIT, seed=RAND_SEED,
                                out_img=OUT_IMG_TRAJECTORY):
    """Simulate one fixed-tau trajectory, add noise, round-trip it through
    every selected PCA model, and plot every model's reconstructed head-node
    trajectory against the true and noisy trajectories on one plot — same
    setup as PCA_Reconstruction.py, extended to every selected model at once.
    """
    model_paths = _model_paths(models_dir, runs)
    fits = [(p, joblib.load(p)) for p in model_paths]

    p = setup_var(sim_duration, x_init, y_init, 0.01)
    p.controller = get_controls
    p = setup_sys(p)
    n_steps = round(sim_duration / p.dt)

    print(f"\nSimulating {sim_duration}s at tau={tau.tolist()}  ({n_steps} steps, dt={p.dt})")

    Xfemi   = np.zeros(p.ns, dtype=np.float64)
    Xrigidi = np.zeros(3, dtype=np.float64)
    Xrigidi[:] = p.x0[0:3]

    Xfem_hist, Xrigid_hist = [Xfemi.copy()], [Xrigidi.copy()]
    for s in range(n_steps):
        Xfemi, Xrigidi = fish_step(p, s * p.dt, Xfemi, Xrigidi, tau)
        Xfem_hist.append(Xfemi.copy())
        Xrigid_hist.append(Xrigidi.copy())
    Xfem_hist   = np.array(Xfem_hist)
    Xrigid_hist = np.array(Xrigid_hist)

    Xfem_noisy = add_noise_to_xfem(Xfem_hist.copy(), NOISE, NOISE_INDEX_GROUPS,
                                    rng=np.random.default_rng(seed))

    def head_xy(Xfem_arr, Xrigid_arr):
        Xstates = np.vstack([Xfem_arr.T, Xrigid_arr.T])
        Xinertial = fem_to_inertial(p, Xstates)
        return Xinertial[p.ix[-1], :], Xinertial[p.iy[-1], :]

    true_x, true_y   = head_xy(Xfem_hist, Xrigid_hist)
    noisy_x, noisy_y = head_xy(Xfem_noisy, Xrigid_hist)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.plot(true_x, true_y, '-', lw=2.5, color='steelblue', label='true', zorder=3)
    ax.scatter(noisy_x, noisy_y, s=8, color='gray', alpha=0.5,
               label='true + noise (no PCA)', zorder=1)

    colors = cm.tab10(np.linspace(0, 1, max(len(fits), 1)))
    for (path, fit), color in zip(fits, colors):
        k = fit['k']
        Xfem_recon = _round_trip(fit, Xfem_noisy)
        recon_x, recon_y = head_xy(Xfem_recon, Xrigid_hist)
        mae = np.abs(np.concatenate([true_x - recon_x, true_y - recon_y])).mean()
        label = f"run {_run_number(path)} (k={k}, MAE={mae:.4f}m)"
        ax.scatter(recon_x, recon_y, s=6, color=color, label=label, zorder=2)

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Head-node trajectory — true vs PCA round-trip of noisy Xfem, {len(fits)} model(s)\n'
                 f'tau={tau.tolist()}  {sim_duration}s')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_img:
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        plt.savefig(out_img, dpi=150)
        print(f"Saved {out_img}")

    return fits


def compare_pca_models(models_dir=PCA_MODELS_DIR, runs=PCA_MODEL_RUNS,
                        data_dir=DATA_POOL_DIR, data_pool_runs=DATA_POOL_RUNS,
                        sample_fraction=SAMPLE_FRACTION, seed=RAND_SEED,
                        tau=TAU, sim_duration=SIM_DURATION, x_init=X_INIT, y_init=Y_INIT):
    """Run both comparisons — reconstruction MSE/variance across models, and
    a shared round-trip trajectory overlay — in one call."""
    results = compare_model_metrics(models_dir, runs, data_dir, data_pool_runs,
                                     sample_fraction, seed)
    compare_model_trajectories(models_dir, runs, tau, sim_duration, x_init, y_init, seed)
    return results


if __name__ == '__main__':
    compare_pca_models()
    plt.show()
