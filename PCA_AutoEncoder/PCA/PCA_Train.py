"""
PCA analysis on Xfem, sourced from the raw data pool, same as
MLP_PCA_Analysis_Data_Pool.py — except PCA is fit on a random *sample* of
the pool's Xfem rows (like MLP_PCA_Analysis_Rollout_Cycles.py's
SAMPLE_FRACTION) with synthetic Gaussian noise added, instead of the full,
noise-free pool. The point is to see how the clean PCA's
variance/components-needed picture changes when the data is noisy (e.g. to
approximate sensor/measurement noise instead of the clean simulated state).

Xfem's P_NS = 2*ndof layout is [x1..xnn, y1..ynn, th1..thnn] (positions)
followed by [vx1..vxnn, vy1..vynn, w1..wnn] (velocities) — see
Fish_Simulation/setup_sys.py's p.ix/p.iy/p.ith. NOISE (imported from
main_PCA_AE.py, the single source of truth for every noise-experiment
setting — see that file) defines one (mean, variance) Gaussian per component
*type* (x, y, th, vx, vy, w), applied identically — but with an independent
random draw per node and per sample — to every node's value of that
component. That's 6 noise distributions total, not one per node.

Nothing is written to disk except the PCA fit itself — the sampled rows and
the noise draw are both seeded (RAND_SEED), so re-running this script
reproduces the exact same noisy data on demand instead of needing a saved
copy of it.

Autoencoder_noise.py imports add_noise_to_xfem / load_noisy_xfem_sample from
here directly rather than duplicating them — the sampling+noise step is
identical for both, only what's fit/trained on the result differs.

Run directly to sample + add noise + fit + report + plot + save; import
run_data_pool_pca_pipeline_noisy / load_noisy_xfem_sample to use elsewhere.
"""

import numpy as np
import os
import re
import glob
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from PCA_AutoEncoder.main_PCA_AE import (
    RAND_SEED, SAMPLE_FRACTION, DATA_POOL_RUNS, NOISE, NOISE_INDEX_GROUPS, P_NS,
)
from paths import DATA_POOL_DIR, PCA_MODELS_DIR, IMAGES_DIR

OUT_IMG = IMAGES_DIR / 'pca_component_analysis_data_pool_noisy.png'

# Cumulative-variance thresholds to report a component count for.
VAR_THRESHOLDS = [0.8, 0.85, 0.9, 0.95]

# How many components to keep in the *saved* fit. None = auto-pick via
# SAVE_VAR_THRESHOLD; set an int to override.
#N_COMPONENTS_TO_SAVE = None
N_COMPONENTS_TO_SAVE = 9
SAVE_VAR_THRESHOLD   = 0.99


def _existing_run_numbers(pattern):
    """All run numbers found among files matching glob pattern (the first
    digit-group in each filename). Scans every match rather than walking
    forward from 1, so a gap from a deleted run can't hide a higher-numbered
    one that still exists."""
    nums = []
    for path in glob.glob(pattern):
        m = re.search(r'(\d+)', os.path.basename(path))
        if m:
            nums.append(int(m.group(1)))
    return nums


def _next_run_number(pattern):
    return max(_existing_run_numbers(pattern), default=0) + 1


def _pool_file_paths(data_dir=DATA_POOL_DIR, runs=DATA_POOL_RUNS):
    """Return fish-simulation data-pool file(s), optionally restricted to
    specific run(s).

    The current generator writes a single full data-pool tensor per run
    (shape [n_tau, n_steps, 1 + P_NS + 3]), while older comments/docs refer to
    per-boundary boundary files. Accept both layouts so the analysis script can
    run against the files that are actually present in this repo.

    runs: None (default, from DATA_POOL_RUNS in main_PCA_AE.py) = every run
    found in data_dir; else an iterable of run numbers (the trailing int in
    each filename, e.g. fish_simulation_data_pool_3.npy -> 3) to restrict to.
    """
    paths = sorted(glob.glob(f'{data_dir}/fish_simulation_data_pool_*.npy'))
    paths = [p for p in paths if 'tau_matrix' not in os.path.basename(p)]
    if runs is not None:
        wanted = set(runs)
        paths = [p for p in paths if _run_number(p) in wanted]
    if not paths:
        raise FileNotFoundError(
            f"No data-pool files found in {data_dir}/ (runs={runs}) — "
            f"run Data_Pool_Generation.py first"
        )
    return paths


def _run_number(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


# ── Sampling + noise injection (nothing written to disk) ────────────────────

def add_noise_to_xfem(X, noise=NOISE, index_groups=NOISE_INDEX_GROUPS, rng=None):
    """Add per-component-type Gaussian noise to an (N, P_NS) Xfem array, in
    place. One (mean, variance) draw distribution per component *type*, but
    an independent random value per sample and per node. Returns X (same
    object, mutated).
    """
    rng = rng if rng is not None else np.random.default_rng()
    for comp, idx in index_groups.items():
        mean, var = noise[comp]
        if mean == 0.0 and var == 0.0:
            continue
        std = np.sqrt(var)
        X[:, idx] += rng.normal(mean, std, size=(X.shape[0], len(idx)))
    return X


def load_noisy_xfem_sample(data_dir=DATA_POOL_DIR, sample_fraction=SAMPLE_FRACTION,
                            noise=NOISE, seed=RAND_SEED, runs=DATA_POOL_RUNS,
                            return_clean=False):
    """Randomly sample a `sample_fraction` of Xfem rows from every data-pool
    file (memory-mapped, so only the sampled rows are actually read) and add
    Gaussian `noise`, all in memory. Same `seed` -> same sampled rows and the
    same noise draw every time, so this is fully reproducible without
    persisting anything to disk.

    runs: which data-pool run(s) to sample from — None (default, from
    DATA_POOL_RUNS in main_PCA_AE.py) pools every run found in data_dir;
    pass a list of run numbers to restrict to specific run(s). See
    _pool_file_paths.

    return_clean: if True, also return the same rows *before* noise was
    added, as a second return value (X_noisy, X_clean) — for evaluating
    reconstruction against ground truth rather than against the noisy input.
    """
    if all(mean == 0.0 and var == 0.0 for mean, var in noise.values()):
        print("NOISE is all (0.0, 0.0) — set mean/var in NOISE (main_PCA_AE.py) "
              "before running for this to actually inject any noise.")

    pool_paths = _pool_file_paths(data_dir, runs)
    print(f"Training on {len(pool_paths)} data-pool file(s): "
          f"{[os.path.basename(p) for p in pool_paths]}")

    rng = np.random.default_rng(seed)
    chunks = []
    for path in pool_paths:
        arr = np.load(path, mmap_mode='r')
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        xfem = arr[:, 1:1 + P_NS]
        n_rows = xfem.shape[0]
        n_keep = max(1, int(round(n_rows * sample_fraction)))
        idx = np.sort(rng.choice(n_rows, size=n_keep, replace=False))
        chunks.append(np.asarray(xfem[idx]))

    X = np.concatenate(chunks, axis=0)
    X = X[np.isfinite(X).all(axis=1)]
    X_clean = X.copy() if return_clean else None
    add_noise_to_xfem(X, noise, NOISE_INDEX_GROUPS, rng)
    return (X, X_clean) if return_clean else X


# ── PCA fit (unchanged from MLP_PCA_Analysis_Data_Pool.py) ──────────────────

def analyze_pca_incremental(get_chunks, n_dims, label, out_img=None, save_path=None,
                             save_n_components=N_COMPONENTS_TO_SAVE,
                             save_var_threshold=SAVE_VAR_THRESHOLD):
    """Memory-safe PCA: standardizes and fits PCA one chunk at a time instead
    of concatenating everything into one array first.

    get_chunks: zero-arg callable returning a *fresh* iterator of (n_i, n_dims)
    arrays each time it's called (called twice here — once to fit the scaler,
    once for the PCA fit — so it must be re-iterable, e.g. a generator
    function, not an already-exhausted generator object).

    Prints an explained-variance report, plots per-component and cumulative
    variance (saved to out_img if given), and returns (ipca, scaler, cumvar).

    If save_path is given, also persists {scaler, pca, k, n_dims, label} via
    joblib — k (the number of components actually kept for downstream use) is
    save_n_components if given, else however many are needed to reach
    save_var_threshold cumulative variance. This is a *smaller* k than the
    full n_dims-component fit used for the report/plot above — PCA components
    are variance-ordered, so keeping the first k of an already-fit n_dims-
    component PCA is equivalent to having fit a k-component PCA directly, no
    refitting needed.
    """
    scaler = StandardScaler()
    n_total = 0
    for chunk in get_chunks():
        chunk = chunk[np.isfinite(chunk).all(axis=1)]
        if len(chunk):
            scaler.partial_fit(chunk)
            n_total += len(chunk)

    ipca = IncrementalPCA(n_components=n_dims)
    for chunk in get_chunks():
        chunk = chunk[np.isfinite(chunk).all(axis=1)]
        if len(chunk) >= n_dims:   # IncrementalPCA needs each batch >= n_components
            ipca.partial_fit(scaler.transform(chunk))

    cumvar = np.cumsum(ipca.explained_variance_ratio_)
    n_comp = len(cumvar)

    print(f"\n{'='*60}\n{label}: {n_total} samples × {n_dims} dims\n{'='*60}")
    print(f"{'components':>10}  {'cum. variance':>14}")
    checkpoints = sorted(set(k for k in [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, n_comp] if k <= n_comp))
    for k in checkpoints:
        print(f"{k:>10}  {cumvar[k-1]*100:>13.3f}%")

    print(f"\nComponents needed to reach a given variance threshold:")
    for thresh in VAR_THRESHOLDS:
        k = min(int(np.searchsorted(cumvar, thresh) + 1), n_comp)
        print(f"  {thresh*100:>6.1f}%  ->  {k:>4} components  (of {n_dims})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n_show = min(50, n_comp)

    axes[0].bar(range(1, n_show + 1), ipca.explained_variance_ratio_[:n_show] * 100)
    axes[0].set_xlabel('component')
    axes[0].set_ylabel('explained variance (%)')
    axes[0].set_title(f'{label}: per-component variance (first {n_show}/{n_comp})')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, n_comp + 1), cumvar * 100)
    for thresh in VAR_THRESHOLDS:
        k = min(int(np.searchsorted(cumvar, thresh) + 1), n_comp)
        axes[1].axhline(thresh * 100, color='gray', linestyle='--', linewidth=0.8)
        axes[1].axvline(k, color='gray', linestyle=':', linewidth=0.8)
    axes[1].set_xlabel('number of components')
    axes[1].set_ylabel('cumulative explained variance (%)')
    axes[1].set_title(f'{label}: cumulative explained variance')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'PCA component analysis — {label}')
    plt.tight_layout()

    if out_img:
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        plt.savefig(out_img, dpi=150)
        print(f"\nSaved plot -> {out_img}")

    if save_path:
        if save_n_components is not None:
            k = save_n_components
        else:
            k = min(int(np.searchsorted(cumvar, save_var_threshold) + 1), n_comp)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({
            'scaler': scaler,
            'pca': ipca,
            'k': k,
            'n_dims': n_dims,
            'var_threshold': save_var_threshold if save_n_components is None else None,
            'label': label,
        }, save_path)
        print(f"Saved PCA fit ({k} component(s), {cumvar[k-1]*100:.3f}% variance) -> {save_path}")

    return ipca, scaler, cumvar


def run_data_pool_pca_pipeline_noisy(data_dir=DATA_POOL_DIR, out_img=OUT_IMG,
                                      sample_fraction=SAMPLE_FRACTION, seed=RAND_SEED,
                                      runs=DATA_POOL_RUNS):
    """Xfem-only PCA on a random noisy sample of the data pool, start to
    finish: sample `sample_fraction` of rows (seeded, reproducible), add
    Gaussian NOISE, fit PCA, save the fit.

    runs: which data-pool run(s) to sample from — None (default, from
    DATA_POOL_RUNS in main_PCA_AE.py) pools every run found in data_dir;
    pass a list of run numbers to restrict to specific run(s).

    Returns the path to the saved PCA fit in the shared PCA model directory,
    using a `pca_xfem_noisy_{n}.joblib` naming convention so it's never
    confused with a clean-data fit.
    """
    X = load_noisy_xfem_sample(data_dir, sample_fraction, NOISE, seed, runs)
    print(f"Sampled {X.shape[0]} noisy Xfem rows ({sample_fraction*100:.0f}% of {data_dir}/, seed={seed})")

    os.makedirs(PCA_MODELS_DIR, exist_ok=True)
    pca_n = _next_run_number(f'{PCA_MODELS_DIR}/pca_xfem_noisy_*.joblib')
    pca_save_path = f'{PCA_MODELS_DIR}/pca_xfem_noisy_{pca_n}.joblib'
    analyze_pca_incremental(
        lambda: iter([X]), P_NS,
        f'Xfem only, noisy data pool sample ({P_NS} dims, {sample_fraction*100:.0f}% sampled, seed={seed})',
        out_img, save_path=pca_save_path,
    )
    return pca_save_path


if __name__ == '__main__':
    run_data_pool_pca_pipeline_noisy()
    plt.show()
