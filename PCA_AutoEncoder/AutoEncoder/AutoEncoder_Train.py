"""
Autoencoder (encoder/decoder MLP) alternative to the PCA compression used
elsewhere in this repo (MLP_PCA_Analysis_*.py). Fits a nonlinear Xfem ->
latent -> Xfem round trip instead of a linear PCA one.

Standalone: not wired into main_MLP_Pipeline.py or the PCA-consuming training
scripts. Run this directly to train + save a fit; use MLP_Autoencoder_Test.py
to load a saved fit and round-trip data through it.

train_autoencoder(X, ...) is data-source-agnostic (any (N, n_dims) array of
Xfem rows) so it can be pointed at Data_Pool, Rollout_Cycles, or FEM data
later. load_xfem_data_pool() is just the default/convenience loader used by
the __main__ pipeline below — like MLP_PCA_Analysis_Rollout_Cycles.py's
SAMPLE_FRACTION (but unlike MLP_PCA_Analysis_Data_Pool.py, which fits on
every row), it only reads a random SAMPLE_FRACTION of each file's rows,
since a nonlinear autoencoder fit doesn't need the full pool to converge.
"""

import os
import sys
# Project root: MLP_Training/Training_Functions/ -> up two levels
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import re
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Fish_Simulation import setup_var
from paths import DATA_POOL_DIR, AUTOENCODER_MODELS_DIR
from PCA_AutoEncoder.main_PCA_AE import (LATENT_DIM, HIDDEN_DIMS, DROPOUT, RAND_SEED, SAMPLE_FRACTION)

# ── Parameters ────────────────────────────────────────────────────────────────
BATCH_SIZE     = 256
NUM_EPOCHS     = 500
LR             = 3e-4
EARLY_STOP_PAT = 20
LR_REDUCE_PAT  = 10
VAL_FRACTION   = 0.2

# Only setup_var() is needed for p.ns (= 2*p.ndof, set by get_sys_vars() inside
# it) — the full setup_sys() FEM eigendecomposition isn't needed just for that.
_p = setup_var(20.0, -0.5, 0.0, 0.01)
P_NS = 2 * _p.ndof


# ── Data intake ───────────────────────────────────────────────────────────────

def _existing_run_numbers(pattern):
    nums = []
    for path in glob.glob(pattern):
        m = re.search(r'(\d+)', os.path.basename(path))
        if m:
            nums.append(int(m.group(1)))
    return nums


def _next_run_number(pattern):
    return max(_existing_run_numbers(pattern), default=0) + 1


def _pool_file_paths(data_dir=DATA_POOL_DIR):
    paths = sorted(glob.glob(f'{data_dir}/fish_simulation_data_pool_*.npy'))
    paths = [p for p in paths if 'tau_matrix' not in os.path.basename(p)]
    if not paths:
        raise FileNotFoundError(
            f"No data-pool files found in {data_dir}/ — run MLP_Data_Pool_Generation.py first"
        )
    return paths


def load_xfem_data_pool(data_dir=DATA_POOL_DIR, sample_fraction=SAMPLE_FRACTION, seed=RAND_SEED):
    """Concatenate a random `sample_fraction` of rows from every data-pool
    file's Xfem block into one (N, P_NS) array (set sample_fraction=1.0 to
    use every row instead).

    Handles both repository layouts: old-style per-boundary arrays
    ([n_rows, 1+P_NS+3]) and the current full tensor ([n_tau, n_steps,
    1+P_NS+3]). Files are memory-mapped so the per-file subsample is read
    without paying for a full-file load first. Rows with any non-finite
    value are dropped.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for path in _pool_file_paths(data_dir):
        arr = np.load(path, mmap_mode='r')
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        xfem = arr[:, 1:1 + P_NS]
        if sample_fraction < 1.0:
            n_rows = xfem.shape[0]
            n_keep = max(1, int(round(n_rows * sample_fraction)))
            idx = np.sort(rng.choice(n_rows, size=n_keep, replace=False))
            xfem = xfem[idx]
        chunks.append(np.asarray(xfem))
    X = np.concatenate(chunks, axis=0)
    X = X[np.isfinite(X).all(axis=1)]
    return X


# ── Model ─────────────────────────────────────────────────────────────────────

def _mlp_layers(in_dim, out_dim, hidden_dims, dropout):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return layers


class Encoder(nn.Module):
    def __init__(self, n_dims, latent_dim, hidden_dims, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(*_mlp_layers(n_dims, latent_dim, hidden_dims, dropout))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_dims, hidden_dims, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(*_mlp_layers(latent_dim, n_dims, list(reversed(hidden_dims)), dropout))

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, n_dims, latent_dim, hidden_dims, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(n_dims, latent_dim, hidden_dims, dropout)
        self.decoder = Decoder(latent_dim, n_dims, hidden_dims, dropout)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


# ── Training ──────────────────────────────────────────────────────────────────

def train_autoencoder(X, latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
                       batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, lr=LR,
                       early_stop_pat=EARLY_STOP_PAT, lr_reduce_pat=LR_REDUCE_PAT,
                       val_fraction=VAL_FRACTION, seed=RAND_SEED):
    """Standardize X and train an encoder/decoder MLP to reconstruct it
    (MSE loss). Returns (model, scaler, history) where model is the
    best-val-loss Autoencoder, scaler is the fitted StandardScaler, and
    history is {'train_losses': [...], 'val_losses': [...]}.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    n_dims = X.shape[1]
    print(f"Training autoencoder: {X.shape[0]} samples x {n_dims} dims -> latent_dim={latent_dim}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val = train_test_split(X_scaled, test_size=val_fraction, random_state=seed)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32)

    X_train_t = to_tensor(X_train).to(device)
    X_val_t = to_tensor(X_val).to(device)
    n_train = X_train_t.shape[0]

    model = Autoencoder(n_dims, latent_dim, hidden_dims, dropout=dropout).to(device)
    print(f"Model: {n_dims} -> {hidden_dims} -> {latent_dim} -> {list(reversed(hidden_dims))} -> {n_dims}  "
          f"(dropout={dropout})")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_reduce_pat, factor=0.5)

    os.makedirs(AUTOENCODER_MODELS_DIR, exist_ok=True)
    best_path = f'{AUTOENCODER_MODELS_DIR}/best_model_temp.pth'
    best_val, patience_ctr = float('inf'), 0
    train_losses, val_losses = [], []
    t0_train = time.time()
    log_every = max(1, num_epochs // 10)   # ~10% progress increments
    log_every = 1

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_train, n_batches = 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            total_train += loss.item()
            n_batches += 1
        avg_train = total_train / n_batches

        model.eval()
        with torch.no_grad():
            avg_val = criterion(model(X_val_t), X_val_t).item()

        scheduler.step(avg_val)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % log_every == 0 or epoch == num_epochs - 1:
            improved = '*' if patience_ctr == 0 else ''
            print(f"Epoch {epoch + 1:4d}/{num_epochs}  train={avg_train:.6f}  val={avg_val:.6f}  "
                  f"best={best_val:.6f}{improved}  patience={patience_ctr}/{early_stop_pat}")

        if patience_ctr >= early_stop_pat:
            print(f"Early stop at epoch {epoch + 1}  best_val={best_val:.6f}")
            break

    print(f"Training done in {time.time() - t0_train:.1f}s")
    model.load_state_dict(torch.load(best_path, weights_only=False))
    os.remove(best_path)

    return model, scaler, {'train_losses': train_losses, 'val_losses': val_losses}


def save_autoencoder(model, scaler, n_dims, latent_dim, hidden_dims, dropout, path, label=None):
    """Persist {encoder+decoder weights, architecture, scaler stats} via
    torch.save — the encoder/decoder analogue of the PCA scripts' joblib.dump
    of {scaler, pca, k, n_dims}.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_dims': n_dims,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'label': label,
    }, path)
    print(f"Saved autoencoder fit (latent_dim={latent_dim}) -> {path}")


def run_autoencoder_training_pipeline(data_dir=DATA_POOL_DIR, latent_dim=LATENT_DIM,
                                       hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
                                       sample_fraction=SAMPLE_FRACTION):
    """Load a random sample of Xfem from the data pool, train an autoencoder,
    save the fit under the same `_{n}` run-numbering convention as the PCA
    scripts. Returns the saved path.
    """
    X = load_xfem_data_pool(data_dir, sample_fraction=sample_fraction)
    print(f"Loaded {X.shape[0]} Xfem rows ({X.shape[1]} dims) from {data_dir}/ "
          f"({sample_fraction*100:.0f}% sample)")

    model, scaler, history = train_autoencoder(X, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)

    n = _next_run_number(f'{AUTOENCODER_MODELS_DIR}/autoencoder_xfem_*.pth')
    save_path = f'{AUTOENCODER_MODELS_DIR}/autoencoder_xfem_{n}.pth'
    save_autoencoder(model, scaler, X.shape[1], latent_dim, hidden_dims, dropout, save_path,
                      label='Xfem, data pool')
    return save_path


if __name__ == '__main__':
    saved_path = run_autoencoder_training_pipeline()
    print(f"\n{'='*70}\nAutoencoder trained: {saved_path}")
