import numpy as np


def get_controls(p, t_curr, tau, Cacc = 1):
    """Port of getControls.m

    k is the 0-based index into p.t (i.e. the same time sample MATLAB's
    1-based p.t(k) would refer to when k_matlab = k + 1).
    """
    # swim straight
    # C_A = 1
    # C_w = 1
    # C_off = 0

    # turning
    C_A = tau[0]
    C_w = 1
    C_off = tau[1]

    ii = Cacc*(1 / 1.33) * np.linspace(1 / p.nin, 1, p.nin)

    #if t_curr < 10:
    #    ii = (1 / 1.33) * np.linspace(1 / p.nin, 1, p.nin)
    #else:
    #    ii = -(1 / 1.33) * np.linspace(1 / p.nin, 1, p.nin)
    uk = C_A * p.A * p.AmpCo * np.sin(2 * np.pi * (ii + C_w * p.w * t_curr)) + C_off * p.offsetCo
    return uk.reshape(-1, 1)


def get_controls_batch(p, t_curr, tau_batch):
    """Batched get_controls: tau_batch is (n_batch, 2); t_curr is a scalar or
    (n_batch,) array (each batch element may be at a different point in time).
    Returns uk of shape (nin, n_batch). Equivalent to get_controls_batch_cacc
    with Cacc=1 (get_controls's own default).
    """
    C_A   = tau_batch[:, 0]   # (n_batch,)
    C_off = tau_batch[:, 1]   # (n_batch,)
    C_w   = 1

    ii    = (1 / 1.33) * np.linspace(1 / p.nin, 1, p.nin)   # (nin,)
    phase = ii[:, None] + C_w * p.w * t_curr                # (nin, n_batch) via broadcasting
    uk = (C_A[None, :] * p.A * p.AmpCo[:, None] * np.sin(2 * np.pi * phase)
          + C_off[None, :] * p.offsetCo[:, None])
    return uk   # (nin, n_batch)


def get_controls_batch_cacc(p, t_curr, tau_batch, Cacc):
    """Batched get_controls with an explicit Cacc (get_controls_batch always
    behaves as Cacc=1). tau_batch is (n_batch, 2); t_curr is a scalar or
    (n_batch,) array. Cacc may be a scalar (same acceleration coefficient
    applied to every sample in the batch) or a (n_batch,) array (a
    per-sample coefficient, e.g. a freshly-drawn continuous Cacc alongside
    C_A/C_offset for each sample) — both broadcast to (n_batch,) first, so
    a scalar call produces exactly the same result as before. Returns uk of
    shape (nin, n_batch).
    """
    C_A   = tau_batch[:, 0]   # (n_batch,)
    C_off = tau_batch[:, 1]   # (n_batch,)
    C_w   = 1

    base     = (1 / 1.33) * np.linspace(1 / p.nin, 1, p.nin)                 # (nin,)
    Cacc_arr = np.broadcast_to(np.asarray(Cacc, dtype=float), C_A.shape)     # (n_batch,)
    ii    = Cacc_arr[None, :] * base[:, None]   # (nin, n_batch) — per-sample phase offset
    phase = ii + C_w * p.w * t_curr             # (nin, n_batch) via broadcasting
    uk = (C_A[None, :] * p.A * p.AmpCo[:, None] * np.sin(2 * np.pi * phase)
          + C_off[None, :] * p.offsetCo[:, None])
    return uk   # (nin, n_batch)
