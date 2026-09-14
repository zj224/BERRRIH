"""
BASS Full Pipeline — main entry point.

Add or remove step names in RUN to select which steps to run.
"""

import subprocess
import sys
import time
import os

from paths import (
    PROJECT_ROOT as ROOT, MLP_TRAINING_DIR,
    SCRIPT_DATA_POOL_GENERATION, SCRIPT_TRAINING_DATA_GENERATION, SCRIPT_ROLLOUT_CYCLES,
    SCRIPT_PCA_ROLLOUT_CYCLES
)

# ── Select steps — add/remove names to run ────────────────────────────────────
# Sections below are grouped by *data type* (what state representation the step
# operates on), not by pipeline stage — e.g. the two "Train Model" variants sit
# under their own data-type sections rather than both under one "Training"
# heading, since they consume/produce genuinely different feature layouts.
RUN = [
    # ── Data Generation ────────────────────────────────────────────────────────────────────────────────────

    "Pool Generation",          # simulate fish over C_A × C_offset grid
    #"Training Data Generation", # sample period-boundary states from pool
    #"Rollout Cycles",           # multi-cycle chained rollout, fresh tau each cycle
    #"PCA Rollout Cycles",        
]

# ── Step registry — maps each name to (script_path, description) ──────────────
REGISTRY = {
    # ── Data Generation (shared) ──────────────────────────────────────────────
    "Pool Generation": (
        SCRIPT_DATA_POOL_GENERATION,
        'Step 1  —  Pool Generation          (C_A × C_offset × Cacc grid simulations)',
    ),
    "Training Data Generation": (
        SCRIPT_TRAINING_DATA_GENERATION,
        'Step 2  —  Training Data Generation  (period-boundary sampling over the C_A × C_offset × Cacc grid)',
    ),
    "Rollout Cycles": (
        SCRIPT_ROLLOUT_CYCLES,
        'Step 3  —  Rollout Cycles            (multi-cycle rollout, single unified dataset, fresh C_A/C_offset/Cacc each cycle)',
    ),
    "PCA Rollout Cycles": (
        SCRIPT_PCA_ROLLOUT_CYCLES,
        'Step 3b —  PCA Rollout Cycles         (dense short rollout, every step saved, continuous Cacc, for PCA fitting)',
    )
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
