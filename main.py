"""
BERRRI — top-level entry point.

Each component keeps its own main_*.py inside its own folder (its step
selection RUN list lives there — edit that file directly to choose what it
runs); this just gives one place to invoke whichever of them you want,
without cd-ing into subfolders or remembering each file's path.

Add or remove names in RUN to select which pipelines to run.
"""

from Fish_Simulation.main_fish_sim import main as run_fish_simulation
from Simulation_Data_Generation.main_Sim_Data_Generation import main as run_sim_data_generation
from PCA_AutoEncoder.main_PCA_AE import main as run_pca_autoencoder

# ── Select pipelines — add/remove names to run ────────────────────────────────
RUN = [
    #"Fish Simulation",              # single simulated trajectory + animation
    #"Simulation Data Generation",   # data-pool / training-data pipeline (Simulation_Data_Generation/)
    "PCA AutoEncoder",              # PCA/autoencoder noise-experiment pipeline (PCA_AutoEncoder/)
]

# ── Pipeline registry — maps each name to its main() function ─────────────────
REGISTRY = {
    "Fish Simulation":            run_fish_simulation,
    "Simulation Data Generation": run_sim_data_generation,
    "PCA AutoEncoder":            run_pca_autoencoder,
}


if __name__ == '__main__':
    if not RUN:
        print("No pipelines selected — add names to the RUN list.")
        print(f"Valid names: {list(REGISTRY)}")
    else:
        unknown = [name for name in RUN if name not in REGISTRY]
        if unknown:
            raise SystemExit(f"Unknown pipeline name(s): {unknown}\nValid names: {list(REGISTRY)}")

        for name in RUN:
            print(f"\n{'='*70}\n  {name}\n{'='*70}")
            REGISTRY[name]()



