"""
Single source of truth for every project file/directory location.

Every other script imports its paths from here instead of hardcoding them,
so moving a file only requires updating its entry in this module.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Data_Collection ─────────────────────────────────────────────────────────
DATA_COLLECTION_DIR          = PROJECT_ROOT / 'Simulation_Data_Generation' / 'Simulation_Data'
DATA_POOL_DIR                = DATA_COLLECTION_DIR / 'Data_Pool'
TRAINING_DATA_DIR            = DATA_COLLECTION_DIR / 'Training_Data'
TRAINING_DATA_COLLECTION_DIR = DATA_COLLECTION_DIR / 'Training_Data_Collection'
ROLLOUT_DATA_DIR             = TRAINING_DATA_COLLECTION_DIR / 'Rollout_Data'
PCA_ROLLOUT_DATA_DIR         = TRAINING_DATA_COLLECTION_DIR / 'PCA_Rollout_Data'
PCA_TRANSFORMED_DATA_DIR     = DATA_COLLECTION_DIR / 'PCA_Transformed_Data'
AUTOENCODER_MODELS_DIR       = DATA_COLLECTION_DIR / 'AutoEncoder_Models'
MODELS_DIR                    = DATA_COLLECTION_DIR / 'Models'

# Frozen sim-trained surrogate, shared (read-only) by the Sim2Real/MPC scripts.
MODEL_PATH = MODELS_DIR / 'model_1.pth'

# ── Results ──────────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / 'Results'
IMAGES_DIR  = RESULTS_DIR / 'Images'

# ── Code directories (sys.path / subprocess PYTHONPATH setup) ────────────────
MLP_TRAINING_DIR        = PROJECT_ROOT / 'MLP_Training'
TRAINING_FUNCTIONS_DIR  = MLP_TRAINING_DIR / 'Training_Functions'
PLOTTING_DIR             = PROJECT_ROOT / 'Plotting'
ANALYSIS_FUNCTIONS_DIR  = PLOTTING_DIR / 'Analysis_Functions'
FISH_SIMULATION_DIR      = PROJECT_ROOT / 'Fish_Simulation'
SIM_DATA_GEN_DIR         = PROJECT_ROOT / 'Simulation_Data_Generation'
PCA_DIR = PROJECT_ROOT / 'PCA_AutoEncoder' / 'PCA'
AUTOENCODER_DIR = PROJECT_ROOT / 'PCA_AutoEncoder' / 'AutoEncoder'
PCA_MODELS_DIR = PCA_DIR / 'PCA_Models'

# ── Pipeline step scripts — main_MLP_Pipeline.py ──────────────────────────────
SCRIPT_DATA_POOL_GENERATION       = SIM_DATA_GEN_DIR / 'Data_Pool_Generation.py'
SCRIPT_TRAINING_DATA_GENERATION   = TRAINING_FUNCTIONS_DIR / 'MLP_Training_Data_Generation.py'
SCRIPT_ROLLOUT_CYCLES             = TRAINING_FUNCTIONS_DIR / 'MLP_Rollout_Cycles.py'
SCRIPT_PCA_ROLLOUT_CYCLES         = TRAINING_FUNCTIONS_DIR / 'MLP_PCA_Rollout_Cycles.py'
SCRIPT_PCA_ANALYSIS_FEM           = TRAINING_FUNCTIONS_DIR / 'MLP_PCA_Analysis_FEM.py'
SCRIPT_PCA_ANALYSIS_DATA_POOL     = TRAINING_FUNCTIONS_DIR / 'MLP_PCA_Analysis_Data_Pool.py'
SCRIPT_PCA_ANALYSIS_ROLLOUT_CYCLES = TRAINING_FUNCTIONS_DIR / 'MLP_PCA_Analysis_Rollout_Cycles.py'
SCRIPT_PCA_APPLY_ROLLOUT_CYCLES   = TRAINING_FUNCTIONS_DIR / 'MLP_PCA_Apply_Rollout_Cycles.py'
SCRIPT_MODEL_TRAINING             = TRAINING_FUNCTIONS_DIR / 'MLP_Model_Training.py'
SCRIPT_PCA_RECONSTRUCTION_PLOT      = PCA_DIR / 'PCA_Reconstruction.py'
SCRIPT_PCA_COMPONENT_ANALYSIS_PLOT  = ANALYSIS_FUNCTIONS_DIR / 'MLP_Plot_PCA_Components.py'
SCRIPT_ROLLING_PREDICTION_PLOT      = ANALYSIS_FUNCTIONS_DIR / 'MLP_Plot_Rolling_Prediction_Cacc.py'

# ── Pipeline step scripts — main_PCA_AE.py ────────────────────────────────────
SCRIPT_PCA_TRAIN               = PCA_DIR / 'PCA_Train.py'
SCRIPT_PCA_MODEL_COMPARISON    = PCA_DIR / 'PCA_Model_Comparison.py'
SCRIPT_AUTOENCODER_TRAIN       = AUTOENCODER_DIR / 'AutoEncoder_Train.py'
SCRIPT_AUTOENCODER_RECONSTRUCTION_PLOT = AUTOENCODER_DIR / 'AutoEncoder_Reconstruction.py'
SCRIPT_PCA_VS_AUTOENCODER_PLOT = ANALYSIS_FUNCTIONS_DIR / 'MLP_Plot_PCA_vs_AutoEncoder.py'
