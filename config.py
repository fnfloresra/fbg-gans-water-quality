"""
Configuration file for FBG-GANs Water Quality Prediction
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Results directories
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    # Data parameters
    'input_steps': 30,              # Number of historical time steps
    'output_steps': 3,              # Number of future steps to predict
    'n_features': 14,               # Number of water quality parameters
    'train_ratio': 0.8,             # Train/test split ratio
    'validation_ratio': 0.1,        # Validation split from training data
    
    # Generator architecture (Bi-GRU)
    'generator_units': [1024, 512, 256],  # Number of units in each Bi-GRU layer
    'generator_dense': [128, 64],         # Dense layers after Bi-GRU
    'generator_dropout': 0.2,             # Dropout rate
    
    # Discriminator architecture (CNN)
    'discriminator_filters': [32, 64, 128],  # Filters in each Conv1D layer
    'discriminator_kernel_size': 3,          # Kernel size for convolutions
    'discriminator_dense': [220, 220],       # Dense layers after CNN
    'discriminator_dropout': 0.3,            # Dropout rate
    
    # Fourier transform
    'fourier_components': [3, 6, 9],      # Number of components to extract
    'use_fourier': True,                   # Enable Fourier feature extraction
    
    # Training parameters
    'batch_size': 64,
    'epochs': 1000,
    'learning_rate': 0.001,
    'beta_1': 0.5,                        # Adam optimizer parameter
    'beta_2': 0.999,                      # Adam optimizer parameter
    
    # Wasserstein GAN parameters
    'lambda_gp': 10,                      # Gradient penalty coefficient
    'n_critic': 5,                        # Discriminator updates per generator update
    
    # Data normalization
    'normalization_method': 'minmax',     # 'minmax' or 'standard'
    'normalization_range': (-1, 1),       # Range for MinMax scaling
    
    # Early stopping
    'patience': 50,                       # Epochs without improvement
    'min_delta': 0.0001,                  # Minimum change to qualify as improvement
    
    # Model saving
    'save_best_only': True,
    'save_frequency': 10,                 # Save model every N epochs
    
    # Logging
    'log_frequency': 10,                  # Print metrics every N epochs
    'verbose': 1,
}

# Water Quality Parameters
# Customize this list based on your specific dataset
WATER_QUALITY_PARAMS = [
    'pH',
    'Temperature',
    'Dissolved_Oxygen',
    'BOD',                    # Biochemical Oxygen Demand
    'COD',                    # Chemical Oxygen Demand
    'TSS',                    # Total Suspended Solids
    'Turbidity',
    'Ammonia',
    'Nitrate',
    'Phosphate',
    'Conductivity',
    'TDS',                    # Total Dissolved Solids
    'Chloride',
    'Hardness',
]

# Evaluation metrics
EVALUATION_METRICS = ['rmse', 'mae', 'r2', 'mape']

# Random seed for reproducibility
RANDOM_SEED = 42

# GPU configuration
GPU_CONFIG = {
    'use_gpu': True,
    'gpu_memory_fraction': 0.8,    # Fraction of GPU memory to use
    'allow_growth': True,           # Allow dynamic memory growth
}

# Baseline models for comparison (optional)
BASELINE_MODELS = [
    'LSTM',
    'GRU',
    'Bi-LSTM',
    'Bi-GRU',
    'GAN-LSTM',
]
