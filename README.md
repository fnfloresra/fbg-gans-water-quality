# FBG-GANs for Water Quality Time Series Prediction

Implementation of **FBG-GANs (Fourier-Bidirectional GRU-Generative Adversarial Networks)** based on the paper:
> Qin, X., Shi, H., Dong, X., & Zhang, S. (2024). A new method based on generative adversarial networks for multivariate time series prediction. Expert Systems, 41(12), e13700.

## Overview

This project implements a state-of-the-art deep learning model for predicting multivariate time series in the water quality domain with 14 parameters.

### Key Features

- **Fourier Transform**: Extracts long and short-term trends from time series data
- **Bi-GRU Generator**: Captures temporal dependencies in both forward and backward directions
- **CNN Discriminator**: Efficiently discriminates between real and generated sequences
- **Wasserstein GAN**: Uses WGAN-GP to prevent mode collapse and gradient vanishing

## Architecture

### Generator (Bi-GRU)
```
Input: (batch_size, input_steps, features)
├── Bi-GRU Layer 1: 1024 neurons
├── Bi-GRU Layer 2: 512 neurons
├── Bi-GRU Layer 3: 256 neurons
├── Dense Layer 1: 128 neurons
├── Dense Layer 2: 64 neurons
└── Output Dense: output_steps
```

### Discriminator (CNN)
```
Input: (batch_size, sequence_length, 1)
├── Conv1D Layer 1: 32 filters
├── Conv1D Layer 2: 64 filters
├── Conv1D Layer 3: 128 filters
├── Flatten
├── Dense Layer 1: 220 neurons
├── Dense Layer 2: 220 neurons
└── Output Dense: 1 (real/fake)
```

## Dataset

**Water Quality Parameters (14 features):**
Your dataset should include parameters such as:
- pH, Temperature, Dissolved Oxygen (DO)
- Biochemical Oxygen Demand (BOD), Chemical Oxygen Demand (COD)
- Total Suspended Solids (TSS), Turbidity
- Ammonia, Nitrate, Phosphate
- Conductivity, Total Dissolved Solids (TDS)
- And other relevant water quality indicators

## Installation

```bash
# Clone the repository
git clone https://github.com/fnfloresra/fbg-gans-water-quality.git
cd fbg-gans-water-quality

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Place your water quality dataset in `data/raw/` directory. Expected format:
- CSV file with columns: timestamp + 14 water quality parameters
- No missing values (or use provided imputation methods)

```python
import pandas as pd
from src.data_preprocessing import WaterQualityPreprocessor

# Load data
df = pd.read_csv('data/raw/water_quality.csv')

# Preprocess
preprocessor = WaterQualityPreprocessor()
X_train, y_train, X_test, y_test = preprocessor.prepare_data(
    df, 
    train_ratio=0.8,
    input_steps=30,
    output_steps=3
)
```

### 2. Feature Engineering

```python
from src.feature_engineering import FourierFeatureExtractor

# Extract Fourier features
fourier_extractor = FourierFeatureExtractor()
X_train_extended = fourier_extractor.transform(X_train)
X_test_extended = fourier_extractor.transform(X_test)
```

### 3. Model Training

```python
from src.models.fbg_gan import FBGGAN
from src.training import train_fbg_gan

# Initialize model
model = FBGGAN(
    input_steps=30,
    output_steps=3,
    n_features=36,  # 14 original + Fourier features
    generator_units=[1024, 512, 256],
    discriminator_filters=[32, 64, 128]
)

# Train
history = train_fbg_gan(
    model,
    X_train_extended,
    y_train,
    epochs=1000,
    batch_size=64,
    learning_rate=0.001
)
```

### 4. Prediction

```python
# Make predictions
predictions = model.predict(X_test_extended)

# Evaluate
from src.evaluation import evaluate_model
metrics = evaluate_model(y_test, predictions)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

### 5. Using Scripts

```bash
# Train model
python scripts/train.py --data data/raw/water_quality.csv \
                        --epochs 1000 \
                        --batch-size 64

# Make predictions
python scripts/predict.py --model results/models/fbg_gan_best.h5 \
                          --data data/raw/water_quality_test.csv \
                          --output results/predictions.csv
```

## Configuration

Edit `config.py` to customize:

```python
CONFIG = {
    'input_steps': 30,          # Historical window size
    'output_steps': 3,          # Prediction horizon
    'n_features': 14,           # Number of water quality parameters
    'train_ratio': 0.8,         # Train/test split
    'batch_size': 64,
    'epochs': 1000,
    'learning_rate': 0.001,
    'lambda_gp': 10,            # Gradient penalty coefficient
    'n_critic': 5,              # Discriminator updates per generator update
}
```

## Performance Metrics

The model is evaluated using:

- **RMSE (Root Mean Square Error)**: Lower is better
- **MAE (Mean Absolute Error)**: Lower is better
- **R² (Coefficient of Determination)**: Higher is better (max 1.0)

## Expected Results

Based on the paper's experiments, FBG-GANs should achieve:
- R² > 0.95 on well-structured water quality data
- Significantly lower RMSE/MAE compared to LSTM, GRU, Bi-LSTM baselines
- Robust performance even during sudden water quality changes

## Project Structure

```
fbg_gans_water_quality/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.py                      # Configuration parameters
├── data/
│   ├── raw/                       # Original datasets
│   └── processed/                 # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── feature_engineering.py     # Fourier transform features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py           # Bi-GRU generator
│   │   ├── discriminator.py       # CNN discriminator
│   │   └── fbg_gan.py            # Complete FBG-GAN model
│   ├── training.py                # Training loop
│   └── evaluation.py              # Evaluation metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA and visualization
│   ├── 02_model_training.ipynb    # Interactive training
│   └── 03_results_analysis.ipynb  # Results visualization
├── scripts/
│   ├── train.py                   # Command-line training
│   └── predict.py                 # Command-line prediction
└── results/
    ├── models/                    # Saved models
    ├── figures/                   # Plots and visualizations
    └── metrics/                   # Performance metrics
```

## Troubleshooting

### Common Issues

1. **Mode Collapse**: Adjust `lambda_gp` and `n_critic` in config
2. **Training Instability**: Reduce learning rate or increase batch size
3. **Poor Performance**: Ensure proper data normalization and sufficient training epochs

## Citation

If you use this implementation, please cite:

```bibtex
@article{qin2024fbggans,
  title={A new method based on generative adversarial networks for multivariate time series prediction},
  author={Qin, Xiwen and Shi, Hongyu and Dong, Xiaogang and Zhang, Siqi},
  journal={Expert Systems},
  volume={41},
  number={12},
  pages={e13700},
  year={2024},
  publisher={Wiley}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.
