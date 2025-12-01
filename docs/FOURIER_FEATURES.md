# Fourier Transform Feature Extraction

This document explains the **Fourier Transform feature extraction** process used in FBG-GANs, as described in the paper and illustrated in **Figure 4**.

## Overview

The Fourier transform is used to **extend the data features** to help the Bi-GRU generator better learn the distributional characteristics of the original time series data. This preprocessing step is crucial for the model's performance.

## Feature Extraction Process

From the paper's Figure 4, we extract **12 features** for each time series variable:

### 1. Moving Averages

#### MA7 (Moving Average 7)
```
MA7 = mean(last 7 values)
```
Captures short-term trends in the data.

#### MA21 (Moving Average 21)
```
MA21 = mean(last 21 values)
```
Captures medium-term trends in the data.

### 2. Bollinger Bands

#### Upper Band
```
Upper Band = MA20 + 2 * std(last 20 values)
```
Indicates the upper volatility boundary.

#### Lower Band
```
Lower Band = MA20 - 2 * std(last 20 values)
```
Indicates the lower volatility boundary.

### 3. Exponential Moving Average (EMA)

```python
alpha = 2 / (span + 1)  # span = 12
EMA[t] = alpha * value[t] + (1 - alpha) * EMA[t-1]
```

Gives more weight to recent observations while still considering historical data.

### 4. Log Momentum

```
log_momentum = log(current_value / previous_value)
```

Captures the rate of change in logarithmic scale, useful for identifying trends and momentum.

### 5. Fourier Components

For each of the Fourier component indices **[3, 6, 9]**, we extract:

#### Absolute Value (Magnitude)
```
abs_k = |FFT[k]|
```
Represents the strength of the k-th frequency component.

#### Angle (Phase)
```
angle_k = atan2(Im(FFT[k]), Re(FFT[k]))
```
Represents the phase of the k-th frequency component.

**This gives us 6 Fourier-based features:**
- `abs_3`, `angle_3` (3rd component)
- `abs_6`, `angle_6` (6th component)
- `abs_9`, `angle_9` (9th component)

## Complete Feature List

For each time series variable, we generate **12 features**:

1. MA7
2. MA21
3. upperband
4. lowerband
5. EMA
6. log_momentum
7. absolute_of_3_comp
8. angle_of_3_comp
9. absolute_of_6_comp
10. angle_of_6_comp
11. absolute_of_9_comp
12. angle_of_9_comp

## Application to Water Quality Data

### Original Data
For a water quality dataset with **14 parameters**:
- pH, Temperature, DO, BOD, COD, TSS, Turbidity, Ammonia, Nitrate, Phosphate, Conductivity, TDS, Chloride, Hardness

### Extended Data
After Fourier transformation:
- **Original features**: 14
- **Fourier features per parameter**: 12
- **Total Fourier features**: 14 × 12 = 168
- **Total extended features**: 14 + 168 = **182 features**

### Input Shape Transformation

```
Original:  (batch_size, 30, 14)
           ↓
           [Fourier Transform]
           ↓
Extended:  (batch_size, 30, 182)
```

Where:
- `batch_size`: Number of sequences in batch
- `30`: Input time steps (historical window)
- `14`: Original water quality parameters
- `182`: Original + Fourier-extended features

## Implementation Example

```python
from src.feature_engineering import FourierFeatureExtractor
import numpy as np

# Your original data
X_train = np.random.randn(100, 30, 14)  # 100 sequences, 30 timesteps, 14 features

# Initialize extractor
extractor = FourierFeatureExtractor(components=[3, 6, 9])

# Transform to extended features
X_train_extended = extractor.transform(X_train)

print(f"Original shape: {X_train.shape}")        # (100, 30, 14)
print(f"Extended shape: {X_train_extended.shape}")  # (100, 30, 182)
```

## Why These Features?

### Technical Indicators
- **MA7, MA21**: Capture different time-scale trends
- **Bollinger Bands**: Measure volatility and potential outliers
- **EMA**: Emphasizes recent changes
- **Log Momentum**: Captures growth rates

### Fourier Components
- **Components 3, 6, 9**: Capture periodicity patterns at different frequencies
- **Magnitude (absolute)**: Strength of periodic patterns
- **Phase (angle)**: Timing/offset of periodic patterns

### Benefits for Water Quality
1. **Seasonal patterns**: Fourier components detect daily, weekly cycles
2. **Trend detection**: Moving averages smooth noise
3. **Anomaly detection**: Bollinger bands identify unusual values
4. **Change sensitivity**: Log momentum and EMA react to shifts

## Matching Paper Architecture

The paper shows `Input(Bs, 30, 36)` in Figure 4. This suggests:
- They may have used fewer original features (e.g., only key water quality indicators)
- Or applied feature selection to reduce from 182 to 36 dimensions

You can use `transform_reduced()` method for this:

```python
# Reduce to match paper's dimension
X_extended = extractor.transform_reduced(X_train, n_top_features=36)
print(X_extended.shape)  # (100, 30, 36)
```

## Mathematical Justification

### Fourier Transform

The Fourier transform decomposes a time series into its frequency components:

```
FFT[k] = Σ(x[n] * e^(-2πikn/N))  for n = 0 to N-1
```

Where:
- `x[n]`: Time series values
- `N`: Length of series
- `k`: Frequency index

### Why Components 3, 6, 9?

These specific indices capture:
- **Low-frequency components** (3, 6, 9 out of N total)
- **Long-term periodic patterns**
- **Stable seasonal trends** rather than high-frequency noise

For a 30-timestep window:
- Component 3: Period of 10 timesteps (e.g., weekly if daily data)
- Component 6: Period of 5 timesteps
- Component 9: Period of ~3.3 timesteps

## Performance Impact

From the paper's results:

| Model | RMSE (Air Quality) | R² |
|-------|-------------------:|----:|
| Bi-GRU (no Fourier) | 0.0064 | 0.9872 |
| **FBG-GANs (with Fourier)** | **0.0015** | **0.9994** |

The Fourier features provide:
- ✅ **76% reduction in RMSE**
- ✅ **1.2% improvement in R²** (very significant near 1.0)
- ✅ Better capturing of periodic patterns
- ✅ More robust to sudden changes

## Tips for Your Dataset

### 1. Feature Selection
If 182 features is too many:
```python
# Use PCA or feature importance
from sklearn.decomposition import PCA

pca = PCA(n_components=36)
X_reduced = pca.fit_transform(X_extended.reshape(-1, 182))
X_reduced = X_reduced.reshape(batch_size, 30, 36)
```

### 2. Adjust Components
For different periodicities:
```python
# For daily data with weekly/monthly patterns
extractor = FourierFeatureExtractor(components=[7, 14, 30])

# For hourly data with daily patterns
extractor = FourierFeatureExtractor(components=[6, 12, 24])
```

### 3. Normalize First
Always normalize before Fourier transform:
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
X_normalized = scaler.fit_transform(X.reshape(-1, 14))
X_normalized = X_normalized.reshape(batch_size, 30, 14)

X_extended = extractor.transform(X_normalized)
```

## References

1. Qin, X., Shi, H., Dong, X., & Zhang, S. (2024). A new method based on generative adversarial networks for multivariate time series prediction. *Expert Systems*, 41(12), e13700.

2. Figure 4: Flowchart of the overall framework of the FBG-GANs (page 6 of paper)

## Summary

The Fourier transform feature extraction:
- ✅ Extends each variable from 1 feature to 13 features (original + 12 Fourier)
- ✅ Captures multiple time-scale patterns (short, medium, long-term)
- ✅ Provides both trend and periodic information
- ✅ Significantly improves model performance
- ✅ Is essential for FBG-GANs to achieve state-of-the-art results
