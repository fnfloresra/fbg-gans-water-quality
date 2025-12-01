"""
Example: Fourier Transform Feature Extraction for Water Quality Data

This script demonstrates how the Fourier transform extends features
from the original 14 water quality parameters to 182 extended features.
"""

import numpy as np
import sys
sys.path.append('..')

from src.feature_engineering import FourierFeatureExtractor


def create_sample_water_quality_data(n_samples=100, timesteps=30, n_features=14):
    """
    Create synthetic water quality time series data
    
    Simulates realistic water quality patterns with:
    - Daily cycles (temperature, DO)
    - Weekly patterns (pollution indicators)
    - Random fluctuations (natural variability)
    """
    print("\n" + "="*70)
    print("Creating Synthetic Water Quality Data")
    print("="*70)
    
    data = []
    
    for i in range(n_samples):
        # Create time-based patterns
        t = np.linspace(0, 2*np.pi, timesteps)
        
        sequence = []
        for step in range(timesteps):
            features = []
            
            # pH (7.0-8.5, slight daily variation)
            ph = 7.5 + 0.5 * np.sin(t[step]) + np.random.normal(0, 0.1)
            features.append(ph)
            
            # Temperature (15-25°C, daily cycle)
            temp = 20 + 3 * np.sin(t[step]) + np.random.normal(0, 0.5)
            features.append(temp)
            
            # Dissolved Oxygen (5-10 mg/L, inverse temp correlation)
            do = 7.5 - 1.5 * np.sin(t[step]) + np.random.normal(0, 0.3)
            features.append(do)
            
            # BOD (1-5 mg/L)
            bod = 3 + np.random.normal(0, 0.5)
            features.append(bod)
            
            # COD (10-30 mg/L)
            cod = 20 + np.random.normal(0, 3)
            features.append(cod)
            
            # TSS (5-50 mg/L)
            tss = 25 + np.random.normal(0, 5)
            features.append(tss)
            
            # Turbidity (1-10 NTU)
            turbidity = 5 + np.random.normal(0, 1)
            features.append(turbidity)
            
            # Ammonia (0.1-2 mg/L)
            ammonia = 1 + np.random.normal(0, 0.2)
            features.append(ammonia)
            
            # Nitrate (1-10 mg/L)
            nitrate = 5 + np.random.normal(0, 1)
            features.append(nitrate)
            
            # Phosphate (0.1-1 mg/L)
            phosphate = 0.5 + np.random.normal(0, 0.1)
            features.append(phosphate)
            
            # Conductivity (200-800 μS/cm)
            conductivity = 500 + np.random.normal(0, 50)
            features.append(conductivity)
            
            # TDS (100-500 mg/L)
            tds = 300 + np.random.normal(0, 30)
            features.append(tds)
            
            # Chloride (10-100 mg/L)
            chloride = 50 + np.random.normal(0, 10)
            features.append(chloride)
            
            # Hardness (50-200 mg/L)
            hardness = 125 + np.random.normal(0, 20)
            features.append(hardness)
            
            sequence.append(features)
        
        data.append(sequence)
    
    X = np.array(data, dtype=np.float32)
    
    print(f"Generated data shape: {X.shape}")
    print(f"  - {n_samples} sequences")
    print(f"  - {timesteps} time steps per sequence")
    print(f"  - {n_features} water quality parameters")
    
    return X


def demonstrate_fourier_extraction():
    """
    Demonstrate the Fourier transform feature extraction process
    """
    # Step 1: Create sample data
    print("\n" + "="*70)
    print("Step 1: Generate Sample Water Quality Data")
    print("="*70)
    
    X = create_sample_water_quality_data(n_samples=10, timesteps=30, n_features=14)
    
    # Step 2: Initialize Fourier extractor
    print("\n" + "="*70)
    print("Step 2: Initialize Fourier Feature Extractor")
    print("="*70)
    
    extractor = FourierFeatureExtractor(components=[3, 6, 9])
    print(f"Fourier components: {extractor.components}")
    print("Features to extract per variable: 12")
    print("  1. MA7 (Moving Average 7)")
    print("  2. MA21 (Moving Average 21)")
    print("  3. Upper Bollinger Band")
    print("  4. Lower Bollinger Band")
    print("  5. EMA (Exponential Moving Average)")
    print("  6. Log Momentum")
    print("  7. Absolute of 3rd Fourier component")
    print("  8. Angle of 3rd Fourier component")
    print("  9. Absolute of 6th Fourier component")
    print(" 10. Angle of 6th Fourier component")
    print(" 11. Absolute of 9th Fourier component")
    print(" 12. Angle of 9th Fourier component")
    
    # Step 3: Extract features
    print("\n" + "="*70)
    print("Step 3: Extract Fourier Features")
    print("="*70)
    
    X_extended = extractor.transform(X)
    
    # Step 4: Analyze results
    print("\n" + "="*70)
    print("Step 4: Feature Extraction Results")
    print("="*70)
    
    print(f"\nOriginal data:")
    print(f"  Shape: {X.shape}")
    print(f"  Features per timestep: {X.shape[-1]}")
    
    print(f"\nExtended data:")
    print(f"  Shape: {X_extended.shape}")
    print(f"  Features per timestep: {X_extended.shape[-1]}")
    
    original_features = X.shape[-1]
    extended_features = X_extended.shape[-1]
    fourier_features = extended_features - original_features
    
    print(f"\nFeature breakdown:")
    print(f"  Original features: {original_features}")
    print(f"  Fourier features added: {fourier_features}")
    print(f"  Total features: {extended_features}")
    print(f"  Expansion ratio: {extended_features / original_features:.1f}x")
    
    # Step 5: Example of extracted features for one timestep
    print("\n" + "="*70)
    print("Step 5: Example Feature Values (First Sequence, Last Timestep)")
    print("="*70)
    
    seq_idx = 0
    time_idx = -1  # Last timestep
    
    print(f"\nOriginal features (14 water quality parameters):")
    param_names = ['pH', 'Temp', 'DO', 'BOD', 'COD', 'TSS', 'Turbidity', 
                   'Ammonia', 'Nitrate', 'Phosphate', 'Conductivity', 
                   'TDS', 'Chloride', 'Hardness']
    
    for i, name in enumerate(param_names):
        print(f"  {name:12s}: {X[seq_idx, time_idx, i]:8.3f}")
    
    print(f"\nExample Fourier features for pH (first 6 of 12):")
    fourier_feature_names = ['MA7', 'MA21', 'Upper Band', 'Lower Band', 'EMA', 'Log Mom']
    
    # pH is the first parameter, so its Fourier features start at index 14
    fourier_start = original_features
    for i, name in enumerate(fourier_feature_names):
        print(f"  {name:12s}: {X_extended[seq_idx, time_idx, fourier_start + i]:8.3f}")
    
    print("\n  (... plus 6 more Fourier component features)")
    
    # Step 6: Ready for model training
    print("\n" + "="*70)
    print("Step 6: Ready for FBG-GAN Training")
    print("="*70)
    
    print(f"\nThe extended data is now ready to be fed into the Bi-GRU generator:")
    print(f"  Input shape to generator: {X_extended.shape}")
    print(f"  Expected by paper: (Batch_size, 30, features)")
    print(f"\nThis extended feature set helps the model learn:")
    print(f"  ✓ Short-term trends (MA7)")
    print(f"  ✓ Medium-term trends (MA21)")
    print(f"  ✓ Volatility patterns (Bollinger Bands)")
    print(f"  ✓ Recent dynamics (EMA, Log Momentum)")
    print(f"  ✓ Periodic patterns (Fourier components)")
    
    return X, X_extended


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  FBG-GANs: Fourier Transform Feature Extraction Demo".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    X_original, X_extended = demonstrate_fourier_extraction()
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use your actual water quality dataset")
    print("  2. Apply the same Fourier transformation")
    print("  3. Train the FBG-GAN model with extended features")
    print("  4. Compare performance with/without Fourier features")
    print("\nSee docs/FOURIER_FEATURES.md for detailed documentation.")
    print("="*70 + "\n")
