"""
Comparison: Full vs Lite Fourier Feature Extraction

This script compares the full 12-feature version with the lite 3-feature version
to help you decide which one to use for your water quality dataset.
"""

import numpy as np
import time
import sys
sys.path.append('..')

from src.feature_engineering import FourierFeatureExtractor
from src.feature_engineering_lite import FourierFeatureExtractorLite, FourierFeatureExtractorCustom


def create_sample_data(n_sequences=100, timesteps=30, n_features=14):
    """Create sample water quality data"""
    return np.random.randn(n_sequences, timesteps, n_features).astype(np.float32)


def compare_versions():
    """Compare full vs lite Fourier feature extraction"""
    
    print("\n" + "="*80)
    print("FOURIER FEATURE EXTRACTION COMPARISON")
    print("Full (12 features/param) vs Lite (3 features/param)")
    print("="*80)
    
    # Create sample data
    print("\n[1] Creating sample water quality data...")
    X = create_sample_data(n_sequences=1000, timesteps=30, n_features=14)
    print(f"    Data shape: {X.shape}")
    print(f"    Size in memory: {X.nbytes / 1024**2:.2f} MB")
    
    # Test Full Version (12 features per parameter)
    print("\n" + "-"*80)
    print("[2] FULL VERSION - 12 features per parameter")
    print("-"*80)
    
    extractor_full = FourierFeatureExtractor(components=[3, 6, 9])
    
    start_time = time.time()
    X_full = extractor_full.transform(X)
    full_time = time.time() - start_time
    
    print(f"\n    Features extracted:")
    print(f"      - MA7, MA21, Upper/Lower Bands, EMA, Log Momentum")
    print(f"      - FFT: 3 components × 2 (magnitude + phase)")
    print(f"      - Total: 12 features per parameter")
    print(f"\n    Result shape: {X_full.shape}")
    print(f"    Original features: {X.shape[-1]}")
    print(f"    Extended features: {X_full.shape[-1]}")
    print(f"    Size in memory: {X_full.nbytes / 1024**2:.2f} MB")
    print(f"    Extraction time: {full_time:.3f} seconds")
    
    # Test Lite Version (3 features per parameter)
    print("\n" + "-"*80)
    print("[3] LITE VERSION - 3 features per parameter")
    print("-"*80)
    
    extractor_lite = FourierFeatureExtractorLite(ma_window=7, fft_component=3)
    
    start_time = time.time()
    X_lite = extractor_lite.transform(X)
    lite_time = time.time() - start_time
    
    print(f"\n    Features extracted:")
    print(f"      - MA7 (Moving Average)")
    print(f"      - Std (Standard Deviation)")
    print(f"      - FFT Magnitude (3rd component)")
    print(f"      - Total: 3 features per parameter")
    print(f"\n    Result shape: {X_lite.shape}")
    print(f"    Original features: {X.shape[-1]}")
    print(f"    Extended features: {X_lite.shape[-1]}")
    print(f"    Size in memory: {X_lite.nbytes / 1024**2:.2f} MB")
    print(f"    Extraction time: {lite_time:.3f} seconds")
    
    # Comparison Summary
    print("\n" + "="*80)
    print("[4] COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Full':<15} {'Lite':<15} {'Speedup/Reduction'}")
    print("-"*80)
    
    print(f"{'Features per parameter':<30} {12:<15} {3:<15} {12/3:.1f}x reduction")
    print(f"{'Total features':<30} {X_full.shape[-1]:<15} {X_lite.shape[-1]:<15} {X_full.shape[-1]/X_lite.shape[-1]:.2f}x reduction")
    print(f"{'Memory (MB)':<30} {X_full.nbytes/1024**2:<15.2f} {X_lite.nbytes/1024**2:<15.2f} {X_full.nbytes/X_lite.nbytes:.2f}x reduction")
    print(f"{'Extraction time (s)':<30} {full_time:<15.3f} {lite_time:<15.3f} {full_time/lite_time:.2f}x faster")
    
    # Test Custom Version
    print("\n" + "-"*80)
    print("[5] CUSTOM VERSION - Choose your own features")
    print("-"*80)
    
    custom_features_list = [
        ['ma7', 'std', 'fft_mag'],
        ['ma7', 'ma21', 'fft_mag'],
        ['ema', 'log_mom', 'fft_mag'],
        ['ma7', 'bollinger_upper', 'bollinger_lower'],
    ]
    
    print("\n    Testing different feature combinations...\n")
    
    for i, features in enumerate(custom_features_list, 1):
        extractor_custom = FourierFeatureExtractorCustom(selected_features=features)
        
        start_time = time.time()
        X_custom = extractor_custom.transform(X)
        custom_time = time.time() - start_time
        
        print(f"    Option {i}: {features}")
        print(f"      Shape: {X_custom.shape}")
        print(f"      Time: {custom_time:.3f}s")
        print()
    
    # Recommendations
    print("\n" + "="*80)
    print("[6] RECOMMENDATIONS")
    print("="*80)
    
    print("\n    Use FULL VERSION when:")
    print("      ✓ Maximum accuracy is critical")
    print("      ✓ Computational resources are available")
    print("      ✓ Dataset is small-medium (<10K sequences)")
    print("      ✓ Training time is not a constraint")
    
    print("\n    Use LITE VERSION when:")
    print("      ✓ Need faster experimentation")
    print("      ✓ Limited memory/GPU resources")
    print("      ✓ Large dataset (>10K sequences)")
    print("      ✓ Real-time prediction required")
    print("      ✓ Edge device deployment")
    
    print("\n    Use CUSTOM VERSION when:")
    print("      ✓ You know which features matter most")
    print("      ✓ Domain expertise suggests specific indicators")
    print("      ✓ Fine-tuning for optimal performance")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")
    
    return X_full, X_lite


if __name__ == "__main__":
    compare_versions()
