"""
Lightweight Fourier Transform Feature Extraction

This simplified version extracts only 3 Fourier features per variable
instead of the full 12 features, making it more computationally efficient
while still capturing essential time series patterns.

For 14 water quality parameters:
- Original: 14 features
- Fourier: 14 × 3 = 42 features
- Total: 14 + 42 = 56 features (vs 182 in full version)

The 3 features selected are:
1. Moving Average (MA) - Captures trend
2. Standard Deviation (Std) - Captures volatility
3. FFT Magnitude (dominant frequency) - Captures periodicity
"""

import numpy as np
from typing import Optional


class FourierFeatureExtractorLite:
    """
    Lightweight Fourier feature extractor with only 3 features per variable
    
    This is a simplified alternative to the full 12-feature extractor,
    designed for faster computation and lower memory usage.
    """
    
    def __init__(self, ma_window: int = 7, fft_component: int = 3):
        """
        Initialize lightweight Fourier feature extractor
        
        Args:
            ma_window: Window size for moving average (default: 7)
            fft_component: Which FFT component to extract (default: 3)
        """
        self.ma_window = ma_window
        self.fft_component = fft_component
        
    def extract_lite_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract 3 essential features from a time series signal
        
        Args:
            signal: 1D array of time series values
            
        Returns:
            Array of 3 features: [MA, Std, FFT_magnitude]
        """
        n = len(signal)
        
        if n == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # 1. Moving Average - captures trend
        window = min(self.ma_window, n)
        ma = np.mean(signal[-window:])
        
        # 2. Standard Deviation - captures volatility/variability
        std = np.std(signal[-window:]) if window > 1 else 0.0
        
        # 3. FFT Magnitude - captures dominant periodic pattern
        if n > self.fft_component:
            fft_result = np.fft.fft(signal)
            fft_mag = np.abs(fft_result[self.fft_component])
        else:
            fft_mag = 0.0
        
        return np.array([ma, std, fft_mag], dtype=np.float32)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input sequences by adding lightweight Fourier features
        
        Args:
            X: Input sequences (n_sequences, input_steps, n_features)
               For example: (batch_size, 30, 14) for 30 timesteps, 14 water quality params
            
        Returns:
            Extended sequences with Fourier features
            Shape: (n_sequences, input_steps, n_features + n_features*3)
            For example: (batch_size, 30, 14 + 42) = (batch_size, 30, 56)
        """
        n_sequences, input_steps, n_features = X.shape
        
        print(f"Extracting lightweight Fourier features...")
        print(f"  Input shape: {X.shape}")
        print(f"  Features per variable: 3 (MA, Std, FFT magnitude)")
        
        X_extended_list = []
        
        # Process each sequence
        for seq_idx in range(n_sequences):
            seq = X[seq_idx]  # Shape: (input_steps, n_features)
            seq_extended = []
            
            # Process each time step
            for t in range(input_steps):
                # Original features at time t
                original_features = seq[t, :]
                
                # Extract Fourier features for each variable
                fourier_features_all = []
                
                for feat_idx in range(n_features):
                    # Get the signal up to current time step
                    signal = seq[:t+1, feat_idx]
                    
                    # Extract 3 lite features
                    lite_feats = self.extract_lite_features(signal)
                    fourier_features_all.extend(lite_feats)
                
                # Combine original + Fourier features
                combined_features = np.concatenate([
                    original_features, 
                    np.array(fourier_features_all)
                ])
                
                seq_extended.append(combined_features)
            
            X_extended_list.append(seq_extended)
        
        X_extended = np.array(X_extended_list, dtype=np.float32)
        
        print(f"  Extended shape: {X_extended.shape}")
        print(f"  Features increased from {n_features} to {X_extended.shape[-1]}")
        print(f"  Memory reduction vs full version: {182/X_extended.shape[-1]:.1f}x")
        
        return X_extended
    
    def get_feature_names(self, original_feature_names: list) -> list:
        """
        Generate feature names for extended dataset
        
        Args:
            original_feature_names: List of original feature names
            
        Returns:
            List of all feature names (original + Fourier)
        """
        feature_names = original_feature_names.copy()
        
        for orig_name in original_feature_names:
            feature_names.append(f"{orig_name}_MA{self.ma_window}")
            feature_names.append(f"{orig_name}_Std")
            feature_names.append(f"{orig_name}_FFT{self.fft_component}")
        
        return feature_names


class FourierFeatureExtractorCustom:
    """
    Customizable Fourier feature extractor
    
    Allows you to select exactly which features to extract
    """
    
    AVAILABLE_FEATURES = {
        'ma7': 'Moving Average (7-step window)',
        'ma21': 'Moving Average (21-step window)',
        'std': 'Standard Deviation',
        'ema': 'Exponential Moving Average',
        'log_mom': 'Log Momentum',
        'fft_mag': 'FFT Magnitude (configurable component)',
        'fft_phase': 'FFT Phase (configurable component)',
        'bollinger_upper': 'Upper Bollinger Band',
        'bollinger_lower': 'Lower Bollinger Band',
    }
    
    def __init__(self, selected_features: list = ['ma7', 'std', 'fft_mag'],
                 fft_component: int = 3, ema_span: int = 12):
        """
        Initialize custom Fourier feature extractor
        
        Args:
            selected_features: List of feature names to extract
            fft_component: Which FFT component to use (default: 3)
            ema_span: EMA span parameter (default: 12)
        """
        self.selected_features = selected_features
        self.fft_component = fft_component
        self.ema_span = ema_span
        
        # Validate features
        for feat in selected_features:
            if feat not in self.AVAILABLE_FEATURES:
                raise ValueError(f"Unknown feature: {feat}. Available: {list(self.AVAILABLE_FEATURES.keys())}")
    
    def extract_custom_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract selected features from signal
        
        Args:
            signal: 1D time series signal
            
        Returns:
            Array of selected features
        """
        n = len(signal)
        features = []
        
        for feat_name in self.selected_features:
            if feat_name == 'ma7':
                window = min(7, n)
                val = np.mean(signal[-window:]) if n > 0 else 0.0
                
            elif feat_name == 'ma21':
                window = min(21, n)
                val = np.mean(signal[-window:]) if n > 0 else 0.0
                
            elif feat_name == 'std':
                window = min(7, n)
                val = np.std(signal[-window:]) if window > 1 else 0.0
                
            elif feat_name == 'ema':
                val = self._calculate_ema(signal, self.ema_span)
                
            elif feat_name == 'log_mom':
                val = self._calculate_log_momentum(signal)
                
            elif feat_name == 'fft_mag':
                if n > self.fft_component:
                    fft_result = np.fft.fft(signal)
                    val = np.abs(fft_result[self.fft_component])
                else:
                    val = 0.0
                    
            elif feat_name == 'fft_phase':
                if n > self.fft_component:
                    fft_result = np.fft.fft(signal)
                    val = np.angle(fft_result[self.fft_component])
                else:
                    val = 0.0
                    
            elif feat_name == 'bollinger_upper':
                window = min(20, n)
                if n >= window:
                    mean = np.mean(signal[-window:])
                    std = np.std(signal[-window:])
                    val = mean + 2 * std
                else:
                    val = np.mean(signal) if n > 0 else 0.0
                    
            elif feat_name == 'bollinger_lower':
                window = min(20, n)
                if n >= window:
                    mean = np.mean(signal[-window:])
                    std = np.std(signal[-window:])
                    val = mean - 2 * std
                else:
                    val = np.mean(signal) if n > 0 else 0.0
            
            features.append(val)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_ema(self, signal: np.ndarray, span: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(signal) == 0:
            return 0.0
        if len(signal) == 1:
            return signal[0]
        
        alpha = 2.0 / (span + 1)
        ema = signal[0]
        for val in signal[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema
    
    def _calculate_log_momentum(self, signal: np.ndarray) -> float:
        """Calculate log momentum"""
        if len(signal) < 2:
            return 0.0
        if signal[-1] <= 0 or signal[-2] <= 0:
            return 0.0
        return np.log(signal[-1] / signal[-2])
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input sequences with custom features
        
        Args:
            X: Input sequences (n_sequences, input_steps, n_features)
            
        Returns:
            Extended sequences
        """
        n_sequences, input_steps, n_features = X.shape
        n_selected = len(self.selected_features)
        
        print(f"Extracting custom Fourier features...")
        print(f"  Input shape: {X.shape}")
        print(f"  Selected features: {self.selected_features}")
        print(f"  Features per variable: {n_selected}")
        
        X_extended_list = []
        
        for seq_idx in range(n_sequences):
            seq = X[seq_idx]
            seq_extended = []
            
            for t in range(input_steps):
                original_features = seq[t, :]
                fourier_features_all = []
                
                for feat_idx in range(n_features):
                    signal = seq[:t+1, feat_idx]
                    custom_feats = self.extract_custom_features(signal)
                    fourier_features_all.extend(custom_feats)
                
                combined_features = np.concatenate([
                    original_features,
                    np.array(fourier_features_all)
                ])
                
                seq_extended.append(combined_features)
            
            X_extended_list.append(seq_extended)
        
        X_extended = np.array(X_extended_list, dtype=np.float32)
        
        print(f"  Extended shape: {X_extended.shape}")
        print(f"  Features increased from {n_features} to {X_extended.shape[-1]}")
        
        return X_extended


def create_lite_features(X: np.ndarray, 
                        ma_window: int = 7, 
                        fft_component: int = 3) -> np.ndarray:
    """
    Convenience function to create lite Fourier features
    
    Args:
        X: Input sequences (n_sequences, input_steps, n_features)
        ma_window: Moving average window size
        fft_component: FFT component index
        
    Returns:
        Extended sequences with lite Fourier features
    """
    extractor = FourierFeatureExtractorLite(
        ma_window=ma_window,
        fft_component=fft_component
    )
    return extractor.transform(X)


def create_custom_features(X: np.ndarray,
                          features: list = ['ma7', 'std', 'fft_mag']) -> np.ndarray:
    """
    Convenience function to create custom Fourier features
    
    Args:
        X: Input sequences
        features: List of feature names to extract
        
    Returns:
        Extended sequences with custom features
    """
    extractor = FourierFeatureExtractorCustom(selected_features=features)
    return extractor.transform(X)
