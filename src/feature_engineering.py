"""
Feature engineering using Fourier Transform
Based on the paper's methodology for extending data features
"""

import numpy as np
from typing import List


class FourierFeatureExtractor:
    """
    Extract Fourier transform features from time series
    This helps the model learn long-term and short-term trends
    """
    
    def __init__(self, components: List[int] = [3, 6, 9]):
        """
        Initialize Fourier feature extractor
        
        Args:
            components: List of component numbers to extract
                       According to paper: [3, 6, 9] components
        """
        self.components = components
        
    def extract_fourier_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract Fourier features from a single signal
        
        Args:
            signal: 1D time series signal
            
        Returns:
            Fourier features: [MA7, MA21, upper_band, lower_band, EMA, log_momentum,
                              abs_3comp, angle_3comp, abs_6comp, angle_6comp, 
                              abs_9comp, angle_9comp]
        """
        features = []
        
        # 1. Moving Averages (MA7, MA21)
        ma7 = self._moving_average(signal, 7)
        ma21 = self._moving_average(signal, 21)
        features.extend([ma7, ma21])
        
        # 2. Bollinger Bands (upper_band, lower_band)
        upper, lower = self._bollinger_bands(signal, window=20)
        features.extend([upper, lower])
        
        # 3. Exponential Moving Average (EMA)
        ema = self._exponential_moving_average(signal, span=12)
        features.append(ema)
        
        # 4. Log Momentum
        log_momentum = self._log_momentum(signal)
        features.append(log_momentum)
        
        # 5. Fourier components (absolute and angle for different components)
        fft = np.fft.fft(signal)
        
        for n_comp in self.components:
            # Get the n_comp-th component
            component = fft[n_comp] if n_comp < len(fft) else 0
            
            # Absolute value (magnitude)
            abs_val = np.abs(component)
            features.append(abs_val)
            
            # Angle (phase)
            angle = np.angle(component)
            features.append(angle)
        
        return np.array(features)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input sequences by adding Fourier features
        
        Args:
            X: Input sequences (n_sequences, input_steps, n_features)
            
        Returns:
            Extended sequences with Fourier features
            Shape: (n_sequences, input_steps, n_features + fourier_features)
        """
        n_sequences, input_steps, n_features = X.shape
        
        # Process each sequence
        X_extended = []
        
        for seq in X:
            # seq shape: (input_steps, n_features)
            seq_features = []
            
            # For each time step, extract features from each variable
            for t in range(input_steps):
                time_features = seq[t, :].copy()  # Original features at time t
                
                # Extract Fourier features for each variable
                for feature_idx in range(n_features):
                    signal = seq[:t+1, feature_idx]  # Signal up to current time
                    
                    # Only extract if we have enough data
                    if len(signal) >= 21:  # Minimum for MA21
                        fourier_feats = self.extract_fourier_features(signal)
                    else:
                        fourier_feats = np.zeros(12)  # MA7, MA21, upper, lower, EMA, log_mom, 3*2 comps
                    
                    time_features = np.concatenate([time_features, fourier_feats])
                
                seq_features.append(time_features)
            
            X_extended.append(seq_features)
        
        X_extended = np.array(X_extended)
        
        print(f"Fourier feature extraction complete:")
        print(f"  Original shape: {X.shape}")
        print(f"  Extended shape: {X_extended.shape}")
        
        return X_extended
    
    def _moving_average(self, signal: np.ndarray, window: int) -> float:
        """
        Calculate moving average for the last window points
        """
        if len(signal) < window:
            return np.mean(signal)
        return np.mean(signal[-window:])
    
    def _exponential_moving_average(self, signal: np.ndarray, span: int) -> float:
        """
        Calculate exponential moving average
        """
        if len(signal) < 2:
            return signal[-1] if len(signal) > 0 else 0
        
        alpha = 2 / (span + 1)
        ema = signal[0]
        for val in signal[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema
    
    def _bollinger_bands(self, signal: np.ndarray, window: int = 20, num_std: float = 2.0):
        """
        Calculate Bollinger Bands
        """
        if len(signal) < window:
            ma = np.mean(signal)
            std = np.std(signal)
        else:
            ma = np.mean(signal[-window:])
            std = np.std(signal[-window:])
        
        upper_band = ma + num_std * std
        lower_band = ma - num_std * std
        
        return upper_band, lower_band
    
    def _log_momentum(self, signal: np.ndarray) -> float:
        """
        Calculate log momentum
        """
        if len(signal) < 2:
            return 0
        
        # Prevent log of zero or negative
        if signal[-1] <= 0 or signal[-2] <= 0:
            return 0
        
        return np.log(signal[-1] / signal[-2])
