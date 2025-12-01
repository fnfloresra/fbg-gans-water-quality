"""
Feature engineering using Fourier Transform
Based on the paper's Figure 4 - Extended Data Features

From the paper (Figure 4), the Fourier features are:
- MA7: Moving Average with window 7
- MA21: Moving Average with window 21
- upperband: Upper Bollinger Band
- lowerband: Lower Bollinger Band
- EMA: Exponential Moving Average
- log_momentum: Log momentum indicator
- absolute of 3 comp: Magnitude of 3rd Fourier component
- angle of 3 comp: Phase of 3rd Fourier component
- absolute of 6 comp: Magnitude of 6th Fourier component
- angle of 6 comp: Phase of 6th Fourier component
- absolute of 9 comp: Magnitude of 9th Fourier component
- angle of 9 comp: Phase of 9th Fourier component

Total: 12 features per variable
For 14 water quality parameters: 14 original + 14*12 Fourier = 182 total features
"""

import numpy as np
from typing import List


class FourierFeatureExtractor:
    """
    Extract Fourier transform features from time series
    This extends the original features to help Bi-GRU learn distributional characteristics
    """
    
    def __init__(self, components: List[int] = [3, 6, 9]):
        """
        Initialize Fourier feature extractor
        
        Args:
            components: Fourier component indices to extract [3, 6, 9] as per paper
        """
        self.components = components
        
    def extract_fourier_features_from_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract 12 Fourier features from a single time series signal
        
        Args:
            signal: 1D array of time series values
            
        Returns:
            Array of 12 features: [MA7, MA21, upper, lower, EMA, log_mom, 
                                   abs_3, ang_3, abs_6, ang_6, abs_9, ang_9]
        """
        n = len(signal)
        features = []
        
        # 1. MA7 - Moving Average with window 7
        window = min(7, n)
        ma7 = np.mean(signal[-window:]) if n > 0 else 0.0
        features.append(ma7)
        
        # 2. MA21 - Moving Average with window 21
        window = min(21, n)
        ma21 = np.mean(signal[-window:]) if n > 0 else 0.0
        features.append(ma21)
        
        # 3 & 4. Bollinger Bands (upper_band, lower_band)
        window = min(20, n)
        if n >= window:
            rolling_mean = np.mean(signal[-window:])
            rolling_std = np.std(signal[-window:])
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std
        else:
            mean_val = np.mean(signal) if n > 0 else 0.0
            std_val = np.std(signal) if n > 1 else 0.0
            upper_band = mean_val + 2 * std_val
            lower_band = mean_val - 2 * std_val
        features.append(upper_band)
        features.append(lower_band)
        
        # 5. EMA - Exponential Moving Average (span=12)
        ema = self._calculate_ema(signal, span=12)
        features.append(ema)
        
        # 6. log_momentum
        log_mom = self._calculate_log_momentum(signal)
        features.append(log_mom)
        
        # 7-12. Fourier components (absolute and angle for components 3, 6, 9)
        if n > max(self.components):
            # Perform FFT
            fft_result = np.fft.fft(signal)
            
            for comp_idx in self.components:
                if comp_idx < len(fft_result):
                    # Absolute value (magnitude)
                    abs_val = np.abs(fft_result[comp_idx])
                    features.append(abs_val)
                    
                    # Angle (phase)
                    angle_val = np.angle(fft_result[comp_idx])
                    features.append(angle_val)
                else:
                    features.extend([0.0, 0.0])
        else:
            # Not enough data for FFT, use zeros
            features.extend([0.0] * 6)  # 3 components * 2 (abs + angle)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_ema(self, signal: np.ndarray, span: int = 12) -> float:
        """
        Calculate Exponential Moving Average
        
        Args:
            signal: Time series signal
            span: EMA span
            
        Returns:
            EMA value
        """
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
        """
        Calculate log momentum: log(current / previous)
        
        Args:
            signal: Time series signal
            
        Returns:
            Log momentum value
        """
        if len(signal) < 2:
            return 0.0
        
        current = signal[-1]
        previous = signal[-2]
        
        # Avoid log of zero or negative values
        if current <= 0 or previous <= 0:
            return 0.0
        
        return np.log(current / previous)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input sequences by adding Fourier features
        
        Args:
            X: Input sequences (n_sequences, input_steps, n_features)
               For example: (batch_size, 30, 14) for 30 timesteps, 14 water quality params
            
        Returns:
            Extended sequences with Fourier features
            Shape: (n_sequences, input_steps, n_features + n_features*12)
            For example: (batch_size, 30, 14 + 14*12) = (batch_size, 30, 182)
                        Or as shown in paper: (Bs, 30, 36) where 36 = original + extended features
        
        Note: The paper shows Input(Bs, 30, 36) in Figure 4. This suggests they may have
              used fewer original features or selected specific Fourier features.
              Adjust based on your dataset.
        """
        n_sequences, input_steps, n_features = X.shape
        
        print(f"Extracting Fourier features...")
        print(f"  Input shape: {X.shape}")
        print(f"  Features per variable: 12 (MA7, MA21, upper, lower, EMA, log_mom, 3*FFT components)")
        
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
                    
                    # Extract 12 Fourier features
                    fourier_feats = self.extract_fourier_features_from_signal(signal)
                    fourier_features_all.extend(fourier_feats)
                
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
        
        return X_extended
    
    def transform_reduced(self, X: np.ndarray, n_top_features: int = None) -> np.ndarray:
        """
        Transform with optional feature reduction
        
        This is useful if you want to match the paper's architecture exactly.
        For example, if paper shows (Bs, 30, 36), you might want to reduce
        from 182 total features to 36.
        
        Args:
            X: Input sequences
            n_top_features: Number of top features to keep (None = keep all)
            
        Returns:
            Extended and optionally reduced sequences
        """
        X_extended = self.transform(X)
        
        if n_top_features is not None and n_top_features < X_extended.shape[-1]:
            print(f"  Reducing features from {X_extended.shape[-1]} to {n_top_features}")
            # Simple approach: take first n_top_features
            # More sophisticated: use feature selection methods
            X_extended = X_extended[:, :, :n_top_features]
        
        return X_extended


def create_extended_features(X: np.ndarray, 
                            components: List[int] = [3, 6, 9]) -> np.ndarray:
    """
    Convenience function to create extended features
    
    Args:
        X: Input sequences (n_sequences, input_steps, n_features)
        components: Fourier component indices
        
    Returns:
        Extended sequences with Fourier features
    """
    extractor = FourierFeatureExtractor(components=components)
    return extractor.transform(X)
