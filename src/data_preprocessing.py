"""
Data preprocessing utilities for water quality time series
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class WaterQualityPreprocessor:
    """
    Preprocessor for water quality multivariate time series data
    """
    
    def __init__(self, normalization_method='minmax', normalization_range=(-1, 1)):
        """
        Initialize preprocessor
        
        Args:
            normalization_method: 'minmax' or 'standard'
            normalization_range: tuple for MinMaxScaler range
        """
        self.normalization_method = normalization_method
        self.normalization_range = normalization_range
        self.scaler = None
        self.feature_names = None
        
    def fit_scaler(self, data: np.ndarray):
        """
        Fit the scaler on training data
        
        Args:
            data: Training data array
        """
        if self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=self.normalization_range)
        elif self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        self.scaler.fit(data)
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler
        
        Args:
            data: Data to transform
            
        Returns:
            Normalized data
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data
        
        Args:
            data: Normalized data
            
        Returns:
            Original scale data
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.inverse_transform(data)
    
    def create_sequences(self, 
                        data: np.ndarray, 
                        input_steps: int, 
                        output_steps: int,
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences using sliding window
        
        Args:
            data: Time series data (n_samples, n_features)
            input_steps: Number of historical time steps
            output_steps: Number of future steps to predict
            stride: Sliding window stride
            
        Returns:
            X: Input sequences (n_sequences, input_steps, n_features)
            y: Output sequences (n_sequences, output_steps, n_features)
        """
        X, y = [], []
        
        for i in range(0, len(data) - input_steps - output_steps + 1, stride):
            X.append(data[i:i + input_steps])
            y.append(data[i + input_steps:i + input_steps + output_steps])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self,
                    df: pd.DataFrame,
                    train_ratio: float = 0.8,
                    input_steps: int = 30,
                    output_steps: int = 3,
                    stride: int = 1,
                    feature_columns: Optional[list] = None) -> Tuple:
        """
        Complete data preparation pipeline
        
        Args:
            df: DataFrame with time series data
            train_ratio: Ratio of training data
            input_steps: Historical window size
            output_steps: Prediction horizon
            stride: Sliding window stride
            feature_columns: List of feature column names (None = all except timestamp)
            
        Returns:
            X_train, y_train, X_test, y_test, scaler
        """
        # Select features
        if feature_columns is None:
            # Assume first column is timestamp, rest are features
            feature_columns = df.columns[1:] if df.columns[0] in ['timestamp', 'date', 'time'] else df.columns
        
        self.feature_names = feature_columns
        data = df[feature_columns].values
        
        # Split train/test
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Fit scaler on training data
        self.fit_scaler(train_data)
        
        # Normalize
        train_normalized = self.transform(train_data)
        test_normalized = self.transform(test_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(
            train_normalized, input_steps, output_steps, stride
        )
        X_test, y_test = self.create_sequences(
            test_normalized, input_steps, output_steps, stride
        )
        
        print(f"Data preparation complete:")
        print(f"  Training sequences: {X_train.shape[0]}")
        print(f"  Test sequences: {X_test.shape[0]}")
        print(f"  Input shape: {X_train.shape}")
        print(f"  Output shape: {y_train.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: DataFrame with potential missing values
            method: 'interpolate', 'forward_fill', or 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        if method == 'interpolate':
            return df.interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            return df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'drop':
            return df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using z-score method
        
        Args:
            data: Data array
            threshold: Z-score threshold
            
        Returns:
            Boolean mask of outliers
        """
        z_scores = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
        return np.any(z_scores > threshold, axis=1)
