"""
LEAF - Baseline Forecasters
===========================
baseline forecasting models: Naive Persistence, Moving Average
for comparing and evaluating the improvement of LightGBM
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base import BaseForecaster


class NaivePersistence(BaseForecaster):
    """
    naive persistence model
    prediction value = last observed value
    """
    
    def __init__(self, lag_col: str = 'lag_1h'):
        super().__init__(name="Naive Persistence")
        self.lag_col = lag_col
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaivePersistence':
        """naive model does not need training"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """use lag value as prediction"""
        if self.lag_col not in X.columns:
            raise ValueError(f"need '{self.lag_col}' column as lag feature")
        return X[self.lag_col].values


class SeasonalNaive(BaseForecaster):
    """
    seasonal naive model
    prediction value = value 24 hours ago (capture daily cycle)
    """
    
    def __init__(self, lag_col: str = 'lag_24h'):
        super().__init__(name="Seasonal Naive (24h)")
        self.lag_col = lag_col
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SeasonalNaive':
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.lag_col not in X.columns:
            raise ValueError(f"need '{self.lag_col}' column as lag feature")
        return X[self.lag_col].values


class MovingAverage(BaseForecaster):
    """
    moving average model
    prediction value = average of last N hours
    """
    
    def __init__(self, rolling_col: str = 'rolling_mean_24h'):
        super().__init__(name="Moving Average")
        self.rolling_col = rolling_col
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MovingAverage':
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.rolling_col not in X.columns:
            raise ValueError(f"need '{self.rolling_col}' column")
        return X[self.rolling_col].values


class HistoricalMean(BaseForecaster):
    """
    historical mean model
    prediction value = mean of training set
    """
    
    def __init__(self):
        super().__init__(name="Historical Mean")
        self.mean_value = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HistoricalMean':
        self.mean_value = y.mean()
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_value)


class HourlyMean(BaseForecaster):
    """
    hourly mean model
    prediction value = historical average of this hour
    """
    
    def __init__(self):
        super().__init__(name="Hourly Mean")
        self.hourly_means = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HourlyMean':
        if 'hour' not in X.columns:
            raise ValueError("need 'hour' column")
        
        df = pd.DataFrame({'hour': X['hour'], 'target': y})
        self.hourly_means = df.groupby('hour')['target'].mean().to_dict()
        self.global_mean = y.mean()
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if 'hour' not in X.columns:
            raise ValueError("need 'hour' column")
        
        predictions = X['hour'].map(self.hourly_means)
        predictions = predictions.fillna(self.global_mean)
        return predictions.values
