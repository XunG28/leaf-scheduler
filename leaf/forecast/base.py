"""
LEAF - Forecast Base Classes
============================
abstract base class and common evaluation methods for forecasting models
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class BaseForecaster(ABC):
    """abstract base class for forecasting models"""
    
    def __init__(self, name: str = "BaseForecaster"):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseForecaster':
        """
        train model
        
        Args:
            X: feature DataFrame
            y: target Series
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        predict
        
        Args:
            X: feature DataFrame
        
        Returns:
            prediction values array
        """
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        evaluate prediction results
        
        Args:
            y_true: true values
            y_pred: prediction values
        
        Returns:
            evaluation metrics dictionary
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # filter NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan}
        
        # MAE - Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # RMSE - Root Mean Squared Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE - Mean Absolute Percentage Error
        # avoid division by zero
        nonzero_mask = y_true != 0
        if nonzero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        else:
            mape = np.nan
        
        # R² - Coefficient of Determination
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 4),
            'r2': round(r2, 4)
        }
    
    def __repr__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"


def print_evaluation_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """
    print evaluation results comparison of multiple models
    
    Args:
        results: {model name: evaluation metrics dictionary}
    """
    print("\n" + "=" * 70)
    print("evaluation results comparison of multiple models")
    print("=" * 70)
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'R²':>10}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f} "
              f"{metrics['mape']:>10.2f} {metrics['r2']:>10.4f}")
    
    print("=" * 70)
