"""
LEAF - LightGBM Forecaster
==========================
forecasting model based on LightGBM

features:
- lightweight: fast training, low resource consumption (符合GREEN AI理念)
- interpretable: support feature importance analysis and SHAP explanation
- efficient: suitable for time series prediction tasks
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Optional, List, Tuple
import warnings
from pathlib import Path

from .base import BaseForecaster

warnings.filterwarnings('ignore', category=UserWarning)


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM forecasting model
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        initialize LightGBM forecasting model
        
        Args:
            params: LightGBM parameters dictionary
            num_boost_round: maximum number of boosting rounds
            early_stopping_rounds: early stopping rounds
            verbose: whether to print training process
        """
        super().__init__(name="LightGBM")
        
        # default parameters (optimized for time series prediction)
        self.params = params or {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbosity': -1,
            'seed': 42,
            'n_jobs': -1
        }
        
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        
        self.model = None
        self.feature_names = None
        self.best_iteration = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'LightGBMForecaster':
        """
        train model
        
        Args:
            X: training features
            y: training target
            X_val: validation features (for early stopping)
            y_val: validation target
        
        Returns:
            self
        """
        self.feature_names = list(X.columns)
        
        # create dataset
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # training callbacks
        callbacks = []
        if self.early_stopping_rounds and X_val is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
        if self.verbose:
            callbacks.append(lgb.log_evaluation(period=100))
        else:
            callbacks.append(lgb.log_evaluation(period=0))
        
        # train
        if self.verbose:
            print(f"   start training LightGBM...")
            print(f"   training samples: {len(X):,}")
            if X_val is not None:
                print(f"   validation samples: {len(X_val):,}")
            print(f"   feature数量: {len(self.feature_names)}")
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.best_iteration = self.model.best_iteration
        self.is_fitted = True
        
        if self.verbose:
            print(f"   training completed!")
            print(f"   best iteration: {self.best_iteration}")
            if X_val is not None:
                val_pred = self.predict(X_val)
                metrics = self.evaluate(y_val.values, val_pred)
                print(f"   validation MAE: {metrics['mae']:.2f} g/kWh")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        predict
        
        Args:
            X: feature DataFrame
        
        Returns:
            prediction values array
        """
        if not self.is_fitted:
            raise RuntimeError("model not trained, please call fit()")
        
        return self.model.predict(X, num_iteration=self.best_iteration)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        get feature importance
        
        Args:
            importance_type: 'gain' or 'split'
        
        Returns:
            feature importance DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("model not trained")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
        
        return df
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        plot feature importance
        
        Args:
            top_n: show top N features
            save_path: save path
        """
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance()[:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = range(len(importance_df))
        ax.barh(y_pos, importance_df['importance_pct'], color='#2E86AB')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'Top {top_n} Feature Importance (LightGBM)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   feature importance plot saved: {save_path}")
        
        return fig, ax
    
    def save_model(self, path: str) -> None:
        """save model"""
        if not self.is_fitted:
            raise RuntimeError("model not trained")
        
        self.model.save_model(path)
        print(f"   model saved: {path}")
    
    def load_model(self, path: str) -> 'LightGBMForecaster':
        """load model"""
        self.model = lgb.Booster(model_file=path)
        self.is_fitted = True
        print(f"   model loaded: {path}")
        return self


def explain_with_shap(
    model: LightGBMForecaster,
    X: pd.DataFrame,
    sample_size: int = 1000,
    save_path: Optional[str] = None
) -> None:
    """
    use SHAP to explain model prediction
    
    Args:
        model: trained LightGBM model
        X: feature data
        sample_size: sample size
        save_path: save path
    """
    import shap
    import matplotlib.pyplot as plt
    
    print("   calculating SHAP values...")
    
    # sample (SHAP calculation may be slow)
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # create explainer
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_sample)
    
    # plot summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   SHAP plot saved: {save_path}")
    
    plt.tight_layout()
    
    return shap_values
