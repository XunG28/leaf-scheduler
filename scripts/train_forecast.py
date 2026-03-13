"""
LEAF - carbon intensity forecasting model training script
==============================
train LightGBM model and compare with baseline models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from leaf.data.features import (
    build_features,
    get_feature_columns,
    split_data_by_date
)
from leaf.forecast.base import print_evaluation_comparison
from leaf.forecast.baseline import (
    NaivePersistence,
    SeasonalNaive,
    MovingAverage,
    HourlyMean
)
from leaf.forecast.lightgbm_model import LightGBMForecaster, explain_with_shap


# =============================================================================
# configuration
# =============================================================================
DATA_PATH = project_root / "data" / "processed" / "energy_data_full.csv"
OUTPUT_DIR = project_root / "data" / "processed"
MODEL_DIR = project_root / "models"
FIGURE_DIR = project_root / "figures"

# data split dates
TRAIN_END = '2026-02-01'
VAL_END = '2026-03-02'

# target column
TARGET_COL = 'CO2_Intensity_gkWh'


def main():
    print("=" * 70)
    print("LEAF - Carbon Intensity Forecast Training")
    print("=" * 70)
    
    # =========================================================================
    # 1. load data
    # =========================================================================
    print("\n   load data...")
    df = pd.read_csv(DATA_PATH)
    df['Start date'] = pd.to_datetime(df['Start date'])
    print(f"   total samples: {len(df):,}")
    
    # =========================================================================
    # 2. feature engineering
    # =========================================================================
    print("\n   feature engineering...")
    df = build_features(
        df,
        target_col=TARGET_COL,
        lag_hours=[1, 2, 3, 6, 12, 24, 48, 168],
        rolling_hours=[6, 12, 24, 168],
        diff_hours=[1, 24]
    )
    
    # =========================================================================
    # 3. dataset splitting
    # =========================================================================
    print("\n   dataset splitting...")
    train_df, val_df, test_df = split_data_by_date(df, TRAIN_END, VAL_END)
    
    # remove rows with NaN (due to lag features)
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()
    
    print(f"   after cleaning - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    # =========================================================================
    # 4. prepare features and target
    # =========================================================================
    feature_cols = get_feature_columns(train_df)
    print(f"\n   using {len(feature_cols)} features")
    
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]
    
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]
    
    # =========================================================================
    # 5. train baseline models
    # =========================================================================
    print("\n" + "=" * 70)
    print("train baseline models")
    print("=" * 70)
    
    baselines = {
        'Naive (t-1h)': NaivePersistence(lag_col='lag_1h'),
        'Seasonal Naive (t-24h)': SeasonalNaive(lag_col='lag_24h'),
        'Moving Average (24h)': MovingAverage(rolling_col='rolling_mean_24h'),
        'Hourly Mean': HourlyMean()
    }
    
    results = {}
    
    for name, model in baselines.items():
        print(f"\n   training {name}...")
        model.fit(X_train, y_train)
        
        # evaluate on test set
        y_pred = model.predict(X_test)
        metrics = model.evaluate(y_test.values, y_pred)
        results[name] = metrics
        print(f"   Test MAE: {metrics['mae']:.2f} g/kWh")
    
    # =========================================================================
    # 6. train LightGBM
    # =========================================================================
    print("\n" + "=" * 70)
    print("train LightGBM")
    print("=" * 70)
    
    lgb_model = LightGBMForecaster(
        params={
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbosity': -1,
            'seed': 42,
            'n_jobs': -1
        },
        num_boost_round=2000,
        early_stopping_rounds=100,
        verbose=True
    )
    
    lgb_model.fit(X_train, y_train, X_val, y_val)
    
    # evaluate on test set
    y_pred_lgb = lgb_model.predict(X_test)
    metrics_lgb = lgb_model.evaluate(y_test.values, y_pred_lgb)
    results['LightGBM'] = metrics_lgb
    
    print(f"\n   LightGBM Test MAE: {metrics_lgb['mae']:.2f} g/kWh")
    
    # =========================================================================
    # 7. result comparison
    # =========================================================================
    print_evaluation_comparison(results)
    
    # =========================================================================
    # 8. 保存模型和结果
    # =========================================================================
    print("\n   save model and results...")
    
    # create directories
    MODEL_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    
    # save model
    lgb_model.save_model(str(MODEL_DIR / "lightgbm_co2_forecast.txt"))
    
    # save feature importance
    importance_df = lgb_model.get_feature_importance()
    importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    print(f"   feature importance saved")
    
    # plot feature importance
    lgb_model.plot_feature_importance(
        top_n=20,
        save_path=str(FIGURE_DIR / "feature_importance.png")
    )
    
    # save evaluation results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(MODEL_DIR / "evaluation_results.csv")
    print(f"   evaluation results saved")
    
    # =========================================================================
    # 9. plot forecast comparison
    # =========================================================================
    print("\n   plot forecast comparison...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # test set time series
    test_dates = test_df['Start date'].values
    
    # plot 1: prediction vs actual
    ax1 = axes[0]
    ax1.plot(test_dates, y_test.values, label='Actual', color='black', alpha=0.7, linewidth=1)
    ax1.plot(test_dates, y_pred_lgb, label='LightGBM', color='#2E86AB', alpha=0.8, linewidth=1)
    ax1.set_ylabel('CO₂ Intensity (g/kWh)')
    ax1.set_title('Carbon Intensity Forecast: LightGBM vs Actual (Test Set)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # plot 2: prediction error
    ax2 = axes[1]
    errors = y_pred_lgb - y_test.values
    ax2.fill_between(test_dates, errors, 0, alpha=0.5, color='#E74C3C', label='Prediction Error')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Error (g/kWh)')
    ax2.set_xlabel('Time')
    ax2.set_title('Prediction Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "forecast_comparison.png", dpi=150, bbox_inches='tight')
    print(f"   forecast comparison plot saved: {FIGURE_DIR / 'forecast_comparison.png'}")
    
    # =========================================================================
    # 10. SHAP explanation (optional)
    # =========================================================================
    try:
        print("\n   generating SHAP explanation...")
        explain_with_shap(
            lgb_model, X_test,
            sample_size=500,
            save_path=str(FIGURE_DIR / "shap_summary.png")
        )
    except Exception as e:
        print(f"   ⚠️ SHAP generation failed: {e}")
    
    # =========================================================================
    # complete
    # =========================================================================
    print("\n" + "=" * 70)
    print("   training completed!")
    print("=" * 70)
    print(f"\n   output files:")
    print(f"   - model: {MODEL_DIR / 'lightgbm_co2_forecast.txt'}")
    print(f"   - feature importance: {MODEL_DIR / 'feature_importance.csv'}")
    print(f"   - evaluation results: {MODEL_DIR / 'evaluation_results.csv'}")
    print(f"   - plots: {FIGURE_DIR}/")
    
    # return final results
    return results, lgb_model


if __name__ == "__main__":
    results, model = main()
