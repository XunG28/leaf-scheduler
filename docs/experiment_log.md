# LEAF Experiment Log

This document tracks all experiments, configurations, and results for the carbon intensity forecasting model.

---

## Experiment Overview

| Version | Date | Description | MAE | RMSE | MAPE | R² |
|---------|------|-------------|-----|------|------|-----|
| v1.0 | 2026-03-12 | Baseline LightGBM | 42.83 | 52.80 | 11.25% | 0.8548 |

---

## Experiment v1.0 - Baseline LightGBM

**Date:** 2026-03-12  
**Status:** Completed  

### Configuration

**Data Split:**
- Train: 2024-03-02 → 2026-01-31 (~64,000 samples)
- Validation: 2026-02-01 → 2026-03-01 (~2,800 samples)
- Test: 2026-03-02 → 2026-03-08 (~670 samples)

**Features:**

| Category | Features |
|----------|----------|
| Time | hour, minute, dayofweek, dayofmonth, month, quarter, weekofyear |
| Cyclic | hour_sin/cos, dayofweek_sin/cos, month_sin/cos |
| Binary | is_weekend |
| Lag | lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h, lag_168h |
| Rolling | rolling_mean/std/max/min for 6h, 12h, 24h, 168h windows |
| Diff | diff_1h, diff_24h |

**Model Parameters:**
```python
{
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
    'num_boost_round': 2000,
    'early_stopping_rounds': 100
}
```

### Results

**Baseline Comparison:**

| Model | MAE | RMSE | MAPE(%) | R² |
|-------|-----|------|---------|-----|
| Naive (t-1h) | 65.33 | 99.83 | 21.14 | 0.4809 |
| Seasonal Naive (t-24h) | 91.99 | 134.32 | 24.97 | 0.0602 |
| Moving Average (24h) | 106.70 | 132.32 | 32.26 | 0.0880 |
| Hourly Mean | 101.64 | 139.69 | 28.82 | -0.0164 |
| **LightGBM** | **42.83** | **52.80** | **11.25** | **0.8548** |

**Improvement over best baseline (Naive t-1h):**
- MAE: -34.4%
- RMSE: -47.1%
- MAPE: -46.8%

### Key Observations

1. **Short-term autocorrelation is strong:** Naive (t-1h) is the best baseline, indicating that recent values are highly predictive.

2. **Daily seasonality is weaker:** Seasonal Naive (t-24h) performs worse than simple persistence, suggesting that intra-day patterns vary significantly.

3. **Historical mean is ineffective:** Hourly Mean has negative R², indicating the data has significant trends or non-stationarity.

4. **LightGBM captures complex patterns:** The 34% improvement over the best baseline demonstrates the value of machine learning.

### Output Files

- Model: `models/lightgbm_co2_forecast.txt`
- Feature Importance: `models/feature_importance.csv`
- Evaluation Results: `models/evaluation_results.csv`
- Figures: `figures/feature_importance.png`, `figures/forecast_comparison.png`

---

## Experiment v1.1 - [TEMPLATE]

**Date:** YYYY-MM-DD  
**Status:** Planned / In Progress / Completed  

### Changes from v1.0

- [ ] Change 1
- [ ] Change 2

### Configuration

(Copy and modify from v1.0)

### Results

| Model | MAE | RMSE | MAPE(%) | R² |
|-------|-----|------|---------|-----|
| LightGBM v1.0 | 42.83 | 52.80 | 11.25 | 0.8548 |
| LightGBM v1.1 | ? | ? | ? | ? |

**Change:** MAE ? (↑/↓ ?%)

### Observations

(Notes)

---

## Future Experiment Ideas

| ID | Idea | Expected Impact | Priority |
|----|------|-----------------|----------|
| E1 | Add weather features (temperature, cloud cover) | Medium-High | Medium |
| E2 | Add German holiday indicator | Low-Medium | Low |
| E3 | Add interaction features (hour × dayofweek) | Medium | High |
| E4 | Hyperparameter tuning with Optuna | Medium | Medium |
| E5 | Try different models (XGBoost, CatBoost) | Low | Low |
| E6 | Increase lag features (lag_336h for 2-week) | Low | Low |

---

## Reproducibility

### Environment
- Python: 3.10
- LightGBM: 4.x
- OS: Linux

### Commands
```bash
# Data preprocessing
python scripts/process_raw_data.py

# Model training
python scripts/train_forecast.py
```

### Random Seeds
- LightGBM seed: 42
- Train/Val/Test split: Temporal (deterministic)
