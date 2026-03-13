"""
LEAF - Feature Engineering
==========================
for carbon intensity prediction, create time features, lag features and rolling features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def create_time_features(df: pd.DataFrame, datetime_col: str = 'Start date') -> pd.DataFrame:
    """
    create time-related features
    
    Args:
        df: input DataFrame
        datetime_col: datetime column
    
    Returns:
        DataFrame with time features
    """
    df = df.copy()
    
    # ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    dt = df[datetime_col]
    
    # basic time features
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['dayofweek'] = dt.dt.dayofweek  # 0=Monday, 6=Sunday
    df['dayofmonth'] = dt.dt.day
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['weekofyear'] = dt.dt.isocalendar().week.astype(int)
    
    # periodic encoding (sine/cosine transformation)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # binary features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # time of day features
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_periods: List[int],
    freq_minutes: int = 15
) -> pd.DataFrame:
    """
    create lag features
    
    Args:
        df: input DataFrame
        target_col: target column
        lag_periods: lag time list (in hours)
        freq_minutes: data frequency (minutes)
    
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    periods_per_hour = 60 // freq_minutes  # 15 minutes data = 4 periods/hour
    
    for lag_hours in lag_periods:
        lag_steps = lag_hours * periods_per_hour
        col_name = f'lag_{lag_hours}h'
        df[col_name] = df[target_col].shift(lag_steps)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int],
    freq_minutes: int = 15
) -> pd.DataFrame:
    """
    create rolling statistical features
    
    Args:
        df: input DataFrame
        target_col: target column
        windows: window size list (in hours)
        freq_minutes: data frequency (minutes)
    
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    periods_per_hour = 60 // freq_minutes
    
    for window_hours in windows:
        window_steps = window_hours * periods_per_hour
        
        # rolling mean
        df[f'rolling_mean_{window_hours}h'] = (
            df[target_col]
            .shift(1)  # avoid data leakage
            .rolling(window=window_steps, min_periods=1)
            .mean()
        )
        
        # rolling standard deviation
        df[f'rolling_std_{window_hours}h'] = (
            df[target_col]
            .shift(1)
            .rolling(window=window_steps, min_periods=1)
            .std()
        )
        
        # rolling maximum and minimum values
        df[f'rolling_max_{window_hours}h'] = (
            df[target_col]
            .shift(1)
            .rolling(window=window_steps, min_periods=1)
            .max()
        )
        
        df[f'rolling_min_{window_hours}h'] = (
            df[target_col]
            .shift(1)
            .rolling(window=window_steps, min_periods=1)
            .min()
        )
    
    return df


def create_diff_features(
    df: pd.DataFrame,
    target_col: str,
    diff_periods: List[int],
    freq_minutes: int = 15
) -> pd.DataFrame:
    """
    create difference features
    
    Args:
        df: input DataFrame
        target_col: target column
        diff_periods: difference period list (in hours)
        freq_minutes: data frequency (minutes)
    
    Returns:
        DataFrame with difference features
    """
    df = df.copy()
    
    periods_per_hour = 60 // freq_minutes
    
    for diff_hours in diff_periods:
        diff_steps = diff_hours * periods_per_hour
        df[f'diff_{diff_hours}h'] = df[target_col].diff(diff_steps)
    
    return df


def build_features(
    df: pd.DataFrame,
    target_col: str = 'CO2_Intensity_gkWh',
    datetime_col: str = 'Start date',
    lag_hours: Optional[List[int]] = None,
    rolling_hours: Optional[List[int]] = None,
    diff_hours: Optional[List[int]] = None,
    freq_minutes: int = 15
) -> pd.DataFrame:
    """
    full feature engineering pipeline
    
    Args:
        df: input DataFrame
        target_col: target column
        datetime_col: datetime column
        lag_hours: lag feature hour list
        rolling_hours: rolling window hour list
        diff_hours: difference feature hour list
        freq_minutes: data frequency
    
    Returns:
        DataFrame with all features
    """
    # default parameters
    if lag_hours is None:
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, ..., 1week
    if rolling_hours is None:
        rolling_hours = [6, 12, 24, 168]  # 6h, 12h, 24h, 1week
    if diff_hours is None:
        diff_hours = [1, 24]  # 1h difference, 24h difference
    
    print(f"   Start feature engineering...")
    print(f"    target column: {target_col}")
    print(f"    lag features: {lag_hours} hours")
    print(f"    rolling features: {rolling_hours} hours window")
    
    # 1. time features
    df = create_time_features(df, datetime_col)
    print(f"      time features created")
    
    # 2. lag features
    df = create_lag_features(df, target_col, lag_hours, freq_minutes)
    print(f"      lag features created")
    
    # 3. rolling features
    df = create_rolling_features(df, target_col, rolling_hours, freq_minutes)
    print(f"      rolling features created")
    
    # 4. difference features
    df = create_diff_features(df, target_col, diff_hours, freq_minutes)
    print(f"      difference features created")
    
    print(f"      feature engineering completed, total features: {len(df.columns)}")
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    get feature columns for training
    
    Args:
        df: DataFrame
        exclude_cols: columns to exclude
    
    Returns:
        feature column names list
    """
    if exclude_cols is None:
        exclude_cols = [
            'Start date', 'End date',
            'CO2_Intensity_gkWh',  # target column
            'Total_Generation_MWh', 'Renewable_Generation_MWh',  # original data
            'time_of_day'  # categorical variable
        ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return feature_cols


def split_data_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    datetime_col: str = 'Start date'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    split data by date
    
    Args:
        df: full data
        train_end: train set end date (exclusive)
        val_end: validation set end date (exclusive)
        datetime_col: datetime column
    
    Returns:
        (train_df, val_df, test_df)
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    train_mask = df[datetime_col] < train_end
    val_mask = (df[datetime_col] >= train_end) & (df[datetime_col] < val_end)
    test_mask = df[datetime_col] >= val_end
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"dataset splitting:")
    print(f"  Train: {len(train_df):,} samples ({train_df[datetime_col].min()} → {train_df[datetime_col].max()})")
    print(f"  Val:   {len(val_df):,} samples ({val_df[datetime_col].min()} → {val_df[datetime_col].max()})")
    print(f"  Test:  {len(test_df):,} samples ({test_df[datetime_col].min()} → {test_df[datetime_col].max()})")
    
    return train_df, val_df, test_df
