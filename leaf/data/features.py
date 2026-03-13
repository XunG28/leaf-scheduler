"""
LEAF - Feature Engineering
==========================
为碳强度预测创建时间特征、滞后特征和滚动特征
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def create_time_features(df: pd.DataFrame, datetime_col: str = 'Start date') -> pd.DataFrame:
    """
    创建时间相关特征
    
    Args:
        df: 输入DataFrame
        datetime_col: 时间列名
    
    Returns:
        添加了时间特征的DataFrame
    """
    df = df.copy()
    
    # 确保是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    dt = df[datetime_col]
    
    # 基础时间特征
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['dayofweek'] = dt.dt.dayofweek  # 0=Monday, 6=Sunday
    df['dayofmonth'] = dt.dt.day
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['weekofyear'] = dt.dt.isocalendar().week.astype(int)
    
    # 周期性编码（正弦/余弦变换）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 二值特征
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # 时段特征
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
    创建滞后特征
    
    Args:
        df: 输入DataFrame
        target_col: 目标列名
        lag_periods: 滞后时间列表（以小时为单位）
        freq_minutes: 数据频率（分钟）
    
    Returns:
        添加了滞后特征的DataFrame
    """
    df = df.copy()
    
    periods_per_hour = 60 // freq_minutes  # 15分钟数据 = 4个周期/小时
    
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
    创建滚动统计特征
    
    Args:
        df: 输入DataFrame
        target_col: 目标列名
        windows: 窗口大小列表（以小时为单位）
        freq_minutes: 数据频率（分钟）
    
    Returns:
        添加了滚动特征的DataFrame
    """
    df = df.copy()
    
    periods_per_hour = 60 // freq_minutes
    
    for window_hours in windows:
        window_steps = window_hours * periods_per_hour
        
        # 滚动均值
        df[f'rolling_mean_{window_hours}h'] = (
            df[target_col]
            .shift(1)  # 避免数据泄露
            .rolling(window=window_steps, min_periods=1)
            .mean()
        )
        
        # 滚动标准差
        df[f'rolling_std_{window_hours}h'] = (
            df[target_col]
            .shift(1)
            .rolling(window=window_steps, min_periods=1)
            .std()
        )
        
        # 滚动最大最小值
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
    创建差分特征
    
    Args:
        df: 输入DataFrame
        target_col: 目标列名
        diff_periods: 差分周期列表（以小时为单位）
        freq_minutes: 数据频率（分钟）
    
    Returns:
        添加了差分特征的DataFrame
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
    完整特征工程流水线
    
    Args:
        df: 输入DataFrame
        target_col: 目标列名
        datetime_col: 时间列名
        lag_hours: 滞后特征的小时数列表
        rolling_hours: 滚动窗口的小时数列表
        diff_hours: 差分特征的小时数列表
        freq_minutes: 数据频率
    
    Returns:
        包含所有特征的DataFrame
    """
    # 默认参数
    if lag_hours is None:
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, ..., 1week
    if rolling_hours is None:
        rolling_hours = [6, 12, 24, 168]  # 6h, 12h, 24h, 1week
    if diff_hours is None:
        diff_hours = [1, 24]  # 1h差分, 24h差分
    
    print(f"⏳ 开始特征工程...")
    print(f"   目标列: {target_col}")
    print(f"   滞后特征: {lag_hours} 小时")
    print(f"   滚动特征: {rolling_hours} 小时窗口")
    
    # 1. 时间特征
    df = create_time_features(df, datetime_col)
    print(f"   ✅ 时间特征创建完成")
    
    # 2. 滞后特征
    df = create_lag_features(df, target_col, lag_hours, freq_minutes)
    print(f"   ✅ 滞后特征创建完成")
    
    # 3. 滚动特征
    df = create_rolling_features(df, target_col, rolling_hours, freq_minutes)
    print(f"   ✅ 滚动特征创建完成")
    
    # 4. 差分特征
    df = create_diff_features(df, target_col, diff_hours, freq_minutes)
    print(f"   ✅ 差分特征创建完成")
    
    print(f"✅ 特征工程完成，总特征数: {len(df.columns)}")
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    获取用于训练的特征列名
    
    Args:
        df: DataFrame
        exclude_cols: 要排除的列
    
    Returns:
        特征列名列表
    """
    if exclude_cols is None:
        exclude_cols = [
            'Start date', 'End date',
            'CO2_Intensity_gkWh',  # 目标
            'Total_Generation_MWh', 'Renewable_Generation_MWh',  # 原始数据
            'time_of_day'  # 分类变量
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
    按日期划分数据集
    
    Args:
        df: 完整数据
        train_end: 训练集结束日期 (exclusive)
        val_end: 验证集结束日期 (exclusive)
        datetime_col: 时间列名
    
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
    
    print(f"数据集划分:")
    print(f"  Train: {len(train_df):,} samples ({train_df[datetime_col].min()} → {train_df[datetime_col].max()})")
    print(f"  Val:   {len(val_df):,} samples ({val_df[datetime_col].min()} → {val_df[datetime_col].max()})")
    print(f"  Test:  {len(test_df):,} samples ({test_df[datetime_col].min()} → {test_df[datetime_col].max()})")
    
    return train_df, val_df, test_df
