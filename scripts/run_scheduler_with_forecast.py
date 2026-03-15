"""
LEAF - Scheduler Demo with LightGBM Forecast
=============================================
Demonstrates carbon-aware scheduling using:
1. Actual CO2 data (oracle/upper bound)
2. Predicted CO2 data (realistic scenario with LightGBM)

This script shows the full pipeline: Forecast → Schedule → Evaluate
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from leaf.data.features import build_features, get_feature_columns
from leaf.scheduler import (
    schedule_fifo,
    schedule_carbon_aware,
    get_default_capacity,
    evaluate_schedule,
    compare_schedules,
    print_comparison,
)


# =============================================================================
# Configuration
# =============================================================================
ENERGY_DATA_PATH = project_root / "data" / "processed" / "energy_data_full.csv"
JOBS_DATA_PATH = project_root / "data" / "sample" / "jobs_pro_2026.csv"
MODEL_PATH = project_root / "models" / "lightgbm_co2_forecast.txt"
OUTPUT_DIR = project_root / "data" / "processed"
FIGURE_DIR = project_root / "figures"

# Column names
CO2_COL = 'CO2_Intensity_gkWh'
CO2_PRED_COL = 'CO2_Intensity_Predicted'
RENEWABLE_COL = 'Renewable_Share_pct'
TARGET_COL = 'CO2_Intensity_gkWh'


def load_energy_data(path: Path) -> pd.DataFrame:
    """Load and prepare energy data."""
    df = pd.read_csv(path)
    df['Start date'] = pd.to_datetime(df['Start date'])
    df = df.sort_values('Start date')
    df = df.drop_duplicates(subset=['Start date'], keep='first')
    return df


def load_jobs_data(path: Path) -> pd.DataFrame:
    """Load jobs data."""
    df = pd.read_csv(path)
    df['arrival'] = pd.to_datetime(df['arrival'])
    df['deadline'] = pd.to_datetime(df['deadline'])
    df = df.sort_values(['arrival', 'priority', 'id']).reset_index(drop=True)
    return df


def generate_predictions(
    energy_df: pd.DataFrame,
    model_path: Path,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Generate CO2 predictions using trained LightGBM model.
    
    Args:
        energy_df: Energy dataframe with datetime and features
        model_path: Path to trained LightGBM model
        target_col: Target column name
    
    Returns:
        DataFrame with predictions added
    """
    print("🔮 Generating CO2 predictions with LightGBM...")
    
    # Build features
    df = energy_df.copy()
    df = build_features(
        df,
        target_col=target_col,
        lag_hours=[1, 2, 3, 6, 12, 24, 48, 168],
        rolling_hours=[6, 12, 24, 168],
        diff_hours=[1, 24]
    )
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Load model
    model = lgb.Booster(model_file=str(model_path))
    print(f"   Model loaded: {model_path.name}")
    
    # Generate predictions (handle NaN from lag features)
    df[CO2_PRED_COL] = np.nan
    
    # Only predict where we have all features
    valid_mask = df[feature_cols].notna().all(axis=1)
    if valid_mask.sum() > 0:
        X = df.loc[valid_mask, feature_cols]
        predictions = model.predict(X)
        df.loc[valid_mask, CO2_PRED_COL] = predictions
    
    # Fill NaN predictions with actual values (for early timestamps without lag)
    df[CO2_PRED_COL] = df[CO2_PRED_COL].fillna(df[target_col])
    
    print(f"   Generated {valid_mask.sum():,} predictions")
    
    return df


def prepare_energy_for_scheduling(
    df: pd.DataFrame,
    jobs_start: pd.Timestamp,
    jobs_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare two versions of energy data for scheduling:
    1. With actual CO2
    2. With predicted CO2
    
    Returns:
        (energy_actual, energy_predicted) - both as indexed DataFrames
    """
    # Filter to job time range (with buffer)
    mask = (df['Start date'] >= jobs_start) & (df['Start date'] <= jobs_end + pd.Timedelta(hours=24))
    df_filtered = df[mask].copy()
    
    # Set index
    df_filtered = df_filtered.set_index('Start date')
    
    # Ensure 15min frequency
    full_index = pd.date_range(df_filtered.index.min(), df_filtered.index.max(), freq='15min')
    df_filtered = df_filtered.reindex(full_index).ffill()
    
    # Create two versions
    energy_actual = df_filtered[[CO2_COL, RENEWABLE_COL]].copy()
    
    energy_predicted = df_filtered[[RENEWABLE_COL]].copy()
    energy_predicted[CO2_COL] = df_filtered[CO2_PRED_COL]
    
    return energy_actual, energy_predicted


def plot_full_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Create visualization comparing all strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = comparison_df['strategy'].tolist()
    
    # Color scheme: FIFO=red, Actual=blue, Predicted=green
    colors = []
    for s in strategies:
        if 'FIFO' in s:
            colors.append('#E74C3C')
        elif 'Actual' in s:
            colors.append('#3498DB')
        elif 'Predicted' in s:
            colors.append('#27AE60')
        else:
            colors.append('#95A5A6')
    
    # Plot 1: Total CO2 Emissions
    ax1 = axes[0, 0]
    emissions = comparison_df['total_emissions_gCO2'].values / 1000
    bars = ax1.bar(strategies, emissions, color=colors)
    ax1.set_ylabel('Total CO₂ Emissions (kg)')
    ax1.set_title('Total Carbon Emissions by Strategy')
    ax1.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, emissions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: CO2 Savings vs FIFO
    ax2 = axes[0, 1]
    fifo_emissions = comparison_df[comparison_df['strategy'] == 'FIFO']['total_emissions_gCO2'].values[0]
    savings = [(fifo_emissions - e) / fifo_emissions * 100 for e in comparison_df['total_emissions_gCO2'].values]
    bars = ax2.bar(strategies, savings, color=colors)
    ax2.set_ylabel('CO₂ Savings vs FIFO (%)')
    ax2.set_title('Carbon Reduction Compared to FIFO')
    ax2.tick_params(axis='x', rotation=15)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, savings):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Renewable Share
    ax3 = axes[1, 0]
    renewable = comparison_df['avg_Renewable_Share_pct'].values
    bars = ax3.bar(strategies, renewable, color=colors)
    ax3.set_ylabel('Avg Renewable Share (%)')
    ax3.set_title('Renewable Energy Usage')
    ax3.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, renewable):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Trade-off scatter
    ax4 = axes[1, 1]
    wait_times = comparison_df['avg_wait_min'].values
    for i, (strat, wait, em) in enumerate(zip(strategies, wait_times, emissions)):
        ax4.scatter(wait, em, s=200, c=colors[i], label=strat, zorder=5)
    ax4.set_xlabel('Average Wait Time (min)')
    ax4.set_ylabel('Total CO₂ Emissions (kg)')
    ax4.set_title('Trade-off: Latency vs Emissions')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison figure saved: {save_path}")


def main():
    print("=" * 70)
    print("LEAF - Carbon-Aware Scheduler with LightGBM Forecast")
    print("=" * 70)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n📁 Loading data...")
    
    energy_df = load_energy_data(ENERGY_DATA_PATH)
    print(f"   Energy data: {len(energy_df):,} records")
    
    jobs_df = load_jobs_data(JOBS_DATA_PATH)
    print(f"   Jobs: {len(jobs_df)} tasks")
    print(f"   Jobs time range: {jobs_df['arrival'].min()} → {jobs_df['deadline'].max()}")
    
    # =========================================================================
    # 2. Generate Predictions
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Generate CO2 Predictions")
    print("=" * 70)
    
    energy_with_pred = generate_predictions(energy_df, MODEL_PATH, TARGET_COL)
    
    # Calculate prediction error on the job time range
    jobs_start = jobs_df['arrival'].min()
    jobs_end = jobs_df['deadline'].max()
    
    job_range_mask = (
        (energy_with_pred['Start date'] >= jobs_start) & 
        (energy_with_pred['Start date'] <= jobs_end)
    )
    job_range_df = energy_with_pred[job_range_mask]
    
    mae = np.abs(job_range_df[CO2_COL] - job_range_df[CO2_PRED_COL]).mean()
    print(f"\n📊 Prediction quality on job time range:")
    print(f"   MAE: {mae:.2f} g/kWh")
    
    # =========================================================================
    # 3. Prepare Energy Data for Scheduling
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Prepare Scheduling Data")
    print("=" * 70)
    
    energy_actual, energy_predicted = prepare_energy_for_scheduling(
        energy_with_pred, jobs_start, jobs_end
    )
    
    print(f"   Actual CO2 data: {len(energy_actual)} time slots")
    print(f"   Predicted CO2 data: {len(energy_predicted)} time slots")
    
    capacity = get_default_capacity(jobs_df)
    print(f"   Resource capacity: {capacity}")
    
    # =========================================================================
    # 4. Run Scheduling Strategies
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Run Scheduling Strategies")
    print("=" * 70)
    
    # FIFO baseline (uses actual CO2 for evaluation)
    print("\n⏳ Running FIFO (baseline)...")
    fifo_schedule = schedule_fifo(energy_actual, jobs_df, capacity, CO2_COL)
    print(f"   ✅ Scheduled {len(fifo_schedule)} jobs")
    
    # Carbon-Aware with ACTUAL CO2 (oracle/upper bound)
    print("\n⏳ Running CarbonAware with ACTUAL CO2 (oracle)...")
    carbon_actual = schedule_carbon_aware(energy_actual, jobs_df, capacity, CO2_COL)
    print(f"   ✅ Scheduled {len(carbon_actual)} jobs")
    
    # Carbon-Aware with PREDICTED CO2 (realistic scenario)
    print("\n⏳ Running CarbonAware with PREDICTED CO2 (realistic)...")
    carbon_predicted = schedule_carbon_aware(energy_predicted, jobs_df, capacity, CO2_COL)
    print(f"   ✅ Scheduled {len(carbon_predicted)} jobs")
    
    # =========================================================================
    # 5. Evaluate ALL schedules against ACTUAL CO2
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Evaluation (all evaluated with ACTUAL CO2)")
    print("=" * 70)
    
    # Important: Evaluate all schedules using ACTUAL CO2 for fair comparison
    schedules = {
        'FIFO': fifo_schedule,
        'CarbonAware (Actual)': carbon_actual,
        'CarbonAware (Predicted)': carbon_predicted,
    }
    
    comparison = compare_schedules(
        energy_actual, schedules,
        co2_col=CO2_COL,
        renewable_col=RENEWABLE_COL,
    )
    
    print_comparison(comparison, baseline='FIFO')
    
    # =========================================================================
    # 6. Save Results
    # =========================================================================
    print("\n💾 Saving results...")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    
    # Save schedules
    carbon_predicted.to_csv(OUTPUT_DIR / "scheduled_carbon_predicted.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "schedule_comparison_with_forecast.csv", index=False)
    print(f"   Results saved to {OUTPUT_DIR}")
    
    # Create visualization
    plot_full_comparison(comparison, FIGURE_DIR / "schedule_comparison_with_forecast.png")
    
    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    fifo_emissions = comparison[comparison['strategy'] == 'FIFO']['total_emissions_gCO2'].values[0]
    actual_emissions = comparison[comparison['strategy'] == 'CarbonAware (Actual)']['total_emissions_gCO2'].values[0]
    pred_emissions = comparison[comparison['strategy'] == 'CarbonAware (Predicted)']['total_emissions_gCO2'].values[0]
    
    actual_saved = (fifo_emissions - actual_emissions) / fifo_emissions * 100
    pred_saved = (fifo_emissions - pred_emissions) / fifo_emissions * 100
    
    print(f"\n📊 Results:")
    print(f"   FIFO (baseline):              {fifo_emissions/1000:.2f} kg CO₂")
    print(f"   CarbonAware (Actual CO2):     {actual_emissions/1000:.2f} kg CO₂  (-{actual_saved:.1f}%) ← Upper Bound")
    print(f"   CarbonAware (Predicted CO2):  {pred_emissions/1000:.2f} kg CO₂  (-{pred_saved:.1f}%) ← Realistic")
    
    print(f"\n🎯 Key Findings:")
    print(f"   • With perfect CO2 knowledge: {actual_saved:.1f}% reduction possible")
    print(f"   • With LightGBM prediction:   {pred_saved:.1f}% reduction achieved")
    print(f"   • Prediction achieves {pred_saved/actual_saved*100:.0f}% of theoretical maximum")
    
    print("\n✅ Demo complete!")
    
    return comparison


if __name__ == "__main__":
    comparison = main()
