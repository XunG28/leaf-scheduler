"""
LEAF - Schedule Evaluator
=========================
Evaluation metrics for scheduled jobs:
- Energy consumption
- CO2 emissions
- Renewable energy share
- Scheduling performance (wait time, tardiness, violations)
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


FREQ = "15min"
FREQ_MINUTES = 15


@dataclass
class ScheduleMetrics:
    """
    Evaluation metrics for a schedule.
    
    Attributes:
        n_jobs: Number of scheduled jobs
        total_energy_kwh: Total energy consumption (kWh)
        total_emissions_g: Total CO2 emissions (grams)
        avg_co2_intensity: Average CO2 intensity (g/kWh)
        avg_renewable_share: Average renewable energy share (%)
        avg_wait_min: Average wait time (minutes)
        avg_tardiness_min: Average tardiness (minutes)
        violation_rate: Fraction of jobs violating deadline
    """
    n_jobs: int
    total_energy_kwh: float
    total_emissions_g: float
    avg_co2_intensity: float
    avg_renewable_share: float
    avg_wait_min: float
    avg_tardiness_min: float
    violation_rate: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'n_jobs': self.n_jobs,
            'total_energy_kWh': round(self.total_energy_kwh, 2),
            'total_emissions_gCO2': round(self.total_emissions_g, 2),
            'avg_CO2_g_per_kWh': round(self.avg_co2_intensity, 2),
            'avg_Renewable_Share_pct': round(self.avg_renewable_share, 2),
            'avg_wait_min': round(self.avg_wait_min, 2),
            'avg_tardiness_min': round(self.avg_tardiness_min, 2),
            'violation_rate': round(self.violation_rate, 4),
        }
    
    def __repr__(self) -> str:
        return (
            f"ScheduleMetrics(\n"
            f"  n_jobs={self.n_jobs},\n"
            f"  total_energy={self.total_energy_kwh:.1f} kWh,\n"
            f"  total_emissions={self.total_emissions_g:.0f} g CO2,\n"
            f"  avg_CO2={self.avg_co2_intensity:.1f} g/kWh,\n"
            f"  avg_renewable={self.avg_renewable_share:.1f}%,\n"
            f"  avg_wait={self.avg_wait_min:.1f} min,\n"
            f"  violation_rate={self.violation_rate:.2%}\n"
            f")"
        )


def evaluate_schedule(
    energy_df: pd.DataFrame,
    scheduled_jobs: pd.DataFrame,
    co2_col: str = 'CO2_Intensity_gkWh',
    renewable_col: str = 'Renewable_Share_pct',
) -> ScheduleMetrics:
    """
    Evaluate a schedule against energy and performance metrics.
    
    Args:
        energy_df: Energy data with datetime index, CO2 and renewable columns
        scheduled_jobs: Scheduled jobs with scheduled_start and scheduled_end
        co2_col: Column name for CO2 intensity (g/kWh)
        renewable_col: Column name for renewable share (%)
    
    Returns:
        ScheduleMetrics object with evaluation results
    """
    idx0 = energy_df.index.min()
    co2 = energy_df[co2_col]
    renewable = energy_df[renewable_col]
    
    total_energy_kwh = 0.0
    total_emissions_g = 0.0
    total_renewable_weighted = 0.0
    
    wait_times = []
    tardiness_values = []
    violations = []
    
    for _, job in scheduled_jobs.iterrows():
        start = pd.Timestamp(job['scheduled_start'])
        end = pd.Timestamp(job['scheduled_end'])
        arrival = pd.Timestamp(job['arrival'])
        deadline = pd.Timestamp(job['deadline'])
        
        duration_min = int(job['duration'])
        power_kw = float(job['power_avg'])
        
        # Scheduling performance metrics
        wait_min = (start - arrival).total_seconds() / 60.0
        wait_times.append(wait_min)
        
        late_min = max(0.0, (end - deadline).total_seconds() / 60.0)
        tardiness_values.append(late_min)
        
        is_violated = 1.0 if end > deadline else 0.0
        violations.append(is_violated)
        
        # Energy and emissions calculation
        start_slot = int((start - idx0) / pd.Timedelta(FREQ))
        n_slots = duration_min // FREQ_MINUTES
        slot_energy_kwh = power_kw * 0.25  # kW * 0.25h = kWh per slot
        
        co2_slice = co2.iloc[start_slot : start_slot + n_slots].to_numpy(dtype=float)
        renewable_slice = renewable.iloc[start_slot : start_slot + n_slots].to_numpy(dtype=float)
        
        job_energy_kwh = slot_energy_kwh * n_slots
        job_emissions_g = float((co2_slice * slot_energy_kwh).sum())
        job_renewable_weighted = float((renewable_slice * slot_energy_kwh).sum())
        
        total_energy_kwh += job_energy_kwh
        total_emissions_g += job_emissions_g
        total_renewable_weighted += job_renewable_weighted
    
    # Calculate averages
    avg_co2 = total_emissions_g / total_energy_kwh if total_energy_kwh > 0 else np.nan
    avg_renewable = total_renewable_weighted / total_energy_kwh if total_energy_kwh > 0 else np.nan
    
    return ScheduleMetrics(
        n_jobs=len(scheduled_jobs),
        total_energy_kwh=total_energy_kwh,
        total_emissions_g=total_emissions_g,
        avg_co2_intensity=avg_co2,
        avg_renewable_share=avg_renewable,
        avg_wait_min=float(np.mean(wait_times)) if wait_times else np.nan,
        avg_tardiness_min=float(np.mean(tardiness_values)) if tardiness_values else np.nan,
        violation_rate=float(np.mean(violations)) if violations else np.nan,
    )


def compare_schedules(
    energy_df: pd.DataFrame,
    schedules: dict[str, pd.DataFrame],
    co2_col: str = 'CO2_Intensity_gkWh',
    renewable_col: str = 'Renewable_Share_pct',
) -> pd.DataFrame:
    """
    Compare multiple scheduling strategies.
    
    Args:
        energy_df: Energy data
        schedules: Dict of {strategy_name: scheduled_jobs_df}
        co2_col: CO2 column name
        renewable_col: Renewable column name
    
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for strategy_name, schedule in schedules.items():
        metrics = evaluate_schedule(energy_df, schedule, co2_col, renewable_col)
        result = {'strategy': strategy_name, **metrics.to_dict()}
        results.append(result)
    
    return pd.DataFrame(results)


def print_comparison(comparison_df: pd.DataFrame, baseline: str = 'FIFO') -> None:
    """
    Print formatted comparison with improvement over baseline.
    
    Args:
        comparison_df: Comparison dataframe from compare_schedules()
        baseline: Name of baseline strategy for comparison
    """
    print("\n" + "=" * 80)
    print("Schedule Comparison")
    print("=" * 80)
    
    # Get baseline metrics
    baseline_row = comparison_df[comparison_df['strategy'] == baseline]
    if len(baseline_row) == 0:
        baseline_emissions = None
    else:
        baseline_emissions = baseline_row['total_emissions_gCO2'].values[0]
    
    # Print header
    print(f"{'Strategy':<20} {'Energy(kWh)':>12} {'CO2(g)':>12} {'Avg CO2':>10} "
          f"{'Renewable%':>10} {'Wait(min)':>10} {'Viol%':>8} {'CO2 Saved':>10}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        co2_saved = ""
        if baseline_emissions is not None and row['strategy'] != baseline:
            saved_pct = (baseline_emissions - row['total_emissions_gCO2']) / baseline_emissions * 100
            co2_saved = f"{saved_pct:+.1f}%"
        
        print(f"{row['strategy']:<20} "
              f"{row['total_energy_kWh']:>12.1f} "
              f"{row['total_emissions_gCO2']:>12.0f} "
              f"{row['avg_CO2_g_per_kWh']:>10.1f} "
              f"{row['avg_Renewable_Share_pct']:>10.1f} "
              f"{row['avg_wait_min']:>10.1f} "
              f"{row['violation_rate']*100:>7.1f}% "
              f"{co2_saved:>10}")
    
    print("=" * 80)
