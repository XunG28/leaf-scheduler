"""
LEAF - Scheduler Demo Script
============================
Demonstrates carbon-aware scheduling with comparison to baselines.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from leaf.scheduler import (
    schedule_fifo,
    schedule_edf,
    schedule_carbon_aware,
    get_default_capacity,
    compare_schedules,
    print_comparison,
)


# =============================================================================
# Configuration
# =============================================================================
ENERGY_DATA_PATH = project_root / "data" / "processed" / "energy_data_full.csv"
JOBS_DATA_PATH = project_root / "data" / "sample" / "jobs_pro_2026.csv"
OUTPUT_DIR = project_root / "data" / "processed"
FIGURE_DIR = project_root / "figures"

# Column names in energy data
CO2_COL = 'CO2_Intensity_gkWh'
RENEWABLE_COL = 'Renewable_Share_pct'


def load_energy_data(path: Path) -> pd.DataFrame:
    """Load and prepare energy data."""
    df = pd.read_csv(path)
    df['Start date'] = pd.to_datetime(df['Start date'])
    df = df.sort_values('Start date')
    
    # Remove duplicate timestamps (keep first)
    df = df.drop_duplicates(subset=['Start date'], keep='first')
    df = df.set_index('Start date')
    
    # Ensure 15min frequency
    full_index = pd.date_range(df.index.min(), df.index.max(), freq='15min')
    df = df.reindex(full_index).ffill()
    
    return df


def load_jobs_data(path: Path) -> pd.DataFrame:
    """Load jobs data."""
    df = pd.read_csv(path)
    df['arrival'] = pd.to_datetime(df['arrival'])
    df['deadline'] = pd.to_datetime(df['deadline'])
    df = df.sort_values(['arrival', 'priority', 'id']).reset_index(drop=True)
    return df


def plot_schedule_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Create visualization of schedule comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    strategies = comparison_df['strategy'].tolist()
    colors = ['#E74C3C', '#3498DB', '#27AE60'][:len(strategies)]
    
    # Plot 1: Total CO2 Emissions
    ax1 = axes[0, 0]
    emissions = comparison_df['total_emissions_gCO2'].values / 1000  # Convert to kg
    bars = ax1.bar(strategies, emissions, color=colors)
    ax1.set_ylabel('Total CO₂ Emissions (kg)')
    ax1.set_title('Total Carbon Emissions by Strategy')
    for bar, val in zip(bars, emissions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Average CO2 Intensity
    ax2 = axes[0, 1]
    avg_co2 = comparison_df['avg_CO2_g_per_kWh'].values
    bars = ax2.bar(strategies, avg_co2, color=colors)
    ax2.set_ylabel('Avg CO₂ Intensity (g/kWh)')
    ax2.set_title('Average Carbon Intensity by Strategy')
    for bar, val in zip(bars, avg_co2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Renewable Share
    ax3 = axes[1, 0]
    renewable = comparison_df['avg_Renewable_Share_pct'].values
    bars = ax3.bar(strategies, renewable, color=colors)
    ax3.set_ylabel('Avg Renewable Share (%)')
    ax3.set_title('Renewable Energy Usage by Strategy')
    for bar, val in zip(bars, renewable):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Wait Time vs Emissions Trade-off
    ax4 = axes[1, 1]
    wait_times = comparison_df['avg_wait_min'].values
    for i, (strat, wait, co2) in enumerate(zip(strategies, wait_times, emissions)):
        ax4.scatter(wait, co2, s=200, c=colors[i], label=strat, zorder=5)
    ax4.set_xlabel('Average Wait Time (min)')
    ax4.set_ylabel('Total CO₂ Emissions (kg)')
    ax4.set_title('Trade-off: Wait Time vs Emissions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison figure saved: {save_path}")


def main():
    print("=" * 70)
    print("LEAF - Carbon-Aware Scheduler Demo")
    print("=" * 70)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n📁 Loading data...")
    
    energy_df = load_energy_data(ENERGY_DATA_PATH)
    print(f"   Energy data: {len(energy_df):,} time slots")
    print(f"   Time range: {energy_df.index.min()} → {energy_df.index.max()}")
    
    jobs_df = load_jobs_data(JOBS_DATA_PATH)
    print(f"   Jobs: {len(jobs_df)} tasks")
    
    # Filter energy data to match jobs time range
    jobs_start = jobs_df['arrival'].min()
    jobs_end = jobs_df['deadline'].max()
    energy_df = energy_df[(energy_df.index >= jobs_start) & (energy_df.index <= jobs_end + pd.Timedelta(hours=24))]
    print(f"   Filtered energy data: {len(energy_df):,} time slots")
    
    # =========================================================================
    # 2. Get Resource Capacity
    # =========================================================================
    capacity = get_default_capacity(jobs_df)
    print(f"\n🔧 Resource capacity: {capacity}")
    
    # =========================================================================
    # 3. Run Scheduling Strategies
    # =========================================================================
    print("\n" + "=" * 70)
    print("Running Scheduling Strategies")
    print("=" * 70)
    
    print("\n⏳ Running FIFO...")
    fifo_schedule = schedule_fifo(energy_df, jobs_df, capacity, CO2_COL)
    print(f"   ✅ Scheduled {len(fifo_schedule)} jobs")
    
    print("\n⏳ Running EDF...")
    edf_schedule = schedule_edf(energy_df, jobs_df, capacity, CO2_COL)
    print(f"   ✅ Scheduled {len(edf_schedule)} jobs")
    
    print("\n⏳ Running Carbon-Aware...")
    carbon_schedule = schedule_carbon_aware(energy_df, jobs_df, capacity, CO2_COL)
    print(f"   ✅ Scheduled {len(carbon_schedule)} jobs")
    
    # =========================================================================
    # 4. Evaluate and Compare
    # =========================================================================
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    schedules = {
        'FIFO': fifo_schedule,
        'EDF': edf_schedule,
        'CarbonAware': carbon_schedule,
    }
    
    comparison = compare_schedules(
        energy_df, schedules,
        co2_col=CO2_COL,
        renewable_col=RENEWABLE_COL,
    )
    
    print_comparison(comparison, baseline='FIFO')
    
    # =========================================================================
    # 5. Save Results
    # =========================================================================
    print("\n💾 Saving results...")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    
    # Save schedules
    fifo_schedule.to_csv(OUTPUT_DIR / "scheduled_fifo.csv", index=False)
    edf_schedule.to_csv(OUTPUT_DIR / "scheduled_edf.csv", index=False)
    carbon_schedule.to_csv(OUTPUT_DIR / "scheduled_carbon_aware.csv", index=False)
    print(f"   Schedules saved to {OUTPUT_DIR}")
    
    # Save comparison
    comparison.to_csv(OUTPUT_DIR / "schedule_comparison.csv", index=False)
    print(f"   Comparison saved: {OUTPUT_DIR / 'schedule_comparison.csv'}")
    
    # Create visualization
    plot_schedule_comparison(comparison, FIGURE_DIR / "schedule_comparison.png")
    
    # =========================================================================
    # 6. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    fifo_emissions = comparison[comparison['strategy'] == 'FIFO']['total_emissions_gCO2'].values[0]
    carbon_emissions = comparison[comparison['strategy'] == 'CarbonAware']['total_emissions_gCO2'].values[0]
    saved_pct = (fifo_emissions - carbon_emissions) / fifo_emissions * 100
    
    print(f"\n🌿 Carbon-Aware scheduling reduced emissions by {saved_pct:.1f}% compared to FIFO")
    print(f"   FIFO:        {fifo_emissions/1000:.2f} kg CO₂")
    print(f"   CarbonAware: {carbon_emissions/1000:.2f} kg CO₂")
    print(f"   Saved:       {(fifo_emissions-carbon_emissions)/1000:.2f} kg CO₂")
    
    print("\n✅ Scheduler demo complete!")
    
    return comparison


if __name__ == "__main__":
    comparison = main()
