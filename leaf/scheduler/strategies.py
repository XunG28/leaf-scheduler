"""
LEAF - Scheduling Strategies
============================
Implementation of various scheduling policies:
- FIFO: First-In-First-Out
- EDF: Earliest Deadline First
- CarbonAware: Two-phase carbon-aware scheduling

The CarbonAware strategy uses a two-phase approach:
1. Generate a feasible FIFO schedule
2. Locally optimize by shifting tasks to lower-carbon time slots
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


# Time resolution
FREQ = "15min"
FREQ_MINUTES = 15

# Carbon-aware optimization parameters
MAX_SHIFT_SLOTS = 96  # Maximum right-shift: 96 slots = 24 hours
SHIFT_PENALTY = 5.0   # Penalty for delaying tasks (balances CO2 vs latency)


@dataclass
class SchedulerConfig:
    """Configuration for scheduler."""
    max_shift_slots: int = MAX_SHIFT_SLOTS
    shift_penalty: float = SHIFT_PENALTY
    freq_minutes: int = FREQ_MINUTES


def _slots_between(t0: pd.Timestamp, t1: pd.Timestamp, index0: pd.Timestamp) -> tuple[int, int]:
    """
    Calculate slot indices between two timestamps.
    
    Returns:
        (start_slot, end_slot) - start inclusive, end exclusive
    """
    s = int((t0 - index0) / pd.Timedelta(FREQ))
    e = int((t1 - index0) / pd.Timedelta(FREQ))
    return s, e


def _find_earliest_feasible_start(
    usage: np.ndarray,
    capacity: int,
    demand: int,
    earliest_slot: int,
    duration_slots: int,
    last_start_slot: Optional[int] = None,
) -> Optional[int]:
    """
    Find the earliest slot where a task can start without violating capacity.
    
    Args:
        usage: Current resource usage array (per slot)
        capacity: Maximum capacity for this resource
        demand: Resource units required by the task
        earliest_slot: Earliest allowed start slot (arrival constraint)
        duration_slots: Task duration in slots
        last_start_slot: Latest allowed start slot (deadline constraint)
    
    Returns:
        Start slot index, or None if no feasible slot exists
    """
    if duration_slots <= 0:
        return earliest_slot
    
    if last_start_slot is None:
        last_start_slot = len(usage) - duration_slots
    
    last_start_slot = min(last_start_slot, len(usage) - duration_slots)
    
    if earliest_slot > last_start_slot:
        return None
    
    for s in range(earliest_slot, last_start_slot + 1):
        block = usage[s : s + duration_slots]
        if np.all(block + demand <= capacity):
            return s
    
    return None


def _find_best_low_co2_start(
    usage: np.ndarray,
    capacity: int,
    demand: int,
    earliest_slot: int,
    duration_slots: int,
    last_start_slot: int,
    co2_intensity: np.ndarray,
    base_slot: int,
    delay_penalty: float = 0.0,
) -> Optional[int]:
    """
    Find the best start slot that minimizes CO2 cost plus delay penalty.
    
    The cost function is:
        cost = sum(CO2[t:t+duration]) + delay_penalty * (start - base_slot)
    
    Args:
        usage: Current resource usage array
        capacity: Maximum capacity
        demand: Resource units required
        earliest_slot: Earliest allowed slot
        duration_slots: Task duration in slots
        last_start_slot: Latest allowed slot
        co2_intensity: CO2 intensity array (g/kWh per slot)
        base_slot: Reference slot for delay calculation
        delay_penalty: Penalty weight for each slot of delay
    
    Returns:
        Best start slot, or None if no feasible slot exists
    """
    best_slot = None
    best_cost = None
    
    for s in range(earliest_slot, last_start_slot + 1):
        # Check capacity constraint
        block = usage[s : s + duration_slots]
        if not np.all(block + demand <= capacity):
            continue
        
        # Calculate cost: CO2 + delay penalty
        co2_cost = float(co2_intensity[s : s + duration_slots].sum())
        delay_cost = float(delay_penalty) * float(max(0, s - base_slot))
        cost = co2_cost + delay_cost
        
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_slot = s
    
    return best_slot


def schedule_fifo(
    energy_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    capacity: dict[str, int],
    co2_col: str = 'CO2_Intensity_gkWh',
) -> pd.DataFrame:
    """
    FIFO (First-In-First-Out) scheduling.
    
    Tasks are scheduled in arrival order at the earliest feasible time.
    
    Args:
        energy_df: Energy data with datetime index
        jobs_df: Jobs dataframe with arrival, deadline, duration, etc.
        capacity: Resource capacity dict {resource_name: capacity}
        co2_col: Column name for CO2 intensity
    
    Returns:
        Scheduled jobs dataframe with scheduled_start and scheduled_end
    """
    idx0 = energy_df.index.min()
    horizon_slots = len(energy_df)
    
    # Initialize resource usage tracking
    usage = {r: np.zeros(horizon_slots, dtype=np.int16) for r in capacity.keys()}
    
    # Sort by arrival time
    out = jobs_df.sort_values(['arrival', 'priority', 'id']).reset_index(drop=True).copy()
    out['scheduled_start'] = pd.NaT
    out['scheduled_end'] = pd.NaT
    
    for row_i, job in out.iterrows():
        resource = job['resource']
        cap = int(capacity[resource])
        demand = int(job['demand'])
        
        arrival = pd.Timestamp(job['arrival'])
        deadline = pd.Timestamp(job['deadline'])
        duration_min = int(job['duration'])
        duration_slots = duration_min // FREQ_MINUTES
        
        # Calculate slot constraints
        earliest_slot, _ = _slots_between(arrival, arrival, idx0)
        latest_start_time = deadline - pd.Timedelta(minutes=duration_min)
        last_start_slot, _ = _slots_between(latest_start_time, latest_start_time, idx0)
        last_start_slot = min(last_start_slot, horizon_slots - duration_slots)
        
        # If deadline is too tight, extend search window (may cause violations)
        if last_start_slot < earliest_slot:
            last_start_slot = horizon_slots - duration_slots
        
        # Find earliest feasible start
        start_slot = _find_earliest_feasible_start(
            usage=usage[resource],
            capacity=cap,
            demand=demand,
            earliest_slot=earliest_slot,
            duration_slots=duration_slots,
            last_start_slot=None,  # Search entire horizon
        )
        
        if start_slot is None:
            raise RuntimeError(f"Cannot schedule job {job['id']} within horizon")
        
        end_slot = start_slot + duration_slots
        
        # Update usage and schedule
        usage[resource][start_slot:end_slot] += demand
        out.at[row_i, 'scheduled_start'] = idx0 + start_slot * pd.Timedelta(FREQ)
        out.at[row_i, 'scheduled_end'] = idx0 + end_slot * pd.Timedelta(FREQ)
    
    return out


def schedule_edf(
    energy_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    capacity: dict[str, int],
    co2_col: str = 'CO2_Intensity_gkWh',
) -> pd.DataFrame:
    """
    EDF (Earliest Deadline First) scheduling.
    
    Tasks are sorted by deadline and scheduled at the earliest feasible time.
    
    Args:
        energy_df: Energy data with datetime index
        jobs_df: Jobs dataframe
        capacity: Resource capacity dict
        co2_col: Column name for CO2 intensity
    
    Returns:
        Scheduled jobs dataframe
    """
    idx0 = energy_df.index.min()
    horizon_slots = len(energy_df)
    
    usage = {r: np.zeros(horizon_slots, dtype=np.int16) for r in capacity.keys()}
    
    # Sort by deadline (earliest first)
    out = jobs_df.sort_values(['deadline', 'priority', 'arrival', 'id']).reset_index(drop=True).copy()
    out['scheduled_start'] = pd.NaT
    out['scheduled_end'] = pd.NaT
    
    for row_i, job in out.iterrows():
        resource = job['resource']
        cap = int(capacity[resource])
        demand = int(job['demand'])
        
        arrival = pd.Timestamp(job['arrival'])
        deadline = pd.Timestamp(job['deadline'])
        duration_min = int(job['duration'])
        duration_slots = duration_min // FREQ_MINUTES
        
        earliest_slot, _ = _slots_between(arrival, arrival, idx0)
        latest_start_time = deadline - pd.Timedelta(minutes=duration_min)
        last_start_slot, _ = _slots_between(latest_start_time, latest_start_time, idx0)
        last_start_slot = min(last_start_slot, horizon_slots - duration_slots)
        
        if last_start_slot < earliest_slot:
            last_start_slot = horizon_slots - duration_slots
        
        start_slot = _find_earliest_feasible_start(
            usage=usage[resource],
            capacity=cap,
            demand=demand,
            earliest_slot=earliest_slot,
            duration_slots=duration_slots,
            last_start_slot=None,
        )
        
        if start_slot is None:
            raise RuntimeError(f"Cannot schedule job {job['id']} within horizon")
        
        end_slot = start_slot + duration_slots
        usage[resource][start_slot:end_slot] += demand
        out.at[row_i, 'scheduled_start'] = idx0 + start_slot * pd.Timedelta(FREQ)
        out.at[row_i, 'scheduled_end'] = idx0 + end_slot * pd.Timedelta(FREQ)
    
    return out


def _build_usage_from_schedule(
    energy_df: pd.DataFrame,
    scheduled_jobs: pd.DataFrame,
    capacity: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Build resource usage arrays from an existing schedule.
    """
    idx0 = energy_df.index.min()
    horizon_slots = len(energy_df)
    usage = {r: np.zeros(horizon_slots, dtype=np.int16) for r in capacity.keys()}
    
    for _, job in scheduled_jobs.iterrows():
        resource = job['resource']
        demand = int(job['demand'])
        start = pd.Timestamp(job['scheduled_start'])
        end = pd.Timestamp(job['scheduled_end'])
        s, e = _slots_between(start, end, idx0)
        
        if s < 0 or e > horizon_slots:
            raise RuntimeError(f"Scheduled job out of horizon: {job['id']}")
        
        usage[resource][s:e] += demand
    
    return usage


def improve_schedule_carbon_aware(
    energy_df: pd.DataFrame,
    fifo_schedule: pd.DataFrame,
    capacity: dict[str, int],
    co2_col: str = 'CO2_Intensity_gkWh',
    config: Optional[SchedulerConfig] = None,
) -> pd.DataFrame:
    """
    Two-phase carbon-aware scheduling optimization.
    
    Starting from a feasible FIFO schedule, locally optimize by shifting
    tasks to lower-carbon time slots while maintaining feasibility.
    
    Key properties:
    - Always maintains feasibility (never breaks a working schedule)
    - Only shifts tasks to the right (later), never earlier
    - Respects deadline constraints
    - Balances CO2 reduction with delay penalty
    
    Args:
        energy_df: Energy data with datetime index and CO2 column
        fifo_schedule: Initial feasible schedule (from FIFO)
        capacity: Resource capacity dict
        co2_col: Column name for CO2 intensity
        config: Scheduler configuration
    
    Returns:
        Optimized schedule with lower carbon footprint
    """
    if config is None:
        config = SchedulerConfig()
    
    idx0 = energy_df.index.min()
    horizon_slots = len(energy_df)
    co2 = energy_df[co2_col].to_numpy(dtype=float)
    
    out = fifo_schedule.copy()
    usage = _build_usage_from_schedule(energy_df, out, capacity)
    
    # Calculate optimization metrics for prioritization
    out['_dur_slots'] = (out['duration'] // FREQ_MINUTES).astype(int)
    out['_energy_kwh'] = out['power_avg'].astype(float) * (out['duration'].astype(float) / 60.0)
    out['_slack_min'] = (
        (pd.to_datetime(out['deadline']) - pd.to_datetime(out['scheduled_end']))
        .dt.total_seconds() / 60.0
    ).fillna(0.0)
    
    # Prioritize tasks: more slack and higher energy consumption first
    # (these have more potential for carbon reduction and are safer to move)
    move_order = out.sort_values(
        ['resource', '_slack_min', '_energy_kwh'],
        ascending=[True, False, False],
    ).index.tolist()
    
    for row_i in move_order:
        job = out.loc[row_i]
        resource = job['resource']
        cap = int(capacity[resource])
        demand = int(job['demand'])
        dur_slots = int(job['_dur_slots'])
        
        arrival = pd.Timestamp(job['arrival'])
        deadline = pd.Timestamp(job['deadline'])
        start = pd.Timestamp(job['scheduled_start'])
        end = pd.Timestamp(job['scheduled_end'])
        
        s_cur, e_cur = _slots_between(start, end, idx0)
        earliest_slot, _ = _slots_between(arrival, arrival, idx0)
        
        latest_start_time = deadline - pd.Timedelta(minutes=int(job['duration']))
        last_start_slot, _ = _slots_between(latest_start_time, latest_start_time, idx0)
        last_start_slot = min(last_start_slot, horizon_slots - dur_slots)
        
        # Search window: from current position to max_shift_slots ahead
        # Only right-shift (don't move earlier) to maintain feasibility
        s_min = max(s_cur, earliest_slot)
        s_max = min(last_start_slot, s_cur + config.max_shift_slots)
        
        if s_max <= s_min:
            continue
        
        # Temporarily remove current task from usage
        usage[resource][s_cur:e_cur] -= demand
        
        # Find better start slot (minimize CO2 + delay penalty)
        s_best = _find_best_low_co2_start(
            usage=usage[resource],
            capacity=cap,
            demand=demand,
            earliest_slot=s_min,
            duration_slots=dur_slots,
            last_start_slot=s_max,
            co2_intensity=co2,
            base_slot=s_cur,
            delay_penalty=config.shift_penalty,
        )
        
        if s_best is None:
            # Restore original position
            usage[resource][s_cur:e_cur] += demand
            continue
        
        # Apply the move
        e_best = s_best + dur_slots
        usage[resource][s_best:e_best] += demand
        out.at[row_i, 'scheduled_start'] = idx0 + s_best * pd.Timedelta(FREQ)
        out.at[row_i, 'scheduled_end'] = idx0 + e_best * pd.Timedelta(FREQ)
    
    # Clean up temporary columns
    out = out.drop(columns=['_dur_slots', '_energy_kwh', '_slack_min'])
    
    return out


def schedule_carbon_aware(
    energy_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    capacity: dict[str, int],
    co2_col: str = 'CO2_Intensity_gkWh',
    config: Optional[SchedulerConfig] = None,
) -> pd.DataFrame:
    """
    Carbon-aware scheduling using two-phase optimization.
    
    Phase 1: Generate feasible FIFO schedule
    Phase 2: Optimize by shifting tasks to lower-carbon slots
    
    Args:
        energy_df: Energy data with datetime index and CO2 column
        jobs_df: Jobs dataframe
        capacity: Resource capacity dict
        co2_col: Column name for CO2 intensity
        config: Scheduler configuration
    
    Returns:
        Carbon-optimized schedule
    """
    # Phase 1: FIFO
    fifo_schedule = schedule_fifo(energy_df, jobs_df, capacity, co2_col)
    
    # Phase 2: Carbon optimization
    optimized = improve_schedule_carbon_aware(
        energy_df, fifo_schedule, capacity, co2_col, config
    )
    
    return optimized


def get_default_capacity(jobs_df: pd.DataFrame) -> dict[str, int]:
    """
    Get default resource capacities based on job types.
    
    Args:
        jobs_df: Jobs dataframe
    
    Returns:
        Capacity dict {resource: capacity}
    """
    defaults = {
        'GPU': 2,
        'Lab_Bench': 1,
        'CPU_Pool': 8,
    }
    
    resources = jobs_df['resource'].unique()
    return {r: defaults.get(r, 1) for r in resources}
