"""
Generate a synthetic job set aligned with 15-minute CO2 data.

This script is a Python version of the experimental notebook
`greenAI/jobs_generation.ipynb`. It produces the CSV file
`data/sample/jobs_pro_2026.csv` used by the scheduling demo.

Usage (from project root):

    python scripts/generate_jobs_pro.py
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def ceil_to_15min(ts: pd.Timestamp) -> pd.Timestamp:
    """Round a timestamp up to the next 15-minute boundary."""
    ts = pd.Timestamp(ts).replace(second=0, microsecond=0)
    minutes_to_add = (15 - ts.minute % 15) % 15
    return ts + timedelta(minutes=int(minutes_to_add))


def generate_pro_jobs(
    num_jobs: int = 200,
    start_date: str = "2026-03-02 00:00:00",
    end_limit: str = "2026-03-09 00:00:00",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate a synthetic job set for the demo, aligned to 15-minute slots.

    Jobs cover three types with different typical durations and resources:
    - Lab_Activity  (Lab_Bench, short jobs, moderate power)
    - AI_Training   (GPU, medium jobs, high power)
    - Data_Process  (CPU_Pool, long jobs, low power)

    All arrivals, deadlines, and durations are multiples of 15 minutes.

    Returns:
        df_jobs: DataFrame with one row per job
        capacity: dict of resource capacities for the scheduler
    """
    rng = np.random.default_rng(seed)
    current_time = ceil_to_15min(pd.Timestamp(start_date))
    limit_time = pd.Timestamp(end_limit)

    # Example resource capacities: 2 GPUs, 1 lab bench, 8 CPU slots
    capacity = {"GPU": 2, "Lab_Bench": 1, "CPU_Pool": 8}

    jobs: list[dict] = []

    for i in range(num_jobs):
        # 1) Sample inter-arrival time (denser during the day, sparser at night)
        hour = current_time.hour
        rate = 40 if 9 <= hour <= 18 else 110  # minutes
        wait_time = rng.exponential(rate)

        # Update current_time and align to 15-minute grid
        current_time = ceil_to_15min(current_time + timedelta(minutes=int(wait_time)))
        arrival = current_time

        # 2) Sample job profile + resource requirements
        r = rng.random()
        if r < 0.65:
            j_type = "Lab_Activity"
            duration = int(rng.choice([15, 30, 45, 60]))
            resource = "Lab_Bench"
            demand = 1
            power = float(np.round(rng.uniform(1.5, 3.0), 2))
            priority = 1
            # slack sampled in 15-minute steps, ensures aligned deadlines
            slack_minutes = int(rng.integers(120 // 15, 300 // 15 + 1) * 15)  # 2–5h
        elif r < 0.90:
            j_type = "AI_Training"
            duration = int(rng.choice([60, 120, 180]))
            resource = "GPU"
            demand = 1
            power = float(np.round(rng.uniform(4.0, 7.0), 2))
            priority = 2
            slack_minutes = int(rng.integers(480 // 15, 960 // 15 + 1) * 15)  # 8–16h
        else:
            j_type = "Data_Process"
            duration = int(rng.choice([300, 360, 480, 600]))
            resource = "CPU_Pool"
            demand = 2
            power = float(np.round(rng.uniform(0.5, 1.2), 2))
            priority = 3
            slack_minutes = int(rng.integers(1200 // 15, 2400 // 15 + 1) * 15)  # 20–40h

        # 3) Stop if we would exceed the energy data window
        if arrival + timedelta(minutes=duration) > limit_time:
            break

        # 4) Deadline within window, with at least zero slack
        deadline = arrival + timedelta(minutes=duration + slack_minutes)
        if deadline > limit_time:
            deadline = limit_time
        earliest_finish = arrival + timedelta(minutes=duration)
        if deadline < earliest_finish:
            deadline = earliest_finish

        jobs.append(
            {
                "id": f"JOB_{i+1:03d}",
                "type": j_type,
                "resource": resource,
                "demand": int(demand),
                "arrival": arrival,
                "deadline": deadline,
                "duration": int(duration),
                "power_avg": float(power),
                "priority": int(priority),
                "scheduled_start": pd.NaT,
                "scheduled_end": pd.NaT,
            }
        )

    df_jobs = pd.DataFrame(jobs)

    # Basic sanity checks: alignment to 15-minute grid and zero seconds
    if len(df_jobs) > 0:
        assert (df_jobs["arrival"].dt.second == 0).all()
        assert (df_jobs["deadline"].dt.second == 0).all()
        assert ((df_jobs["arrival"].dt.minute % 15) == 0).all()
        assert ((df_jobs["deadline"].dt.minute % 15) == 0).all()
        assert ((df_jobs["duration"] % 15) == 0).all()

    return df_jobs, capacity


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_path = project_root / "data" / "sample" / "jobs_pro_2026.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_jobs, capacity = generate_pro_jobs()
    print(f"✅ Generated professional demo job set. Total jobs: {len(df_jobs)}")
    print("   capacity =", capacity)
    if len(df_jobs) > 0:
        print(df_jobs[["id", "arrival", "duration", "resource", "demand"]].head())

    # When reading, use parse_dates for the datetime columns.
    df_jobs.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")


if __name__ == "__main__":
    main()

