"""
LEAF - Task Definition
======================
Data structures for compute jobs and lab activities.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class Task:
    """
    Represents a schedulable task (compute job or lab activity).
    
    Attributes:
        id: Unique task identifier
        task_type: Type of task (e.g., 'AI_Training', 'Lab_Activity', 'Data_Process')
        resource: Required resource type (e.g., 'GPU', 'Lab_Bench', 'CPU_Pool')
        demand: Number of resource units required
        arrival: Earliest start time (task becomes available)
        deadline: Latest completion time (hard constraint)
        duration: Task duration in minutes (must be multiple of 15)
        power_avg: Average power consumption in kW
        priority: Priority level (1=highest, 3=lowest)
    """
    id: str
    task_type: str
    resource: str
    demand: int
    arrival: datetime
    deadline: datetime
    duration: int
    power_avg: float
    priority: int
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    
    @property
    def duration_slots(self) -> int:
        """Duration in 15-minute slots."""
        return self.duration // 15
    
    @property
    def energy_kwh(self) -> float:
        """Total energy consumption in kWh."""
        return self.power_avg * (self.duration / 60.0)
    
    @property
    def is_scheduled(self) -> bool:
        """Check if task has been scheduled."""
        return self.scheduled_start is not None and self.scheduled_end is not None
    
    @property
    def slack_minutes(self) -> float:
        """
        Available slack time in minutes.
        Slack = deadline - arrival - duration
        """
        window = (self.deadline - self.arrival).total_seconds() / 60.0
        return window - self.duration
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'type': self.task_type,
            'resource': self.resource,
            'demand': self.demand,
            'arrival': self.arrival,
            'deadline': self.deadline,
            'duration': self.duration,
            'power_avg': self.power_avg,
            'priority': self.priority,
            'scheduled_start': self.scheduled_start,
            'scheduled_end': self.scheduled_end,
        }


def load_tasks_from_csv(filepath: str) -> list[Task]:
    """
    Load tasks from CSV file.
    
    Args:
        filepath: Path to CSV file with task definitions
    
    Returns:
        List of Task objects
    """
    df = pd.read_csv(filepath)
    
    tasks = []
    for _, row in df.iterrows():
        task = Task(
            id=str(row['id']),
            task_type=str(row['type']),
            resource=str(row['resource']),
            demand=int(row['demand']),
            arrival=pd.to_datetime(row['arrival']),
            deadline=pd.to_datetime(row['deadline']),
            duration=int(row['duration']),
            power_avg=float(row['power_avg']),
            priority=int(row['priority']),
        )
        tasks.append(task)
    
    return tasks


def tasks_to_dataframe(tasks: list[Task]) -> pd.DataFrame:
    """
    Convert list of tasks to DataFrame.
    
    Args:
        tasks: List of Task objects
    
    Returns:
        DataFrame with task information
    """
    records = [t.to_dict() for t in tasks]
    return pd.DataFrame(records)
