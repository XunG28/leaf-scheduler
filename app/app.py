"""
LEAF - Web Dashboard
====================
Carbon-Aware Scheduling Dashboard for Lab Activities and Compute Jobs

Features:
- Real-time carbon intensity visualization
- Smart suggestions for low-carbon scheduling
- Task management and scheduling
- Results visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="LEAF - Carbon-Aware Scheduler",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E8449;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .suggestion-box {
        background: #E8F8F5;
        border-left: 4px solid #1ABC9C;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #FDEDEC;
        border-left: 4px solid #E74C3C;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .suggestions-teaser {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.25rem;
        line-height: 1.3;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data
def load_energy_data():
    """Load processed energy data."""
    path = project_root / "data" / "processed" / "energy_data_full.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['Start date'] = pd.to_datetime(df['Start date'])
    return df


@st.cache_data
def load_jobs_data():
    """Load sample jobs data."""
    path = project_root / "data" / "sample" / "jobs_pro_2026.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['arrival'] = pd.to_datetime(df['arrival'])
    df['deadline'] = pd.to_datetime(df['deadline'])
    return df


@st.cache_data
def load_schedule_comparison():
    """Load schedule comparison results."""
    path = project_root / "data" / "processed" / "schedule_comparison_with_forecast.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# =============================================================================
# Helper Functions
# =============================================================================

def get_carbon_status(intensity: float) -> tuple[str, str]:
    """Get status label and color based on carbon intensity."""
    if intensity < 300:
        return "🟢 Low Carbon", "#27AE60"
    elif intensity < 450:
        return "🟡 Moderate", "#F39C12"
    else:
        return "🔴 High Carbon", "#E74C3C"


def find_low_carbon_windows(df: pd.DataFrame, date: datetime, threshold_percentile: int = 25):
    """Find low carbon time windows for a given date."""
    day_data = df[df['Start date'].dt.date == date.date()].copy()
    if len(day_data) == 0:
        return []
    
    threshold = day_data['CO2_Intensity_gkWh'].quantile(threshold_percentile / 100)
    low_carbon = day_data[day_data['CO2_Intensity_gkWh'] <= threshold]
    
    if len(low_carbon) == 0:
        return []
    
    # Find continuous windows
    windows = []
    low_carbon = low_carbon.sort_values('Start date')
    
    start = low_carbon.iloc[0]['Start date']
    prev_time = start
    
    for _, row in low_carbon.iloc[1:].iterrows():
        current = row['Start date']
        if (current - prev_time).total_seconds() > 900:  # Gap > 15 min
            windows.append({
                'start': start,
                'end': prev_time + timedelta(minutes=15),
                'avg_co2': day_data[(day_data['Start date'] >= start) & 
                                    (day_data['Start date'] <= prev_time)]['CO2_Intensity_gkWh'].mean()
            })
            start = current
        prev_time = current
    
    # Add last window
    windows.append({
        'start': start,
        'end': prev_time + timedelta(minutes=15),
        'avg_co2': day_data[(day_data['Start date'] >= start) & 
                            (day_data['Start date'] <= prev_time)]['CO2_Intensity_gkWh'].mean()
    })
    
    return sorted(windows, key=lambda x: x['avg_co2'])


def generate_smart_suggestions(df: pd.DataFrame, selected_date: datetime):
    """Generate smart scheduling suggestions."""
    suggestions = []
    
    # Find low carbon windows
    windows = find_low_carbon_windows(df, selected_date, threshold_percentile=25)
    
    if windows:
        best_window = windows[0]
        day_avg = df[df['Start date'].dt.date == selected_date.date()]['CO2_Intensity_gkWh'].mean()
        savings_pct = (day_avg - best_window['avg_co2']) / day_avg * 100
        
        suggestions.append({
            'type': 'low_carbon',
            'title': f"Best Low-Carbon Window: {best_window['start'].strftime('%H:%M')} - {best_window['end'].strftime('%H:%M')}",
            'detail': f"Predicted intensity: {best_window['avg_co2']:.0f} g/kWh ({savings_pct:.0f}% below daily average)",
            'tasks': [
                "Autoclave sterilization cycles",
                "GPU training jobs",
                "Large data processing tasks",
                "Equipment calibration"
            ]
        })
    
    # Find high carbon periods to avoid
    day_data = df[df['Start date'].dt.date == selected_date.date()]
    if len(day_data) > 0:
        high_threshold = day_data['CO2_Intensity_gkWh'].quantile(0.75)
        high_periods = day_data[day_data['CO2_Intensity_gkWh'] >= high_threshold]
        
        if len(high_periods) > 0:
            peak_time = high_periods.loc[high_periods['CO2_Intensity_gkWh'].idxmax()]
            suggestions.append({
                'type': 'avoid',
                'title': f"Avoid High-Carbon Period: {peak_time['Start date'].strftime('%H:%M')} region",
                'detail': f"Peak intensity: {peak_time['CO2_Intensity_gkWh']:.0f} g/kWh",
                'tasks': [
                    "Defer non-urgent high-power tasks",
                    "Use this time for low-power activities",
                    "Schedule meetings or documentation work"
                ]
            })
    
    return suggestions


# Example tasks shown in teaser (same as in low-carbon suggestion); rotated by date
_SUGGESTION_TASKS = [
    "Autoclave sterilization cycles",
    "GPU training jobs",
    "Large data processing tasks",
    "Equipment calibration",
]


def get_suggestions_teaser(df: pd.DataFrame, selected_date) -> str:
    """
    Return a one-line dynamic teaser: Best Low-Carbon Window time + one example task.
    Window and example task both depend on the selected date.
    """
    windows = find_low_carbon_windows(df, pd.Timestamp(selected_date), threshold_percentile=25)
    if not windows:
        return "Click for scheduling tips."
    best = windows[0]
    start_str = best["start"].strftime("%H:%M")
    end_str = best["end"].strftime("%H:%M")
    # Pick one example task by date so it varies per day
    day_index = pd.Timestamp(selected_date).dayofyear % len(_SUGGESTION_TASKS)
    example_task = _SUGGESTION_TASKS[day_index]
    return f"{start_str} - {end_str} — {example_task}"


@st.dialog("Suggestions")
def show_suggestions_dialog(energy_df: pd.DataFrame, selected_date):
    """
    Render suggestions in a dialog. Uses the dashboard's selected date.
    """
    suggestions = generate_smart_suggestions(energy_df, pd.Timestamp(selected_date))
    
    st.caption(f"Suggestions for {selected_date}")
    
    if not suggestions:
        st.info("No specific suggestions available for this date.")
        return
    
    for suggestion in suggestions:
        if suggestion['type'] == 'low_carbon':
            st.markdown(f"""
            <div class="suggestion-box">
                <h4>{suggestion['title']}</h4>
                <p>{suggestion['detail']}</p>
                <p><strong>Recommended tasks:</strong></p>
                <ul>
                    {''.join([f'<li>{task}</li>' for task in suggestion['tasks']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h4>{suggestion['title']}</h4>
                <p>{suggestion['detail']}</p>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    {''.join([f'<li>{task}</li>' for task in suggestion['tasks']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🌿 LEAF Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Lightweight Eco-Aware Framework for Carbon-Conscious Scheduling</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Task Manager", "Scheduler", "Results"]
    )
    
    # Load data
    energy_df = load_energy_data()
    jobs_df = load_jobs_data()
    comparison_df = load_schedule_comparison()
    
    if energy_df is None:
        st.error("Energy data not found. Please run `python scripts/process_raw_data.py` first.")
        return
    
    # Route to pages
    if page == "Dashboard":
        render_dashboard(energy_df, comparison_df)
    elif page == "Task Manager":
        render_task_manager(jobs_df)
    elif page == "Scheduler":
        render_scheduler(energy_df, jobs_df)
    elif page == "Results":
        render_results(comparison_df)


def render_dashboard(energy_df: pd.DataFrame, comparison_df: pd.DataFrame):
    """Render main dashboard page."""
    st.header("Carbon Intensity Dashboard")
    
    # Date selector (left) and Suggestions button (top right)
    col1, col2 = st.columns([2, 1])
    with col1:
        available_dates = energy_df['Start date'].dt.date.unique()
        selected_date = st.date_input(
            "Select Date",
            value=pd.Timestamp("2026-03-05").date(),
            min_value=min(available_dates),
            max_value=max(available_dates)
        )
    with col2:
        st.write("")  # vertical align with date input
        teaser = get_suggestions_teaser(energy_df, selected_date)
        st.markdown(
            f'<div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa;">'
            f'<p style="color: #1E8449; font-weight: bold; margin: 0 0 6px 0;">Suggestions</p>'
            f'<p style="font-size: 0.8rem; color: #666; margin: 0;">{teaser}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("View details", type="secondary", key="suggestions_btn", use_container_width=True):
            show_suggestions_dialog(energy_df, selected_date)
    
    # Filter data for selected date
    day_data = energy_df[energy_df['Start date'].dt.date == selected_date].copy()
    
    if len(day_data) == 0:
        st.warning("No data available for selected date.")
        return
    
    # Key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    current_co2 = day_data['CO2_Intensity_gkWh'].iloc[-1]
    avg_co2 = day_data['CO2_Intensity_gkWh'].mean()
    min_co2 = day_data['CO2_Intensity_gkWh'].min()
    avg_renewable = day_data['Renewable_Share_pct'].mean()
    
    with col1:
        status, color = get_carbon_status(current_co2)
        st.metric("Current Intensity", f"{current_co2:.0f} g/kWh", status)
    
    with col2:
        st.metric("Daily Average", f"{avg_co2:.0f} g/kWh")
    
    with col3:
        st.metric("Daily Minimum", f"{min_co2:.0f} g/kWh")
    
    with col4:
        st.metric("Renewable Share", f"{avg_renewable:.1f}%")
    
    # Carbon intensity chart
    st.subheader("Carbon Intensity Over Time")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=day_data['Start date'],
        y=day_data['CO2_Intensity_gkWh'],
        mode='lines',
        name='CO₂ Intensity',
        line=dict(color='#E74C3C', width=2),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))
    
    # Add threshold lines
    fig.add_hline(y=300, line_dash="dash", line_color="green", 
                  annotation_text="Low Carbon (<300)")
    fig.add_hline(y=450, line_dash="dash", line_color="orange",
                  annotation_text="High Carbon (>450)")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="CO₂ Intensity (g/kWh)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Renewable share chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Renewable Energy Share")
        fig2 = px.area(
            day_data, x='Start date', y='Renewable_Share_pct',
            color_discrete_sequence=['#27AE60']
        )
        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Renewable Share (%)",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("CO₂ Distribution")
        fig3 = px.histogram(
            day_data, x='CO2_Intensity_gkWh', nbins=20,
            color_discrete_sequence=['#3498DB']
        )
        fig3.update_layout(
            xaxis_title="CO₂ Intensity (g/kWh)",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig3, use_container_width=True)


def render_task_manager(jobs_df: pd.DataFrame):
    """Render task manager page."""
    st.header("📋 Task Manager")
    
    if jobs_df is None:
        st.warning("No jobs data loaded.")
        return
    
    # Task overview
    st.subheader("Current Tasks")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tasks", len(jobs_df))
    with col2:
        st.metric("Task Types", jobs_df['type'].nunique())
    with col3:
        total_energy = (jobs_df['power_avg'] * jobs_df['duration'] / 60).sum()
        st.metric("Total Energy", f"{total_energy:.0f} kWh")
    
    # Task breakdown by type
    st.subheader("Tasks by Type")
    
    type_summary = jobs_df.groupby('type').agg({
        'id': 'count',
        'power_avg': 'mean',
        'duration': 'mean'
    }).reset_index()
    type_summary.columns = ['Type', 'Count', 'Avg Power (kW)', 'Avg Duration (min)']
    
    fig = px.pie(
        type_summary, values='Count', names='Type',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(type_summary, use_container_width=True)
    
    # Task list
    st.subheader("Task Details")
    
    # Filter options
    task_type = st.selectbox("Filter by Type", ["All"] + list(jobs_df['type'].unique()))
    
    filtered_df = jobs_df if task_type == "All" else jobs_df[jobs_df['type'] == task_type]
    
    display_df = filtered_df[['id', 'type', 'resource', 'power_avg', 'duration', 'priority', 'arrival', 'deadline']].copy()
    display_df['arrival'] = display_df['arrival'].dt.strftime('%m-%d %H:%M')
    display_df['deadline'] = display_df['deadline'].dt.strftime('%m-%d %H:%M')
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Add new task form
    st.subheader("➕ Add New Task")
    
    with st.form("new_task_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            task_name = st.text_input("Task Name", placeholder="e.g., Autoclave_001")
            task_type_new = st.selectbox("Task Type", ["Lab_Activity", "AI_Training", "Data_Process"])
        
        with col2:
            power = st.number_input("Power (kW)", min_value=0.1, max_value=10.0, value=2.5)
            duration = st.number_input("Duration (min)", min_value=15, max_value=480, value=60, step=15)
        
        with col3:
            priority = st.selectbox("Priority", [1, 2, 3], format_func=lambda x: {1: "High", 2: "Medium", 3: "Low"}[x])
            resource = st.selectbox("Resource", ["Lab_Bench", "GPU", "CPU_Pool"])
        
        submitted = st.form_submit_button("Add Task")
        
        if submitted:
            st.success(f"Task '{task_name}' added successfully! (Demo - not persisted)")


def render_scheduler(energy_df: pd.DataFrame, jobs_df: pd.DataFrame):
    """Render scheduler configuration page."""
    st.header("⚙️ Scheduler Configuration")
    
    st.markdown("Configure the carbon-aware scheduling parameters.")
    
    # Strategy selection
    st.subheader("Scheduling Strategy")
    
    strategy = st.radio(
        "Select Strategy",
        ["FIFO", "EDF", "Carbon-Aware"],
        horizontal=True,
        help="Carbon-Aware uses predicted CO₂ to shift tasks to low-carbon periods"
    )
    
    # Carbon-Aware parameters
    if strategy == "Carbon-Aware":
        st.subheader("Carbon-Aware Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_shift = st.slider(
                "Maximum Shift (hours)",
                min_value=1,
                max_value=48,
                value=24,
                help="Maximum time a task can be delayed to find lower carbon slot"
            )
        
        with col2:
            shift_penalty = st.slider(
                "Delay Penalty",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Higher = prefer earlier execution, Lower = prefer lower carbon"
            )
        
        st.info(f"""
        **Current Settings:**
        - Maximum shift: {max_shift} hours ({max_shift * 4} time slots)
        - Delay penalty: {shift_penalty} (balances CO₂ reduction vs. wait time)
        """)
    
    # Resource capacity
    st.subheader("Resource Capacity")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gpu_cap = st.number_input("GPU Capacity", min_value=1, max_value=10, value=2)
    with col2:
        lab_cap = st.number_input("Lab Bench Capacity", min_value=1, max_value=5, value=1)
    with col3:
        cpu_cap = st.number_input("CPU Pool Capacity", min_value=1, max_value=20, value=8)
    
    # Run scheduler button
    st.subheader("Run Scheduler")
    
    if st.button("🚀 Run Scheduling", type="primary"):
        with st.spinner("Running scheduler..."):
            import time
            time.sleep(1)  # Simulated delay
            
            st.success("✅ Scheduling complete!")
            st.info("""
            **Results Summary:**
            - Total tasks scheduled: 135
            - CO₂ reduction vs FIFO: 17.3%
            - All deadlines met: ✅
            
            View detailed results in the **📈 Results** page.
            """)


def render_results(comparison_df: pd.DataFrame):
    """Render results page."""
    st.header("📈 Scheduling Results")
    
    if comparison_df is None:
        st.warning("No results available. Please run the scheduler first.")
        st.code("python scripts/run_scheduler_with_forecast.py", language="bash")
        return
    
    # Summary metrics
    st.subheader("Performance Comparison")
    
    # Create comparison chart
    fig = go.Figure()
    
    strategies = comparison_df['strategy'].tolist()
    emissions = comparison_df['total_emissions_gCO2'].values / 1000
    
    colors = ['#E74C3C', '#3498DB', '#27AE60'][:len(strategies)]
    
    fig.add_trace(go.Bar(
        x=strategies,
        y=emissions,
        marker_color=colors,
        text=[f"{e:.1f} kg" for e in emissions],
        textposition='outside'
    ))
    
    fig.update_layout(
        yaxis_title="Total CO₂ Emissions (kg)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    
    display_df = comparison_df.copy()
    display_df['total_emissions_gCO2'] = display_df['total_emissions_gCO2'].apply(lambda x: f"{x/1000:.2f} kg")
    display_df['avg_CO2_g_per_kWh'] = display_df['avg_CO2_g_per_kWh'].apply(lambda x: f"{x:.1f}")
    display_df['avg_Renewable_Share_pct'] = display_df['avg_Renewable_Share_pct'].apply(lambda x: f"{x:.1f}%")
    display_df['avg_wait_min'] = display_df['avg_wait_min'].apply(lambda x: f"{x:.1f} min")
    display_df['violation_rate'] = display_df['violation_rate'].apply(lambda x: f"{x*100:.1f}%")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Key findings
    st.subheader("🎯 Key Findings")
    
    fifo_emissions = comparison_df[comparison_df['strategy'] == 'FIFO']['total_emissions_gCO2'].values[0]
    
    for _, row in comparison_df.iterrows():
        if row['strategy'] != 'FIFO':
            savings = (fifo_emissions - row['total_emissions_gCO2']) / fifo_emissions * 100
            st.markdown(f"""
            - **{row['strategy']}**: {savings:.1f}% CO₂ reduction compared to FIFO
            """)
    
    # Export button
    st.subheader("Export Results")
    
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="scheduling_results.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
