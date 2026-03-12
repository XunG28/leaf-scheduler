"""
LEAF - Data Preprocessor
========================
process SMARD electricity data, calculate carbon intensity (CO2 Intensity) and renewable energy share

emission factor source:
    Umweltbundesamt (2023). Entwicklung der spezifischen 
    Treibhausgas-Emissionen des deutschen Strommix.
    https://www.umweltbundesamt.de/publikationen
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional


# =============================================================================
# emission factor configuration (g CO2 / kWh)
# =============================================================================
EMISSION_FACTORS: Dict[str, float] = {
    'Hard coal [MWh] Original resolutions': 820.0,      # hard coal - UBA 2023
    'Fossil gas [MWh] Original resolutions': 420.0,     # natural gas - UBA 2023
    'Other conventional [MWh] Original resolutions': 700.0,  # other conventional - conservative estimate
}

# renewable energy columns (emission factor is 0)
RENEWABLE_COLS = [
    'Biomass [MWh] Original resolutions',
    'Hydropower [MWh] Original resolutions',
    'Wind onshore [MWh] Original resolutions',
    'Photovoltaics [MWh] Original resolutions',
    'Other renewable [MWh] Original resolutions',
]

# low-carbon non-renewable (emission factor is 0)
LOW_CARBON_COLS = [
    'Nuclear [MWh] Original resolutions',
    'Hydro pumped storage [MWh] Original resolutions',
]

# all columns used for calculation
ALL_GENERATION_COLS = list(EMISSION_FACTORS.keys()) + RENEWABLE_COLS + LOW_CARBON_COLS


def clean_numeric_value(val) -> float:
    """
    clean numeric value: process German number format and missing values
    
    Args:
        val: original value (string, number or missing)
    
    Returns:
        float: cleaned numeric value
    """
    if pd.isna(val) or val == '-' or val == '':
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    # German number format: "1.234,56" -> 1234.56
    val_str = str(val).strip()
    val_str = val_str.replace('.', '').replace(',', '.')
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def load_smard_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    load SMARD raw CSV data
    
    Args:
        filepath: CSV file path
    
    Returns:
        pd.DataFrame: raw data
    """
    filepath = Path(filepath)
    
    # SMARD CSV uses semicolon delimiter
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    
    print(f"   load data: {filepath.name}")
    print(f"   number of rows: {len(df):,}")
    print(f"   number of columns: {len(df.columns)}")
    print(f"   time range: {df['Start date'].iloc[0]} → {df['Start date'].iloc[-1]}")
    
    return df


def preprocess_smard_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    preprocess SMARD data: timestamp conversion, numeric cleaning, calculate metrics
    
    Args:
        df: raw DataFrame
    
    Returns:
        pd.DataFrame: processed DataFrame
    """
    df = df.copy()
    
    # =========================================================================
    # Step 1: timestamp conversion
    # =========================================================================
    print("⏳ Step 1: timestamp conversion...")
    df['Start date'] = pd.to_datetime(df['Start date'], format='mixed')
    df['End date'] = pd.to_datetime(df['End date'], format='mixed')
    
    # =========================================================================
    # Step 2: numeric cleaning
    # =========================================================================
    print("⏳ Step 2: numeric cleaning...")
    for col in ALL_GENERATION_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric_value)
        else:
            print(f"   ⚠️ column not found, skip: {col}")
    
    # =========================================================================
    # Step 3: calculate total generation (denominator)
    # =========================================================================
    print("⏳ Step 3: calculate total generation...")
    existing_cols = [c for c in ALL_GENERATION_COLS if c in df.columns]
    df['Total_Generation_MWh'] = df[existing_cols].sum(axis=1)
    
    # =========================================================================
    # Step 4: calculate renewable energy generation
    # =========================================================================
    print("⏳ Step 4: calculate renewable energy generation...")
    renewable_existing = [c for c in RENEWABLE_COLS if c in df.columns]
    df['Renewable_Generation_MWh'] = df[renewable_existing].sum(axis=1)
    
    # =========================================================================
    # Step 5: calculate CO2 emissions (grams)
    # =========================================================================
    print("⏳ Step 5: calculate CO2 emissions...")
    df['CO2_Emissions_g'] = 0.0
    for col, factor in EMISSION_FACTORS.items():
        if col in df.columns:
            # MWh × g/kWh = MWh × g/kWh (intermediate value, divided by MWh to get g/kWh)
            df['CO2_Emissions_g'] += df[col] * factor
    
    # =========================================================================
    # Step 6: calculate final metrics
    # =========================================================================
    print("⏳ Step 6: calculate final metrics...")
    
    # filter out zero generation (avoid division by zero)
    valid_mask = df['Total_Generation_MWh'] > 0
    
    # renewable energy share (%)
    df['Renewable_Share_pct'] = np.where(
        valid_mask,
        df['Renewable_Generation_MWh'] / df['Total_Generation_MWh'] * 100,
        0.0
    )
    
    # carbon intensity (g CO2 / kWh)
    df['CO2_Intensity_gkWh'] = np.where(
        valid_mask,
        df['CO2_Emissions_g'] / df['Total_Generation_MWh'],
        0.0
    )
    
    # =========================================================================
    # Step 7: data quality check
    # =========================================================================
    print("⏳ Step 7: data quality check...")
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"    found {invalid_count} rows with zero generation")
    
    print(f"   preprocessing completed")
    print(f"   valid data rows: {valid_mask.sum():,}")
    print(f"   CO2 intensity range: {df['CO2_Intensity_gkWh'].min():.1f} - {df['CO2_Intensity_gkWh'].max():.1f} g/kWh")
    print(f"   renewable share range: {df['Renewable_Share_pct'].min():.1f} - {df['Renewable_Share_pct'].max():.1f} %")
    
    return df


def export_clean_data(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    columns: Optional[list] = None
) -> None:
    """
    export processed data
    
    Args:
        df: processed DataFrame
        output_path: output path
        columns: columns to export (default: core columns)
    """
    if columns is None:
        columns = [
            'Start date',
            'End date',
            'Total_Generation_MWh',
            'Renewable_Generation_MWh',
            'Renewable_Share_pct',
            'CO2_Intensity_gkWh',
        ]
    
    output_path = Path(output_path)
    df[columns].to_csv(output_path, index=False)
    print(f"    data exported: {output_path}")


def process_smard_pipeline(
    input_path: Union[str, Path],
    output_path: Union[str, Path]
) -> pd.DataFrame:
    """
    complete processing pipeline
    
    Args:
        input_path: raw data path
        output_path: output data path
    
    Returns:
        pd.DataFrame: processed DataFrame
    """
    print("=" * 60)
    print("LEAF Data Preprocessing Pipeline")
    print("=" * 60)
    
    # load data
    df = load_smard_data(input_path)
    
    # process data
    df = preprocess_smard_data(df)
    
    # export data
    export_clean_data(df, output_path)
    
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    
    return df


# =============================================================================
# command line entry
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python preprocessor.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_smard_pipeline(input_file, output_file)
