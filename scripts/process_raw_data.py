"""
LEAF - initialize raw data processing script
============================================
process SMARD raw data to be used for training
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from leaf.data.preprocessor import process_smard_pipeline

# =============================================================================
# configuration
# =============================================================================
RAW_DATA_PATH = project_root / "data" / "raw" / "Actual_generation_202403020000_202603090000_Quarterhour.csv"
OUTPUT_PATH = project_root / "data" / "processed" / "energy_data_full.csv"

# =============================================================================
# execute
# =============================================================================
if __name__ == "__main__":
    print(f"Input:  {RAW_DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    
    # ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # run processing pipeline
    df = process_smard_pipeline(RAW_DATA_PATH, OUTPUT_PATH)
    
    # display dataset splitting information
    print("\n" + "=" * 60)
    print("dataset splitting suggestion")
    print("=" * 60)
    
    df['Start date'] = df['Start date'] if df['Start date'].dtype == 'datetime64[ns]' else df['Start date']
    
    train_mask = (df['Start date'] >= '2024-03-02') & (df['Start date'] < '2026-02-01')
    val_mask = (df['Start date'] >= '2026-02-01') & (df['Start date'] < '2026-03-02')
    test_mask = (df['Start date'] >= '2026-03-02')
    
    print(f"Train:      2024-03-02 → 2026-01-31  ({train_mask.sum():,} samples)")
    print(f"Validation: 2026-02-01 → 2026-03-01  ({val_mask.sum():,} samples)")
    print(f"Test:       2026-03-02 → 2026-03-08  ({test_mask.sum():,} samples)")
