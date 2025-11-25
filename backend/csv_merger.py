"""
Utility functions for merging lap timing CSVs with telemetry data
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


def merge_lap_csvs(
    lap_start_df: pd.DataFrame,
    lap_end_df: pd.DataFrame,
    lap_time_df: pd.DataFrame,
    telemetry_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge lap timing data with telemetry data.
    
    Args:
        lap_start_df: DataFrame with lap start information (lap, start_time/start_distance)
        lap_end_df: DataFrame with lap end information (lap, end_time/end_distance)
        lap_time_df: DataFrame with lap timing (lap, lap_time)
        telemetry_df: DataFrame with telemetry data
    
    Returns:
        Merged DataFrame with all data combined
    """
    print(f"[INFO] Merging lap CSVs...")
    print(f"  - Lap Start: {len(lap_start_df)} rows, columns: {list(lap_start_df.columns)}")
    print(f"  - Lap End: {len(lap_end_df)} rows, columns: {list(lap_end_df.columns)}")
    print(f"  - Lap Time: {len(lap_time_df)} rows, columns: {list(lap_time_df.columns)}")
    print(f"  - Telemetry: {len(telemetry_df)} rows, columns: {list(telemetry_df.columns)}")
    
    # Standardize column names (case-insensitive)
    def standardize_columns(df):
        df.columns = df.columns.str.lower().str.strip()
        return df
    
    lap_start_df = standardize_columns(lap_start_df)
    lap_end_df = standardize_columns(lap_end_df)
    lap_time_df = standardize_columns(lap_time_df)
    telemetry_df = standardize_columns(telemetry_df)
    
    # Ensure telemetry has a lap column
    if 'lap' in telemetry_df.columns:
        print(f"[INFO] Telemetry already has 'lap' column with {telemetry_df['lap'].nunique()} unique laps")
    else:
        print("[WARNING] Telemetry data missing 'lap' column. Attempting to infer from timing data...")
        
        # Try to match based on timestamp if available
        if 'timestamp' in telemetry_df.columns and 'timestamp' in lap_start_df.columns:
            print("[INFO] Using vectorized timestamp matching for lap assignment...")
            telemetry_df['timestamp'] = pd.to_datetime(telemetry_df['timestamp'], errors='coerce')
            lap_start_df['timestamp'] = pd.to_datetime(lap_start_df['timestamp'], errors='coerce')
            lap_end_df['timestamp'] = pd.to_datetime(lap_end_df['timestamp'], errors='coerce')
            
            # Merge lap timing data to get start and end times per lap
            lap_ranges = lap_start_df[['lap', 'timestamp']].rename(columns={'timestamp': 'start_time'})
            lap_end_times = lap_end_df[['lap', 'timestamp']].rename(columns={'timestamp': 'end_time'})
            lap_ranges = lap_ranges.merge(lap_end_times, on='lap', how='inner')
            
            # Use merge_asof for efficient timestamp-based matching
            # Sort both dataframes by timestamp
            telemetry_df = telemetry_df.sort_values('timestamp')
            lap_ranges = lap_ranges.sort_values('start_time')
            
            # Assign lap numbers using merge_asof (much faster than row-by-row)
            telemetry_df = pd.merge_asof(
                telemetry_df,
                lap_ranges,
                left_on='timestamp',
                right_on='start_time',
                direction='backward',
                suffixes=('', '_lap')
            )
            
            # Filter to only keep rows within lap end time
            telemetry_df = telemetry_df[telemetry_df['timestamp'] <= telemetry_df['end_time']]
            telemetry_df = telemetry_df.drop(columns=['start_time', 'end_time'], errors='ignore')
            telemetry_df = telemetry_df.dropna(subset=['lap'])
            telemetry_df['lap'] = telemetry_df['lap'].astype(int)
            print(f"[INFO] Assigned lap numbers to {len(telemetry_df)} telemetry rows")
        else:
            raise ValueError("Cannot infer lap numbers: telemetry has no 'lap' column and timestamp matching is not possible")
    
    # Merge lap timing metadata (optional - adds extra columns if available)
    # Skip if lap files have the same structure as telemetry (no unique metadata)
    telemetry_cols = set(telemetry_df.columns)
    lap_start_unique = set(lap_start_df.columns) - telemetry_cols - {'lap'}
    lap_end_unique = set(lap_end_df.columns) - telemetry_cols - {'lap'}
    lap_time_unique = set(lap_time_df.columns) - telemetry_cols - {'lap'}
    
    if lap_start_unique or lap_end_unique or lap_time_unique:
        print(f"[INFO] Merging unique lap metadata columns...")
        lap_metadata = lap_start_df[['lap'] + list(lap_start_unique)].copy() if lap_start_unique else lap_start_df[['lap']].copy()
        
        if lap_end_unique:
            lap_end_subset = lap_end_df[['lap'] + list(lap_end_unique)]
            lap_metadata = lap_metadata.merge(lap_end_subset, on='lap', how='outer', suffixes=('_start', '_end'))
        
        if lap_time_unique:
            lap_time_subset = lap_time_df[['lap'] + list(lap_time_unique)]
            lap_metadata = lap_metadata.merge(lap_time_subset, on='lap', how='outer')
        
        # Only merge if we actually have unique metadata
        if len(lap_metadata.columns) > 1:
            result = telemetry_df.merge(lap_metadata, on='lap', how='left', suffixes=('', '_meta'))
        else:
            result = telemetry_df
    else:
        print(f"[INFO] Lap files have no unique columns - using telemetry data only")
        result = telemetry_df
    
    print(f"[INFO] ✅ Merge complete: {len(result)} rows, {len(result.columns)} columns")
    return result


def validate_lap_csvs(files: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    """
    Validate that the uploaded CSV files have the expected structure.
    
    Returns:
        (is_valid, error_message)
    """
    required_files = ['lap_start', 'lap_end', 'lap_time', 'telemetry']
    
    for file_key in required_files:
        if file_key not in files:
            return False, f"Missing required file: {file_key}.csv"
        
        if files[file_key].empty:
            return False, f"{file_key}.csv is empty"
    
    # Check for lap column in lap files
    for file_key in ['lap_start', 'lap_end', 'lap_time']:
        df = files[file_key]
        if 'lap' not in df.columns and 'Lap' not in df.columns and 'LAP' not in df.columns:
            return False, f"{file_key}.csv must have a 'lap' column"
    
    return True, ""
