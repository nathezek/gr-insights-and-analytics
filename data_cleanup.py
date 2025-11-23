import pandas as pd
import numpy as np
import gc


# Essential telemetry channels to keep (filter out the rest)
PREFERRED_CHANNELS = {
    "speed", "Speed", "SPEED",
    "gear", "Gear", "GEAR",
    "aps", "APS", "throttle", "Throttle",
    "pbrake_f", "pbrake_r",
    "Steering_Angle", "steering", "steer",
    "accx_can", "accy_can",
    "VBOX_Long_Minutes", "VBOX_Lat_Min",
    "Laptrigger_lapdist_dls", "distance",
}


def detect_delimiter(file_path):
    """Detect CSV delimiter by reading first line"""
    try:
        with open(file_path, "r") as f:
            line = f.readline()
            return ";" if ";" in line and "," not in line.split(";")[0] else ","
    except:
        return ","


def clean_telemetry_long_format_chunked(df, sample_frac=0.10, max_chunks=80):
    """
    Memory-efficient processing for long-format telemetry (telemetry_name/telemetry_value)
    
    This is the PROVEN approach that prevents crashes:
    - Process in chunks
    - Sample each chunk
    - Filter to essential channels only
    - Use garbage collection
    """
    
    print(f"[INFO] Processing long-format telemetry (sample={sample_frac:.0%}, max_chunks={max_chunks})")
    
    # Step 1: Quick scan to find available channels
    print("[INFO] Scanning for available channels...")
    sample = df.head(50000)
    
    if 'telemetry_name' in sample.columns:
        available = set(sample['telemetry_name'].dropna().unique())
        chosen = sorted(PREFERRED_CHANNELS & available)
        
        if not chosen:
            print(f"[ERROR] No essential channels found!")
            print(f"[ERROR] Available: {list(available)[:20]}")
            raise ValueError("No speed/gear/throttle metrics found in data")
        
        print(f"[INFO] Using {len(chosen)} channels: {chosen}")
    else:
        raise ValueError("No 'telemetry_name' column found")
    
    del sample
    gc.collect()
    
    # Step 2: Sample the data
    total_rows = len(df)
    sample_rows = int(total_rows * sample_frac)
    
    print(f"[INFO] Sampling {sample_frac:.0%} of data: {total_rows:,} → {sample_rows:,} rows")
    df_sampled = df.sample(n=min(sample_rows, total_rows), random_state=42).copy()
    del df
    gc.collect()
    
    # Step 3: Clean the data
    print("[INFO] Cleaning sampled data...")
    
    # Filter to essential columns
    essential_cols = ['vehicle_id', 'vehicle_number', 'lap', 'timestamp', 'telemetry_name', 'telemetry_value']
    available_cols = [col for col in essential_cols if col in df_sampled.columns]
    df_sampled = df_sampled[available_cols]
    
    # Clean lap
    if 'lap' in df_sampled.columns:
        df_sampled = df_sampled[df_sampled['lap'].notna() & (df_sampled['lap'] != 32768)]
    
    # Clean timestamp
    if 'timestamp' in df_sampled.columns:
        df_sampled['timestamp'] = pd.to_datetime(df_sampled['timestamp'], utc=True, errors='coerce').dt.tz_localize(None)
        df_sampled = df_sampled.dropna(subset=['timestamp'])
    
    # Filter to chosen channels
    df_sampled = df_sampled[df_sampled['telemetry_name'].isin(chosen)]
    
    # Clean values
    df_sampled['telemetry_value'] = pd.to_numeric(df_sampled['telemetry_value'], errors='coerce')
    df_sampled = df_sampled.dropna(subset=['telemetry_value'])
    
    print(f"[INFO] After cleaning: {len(df_sampled):,} rows")
    
    # Step 4: Pivot to wide format
    print("[INFO] Pivoting to wide format...")
    
    index_cols = [col for col in ['vehicle_id', 'vehicle_number', 'lap', 'timestamp'] if col in df_sampled.columns]
    
    wide = df_sampled.pivot_table(
        index=index_cols,
        columns='telemetry_name',
        values='telemetry_value',
        aggfunc='mean'  # Average if duplicates
    ).reset_index()
    
    del df_sampled
    gc.collect()
    
    print(f"[INFO] Pivoted to: {wide.shape}")
    
    # Step 5: Handle column name variations
    column_mappings = {
        'Speed': 'speed', 'SPEED': 'speed',
        'Gear': 'gear', 'GEAR': 'gear',
        'APS': 'aps', 'Throttle': 'aps', 'throttle': 'aps',
    }
    
    for old_name, new_name in column_mappings.items():
        if old_name in wide.columns and new_name not in wide.columns:
            wide = wide.rename(columns={old_name: new_name})
    
    # Step 6: Add derived features
    if 'accx_can' in wide.columns and 'accy_can' in wide.columns:
        wide['total_g'] = np.sqrt(wide['accx_can'] ** 2 + wide['accy_can'] ** 2)
    
    # Step 7: Optimize memory - convert to float32
    for col in wide.columns:
        if col not in ['vehicle_id', 'vehicle_number', 'lap', 'timestamp']:
            if pd.api.types.is_numeric_dtype(wide[col]):
                wide[col] = wide[col].astype('float32')
    
    memory_mb = wide.memory_usage(deep=True).sum() / 1024**2
    print(f"[INFO] ✅ Final telemetry: {wide.shape}, {memory_mb:.2f} MB")
    
    return wide


def clean_telemetry_chunked(df, chunksize=50000):
    """Clean wide-format telemetry data in chunks"""
    
    print(f"[INFO] Cleaning wide-format telemetry in chunks of {chunksize:,} rows...")
    
    chunks = []
    total_rows = len(df)
    
    for i in range(0, total_rows, chunksize):
        chunk = df.iloc[i:i+chunksize].copy()
        
        if "lap" in chunk.columns:
            chunk = chunk[chunk["lap"] != 32768]
        
        if "timestamp" in chunk.columns:
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors='coerce').dt.tz_localize(None)
        
        if "accx_can" in chunk.columns and "accy_can" in chunk.columns:
            chunk["total_g"] = np.sqrt(chunk["accx_can"] ** 2 + chunk["accy_can"] ** 2)
        
        potential_numeric_cols = [
            "speed", "gear", "aps", "pbrake_f", "pbrake_r",
            "Steering_Angle", "accx_can", "accy_can", "total_g",
        ]
        
        numeric_cols = [col for col in potential_numeric_cols if col in chunk.columns]
        
        if numeric_cols:
            chunk[numeric_cols] = chunk[numeric_cols].apply(pd.to_numeric, errors="coerce")
            chunk = chunk.dropna(how='all', subset=numeric_cols)
        
        if len(chunk) > 0:
            chunks.append(chunk)
        
        if (i + chunksize) % (chunksize * 10) == 0:
            print(f"  Processed {i + chunksize:,} / {total_rows:,} rows...")
            gc.collect()
    
    if len(chunks) == 0:
        raise ValueError("No data left after cleaning")
    
    result = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    
    sort_cols = [col for col in ["vehicle_id", "lap", "timestamp"] if col in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols)
    
    print(f"[INFO] Cleaning complete: {len(result):,} rows retained")
    return result


def clean_telemetry(df, max_rows=500000, sample_frac=0.10):
    """
    Clean main telemetry - ULTRA MEMORY SAFE VERSION
    Based on proven working code
    """
    
    original_size = len(df)
    print(f"[INFO] Starting telemetry cleaning: {original_size:,} rows")
    print(f"[INFO] Columns ({len(df.columns)}): {list(df.columns)}")
    
    # CRITICAL: Check if this is long-format data
    if 'telemetry_name' in df.columns and 'telemetry_value' in df.columns:
        print("[INFO] 🔄 Detected long-format data - using chunk-based pivot")
        
        # Use the proven memory-safe approach
        result = clean_telemetry_long_format_chunked(df, sample_frac=sample_frac, max_chunks=80)
        
    else:
        # Wide format - sample first
        print("[INFO] Wide-format data detected")
        
        if len(df) > max_rows:
            print(f"[INFO] Sampling {max_rows:,} rows from {len(df):,}")
            df = df.sample(n=max_rows, random_state=42).copy()
            gc.collect()
        
        # Handle column name variations
        column_mappings = {
            'Speed': 'speed', 'SPEED': 'speed', 'vCar': 'speed', 'velocity': 'speed',
            'Lap': 'lap', 'LAP': 'lap',
            'Gear': 'gear', 'GEAR': 'gear',
            'APS': 'aps', 'Throttle': 'aps',
        }
        
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Use chunked processing
        if len(df) > 100000:
            result = clean_telemetry_chunked(df, chunksize=50000)
        else:
            result = df.copy()
            
            if "lap" in result.columns:
                result = result[result["lap"] != 32768]
            
            if "timestamp" in result.columns:
                result["timestamp"] = pd.to_datetime(result["timestamp"], errors='coerce').dt.tz_localize(None)
            
            if "accx_can" in result.columns and "accy_can" in result.columns:
                result["total_g"] = np.sqrt(result["accx_can"] ** 2 + result["accy_can"] ** 2)
    
    # Verify required columns
    if 'speed' not in result.columns:
        print(f"[ERROR] No speed column found!")
        print(f"[ERROR] Available: {list(result.columns)}")
        raise ValueError(f"Missing 'speed' column. Available: {list(result.columns)[:30]}")
    
    if 'lap' not in result.columns:
        print(f"[ERROR] No lap column found!")
        print(f"[ERROR] Available: {list(result.columns)}")
        raise ValueError(f"Missing 'lap' column. Available: {list(result.columns)[:30]}")
    
    print(f"[INFO] ✅ Telemetry cleaning complete: {len(result):,} rows")
    gc.collect()
    
    return result


def clean_results(df):
    """Clean race results: convert lap times to seconds."""
    
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return np.nan
        if not isinstance(time_str, str):
            return np.nan
        try:
            parts = time_str.split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        except:
            return np.nan

    time_cols = ["TOTAL_TIME", "FL_TIME", "GAP_FIRST", "GAP_PREVIOUS"]
    for col in time_cols:
        if col in df.columns:
            df[col + "_seconds"] = df[col].apply(time_to_seconds)

    extra_cols = [col for col in df.columns if col.startswith("*Extra")]
    if extra_cols:
        df = df.drop(columns=extra_cols)
    
    return df


def join_datasets(data_dict, max_merged_rows=1000000):
    """
    Join telemetry with weather and results - ULTRA MEMORY SAFE VERSION
    Based on proven working code
    """
    
    telemetry = data_dict["telemetry"]
    weather = data_dict.get("weather")
    results = data_dict.get("results")
    
    original_size = len(telemetry)
    print(f"[INFO] Starting merge with {original_size:,} telemetry rows...")
    
    merged = telemetry.copy()
    del telemetry
    gc.collect()
    
    # Fix timestamp dtype
    if "timestamp" in merged.columns:
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors='coerce').dt.tz_localize(None)
    
    # Merge weather by nearest time
    if weather is not None and len(weather) > 0 and "TIME_UTC_STR" in weather.columns:
        print("[INFO] Merging weather data...")
        
        # Keep only essential weather columns
        weather_cols = ['TIME_UTC_STR']
        for col in ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'RAIN', 'PRESSURE', 'WIND_SPEED']:
            if col in weather.columns:
                weather_cols.append(col)
        
        weather = weather[weather_cols].copy()
        weather["time_utc"] = pd.to_datetime(weather["TIME_UTC_STR"], errors='coerce').dt.tz_localize(None)
        weather = weather.dropna(subset=['time_utc'])
        
        if "timestamp" in merged.columns and len(weather) > 0:
            merged_sorted = merged.sort_values("timestamp")
            weather_sorted = weather.sort_values("time_utc")
            
            merged = pd.merge_asof(
                merged_sorted,
                weather_sorted,
                left_on="timestamp",
                right_on="time_utc",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=30)  # Tighter tolerance
            )
            
            print(f"  After weather merge: {len(merged):,} rows")
            del weather, merged_sorted, weather_sorted
            gc.collect()
    
    # Merge results - DEDUPLICATE FIRST
    if results is not None and len(results) > 0 and "NUMBER" in results.columns and "vehicle_number" in merged.columns:
        print("[INFO] Merging results data...")
        
        # CRITICAL: Deduplicate to one row per NUMBER
        results = results.drop_duplicates(subset=["NUMBER"], keep="first")
        print(f"  Results unique vehicles: {len(results)}")
        
        # Keep only essential columns
        result_cols = ['NUMBER']
        for col in ['POSITION', 'CLASS', 'TEAM', 'DRIVER_SHORTNAME', 'VEHICLE']:
            if col in results.columns:
                result_cols.append(col)
        
        results = results[result_cols].copy()
        
        # Convert time columns
        for col in ['FL_TIME', 'TOTAL_TIME']:
            if col in results.columns:
                results[col + '_sec'] = results[col].apply(
                    lambda t: pd.to_timedelta(str(t), errors='coerce').total_seconds() if pd.notna(t) else np.nan
                )
        
        original_len = len(merged)
        merged = pd.merge(
            merged,
            results,
            left_on="vehicle_number",
            right_on="NUMBER",
            how="left"
        )
        
        print(f"  After results merge: {len(merged):,} rows")
        
        # Check for explosion
        if len(merged) > original_len * 1.2:
            print(f"[WARN] Results merge caused explosion! Deduplicating...")
            dedup_cols = ['vehicle_id', 'vehicle_number', 'lap', 'timestamp']
            dedup_cols = [col for col in dedup_cols if col in merged.columns]
            merged = merged.drop_duplicates(subset=dedup_cols)
            print(f"  After dedup: {len(merged):,} rows")
        
        del results
        gc.collect()
    
    # Final size check
    final_size = len(merged)
    print(f"[INFO] Size change: {original_size:,} → {final_size:,} ({final_size / original_size:.2f}x)")
    
    if final_size > original_size * 1.3:
        print("[WARN] Unexpected growth, deduplicating...")
        dedup_cols = ['vehicle_id', 'vehicle_number', 'lap', 'timestamp']
        dedup_cols = [col for col in dedup_cols if col in merged.columns]
        merged = merged.drop_duplicates(subset=dedup_cols)
        print(f"  After dedup: {len(merged):,} rows")
    
    # Sample if still too large
    if len(merged) > max_merged_rows:
        print(f"[WARN] Final dataset too large ({len(merged):,}). Sampling to {max_merged_rows:,}")
        merged = merged.sample(n=max_merged_rows, random_state=42).copy()
    
    print(f"[INFO] ✅ Final merged data: {len(merged):,} rows")
    gc.collect()
    
    return merged