import pandas as pd
import numpy as np
import os
import gc

os.makedirs("processed_data", exist_ok=True)

PREFERRED_CHANNELS = {
    "speed",
    "gear",
    "aps",
    "pbrake_f",
    "pbrake_r",
    "Steering_Angle",
    "accx_can",
    "accy_can",
    "VBOX_Long_Minutes",
    "VBOX_Lat_Min",
    "Laptrigger_lapdist_dls",
}


def detect_delimiter(file_path: str) -> str:
    with open(file_path, "r") as f:
        line = f.readline()
        return ";" if ";" in line and "," not in line.split(";")[0] else ","


def clean_telemetry_wide_chunked(
    csv_path: str, sample_frac=0.10, max_chunks=80
) -> pd.DataFrame:
    """Memory-efficient telemetry processing"""
    print(
        f"[INFO] Loading telemetry: {csv_path} (sample={sample_frac}, max_chunks={max_chunks})"
    )

    sample = pd.read_csv(csv_path, nrows=50_000, usecols=["telemetry_name"])
    available = set(sample["telemetry_name"].dropna().unique())
    chosen = sorted(PREFERRED_CHANNELS & available)

    if not chosen:
        print(f"[ERROR] No channels found!")
        return pd.DataFrame()

    print(f"[INFO] Using {len(chosen)} channels: {chosen}")
    del sample
    gc.collect()

    usecols_list = [
        "vehicle_id",
        "vehicle_number",
        "lap",
        "timestamp",
        "telemetry_name",
        "telemetry_value",
    ]

    print(f"[INFO] Processing telemetry in chunks...")
    chunk_list = []
    chunk_count = 0

    for chunk in pd.read_csv(
        csv_path, usecols=usecols_list, chunksize=200000, low_memory=False
    ):
        chunk = chunk.sample(frac=sample_frac, random_state=42)

        chunk = chunk[chunk["lap"].notna() & (chunk["lap"] != 32768)]
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], utc=True, errors="coerce"
        ).dt.tz_localize(None)
        chunk = chunk.dropna(subset=["timestamp", "telemetry_name", "telemetry_value"])
        chunk = chunk[chunk["telemetry_name"].isin(chosen)]
        chunk["telemetry_value"] = pd.to_numeric(
            chunk["telemetry_value"], errors="coerce"
        )
        chunk = chunk.dropna(subset=["telemetry_value"])

        if len(chunk) > 0:
            chunk_list.append(chunk)

        chunk_count += 1
        if chunk_count % 20 == 0:
            print(
                f"  Processed {chunk_count} chunks, rows: {sum(len(c) for c in chunk_list):,}"
            )
            gc.collect()

        if chunk_count >= max_chunks:
            print(f"[INFO] Reached max chunks ({max_chunks})")
            break

    print(f"[INFO] Concatenating chunks...")
    df = pd.concat(chunk_list, ignore_index=True)
    del chunk_list
    gc.collect()

    print(f"[INFO] Clean long format → {len(df):,} rows")

    print("[INFO] Pivoting to wide format...")
    wide = df.pivot_table(
        index=["vehicle_id", "vehicle_number", "lap", "timestamp"],
        columns="telemetry_name",
        values="telemetry_value",
        aggfunc="mean",
    ).reset_index()

    del df
    gc.collect()

    if {"accx_can", "accy_can"} <= set(wide.columns):
        wide["total_g"] = np.sqrt(wide["accx_can"] ** 2 + wide["accy_can"] ** 2)

    for col in wide.columns:
        if col not in ["vehicle_id", "vehicle_number", "lap", "timestamp"]:
            if pd.api.types.is_numeric_dtype(wide[col]):
                wide[col] = wide[col].astype("float32")

    print(
        f"[INFO] Final telemetry: {wide.shape}, {wide.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )
    return wide


def time_to_seconds(t) -> float:
    if pd.isna(t):
        return np.nan
    parts = str(t).split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return np.nan


paths = {
    "results": "./raw_training_data/cota/Race_1/03_Provisional Results_Race 1.CSV",
    "telemetry": "./raw_training_data/cota/Race_1/R1_COTA_telemetry_data.csv",
    "weather": "./raw_training_data/cota/Race_1/26_Weather_Race 1.CSV",
    "analysis": "./raw_training_data/cota/Race_1/23_AnalysisEnduranceWithSections_Race 1.CSV",
    "best_laps": "./raw_training_data/cota/Race_1/99_Best 10 Laps By Driver_Race 1.CSV",
    "lap_start": "./raw_training_data/cota/Race_1/R1_COTA_lap_start.csv",
    "lap_time": "./raw_training_data/cota/Race_1/R1_COTA_lap_time.csv",
    "lap_end": "./raw_training_data/cota/Race_1/R1_COTA_lap_end.csv",
}

print("=" * 60)
print("=== Loading Race 3 (cota) - FULL MERGE ===")
print("=" * 60)

# Load telemetry
telemetry = clean_telemetry_wide_chunked(
    paths["telemetry"], sample_frac=0.10, max_chunks=80
)
gc.collect()

if telemetry.empty:
    raise ValueError("No telemetry data loaded!")

original_size = len(telemetry)
print(f"\n[INFO] Original telemetry size: {original_size:,} rows")

merged = telemetry.copy()
del telemetry
gc.collect()

# Weather
if os.path.exists(paths["weather"]):
    print("\n[INFO] Merging weather data...")
    weather = pd.read_csv(
        paths["weather"], delimiter=detect_delimiter(paths["weather"])
    )
    print(f"  Weather columns: {list(weather.columns)}")

    weather["time_utc"] = pd.to_datetime(
        weather["TIME_UTC_STR"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )

    # Get available weather columns
    weather_cols = ["time_utc"]
    for col in weather.columns:
        if col not in ["TIME_UTC_STR", "time_utc"]:
            weather_cols.append(col)

    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        weather[weather_cols].sort_values("time_utc"),
        left_on="timestamp",
        right_on="time_utc",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=30),
    )
    print(f"  After weather merge: {merged.shape}")
    del weather
    gc.collect()

# Results - DEDUPLICATE FIRST
if os.path.exists(paths["results"]):
    print("\n[INFO] Merging results data...")
    results = pd.read_csv(
        paths["results"], delimiter=detect_delimiter(paths["results"])
    )
    results.columns = results.columns.str.strip()

    if "NUMBER" in results.columns:
        # CRITICAL: One row per NUMBER
        results = results.drop_duplicates(subset=["NUMBER"], keep="first")
        print(f"  Results unique vehicles: {len(results)}")

        merged = pd.merge(
            merged,
            results,
            left_on="vehicle_number",
            right_on="NUMBER",
            how="left",
        )
        print(f"  After results merge: {merged.shape}")

        # Convert time columns
        for col in ["FL_TIME", "TOTAL_TIME"]:
            if col in merged.columns:
                merged[col + "_sec"] = merged[col].apply(time_to_seconds)

        del results
        gc.collect()

# Analysis - DEDUPLICATE FIRST
if os.path.exists(paths["analysis"]):
    print("\n[INFO] Merging analysis data...")
    analysis = pd.read_csv(
        paths["analysis"], delimiter=detect_delimiter(paths["analysis"])
    )
    analysis.columns = analysis.columns.str.strip()

    # Find lap and number columns
    analysis_lap_col = None
    analysis_num_col = None

    for col in ["LAP_NUMBER", "Lap_Number", "lap", "LAP"]:
        if col in analysis.columns:
            analysis_lap_col = col
            break

    for col in ["NUMBER", "Number", "vehicle_number"]:
        if col in analysis.columns:
            analysis_num_col = col
            break

    if analysis_lap_col and analysis_num_col:
        # CRITICAL: One row per (NUMBER, LAP_NUMBER)
        analysis = analysis.drop_duplicates(
            subset=[analysis_num_col, analysis_lap_col], keep="first"
        )
        print(f"  Analysis unique (vehicle, lap): {len(analysis)}")

        merged = pd.merge(
            merged,
            analysis,
            left_on=["vehicle_number", "lap"],
            right_on=[analysis_num_col, analysis_lap_col],
            how="left",
            suffixes=("", "_analysis"),
        )
        print(f"  After analysis merge: {merged.shape}")

        # Convert time columns
        for col in ["LAP_TIME", "S1", "S2", "S3", "ELAPSED"]:
            if col in merged.columns:
                merged[col + "_sec"] = merged[col].apply(time_to_seconds)

        del analysis
        gc.collect()

# Best laps - DEDUPLICATE FIRST
if os.path.exists(paths["best_laps"]):
    print("\n[INFO] Merging best laps data...")
    best_laps = pd.read_csv(
        paths["best_laps"], delimiter=detect_delimiter(paths["best_laps"])
    )
    best_laps.columns = best_laps.columns.str.strip()

    if "NUMBER" in best_laps.columns:
        # CRITICAL: One row per NUMBER
        best_laps = best_laps.drop_duplicates(subset=["NUMBER"], keep="first")
        print(f"  Best laps unique vehicles: {len(best_laps)}")

        merged = pd.merge(
            merged,
            best_laps,
            left_on="vehicle_number",
            right_on="NUMBER",
            how="left",
            suffixes=("", "_best"),
        )
        print(f"  After best_laps merge: {merged.shape}")
        del best_laps
        gc.collect()

# Lap start - AGGREGATE FIRST
if os.path.exists(paths["lap_start"]):
    print("\n[INFO] Merging lap start data...")
    lap_start = pd.read_csv(paths["lap_start"])

    if all(k in lap_start.columns for k in ["lap", "vehicle_id"]):
        # AGGREGATE to one row per (lap, vehicle_id)
        agg_dict = {
            col: "first"
            for col in lap_start.columns
            if col not in ["lap", "vehicle_id"]
        }
        lap_start = lap_start.groupby(["lap", "vehicle_id"], as_index=False).agg(
            agg_dict
        )
        print(f"  Lap start aggregated: {len(lap_start)}")

        merged = pd.merge(
            merged,
            lap_start,
            on=["lap", "vehicle_id"],
            how="left",
            suffixes=("", "_start"),
        )
        print(f"  After lap_start merge: {merged.shape}")
        del lap_start
        gc.collect()

# Lap time - AGGREGATE FIRST
if os.path.exists(paths["lap_time"]):
    print("\n[INFO] Merging lap time data...")
    lap_time = pd.read_csv(paths["lap_time"])

    if all(k in lap_time.columns for k in ["lap", "vehicle_id"]):
        # AGGREGATE to one row per (lap, vehicle_id)
        agg_dict = {
            col: "first" for col in lap_time.columns if col not in ["lap", "vehicle_id"]
        }
        lap_time = lap_time.groupby(["lap", "vehicle_id"], as_index=False).agg(agg_dict)
        print(f"  Lap time aggregated: {len(lap_time)}")

        merged = pd.merge(
            merged,
            lap_time,
            on=["lap", "vehicle_id"],
            how="left",
            suffixes=("", "_time"),
        )
        print(f"  After lap_time merge: {merged.shape}")
        del lap_time
        gc.collect()

# Lap end - AGGREGATE FIRST
if os.path.exists(paths["lap_end"]):
    print("\n[INFO] Merging lap end data...")
    lap_end = pd.read_csv(paths["lap_end"])

    if all(k in lap_end.columns for k in ["lap", "vehicle_id"]):
        # AGGREGATE to one row per (lap, vehicle_id)
        agg_dict = {
            col: "first" for col in lap_end.columns if col not in ["lap", "vehicle_id"]
        }
        lap_end = lap_end.groupby(["lap", "vehicle_id"], as_index=False).agg(agg_dict)
        print(f"  Lap end aggregated: {len(lap_end)}")

        merged = pd.merge(
            merged,
            lap_end,
            on=["lap", "vehicle_id"],
            how="left",
            suffixes=("", "_end"),
        )
        print(f"  After lap_end merge: {merged.shape}")
        del lap_end
        gc.collect()

# Final check
final_size = len(merged)
print(
    f"\n[INFO] Size change: {original_size:,} → {final_size:,} ({final_size / original_size:.2f}x)"
)

if final_size > original_size * 1.3:
    print("[WARN] Unexpected growth, deduplicating...")
    merged = merged.drop_duplicates(
        subset=["vehicle_id", "vehicle_number", "lap", "timestamp"]
    )
    print(f"  After dedup: {merged.shape}")

# Save
output_path = "processed_data/merged_r5.csv"
print(f"\n[INFO] Saving to {output_path}...")
merged.to_csv(output_path, index=False)

print(f"\n✅ SUCCESS! Saved → {output_path}")
print(f"Final shape: {merged.shape}")
print(f"Columns ({len(merged.columns)}): {list(merged.columns[:40])} ...")

gc.collect()

