import pandas as pd
from typing import cast
import numpy as np
import os

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


def clean_telemetry_wide(csv_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading telemetry: {csv_path}")

    # Discover available channels
    sample = pd.read_csv(csv_path, nrows=100_000, usecols=["telemetry_name"])  # type: ignore
    available = set(sample["telemetry_name"].dropna().unique())
    chosen = sorted(PREFERRED_CHANNELS & available)

    if not chosen:
        counts = sample["telemetry_name"].value_counts()
        chosen = counts.head(12).index.tolist()
        print(f"[WARN] Preferred channels missing → using top 12: {chosen}")
    else:
        print(f"[INFO] Using channels: {chosen}")

    # Load only needed columns
    usecols_list = [
        "vehicle_id",
        "vehicle_number",
        "lap",
        "timestamp",
        "telemetry_name",
        "telemetry_value",
    ]
    df = pd.read_csv(csv_path, usecols=usecols_list)  # type: ignore

    # Basic cleaning
    df = df[df["lap"].notna() & (df["lap"] != 32768)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["timestamp", "telemetry_name", "telemetry_value"])
    df = df[df["telemetry_name"].isin(chosen)]
    df["telemetry_value"] = pd.to_numeric(df["telemetry_value"], errors="coerce")
    df = df.dropna(subset=["telemetry_value"])
    df = df.drop_duplicates(
        subset=["vehicle_id", "vehicle_number", "lap", "timestamp", "telemetry_name"]
    )

    print(f"[INFO] Clean long format → {len(df):,} rows")

    # Pivot to wide
    wide = df.pivot_table(
        index=["vehicle_id", "vehicle_number", "lap", "timestamp"],
        columns="telemetry_name",
        values="telemetry_value",
        aggfunc="mean",
    ).reset_index()

    # Convert to numeric + total_g
    for col in chosen:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    if {"accx_can", "accy_can"} <= set(wide.columns):
        wide["total_g"] = np.sqrt(wide["accx_can"] ** 2 + wide["accy_can"] ** 2)

    wide = wide.sort_values(["vehicle_id", "lap", "timestamp"]).reset_index(drop=True)
    print(f"[INFO] Final wide telemetry shape: {wide.shape}")
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
    "results": (
        "./raw_training_data/barber/Race_1/03_Provisional Results_Race 1_Anonymized.CSV"
    ),
    "telemetry": ("./raw_training_data/barber/Race_1/R1_barber_telemetry_data.csv"),
    "weather": "./raw_training_data/barber/Race_1/Weather.csv",
    "analysis": (
        "./raw_training_data/barber/Race_1/23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV"
    ),
    "best_laps": (
        "./raw_training_data/barber/Race_1/99_Best 10 Laps By Driver_Race 1_Anonymized.CSV"
    ),
    "lap_start": "./raw_training_data/barber/Race_1/R1_barber_lap_start.csv",
    "lap_time": "./raw_training_data/barber/Race_1/R1_barber_lap_time.csv",
    "lap_end": "./raw_training_data/barber/Race_1/R1_barber_lap_end.csv",
}

print("=== Loading Race 1 data ===")
telemetry = clean_telemetry_wide(paths["telemetry"])

print("\n[INFO] Loading other CSV files...")
results = pd.read_csv(paths["results"], delimiter=detect_delimiter(paths["results"]))
weather = pd.read_csv(paths["weather"], delimiter=detect_delimiter(paths["weather"]))
analysis = pd.read_csv(paths["analysis"], delimiter=detect_delimiter(paths["analysis"]))
best_laps = pd.read_csv(
    paths["best_laps"], delimiter=detect_delimiter(paths["best_laps"])
)
lap_start = pd.read_csv(paths["lap_start"], delimiter=",")
lap_time = pd.read_csv(paths["lap_time"], delimiter=",")
lap_end = pd.read_csv(paths["lap_end"], delimiter=",")

# Debug: Print column names
print("\n=== Debug: Column Names ===")
print(f"Telemetry columns: {list(telemetry.columns)}")
print(f"Analysis columns: {list(analysis.columns)}")
print(f"Results columns: {list(results.columns)}")
print(f"Best_laps columns: {list(best_laps.columns)}")
print(f"Lap_start columns: {list(lap_start.columns)}")
print(f"Lap_time columns: {list(lap_time.columns)}")
print(f"Lap_end columns: {list(lap_end.columns)}")

# Normalize column names to handle case sensitivity and whitespace
for df in (analysis, results, best_laps):
    df.columns = df.columns.str.strip()

# Convert time columns to seconds
for df in (analysis, results):
    for col in ["LAP_TIME", "FL_TIME", "S1", "S2", "S3", "ELAPSED", "TOTAL_TIME"]:
        if col in df.columns:
            df[col + "_sec"] = df[col].apply(time_to_seconds)

merged = telemetry.copy()

# Merge with weather
print("\n[INFO] Merging weather data...")
weather["time_utc"] = pd.to_datetime(
    weather["TIME_UTC_STR"], format="%m/%d/%Y %I:%M:%S %p"
)
merged = pd.merge_asof(
    merged.sort_values("timestamp"),
    weather.sort_values("time_utc"),
    left_on="timestamp",
    right_on="time_utc",
    direction="nearest",
    tolerance=cast(pd.Timedelta, pd.Timedelta("5s")),
)

print(f"[INFO] After weather merge: {merged.shape}")

# Merge with results
print("[INFO] Merging results data...")
if "NUMBER" in results.columns:
    merged = pd.merge(
        merged,
        results,
        left_on="vehicle_number",
        right_on="NUMBER",
        how="left",
        suffixes=("", "_results"),
    )
    print(f"[INFO] After results merge: {merged.shape}")
else:
    print(
        f"[WARN] 'NUMBER' column not found in results. Available: {
            list(results.columns)
        }"
    )

# Merge with analysis - with flexible column detection
print("[INFO] Merging analysis data...")
analysis_lap_col = None
analysis_num_col = None

# Try to find the lap column
for col in ["LAP_NUMBER", "Lap_Number", "lap", "LAP", "Lap"]:
    if col in analysis.columns:
        analysis_lap_col = col
        break

# Try to find the number/vehicle column
for col in ["NUMBER", "Number", "vehicle_number", "VEHICLE_NUMBER"]:
    if col in analysis.columns:
        analysis_num_col = col
        break

if analysis_lap_col and analysis_num_col:
    print(f"[INFO] Using analysis columns: {analysis_lap_col}, {analysis_num_col}")
    merged = pd.merge(
        merged,
        analysis,
        left_on=["lap", "vehicle_number"],
        right_on=[analysis_lap_col, analysis_num_col],
        how="left",
        suffixes=("", "_analysis"),
    )
    print(f"[INFO] After analysis merge: {merged.shape}")
else:
    print("[WARN] Cannot merge analysis. Could not find lap/number columns.")
    print(f"[WARN] Available columns: {list(analysis.columns)}")

# Merge with best_laps
print("[INFO] Merging best laps data...")
if "NUMBER" in best_laps.columns:
    merged = pd.merge(
        merged,
        best_laps,
        left_on="vehicle_number",
        right_on="NUMBER",
        how="left",
        suffixes=("", "_best"),
    )
    print(f"[INFO] After best_laps merge: {merged.shape}")
else:
    print(
        f"[WARN] 'NUMBER' column not found in best_laps. Available: {
            list(best_laps.columns)
        }"
    )

# Merge with lap_start
print("[INFO] Merging lap start data...")
lap_start_keys = ["lap", "vehicle_id"]
if all(k in lap_start.columns for k in lap_start_keys):
    merged = pd.merge(
        merged,
        lap_start,
        on=lap_start_keys,
        how="left",
        suffixes=("", "_start"),
    )
    print(f"[INFO] After lap_start merge: {merged.shape}")
else:
    print(
        f"[WARN] Cannot merge lap_start. Available columns: {list(lap_start.columns)}"
    )

# Merge with lap_time
print("[INFO] Merging lap time data...")
lap_time_keys = ["lap", "vehicle_id"]
if all(k in lap_time.columns for k in lap_time_keys):
    merged = pd.merge(
        merged,
        lap_time,
        on=lap_time_keys,
        how="left",
        suffixes=("", "_time"),
    )
    print(f"[INFO] After lap_time merge: {merged.shape}")
else:
    print(f"[WARN] Cannot merge lap_time. Available columns: {list(lap_time.columns)}")

# Merge with lap_end
print("[INFO] Merging lap end data...")
lap_end_keys = ["lap", "vehicle_id"]
if all(k in lap_end.columns for k in lap_end_keys):
    merged = pd.merge(
        merged,
        lap_end,
        on=lap_end_keys,
        how="left",
        suffixes=("", "_end"),
    )
    print(f"[INFO] After lap_end merge: {merged.shape}")
else:
    print(f"[WARN] Cannot merge lap_end. Available columns: {list(lap_end.columns)}")

# Save output
output_path = "processed_data/merged_r1.csv"
merged.to_csv(output_path, index=False)
print(f"\n✅ SUCCESS! Saved → {output_path}")
print(f"Final shape: {merged.shape}")
print(f"Columns ({len(merged.columns)}): {list(merged.columns[:20])} ...")
