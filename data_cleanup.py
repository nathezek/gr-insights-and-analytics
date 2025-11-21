import pandas as pd
import numpy as np


def clean_telemetry(df):
    """Clean main telemetry: fix laps, convert times, drop bad rows."""
    df = df[df["lap"] != 32768]  # Remove bad lap numbers
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["total_g"] = np.sqrt(df["accx_can"] ** 2 + df["accy_can"] ** 2)
    df = df.sort_values(["vehicle_id", "lap", "timestamp"])
    numeric_cols = [
        "speed",
        "gear",
        "aps",
        "pbrake_f",
        "pbrake_r",
        "Steering_Angle",
        "accx_can",
        "accy_can",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=numeric_cols)


def clean_results(df):
    """Clean race results: convert lap times to seconds."""

    def time_to_seconds(time_str):
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

    df = df.drop(columns=[col for col in df.columns if col.startswith("*Extra")])
    return df


def join_datasets(data_dict):
    """Join telemetry with weather and results by time/lap."""
    telemetry = data_dict["telemetry"]
    weather = data_dict["weather"]
    results = data_dict["results"]
    # Add handling for other dict keys like 'analysis', 'best_laps', etc.

    # Fix dtypes
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"]).dt.tz_localize(None)
    weather["time_utc"] = pd.to_datetime(weather["TIME_UTC_STR"]).dt.tz_localize(
        None
    )  # Rename for clarity

    # Merge weather by nearest time
    telemetry = pd.merge_asof(
        telemetry.sort_values("timestamp"),
        weather.sort_values("time_utc"),
        left_on="timestamp",
        right_on="time_utc",
        direction="nearest",
    )

    # Merge results by vehicle_number (assuming NUMBER in results)
    telemetry = pd.merge(
        telemetry, results, left_on="vehicle_number", right_on="NUMBER", how="left"
    )

    # Add more merges, e.g., for analysis:
    # if 'analysis' in data_dict:
    #     telemetry = pd.merge(telemetry, data_dict['analysis'], left_on=['lap', 'vehicle_number'], right_on=['LAP_NUMBER', 'NUMBER'], how='left')

    # Similarly for best_laps, lap_start/time/end (merge on lap, vehicle_id)

    return telemetry
