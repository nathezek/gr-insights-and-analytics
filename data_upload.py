import pandas as pd
import streamlit as st


def detect_delimiter(file):
    """Detect CSV delimiter by peeking at first line."""
    file.seek(0)  # Reset file pointer
    line = file.readline().decode("utf-8")
    file.seek(0)
    if ";" in line and "," not in line.split(";")[0]:
        return ";"
    return ","


def load_csv(file_upload):
    """Load a CSV from uploaded file, detecting delimiter."""
    delimiter = detect_delimiter(file_upload)
    file_upload.seek(0)
    return pd.read_csv(file_upload, delimiter=delimiter)


def upload_data():
    """Let user upload main files."""
    st.sidebar.header("Upload Your Data")
    telemetry_file = st.sidebar.file_uploader("Upload Main Telemetry CSV", type="csv")
    results_file = st.sidebar.file_uploader("Upload Race Results CSV (Optional)", type="csv")
    weather_file = st.sidebar.file_uploader("Upload Weather CSV (Optional)", type="csv")

    if telemetry_file:
        telemetry = load_csv(telemetry_file)
        results = load_csv(results_file) if results_file else None
        weather = load_csv(weather_file) if weather_file else None
        return {"telemetry": telemetry, "results": results, "weather": weather}
    else:
        return None