import streamlit as st
import plotly.express as px
from data_upload import upload_data
from data_cleanup import clean_telemetry, clean_results, join_datasets
from ai_magic import get_model, predict_mistakes

st.title("GAZOO Analyst: Race Data Dashboard")

# Step 1: Get data
raw_data = upload_data()
if raw_data:
    # Step 2: Clean it
    cleaned_telemetry = clean_telemetry(raw_data["telemetry"])
    cleaned_results = clean_results(raw_data["results"])
    raw_data["telemetry"] = cleaned_telemetry  # Update dict
    raw_data["results"] = cleaned_results
    all_cleaned = join_datasets(raw_data)

    # Step 3: AI magic
    model = get_model()
    # Example: pick lap 1 (add selector later)
    selected_lap = all_cleaned[all_cleaned["lap"] == 1]
    predicted_data, insights = predict_mistakes(model, selected_lap)

    # Step 4: Show dashboard
    st.header("Track Map")
    fig_map = px.line_mapbox(
        predicted_data,
        lat="VBOX_Lat_Min",
        lon="VBOX_Long_Minutes",
        color="speed",
        zoom=12,
        height=400,
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map)

    st.header("Speed Graph: Real vs Predicted")
    fig_speed = px.line(
        predicted_data,
        x="Laptrigger_lapdist_dls",
        y=["speed", "predicted_speed"],
        labels={"value": "Speed (km/h)", "variable": "Type"},
    )
    st.plotly_chart(fig_speed)

    st.header("AI Insights")
    for insight in insights:
        st.write(insight)

    # Add more: compare mode, export button, etc.
