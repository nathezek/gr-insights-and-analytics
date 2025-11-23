import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import gc
from data_upload import upload_data
from data_cleanup import clean_telemetry, clean_results, join_datasets
from ai_magic import get_model, predict_mistakes

st.set_page_config(page_title="GAZOO Analyst", layout="wide")
st.title("🏎️ GAZOO Analyst: AI-Powered Race Analysis")

# Sidebar configuration
st.sidebar.header("⚙️ Settings")

# Memory/Performance settings - REDUCED DEFAULTS TO PREVENT CRASHES
max_telemetry_rows = st.sidebar.select_slider(
    "Max Telemetry Rows (prevents crashes)",
    options=[50000, 100000, 250000, 500000],
    value=100000,  # Reduced from 500000
    help="Larger files will be sampled to this size. Start small if experiencing crashes."
)

# Sample fraction for long-format data
sample_frac = st.sidebar.select_slider(
    "Data Sample % (for large files)",
    options=[0.05, 0.10, 0.20, 0.50],
    value=0.10,
    format_func=lambda x: f"{int(x*100)}%",
    help="For 11M+ row files, use smaller % to prevent crashes. 10% = ~1M rows before pivot."
)

chunk_size = st.sidebar.select_slider(
    "AI Processing Speed",
    options=[5000, 10000, 20000, 50000],
    value=10000,
    help="Larger = faster but uses more memory"
)

# Load model
@st.cache_resource
def load_model_cached():
    return get_model("saved/speed_model_v5.pkl")

try:
    model, metadata, imputer = load_model_cached()
    
    if metadata and isinstance(metadata, dict) and metadata.get('test_r2_score'):
        r2_display = f"{metadata['test_r2_score']:.3f}"
    else:
        r2_display = "N/A"
    
    st.sidebar.success(f"✅ Model v5 (R²={r2_display})")
    
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Data upload
raw_data = upload_data()

if raw_data:
    
    # Show upload info
    with st.expander("📁 Uploaded Files Info"):
        for key, df in raw_data.items():
            if df is not None:
                st.write(f"**{key.title()}**: {len(df):,} rows × {len(df.columns)} columns")
                
                # Show sample columns (helps with debugging)
                if len(df.columns) <= 20:
                    st.write(f"Columns: `{', '.join(list(df.columns))}`")
                else:
                    st.write(f"First 20 columns: `{', '.join(list(df.columns)[:20])}`")
                
                # Check if long-format data
                if 'telemetry_name' in df.columns and 'telemetry_value' in df.columns:
                    st.info(f"ℹ️ {key} is in long-format (will be pivoted)")
                    unique_metrics = df['telemetry_name'].nunique()
                    st.write(f"Contains {unique_metrics} different metrics")
                    
                    # Show sample metrics
                    sample_metrics = df['telemetry_name'].unique()[:20]
                    st.write(f"Sample metrics: `{', '.join(sample_metrics)}`")
                    
                    # Calculate expected size after pivot
                    estimated_after_pivot = int(len(df) * sample_frac / unique_metrics)
                    st.write(f"Estimated rows after {int(sample_frac*100)}% sample + pivot: ~{estimated_after_pivot:,}")
                
                # Warn if very large
                if len(df) > max_telemetry_rows:
                    st.warning(f"⚠️ {key} has {len(df):,} rows. Will be sampled to {max_telemetry_rows:,}")
    
    # Processing with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Clean telemetry (with automatic sampling and pivoting)
        status_text.text("🔧 Cleaning telemetry data...")
        progress_bar.progress(20)
        
        cleaned_telemetry = clean_telemetry(
            raw_data["telemetry"], 
            max_rows=max_telemetry_rows,
            sample_frac=sample_frac  # Pass sample fraction
        )
        
        # Force garbage collection
        gc.collect()
        
        # Clean results
        status_text.text("🔧 Cleaning results...")
        progress_bar.progress(40)
        
        if "results" in raw_data and raw_data["results"] is not None:
            cleaned_results = clean_results(raw_data["results"])
            raw_data["results"] = cleaned_results
        
        raw_data["telemetry"] = cleaned_telemetry
        
        # Join datasets (memory-safe)
        status_text.text("🔗 Merging datasets (memory-safe mode)...")
        progress_bar.progress(60)
        
        all_cleaned = join_datasets(
            raw_data,
            max_merged_rows=max_telemetry_rows
        )
        
        # Force garbage collection
        gc.collect()
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.success(f"✅ Ready to analyze: {len(all_cleaned):,} data points")
        
        # Show memory usage
        memory_mb = all_cleaned.memory_usage(deep=True).sum() / 1024**2
        st.info(f"💾 Current memory usage: {memory_mb:.1f} MB")
        
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        with st.expander("🐛 Debug Info"):
            st.write(f"**Error:** {str(e)}")
            
            # Show what columns we found
            if "telemetry" in raw_data and raw_data["telemetry"] is not None:
                st.write("**Your telemetry columns:**")
                st.code(', '.join(list(raw_data["telemetry"].columns)))
                
                if 'telemetry_name' in raw_data["telemetry"].columns:
                    st.write("**Your telemetry metrics:**")
                    metrics = raw_data["telemetry"]['telemetry_name'].unique()
                    st.code(', '.join(list(metrics)[:50]))
            
            import traceback
            st.code(traceback.format_exc())
        
        st.info("""
        💡 **Troubleshooting:**
        - Try reducing 'Data Sample %' to 5%
        - Try reducing 'Max Telemetry Rows' to 50,000
        - Check that your data includes speed measurements
        - Restart the dashboard if it's frozen
        - Close other applications to free up memory
        """)
        st.stop()
        
    finally:
        progress_bar.empty()
        status_text.empty()
    
    # Validate critical columns (lightweight - no data copying)
    if 'speed' not in all_cleaned.columns:
        st.error("❌ 'speed' column not found in processed data")
        st.info(f"Available columns: {', '.join(list(all_cleaned.columns)[:20])}")
        st.stop()
    
    if 'lap' not in all_cleaned.columns:
        st.error("❌ 'lap' column not found in processed data")
        st.info(f"Available columns: {', '.join(list(all_cleaned.columns)[:20])}")
        st.stop()
    
    # Filters
    st.sidebar.header("🔍 Filters")
    
    # Lap selection - Fix data type
    try:
        all_cleaned['lap'] = pd.to_numeric(all_cleaned['lap'], errors='coerce')
        all_cleaned = all_cleaned.dropna(subset=['lap'])
        all_cleaned['lap'] = all_cleaned['lap'].astype(int)
    except Exception as e:
        st.error(f"❌ Error processing lap column: {e}")
        st.stop()
    
    available_laps = sorted(all_cleaned['lap'].unique())
    
    if len(available_laps) == 0:
        st.error("❌ No valid laps found")
        st.stop()
    
    selected_lap = st.sidebar.selectbox("Select Lap", available_laps)
    
    # Filter by lap FIRST (reduce data early)
    lap_data = all_cleaned[all_cleaned['lap'] == selected_lap].copy()
    
    # Free memory
    del all_cleaned
    gc.collect()
    
    # Vehicle selection (on already filtered data)
    if 'vehicle_number' in lap_data.columns:
        available_vehicles = sorted(lap_data['vehicle_number'].dropna().unique())
        if len(available_vehicles) > 1:
            selected_vehicle = st.sidebar.selectbox("Select Vehicle", available_vehicles)
            lap_data = lap_data[lap_data['vehicle_number'] == selected_vehicle]
        elif len(available_vehicles) == 1:
            st.sidebar.info(f"Vehicle: {available_vehicles[0]}")
    
    # Check if lap_data is empty
    if len(lap_data) == 0:
        st.error(f"❌ No data found for Lap {selected_lap}")
        st.info("💡 Try selecting a different lap")
        if len(available_laps) > 0:
            st.sidebar.write(f"**Available laps:** {available_laps[:10]}")
        st.stop()
    
    # Verify speed column exists in filtered data
    if 'speed' not in lap_data.columns:
        st.error("❌ 'speed' column missing from lap data")
        st.stop()
    
    # Show selection metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Lap", selected_lap)
    col2.metric("Data Points", f"{len(lap_data):,}")
    col3.metric("Memory", f"{lap_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Threshold
    threshold = st.sidebar.slider("Mistake Threshold (km/h)", 5, 20, 10)
    
    # Analysis button
    if st.button("🔮 Run AI Analysis", type="primary"):
        
        # Final check
        if 'speed' not in lap_data.columns:
            st.error("❌ 'speed' column required")
            st.stop()
        
        # Check lap data size
        if len(lap_data) > 100000:
            st.warning(f"⚠️ Large lap ({len(lap_data):,} points). This may take a moment...")
        
        analysis_progress = st.progress(0)
        analysis_status = st.empty()
        
        try:
            analysis_status.text("🤖 AI analyzing lap...")
            analysis_progress.progress(10)
            
            predicted_data, insights = predict_mistakes(
                model, 
                lap_data, 
                imputer,
                threshold_kmh=threshold,
                chunksize=chunk_size
            )
            
            analysis_progress.progress(100)
            analysis_status.text("✅ Complete!")
            
            # Free memory
            gc.collect()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            with st.expander("🐛 Debug"):
                st.write(f"**Error:** {str(e)}")
                st.write(f"**Lap data shape:** {lap_data.shape}")
                st.write(f"**Columns:** {list(lap_data.columns)[:30]}")
                import traceback
                st.code(traceback.format_exc())
            st.stop()
            
        finally:
            analysis_progress.empty()
            analysis_status.empty()
        
        # Display insights
        st.header("🧠 AI Insights")
        for insight in insights:
            st.info(insight)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📍 Track Map", 
            "📊 Speed Analysis", 
            "⚠️ Mistakes", 
            "📈 Details"
        ])
        
        with tab1:
            st.subheader("Track Map")
            
            if 'VBOX_Lat_Min' in predicted_data.columns and 'VBOX_Long_Minutes' in predicted_data.columns:
                map_data = predicted_data.dropna(subset=['VBOX_Lat_Min', 'VBOX_Long_Minutes'])
                
                # Downsample for performance
                if len(map_data) > 5000:
                    map_data = map_data.iloc[::len(map_data)//5000]
                
                if len(map_data) > 0:
                    fig_map = px.scatter_mapbox(
                        map_data,
                        lat="VBOX_Lat_Min",
                        lon="VBOX_Long_Minutes",
                        color="mistake_severity",
                        color_continuous_scale="RdYlGn_r",
                        hover_data=['speed', 'predicted_speed'],
                        zoom=13,
                        height=600,
                    )
                    fig_map.update_layout(mapbox_style="open-street-map")
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("⚠️ No valid GPS coordinates")
            else:
                st.warning("⚠️ GPS data not available")
                st.info("Available GPS columns might have different names. Check your data.")
        
        with tab2:
            st.subheader("Speed Comparison")
            
            # Downsample for plotting
            plot_data = predicted_data.copy()
            if len(plot_data) > 10000:
                st.caption(f"📊 Showing 10,000 of {len(plot_data):,} points for performance")
                plot_data = plot_data.iloc[::len(plot_data)//10000]
            
            x_axis = plot_data['Laptrigger_lapdist_dls'] if 'Laptrigger_lapdist_dls' in plot_data.columns else plot_data.index
            x_label = 'Distance (m)' if 'Laptrigger_lapdist_dls' in plot_data.columns else 'Sample Index'
            
            fig = go.Figure()
            
            # Actual speed
            fig.add_trace(go.Scatter(
                x=x_axis, 
                y=plot_data['speed'], 
                name='Actual Speed', 
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{y:.1f} km/h<extra></extra>'
            ))
            
            # Predicted speed
            fig.add_trace(go.Scatter(
                x=x_axis, 
                y=plot_data['predicted_speed'], 
                name='AI Expected Speed', 
                line=dict(color='#ff7f0e', dash='dash', width=2),
                hovertemplate='%{y:.1f} km/h<extra></extra>'
            ))
            
            # Mistakes
            mistakes = plot_data[plot_data['mistake_type'] != 'OK']
            if len(mistakes) > 0:
                mistake_x = mistakes['Laptrigger_lapdist_dls'] if 'Laptrigger_lapdist_dls' in mistakes.columns else mistakes.index
                fig.add_trace(go.Scatter(
                    x=mistake_x, 
                    y=mistakes['speed'], 
                    mode='markers', 
                    name='Mistakes', 
                    marker=dict(color='#d62728', size=8, symbol='x'),
                    hovertemplate='Error: %{text:.1f} km/h<extra></extra>',
                    text=mistakes['speed_error'].abs()
                ))
            
            fig.update_layout(
                xaxis_title=x_label, 
                yaxis_title='Speed (km/h)', 
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Mistakes Summary")
            
            col1, col2, col3 = st.columns(3)
            mistakes = predicted_data[predicted_data['mistake_type'] != 'OK']
            too_slow = predicted_data[predicted_data['mistake_type'] == 'TOO_SLOW']
            too_fast = predicted_data[predicted_data['mistake_type'] == 'TOO_FAST']
            
            col1.metric("Total Mistakes", f"{len(mistakes):,}", 
                       delta=f"{len(mistakes)/len(predicted_data)*100:.1f}% of lap")
            
            if len(too_slow) > 0:
                avg_slow_loss = abs(too_slow['speed_error'].mean())
                col2.metric("Too Slow", f"{len(too_slow):,}", 
                           delta=f"-{avg_slow_loss:.1f} km/h avg",
                           delta_color="inverse")
            else:
                col2.metric("Too Slow", "0")
            
            if len(too_fast) > 0:
                avg_fast_excess = too_fast['speed_error'].mean()
                col3.metric("Too Fast", f"{len(too_fast):,}", 
                           delta=f"+{avg_fast_excess:.1f} km/h avg",
                           delta_color="inverse")
            else:
                col3.metric("Too Fast", "0")
            
            if len(mistakes) > 0:
                st.subheader("Worst 10 Mistakes")
                worst_cols = ['speed', 'predicted_speed', 'speed_error', 'mistake_type']
                
                # Add distance if available
                if 'Laptrigger_lapdist_dls' in mistakes.columns:
                    worst_cols.insert(0, 'Laptrigger_lapdist_dls')
                
                worst = mistakes.nlargest(10, 'abs_error')[worst_cols].copy()
                
                # Format for display
                if 'Laptrigger_lapdist_dls' in worst.columns:
                    worst['Laptrigger_lapdist_dls'] = worst['Laptrigger_lapdist_dls'].round(1)
                worst['speed'] = worst['speed'].round(1)
                worst['predicted_speed'] = worst['predicted_speed'].round(1)
                worst['speed_error'] = worst['speed_error'].round(1)
                
                st.dataframe(worst, use_container_width=True)
                
                # Mistake distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Mistake Types")
                    mistake_counts = predicted_data['mistake_type'].value_counts()
                    fig_pie = px.pie(
                        values=mistake_counts.values,
                        names=mistake_counts.index,
                        title="Distribution",
                        color_discrete_map={'OK': '#2ecc71', 'TOO_SLOW': '#f39c12', 'TOO_FAST': '#e74c3c'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("Error Distribution")
                    fig_hist = px.histogram(
                        mistakes,
                        x='speed_error',
                        nbins=30,
                        title="Speed Error Distribution",
                        labels={'speed_error': 'Speed Error (km/h)'},
                        color_discrete_sequence=['#e74c3c']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
            else:
                st.success("🎉 Perfect lap! No mistakes detected!")
                st.balloons()
        
        with tab4:
            st.subheader("Export Data")
            
            csv = predicted_data.to_csv(index=False)
            st.download_button(
                "📥 Download Full Analysis (CSV)",
                csv,
                f"lap_{selected_lap}_analysis.csv",
                "text/csv",
                key='download-csv'
            )
            
            st.subheader("Data Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{len(predicted_data):,}")
            col2.metric("Total Columns", f"{len(predicted_data.columns)}")
            memory_size = predicted_data.memory_usage(deep=True).sum() / 1024**2
            col3.metric("Memory", f"{memory_size:.1f} MB")
            
            # Statistics
            st.subheader("Speed Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Actual Speed**")
                speed_stats = predicted_data['speed'].describe()
                st.dataframe(speed_stats, use_container_width=True)
            
            with col2:
                st.write("**Predicted Speed**")
                pred_speed_stats = predicted_data['predicted_speed'].describe()
                st.dataframe(pred_speed_stats, use_container_width=True)
            
            # Error statistics
            st.subheader("Prediction Error Statistics")
            error_stats = predicted_data['speed_error'].describe()
            st.dataframe(error_stats, use_container_width=True)
            
            # Show available columns
            with st.expander("📋 View All Columns"):
                st.write(f"**{len(predicted_data.columns)} columns:**")
                st.write(list(predicted_data.columns))
            
            # Show sample data
            with st.expander("👀 Preview Data (First 100 rows)"):
                st.dataframe(predicted_data.head(100), use_container_width=True)

else:
    st.info("👈 Upload data files to begin!")
    
    st.markdown(f"""
    ### ⚙️ Current Settings:
    - **Max rows**: {max_telemetry_rows:,} (samples large files to prevent crashes)
    - **Sample fraction**: {int(sample_frac*100)}% (for long-format data before pivot)
    - **Processing speed**: {chunk_size:,} rows/chunk
    
    ### 💡 Performance Tips:
    - **For files > 10M rows**: Use 5-10% sample, 50k-100k max rows
    - **For files 1-10M rows**: Use 10-20% sample, 100k-250k max rows
    - **For files < 1M rows**: Use 50-100% sample, 250k-500k max rows
    - **If VSCode crashes**: Reduce sample % to 5% and max rows to 50k
    
    ### 📋 Required Files:
    
    **Telemetry CSV** (Required)
    - Must include speed measurements and lap information
    - Supported formats:
      - **Wide format**: Columns like speed, gear, throttle, etc.
      - **Long format**: Columns telemetry_name + telemetry_value (auto-pivoted)
    
    **Results CSV** (Optional)
    - Race results with driver/team info
    
    **Weather CSV** (Optional)
    - Weather conditions during race
    
    ### 📊 Data Format Examples:
    """)
    
    # Example data format
    with st.expander("📖 Wide Format Example (Standard)"):
        st.code("""
lap,timestamp,vehicle_number,speed,gear,aps,pbrake_f,Steering_Angle
1,2024-01-01 10:00:00,101,120.5,3,75.2,0.0,15.3
1,2024-01-01 10:00:01,101,125.3,4,80.1,0.0,12.7
1,2024-01-01 10:00:02,101,128.7,4,85.5,0.0,8.2
        """, language="csv")
    
    with st.expander("📖 Long Format Example (Auto-Pivoted)"):
        st.code("""
lap,timestamp,vehicle_number,telemetry_name,telemetry_value
1,2024-01-01 10:00:00,101,speed,120.5
1,2024-01-01 10:00:00,101,gear,3
1,2024-01-01 10:00:00,101,aps,75.2
1,2024-01-01 10:00:01,101,speed,125.3
1,2024-01-01 10:00:01,101,gear,4
        """, language="csv")
        st.info("Long-format data is automatically converted to wide format during processing")
    
    with st.expander("🔍 Essential Metrics"):
        st.write("""
        The system looks for these telemetry metrics:
        - **speed** (required) - Vehicle speed
        - **gear** - Current gear
        - **aps** / **throttle** - Throttle position
        - **pbrake_f** / **pbrake_r** - Front/rear brake pressure
        - **Steering_Angle** - Steering wheel angle
        - **accx_can** / **accy_can** - Lateral/longitudinal acceleration
        - **Laptrigger_lapdist_dls** - Distance along lap
        - **VBOX_Lat_Min** / **VBOX_Long_Minutes** - GPS coordinates
        
        Only essential metrics are kept to reduce memory usage.
        """)