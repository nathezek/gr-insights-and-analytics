import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import gc
from data_upload import upload_data
from data_cleanup import clean_telemetry, clean_results, join_datasets
from ai_magic import get_model, predict_mistakes

st.set_page_config(page_title="GAZOO Analyst", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏎️ GAZOO Analyst: AI-Powered Race Analysis")

# Sidebar configuration
st.sidebar.header("⚙️ Settings")

# Memory/Performance settings
max_telemetry_rows = st.sidebar.select_slider(
    "Max Telemetry Rows",
    options=[50000, 100000, 250000, 500000],
    value=100000,
    help="Maximum rows to process"
)

sample_frac = st.sidebar.select_slider(
    "Data Sample %",
    options=[0.05, 0.10, 0.20, 0.50],
    value=0.10,
    format_func=lambda x: f"{int(x*100)}%",
    help="Percentage of data to sample for large files"
)

chunk_size = st.sidebar.select_slider(
    "AI Processing Speed",
    options=[5000, 10000, 20000, 50000],
    value=10000,
    help="Processing chunk size"
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

# ADD: single shared friendly name mapping (available to all tabs)
feature_name_mapping = {
    'speed': 'Speed',
    'gear': 'Gear',
    'total_g': 'Total G-Force',
    'accx_can': 'Lateral Acceleration',
    'Steering_Angle': 'Steering Angle',
    'aps': 'Throttle Position',
    'accy_can': 'Longitudinal Acceleration',
    'pbrake_f': 'Front Brake Pressure',
    'pbrake_r': 'Rear Brake Pressure',
    'Laptrigger_lapdist_dls': 'Lap Distance',
    'acc_angle': 'Acceleration Angle',
    'gear_aps': 'Gear × Throttle',
    'total_brake': 'Total Brake Pressure',
    'brake_balance': 'Brake Balance (F/R)',
    'g_per_gear': 'G-Force per Gear',
    'abs_steering': 'Absolute Steering Angle'
}

# Data upload
raw_data = upload_data()

if raw_data:
    
    # Show upload info
    with st.expander("📁 Uploaded Files Info", expanded=False):
        for key, df in raw_data.items():
            if df is not None:
                st.write(f"**{key.title()}**: {len(df):,} rows × {len(df.columns)} columns")
                
                if 'telemetry_name' in df.columns and 'telemetry_value' in df.columns:
                    st.info(f"ℹ️ {key} is in long-format (will be pivoted)")
                    unique_metrics = df['telemetry_name'].nunique()
                    st.write(f"Contains {unique_metrics} different metrics")
                
                if len(df) > max_telemetry_rows:
                    st.warning(f"⚠️ Will be sampled to {max_telemetry_rows:,} rows")
    
    # Processing with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔧 Cleaning telemetry data...")
        progress_bar.progress(20)
        
        cleaned_telemetry = clean_telemetry(
            raw_data["telemetry"], 
            max_rows=max_telemetry_rows,
            sample_frac=sample_frac
        )
        
        gc.collect()
        
        status_text.text("🔧 Cleaning results...")
        progress_bar.progress(40)
        
        if "results" in raw_data and raw_data["results"] is not None:
            cleaned_results = clean_results(raw_data["results"])
            raw_data["results"] = cleaned_results
        
        raw_data["telemetry"] = cleaned_telemetry
        
        status_text.text("🔗 Merging datasets...")
        progress_bar.progress(60)
        
        all_cleaned = join_datasets(
            raw_data,
            max_merged_rows=max_telemetry_rows
        )
        
        gc.collect()
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.success(f"✅ Ready to analyze: {len(all_cleaned):,} data points")
        
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        with st.expander("🐛 Debug Info"):
            st.write(f"**Error:** {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        st.stop()
        
    finally:
        progress_bar.empty()
        status_text.empty()
    
    # Validate critical columns
    if 'speed' not in all_cleaned.columns:
        st.error("❌ 'speed' column not found")
        st.stop()
    
    if 'lap' not in all_cleaned.columns:
        st.error("❌ 'lap' column not found")
        st.stop()
    
    # Drop rows with NaN in speed or lap (and timestamp if present)
    critical_cols = ['speed', 'lap']
    if 'timestamp' in all_cleaned.columns:
        critical_cols.append('timestamp')
    all_cleaned = all_cleaned.dropna(subset=critical_cols)
    if len(all_cleaned) == 0:
        st.error("❌ No valid telemetry data after cleaning (missing speed/lap/timestamp)")
        st.stop()

    # Filters
    st.sidebar.header("🔍 Filters")
    
    # Lap selection
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
    
    # Filter by lap
    lap_data = all_cleaned[all_cleaned['lap'] == selected_lap].copy()
    
    # Vehicle selection
    if 'vehicle_number' in lap_data.columns:
        available_vehicles = sorted(lap_data['vehicle_number'].dropna().unique())
        if len(available_vehicles) > 1:
            selected_vehicle = st.sidebar.selectbox("Select Vehicle", available_vehicles)
            lap_data = lap_data[lap_data['vehicle_number'] == selected_vehicle]
        elif len(available_vehicles) == 1:
            st.sidebar.info(f"Vehicle: {available_vehicles[0]}")
    
    # Comparison lap (for best vs worst)
    if len(available_laps) > 1:
        comparison_lap = st.sidebar.selectbox(
            "Compare with Lap (optional)",
            options=[None] + available_laps,
            format_func=lambda x: "None" if x is None else str(x)
        )
    else:
        comparison_lap = None
    
    # Free memory
    comparison_lap_data = None
    if comparison_lap is not None and comparison_lap != selected_lap:
        comparison_lap_data = all_cleaned[all_cleaned['lap'] == comparison_lap].copy()
        if 'vehicle_number' in comparison_lap_data.columns and 'vehicle_number' in lap_data.columns:
            vehicle_nums = lap_data['vehicle_number'].dropna().unique()
            if len(vehicle_nums) == 1:
                comparison_lap_data = comparison_lap_data[comparison_lap_data['vehicle_number'] == vehicle_nums[0]]
    
    del all_cleaned
    gc.collect()
    
    # Check if lap_data is empty
    if len(lap_data) == 0:
        st.error(f"❌ No data found for Lap {selected_lap}")
        st.stop()
    
    if 'speed' not in lap_data.columns:
        st.error("❌ 'speed' column missing")
        st.stop()
    
    # Show selection metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lap", selected_lap)
    col2.metric("Data Points", f"{len(lap_data):,}")
    col3.metric("Memory", f"{lap_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Calculate lap time if available
    if 'timestamp' in lap_data.columns:
        lap_duration = (lap_data['timestamp'].max() - lap_data['timestamp'].min()).total_seconds()
        col4.metric("Lap Time", f"{lap_duration:.2f}s")
    else:
        col4.metric("Lap Time", "N/A")
    
    # Threshold
    threshold = st.sidebar.slider("Mistake Threshold (km/h)", 5, 20, 10)
    
    # Analysis button
    if st.button("🔮 Run AI Analysis", type="primary", use_container_width=True):
        
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
            
            gc.collect()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            with st.expander("🐛 Debug"):
                st.write(f"**Error:** {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            st.stop()
            
        finally:
            analysis_progress.empty()
            analysis_status.empty()
        
        # AI Insights at top
        st.header("🧠 AI Insights")
        
        # Display insights in columns
        num_insights = len(insights)
        if num_insights > 0:
            cols_per_row = min(3, num_insights)
            rows_needed = (num_insights + cols_per_row - 1) // cols_per_row
            
            for row in range(rows_needed):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    insight_idx = row * cols_per_row + col_idx
                    if insight_idx < num_insights:
                        with cols[col_idx]:
                            st.info(insights[insight_idx])
        
        # Main Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Telemetry Overview",
            "🎯 AI Analysis", 
            "📍 Sector Analysis",
            "🏁 Lap Comparison",
            "📈 Export & Details"
        ])
        
        # ======================
        # TAB 1: TELEMETRY OVERVIEW
        # ======================
        with tab1:
            st.subheader("Complete Telemetry Analysis")
            
            # Prepare x-axis
            if 'Laptrigger_lapdist_dls' in predicted_data.columns:
                x_axis = predicted_data['Laptrigger_lapdist_dls']
                x_label = 'Distance (m)'
            else:
                x_axis = predicted_data.index
                x_label = 'Sample Index'
            
            # Downsample for plotting
            plot_data = predicted_data.copy()
            if len(plot_data) > 10000:
                st.caption(f"📊 Displaying 10,000 of {len(plot_data):,} points for performance")
                step = len(plot_data) // 10000
                plot_data = plot_data.iloc[::step].copy()
                if 'Laptrigger_lapdist_dls' in plot_data.columns:
                    x_axis = plot_data['Laptrigger_lapdist_dls'].values
                else:
                    x_axis = plot_data.index.values
            
            # Create subplots
            available_channels = []
            if 'speed' in plot_data.columns:
                available_channels.append('speed')
            if 'gear' in plot_data.columns:
                available_channels.append('gear')
            if 'aps' in plot_data.columns:
                available_channels.append('aps')
            if 'pbrake_f' in plot_data.columns or 'pbrake_r' in plot_data.columns:
                available_channels.append('brake')
            if 'Steering_Angle' in plot_data.columns:
                available_channels.append('steering')

            # Use friendly names for subplot titles
            subplot_titles = []
            for ch in available_channels:
                if ch == 'brake':
                    subplot_titles.append('Brake Pressure')
                elif ch == 'steering':
                    subplot_titles.append(feature_name_mapping.get('Steering_Angle', 'Steering Angle'))
                else:
                    subplot_titles.append(feature_name_mapping.get(ch, ch).title())

            num_plots = len(available_channels)

            if num_plots > 0:
                fig = make_subplots(
                    rows=num_plots, 
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=subplot_titles,
                    row_heights=[1] * num_plots
                )
                row = 1

                # Speed
                if 'speed' in available_channels:
                    fig.add_trace(
                        go.Scatter(x=x_axis, y=plot_data['speed'].values, 
                                   name=feature_name_mapping.get('speed', 'Speed'), line=dict(color='#1f77b4', width=2)),
                        row=row, col=1
                    )
                    fig.update_yaxes(title_text="Speed (km/h)", row=row, col=1)
                    row += 1

                # Gear
                if 'gear' in available_channels:
                    fig.add_trace(
                        go.Scatter(x=x_axis, y=plot_data['gear'].values, 
                                   name=feature_name_mapping.get('gear', 'Gear'), line=dict(color='#ff7f0e', width=2),
                                   mode='lines+markers', marker=dict(size=3)),
                        row=row, col=1
                    )
                    fig.update_yaxes(title_text=feature_name_mapping.get('gear', 'Gear'), row=row, col=1)
                    row += 1

                # Throttle
                if 'aps' in available_channels:
                    fig.add_trace(
                        go.Scatter(x=x_axis, y=plot_data['aps'].values, 
                                   name=feature_name_mapping.get('aps', 'Throttle'), line=dict(color='#2ca02c', width=2)),
                        row=row, col=1
                    )
                    fig.update_yaxes(title_text=feature_name_mapping.get('aps', 'Throttle Position') + " (%)", row=row, col=1)
                    row += 1

                # Brakes
                if 'brake' in available_channels:
                    if 'pbrake_f' in plot_data.columns:
                        fig.add_trace(
                            go.Scatter(x=x_axis, y=plot_data['pbrake_f'].values, 
                                       name=feature_name_mapping.get('pbrake_f', 'Brake Front'), line=dict(color='#d62728', width=2)),
                            row=row, col=1
                        )
                    if 'pbrake_r' in plot_data.columns:
                        fig.add_trace(
                            go.Scatter(x=x_axis, y=plot_data['pbrake_r'].values, 
                                       name=feature_name_mapping.get('pbrake_r', 'Brake Rear'), line=dict(color='#ff9896', width=2, dash='dash')),
                            row=row, col=1
                        )
                    fig.update_yaxes(title_text="Brake Pressure", row=row, col=1)
                    row += 1

                # Steering
                if 'steering' in available_channels:
                    fig.add_trace(
                        go.Scatter(x=x_axis, y=plot_data['Steering_Angle'].values, 
                                   name=feature_name_mapping.get('Steering_Angle', 'Steering Angle'), line=dict(color='#9467bd', width=2)),
                        row=row, col=1
                    )
                    fig.update_yaxes(title_text=feature_name_mapping.get('Steering_Angle', 'Steering Angle') + " (°)", row=row, col=1)
                    row += 1

                fig.update_xaxes(title_text=x_label, row=num_plots, col=1)
                fig.update_layout(
                    height=250 * num_plots,
                    showlegend=True,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No telemetry channels available for visualization")
        
        # ======================
        # TAB 2: AI ANALYSIS
        # ======================
        with tab2:
            st.subheader("AI Prediction Analysis")
            
            # Prepare data
            plot_data = predicted_data.copy()
            if len(plot_data) > 10000:
                step = len(plot_data) // 10000
                plot_data = plot_data.iloc[::step].copy()
            
            if 'Laptrigger_lapdist_dls' in plot_data.columns:
                x_axis = plot_data['Laptrigger_lapdist_dls'].values
                x_label = 'Distance (m)'
            else:
                x_axis = plot_data.index.values
                x_label = 'Sample Index'
            
            # Speed comparison with error bands
            fig = go.Figure()
            
            # Actual speed
            fig.add_trace(go.Scatter(
                x=x_axis, 
                y=plot_data['speed'].values, 
                name='Actual Speed',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='%{y:.1f} km/h<extra></extra>'
            ))
            
            # Predicted speed
            fig.add_trace(go.Scatter(
                x=x_axis, 
                y=plot_data['predicted_speed'].values, 
                name='AI Expected Speed',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                hovertemplate='%{y:.1f} km/h<extra></extra>'
            ))
            
            # Error threshold bands
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['predicted_speed'].values + threshold,
                name='Upper Threshold',
                line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dot'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['predicted_speed'].values - threshold,
                name='Lower Threshold',
                line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                showlegend=False
            ))
            
            # Mistakes
            mistakes = plot_data[plot_data['mistake_type'] != 'OK']
            if len(mistakes) > 0:
                mistake_x = mistakes['Laptrigger_lapdist_dls'].values if 'Laptrigger_lapdist_dls' in mistakes.columns else mistakes.index.values
                fig.add_trace(go.Scatter(
                    x=mistake_x,
                    y=mistakes['speed'].values,
                    mode='markers',
                    name='Mistakes',
                    marker=dict(
                        color=mistakes['speed_error'].values,
                        size=10,
                        colorscale='RdYlGn',
                        reversescale=True,
                        showscale=True,
                        colorbar=dict(title="Error (km/h)"),
                        line=dict(color='black', width=1)
                    ),
                    hovertemplate='Error: %{marker.color:.1f} km/h<extra></extra>'
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
            
            # Mistake Summary
            st.subheader("Mistake Summary")
            
            col1, col2, col3 = st.columns(3)
            mistakes_all = predicted_data[predicted_data['mistake_type'] != 'OK']
            too_slow = predicted_data[predicted_data['mistake_type'] == 'TOO_SLOW']
            too_fast = predicted_data[predicted_data['mistake_type'] == 'TOO_FAST']
            
            col1.metric("Total Mistakes", f"{len(mistakes_all):,}", 
                       delta=f"{len(mistakes_all)/len(predicted_data)*100:.1f}% of lap")
            
            if len(too_slow) > 0:
                avg_slow_loss = abs(too_slow['speed_error'].mean())
                col2.metric("Too Slow Points", f"{len(too_slow):,}", 
                           delta=f"-{avg_slow_loss:.1f} km/h avg",
                           delta_color="inverse")
            else:
                col2.metric("Too Slow Points", "0")
            
            if len(too_fast) > 0:
                avg_fast_excess = too_fast['speed_error'].mean()
                col3.metric("Too Fast Points", f"{len(too_fast):,}", 
                           delta=f"+{avg_fast_excess:.1f} km/h avg",
                           delta_color="inverse")
            else:
                col3.metric("Too Fast Points", "0")
            
            # Top 10 Mistakes Table
            if len(mistakes_all) > 0:
                st.subheader("Top 10 Worst Mistakes")
                
                worst_cols = ['speed', 'predicted_speed', 'speed_error', 'mistake_type']
                if 'Laptrigger_lapdist_dls' in mistakes_all.columns:
                    worst_cols.insert(0, 'Laptrigger_lapdist_dls')
                
                worst = mistakes_all.nlargest(10, 'abs_error')[worst_cols].copy()
                
                # Format columns
                for col in ['speed', 'predicted_speed', 'speed_error']:
                    if col in worst.columns:
                        worst[col] = worst[col].round(1)
                if 'Laptrigger_lapdist_dls' in worst.columns:
                    worst['Laptrigger_lapdist_dls'] = worst['Laptrigger_lapdist_dls'].round(1)
                
                st.dataframe(worst, use_container_width=True)
            
            # Feature Importance with REAL names
            st.subheader("Top Contributing Features")
            
            try:
                if hasattr(model, 'feature_importances_') and metadata is not None:
                    importances = model.feature_importances_
                    
                    # Try to get feature names from metadata
                    feature_names = metadata.get('features', [])
                    
                    # If no feature names in metadata, reconstruct them
                    if len(feature_names) != len(importances):
                        # Use the standard feature list
                        feature_names = [
                            'gear', 'total_g', 'accx_can', 'Steering_Angle', 'aps', 
                            'accy_can', 'pbrake_f', 'pbrake_r', 'Laptrigger_lapdist_dls',
                            'acc_angle', 'gear_aps', 'total_brake', 'brake_balance', 
                            'g_per_gear', 'abs_steering'
                        ][:len(importances)]  # Trim to actual length
                    
                    if len(feature_names) == len(importances) and len(feature_names) > 0:
                        # Map to friendly names
                        friendly_names = [feature_name_mapping.get(f, f) for f in feature_names]
                        
                        feature_importance = pd.DataFrame({
                            'Feature': friendly_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        fig_importance = px.bar(
                            feature_importance,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Most Important Telemetry Features',
                            color='Importance',
                            color_continuous_scale='Viridis',
                            labels={'Importance': 'Importance Score', 'Feature': 'Telemetry Feature'}
                        )
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("ℹ️ Feature names not available in model metadata")
                        
                        # Show importances without names
                        if len(importances) > 0:
                            feature_importance = pd.DataFrame({
                                'Feature': [f'Feature {i}' for i in range(len(importances))],
                                'Importance': importances
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            fig_importance = px.bar(
                                feature_importance,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 10 Features by Importance',
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            fig_importance.update_layout(height=400)
                            st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("ℹ️ Model does not support feature importance extraction")
            except Exception as e:
                st.info(f"ℹ️ Feature importance not available: {str(e)}")
        
        # ======================
        # TAB 3: SECTOR ANALYSIS
        # ======================
        with tab3:
            st.subheader("Sector-by-Sector Analysis")
            
            # Divide lap into sectors
            num_sectors = st.slider("Number of Sectors", 3, 10, 5)
            
            if 'Laptrigger_lapdist_dls' in predicted_data.columns:
                # Create a copy for sector analysis
                sector_data = predicted_data.copy()
                
                # Remove NaN values from distance column
                sector_data = sector_data.dropna(subset=['Laptrigger_lapdist_dls'])
                
                if len(sector_data) > 0:
                    max_distance = sector_data['Laptrigger_lapdist_dls'].max()
                    sector_length = max_distance / num_sectors
                    
                    # Calculate sector with proper NaN handling
                    sector_data['sector'] = (sector_data['Laptrigger_lapdist_dls'] // sector_length).fillna(0).astype(int)
                    sector_data['sector'] = sector_data['sector'].clip(0, num_sectors - 1)
                    
                    # Calculate sector statistics
                    sector_stats = sector_data.groupby('sector').agg({
                        'speed_error': ['mean', 'std', 'min', 'max'],
                        'abs_error': 'mean',
                        'mistake_type': lambda x: (x != 'OK').sum()
                    }).round(2)
                    
                    sector_stats.columns = ['Avg Error', 'Std Error', 'Min Error', 'Max Error', 'Avg Abs Error', 'Mistakes']
                    sector_stats.index = [f'Sector {i+1}' for i in sector_stats.index]
                    
                    # Display sector table
                    # Rename columns for friendly names
                    sector_stats = sector_stats.rename(columns={
                        'Avg Error': 'Avg Speed Error (km/h)',
                        'Std Error': 'Std Speed Error (km/h)',
                        'Min Error': 'Min Speed Error (km/h)',
                        'Max Error': 'Max Speed Error (km/h)',
                        'Avg Abs Error': 'Avg Absolute Error (km/h)',
                        'Mistakes': 'Mistake Count'
                    })
                    st.dataframe(sector_stats, use_container_width=True)
                    
                    # Sector visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Average error by sector
                        fig_sector_error = px.bar(
                            x=sector_stats.index,
                            y=sector_stats['Avg Speed Error (km/h)'],
                            title='Average Speed Error by Sector',
                            labels={'x': 'Sector', 'y': 'Avg Speed Error (km/h)'},
                            color=sector_stats['Avg Speed Error (km/h)'],
                            color_continuous_scale='RdYlGn',
                            color_continuous_midpoint=0
                        )
                        st.plotly_chart(fig_sector_error, use_container_width=True)
                    
                    with col2:
                        # Mistakes by sector
                        fig_sector_mistakes = px.bar(
                            x=sector_stats.index,
                            y=sector_stats['Mistake Count'],
                            title='Mistakes by Sector',
                            labels={'x': 'Sector', 'y': 'Number of Mistakes'},
                            color=sector_stats['Mistake Count'],
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_sector_mistakes, use_container_width=True)
                    
                    # Sector map
                    st.subheader("Track Map by Sector")
                    if 'VBOX_Lat_Min' in sector_data.columns and 'VBOX_Long_Minutes' in sector_data.columns:
                        map_data = sector_data.dropna(subset=['VBOX_Lat_Min', 'VBOX_Long_Minutes'])
                        
                        if len(map_data) > 5000:
                            map_data = map_data.iloc[::len(map_data)//5000]
                        
                        if len(map_data) > 0:
                            fig_map = px.scatter_mapbox(
                                map_data,
                                lat="VBOX_Lat_Min",
                                lon="VBOX_Long_Minutes",
                                color="sector",
                                hover_data=['speed', 'predicted_speed', 'speed_error'],
                                zoom=13,
                                height=600,
                                color_continuous_scale='Viridis'
                            )
                            fig_map.update_layout(mapbox_style="open-street-map")
                            st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("⚠️ No valid distance data after removing NaN values")
            else:
                st.warning("⚠️ Distance data not available for sector analysis")
        
        # ======================
        # TAB 4: LAP COMPARISON
        # ======================
        with tab4:
            st.subheader("Lap Comparison Analysis")
            
            if comparison_lap_data is not None and len(comparison_lap_data) > 0:
                
                # Align data by distance or index
                if 'Laptrigger_lapdist_dls' in predicted_data.columns and 'Laptrigger_lapdist_dls' in comparison_lap_data.columns:
                    
                    # Downsample both
                    if len(predicted_data) > 5000:
                        data1 = predicted_data.iloc[::len(predicted_data)//5000].copy()
                    else:
                        data1 = predicted_data.copy()
                    
                    if len(comparison_lap_data) > 5000:
                        data2 = comparison_lap_data.iloc[::len(comparison_lap_data)//5000].copy()
                    else:
                        data2 = comparison_lap_data.copy()
                    
                    # Remove NaN from distance
                    data1 = data1.dropna(subset=['Laptrigger_lapdist_dls', 'speed'])
                    data2 = data2.dropna(subset=['Laptrigger_lapdist_dls', 'speed'])
                    
                    if len(data1) > 0 and len(data2) > 0:
                        # Create comparison plot
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            subplot_titles=('Speed Comparison', 'Delta'),
                            vertical_spacing=0.1
                        )
                        
                        # Speed traces
                        fig.add_trace(
                            go.Scatter(
                                x=data1['Laptrigger_lapdist_dls'].values,
                                y=data1['speed'].values,
                                name=f'Lap {selected_lap}',
                                line=dict(color='#1f77b4', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=data2['Laptrigger_lapdist_dls'].values,
                                y=data2['speed'].values,
                                name=f'Lap {comparison_lap}',
                                line=dict(color='#ff7f0e', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        # Calculate delta (interpolate for alignment)
                        try:
                            from scipy import interpolate
                            
                            f1 = interpolate.interp1d(
                                data1['Laptrigger_lapdist_dls'], 
                                data1['speed'], 
                                bounds_error=False, 
                                fill_value='extrapolate'
                            )
                            f2 = interpolate.interp1d(
                                data2['Laptrigger_lapdist_dls'], 
                                data2['speed'], 
                                bounds_error=False, 
                                fill_value='extrapolate'
                            )
                            
                            common_x = np.linspace(
                                max(data1['Laptrigger_lapdist_dls'].min(), data2['Laptrigger_lapdist_dls'].min()),
                                min(data1['Laptrigger_lapdist_dls'].max(), data2['Laptrigger_lapdist_dls'].max()),
                                1000
                            )
                            
                            delta = f1(common_x) - f2(common_x)
                            
                            # Delta trace
                            fig.add_trace(
                                go.Scatter(
                                    x=common_x,
                                    y=delta,
                                    name='Delta',
                                    fill='tozeroy',
                                    line=dict(color='green', width=2),
                                    fillcolor='rgba(0,255,0,0.2)'
                                ),
                                row=2, col=1
                            )
                            
                            # Add zero line
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                            
                            # Update axes
                            fig.update_xaxes(title_text="Distance (m)", row=2, col=1)
                            fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
                            fig.update_yaxes(title_text="Delta (km/h)", row=2, col=1)
                            
                            fig.update_layout(height=700, hovermode='x unified')
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            
                            avg_delta = np.mean(delta)
                            max_advantage = np.max(delta)
                            max_disadvantage = np.min(delta)
                            
                            col1.metric("Average Delta", f"{avg_delta:.2f} km/h")
                            col2.metric("Max Advantage", f"{max_advantage:.2f} km/h")
                            col3.metric("Max Disadvantage", f"{max_disadvantage:.2f} km/h")
                            
                        except Exception as e:
                            st.error(f"Error calculating delta: {e}")
                            st.info("Showing speed comparison only")
                            
                            fig.update_xaxes(title_text="Distance (m)", row=1, col=1)
                            fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
                            fig.update_layout(height=400, hovermode='x unified')
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ No valid data after removing NaN values")
                    
                else:
                    st.warning("⚠️ Distance data not available for comparison")
            
            else:
                st.info("ℹ️ Select a comparison lap in the sidebar to enable lap comparison")
        
        # ======================
        # TAB 5: EXPORT & DETAILS
        # ======================
        with tab5:
            st.subheader("Export Data")
            
            csv = predicted_data.to_csv(index=False)
            st.download_button(
                "📥 Download Full Analysis (CSV)",
                csv,
                f"lap_{selected_lap}_analysis.csv",
                "text/csv",
                key='download-csv',
                use_container_width=True
            )
            
            st.subheader("Data Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{len(predicted_data):,}")
            col2.metric("Total Columns", f"{len(predicted_data.columns)}")
            memory_size = predicted_data.memory_usage(deep=True).sum() / 1024**2
            col3.metric("Memory", f"{memory_size:.1f} MB")
            
            # Statistics
            st.subheader("Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Speed Statistics**")
                speed_stats = predicted_data[['speed', 'predicted_speed']].describe()
                st.dataframe(speed_stats, use_container_width=True)
            
            with col2:
                st.write("**Error Statistics**")
                error_stats = predicted_data[['speed_error', 'abs_error']].describe()
                st.dataframe(error_stats, use_container_width=True)
            
            # Available columns
            with st.expander("📋 View All Columns"):
                st.write(f"**{len(predicted_data.columns)} columns:**")
                st.write(list(predicted_data.columns))
            
            # Sample data
            with st.expander("👀 Preview Data"):
                st.dataframe(predicted_data.head(100), use_container_width=True)

else:
    # Welcome screen
    st.info("👈 Upload data files to begin!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ⚙️ Current Settings:
        - **Max rows**: {0:,}
        - **Sample fraction**: {1}%
        - **Processing speed**: {2:,} rows/chunk
        
        ### 💡 Performance Tips:
        - **Files > 10M rows**: 5-10% sample, 50k-100k max
        - **Files 1-10M rows**: 10-20% sample, 100k-250k max
        - **Files < 1M rows**: 50-100% sample, 250k-500k max
        """.format(max_telemetry_rows, int(sample_frac*100), chunk_size))
    
    with col2:
        st.markdown("""
        ### 📋 Required Files:
        
        **Telemetry CSV** (Required)
        - Speed, lap, and other metrics
        - Wide or long format supported
        
        **Results CSV** (Optional)
        - Race results and standings
        
        **Weather CSV** (Optional)
        - Weather conditions
        
        ### 📊 Features:
        - ✅ Complete telemetry visualization
        - ✅ AI-powered mistake detection
        - ✅ Sector-by-sector analysis
        - ✅ Lap comparison
        - ✅ Feature importance analysis
        """)