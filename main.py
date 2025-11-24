from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from data_cleanup import clean_telemetry
from ai_magic import get_model, predict_mistakes
from csv_merger import merge_lap_csvs, validate_lap_csvs

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None
imputer = None
metadata = None

# Data directory
DATA_DIR = Path("data_processed")
DATA_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global model, imputer, metadata
    try:
        # Adjust path if needed (now in root)
        model, metadata, imputer = get_model("saved/speed_model_v5.pkl")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        print("   Server will run without AI predictions")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Optimize: Read directly from file object instead of loading into memory
        df = pd.read_csv(file.file)
        
        # Clean data (Limit to 200k rows for AI processing)
        cleaned_df = clean_telemetry(df, max_rows=200000)
        
        # Predict
        if model:
            predicted_df, insights = predict_mistakes(model, cleaned_df, imputer)
            
            # Filter for valid laps (remove incomplete/short laps)
            if 'lap' in predicted_df.columns and 'timestamp' in predicted_df.columns:
                # Calculate lap times
                lap_times = predicted_df.groupby('lap')['timestamp'].apply(lambda x: x.max() - x.min())
                
                # Filter out incomplete laps (too short, e.g., < 30s)
                valid_laps = lap_times[lap_times.dt.total_seconds() > 30].index
                
                if not valid_laps.empty:
                    print(f"Retaining {len(valid_laps)} valid laps: {list(valid_laps)}")
                    predicted_df = predicted_df[predicted_df['lap'].isin(valid_laps)].copy()
            
            # Normalize lap distance to start from 0 for each lap
            if 'lap' in predicted_df.columns and 'Laptrigger_lapdist_dls' in predicted_df.columns:
                def normalize_distance(group):
                    # Don't modify the lap column, only the distance
                    result = group.copy()
                    result['Laptrigger_lapdist_dls'] = result['Laptrigger_lapdist_dls'] - result['Laptrigger_lapdist_dls'].min()
                    return result
                
                predicted_df = predicted_df.groupby('lap', group_keys=False).apply(normalize_distance)
            
            # Downsample for frontend visualization (Limit to 8k points)
            # Reduced to 8k to stay under localStorage limit (~5MB) while supporting 2-3 laps
            MAX_FRONTEND_POINTS = 8000
            if len(predicted_df) > MAX_FRONTEND_POINTS:
                # Use uniform sampling per lap to maintain even distribution
                if 'lap' in predicted_df.columns:
                    sampled_laps = []
                    for lap_num in predicted_df['lap'].unique():
                        lap_data = predicted_df[predicted_df['lap'] == lap_num].sort_values('Laptrigger_lapdist_dls')
                        # Calculate points per lap proportionally
                        lap_points = int((len(lap_data) / len(predicted_df)) * MAX_FRONTEND_POINTS)
                        if lap_points > 0 and len(lap_data) > lap_points:
                            step = len(lap_data) // lap_points
                            sampled_laps.append(lap_data.iloc[::step].iloc[:lap_points])
                        else:
                            sampled_laps.append(lap_data)
                    predicted_df = pd.concat(sampled_laps, ignore_index=True)
                else:
                    predicted_df = predicted_df.sort_values('Laptrigger_lapdist_dls')
                    step = len(predicted_df) // MAX_FRONTEND_POINTS
                    predicted_df = predicted_df.iloc[::step].iloc[:MAX_FRONTEND_POINTS]
            
            # Convert to JSON-friendly format (Handle NaN/Inf)
            # Use pandas to_json which handles NaNs correctly (converts to null)
            import json
            result = json.loads(predicted_df.to_json(orient="records"))
            
            # Calculate lap metadata
            lap_metadata = []
            if 'lap' in predicted_df.columns:
                for lap_num in sorted(predicted_df['lap'].unique()):
                    lap_data = predicted_df[predicted_df['lap'] == lap_num]
                    
                    metadata = {
                        'lap': int(lap_num),
                        'points': len(lap_data)
                    }
                    
                    # Add timing info if available
                    if 'timestamp' in lap_data.columns:
                        start_time = lap_data['timestamp'].min()
                        end_time = lap_data['timestamp'].max()
                        duration = (end_time - start_time).total_seconds()
                        
                        metadata['start_time'] = start_time.isoformat() if pd.notna(start_time) else None
                        metadata['end_time'] = end_time.isoformat() if pd.notna(end_time) else None
                        metadata['duration_seconds'] = float(duration) if pd.notna(duration) else None
                    
                    # Add distance info if available
                    if 'Laptrigger_lapdist_dls' in lap_data.columns:
                        max_distance = lap_data['Laptrigger_lapdist_dls'].max()
                        metadata['max_distance_m'] = float(max_distance) if pd.notna(max_distance) else None
                    
                    lap_metadata.append(metadata)
            
            return {
                "data": result,
                "insights": insights,
                "laps": lap_metadata,
                "metadata": {
                    "rows": len(predicted_df),
                    "columns": list(predicted_df.columns)
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-session")
async def upload_session(
    lap_start: UploadFile = File(...),
    lap_end: UploadFile = File(...),
    lap_time: UploadFile = File(...),
    telemetry: UploadFile = File(...)
):
    """
    Upload a complete session with 4 CSV files.
    Processes data and stores each lap separately.
    """
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Processing new session: {session_id}")
        
        # Read all CSV files
        files = {
            'lap_start': pd.read_csv(lap_start.file),
            'lap_end': pd.read_csv(lap_end.file),
            'lap_time': pd.read_csv(lap_time.file),
            'telemetry': pd.read_csv(telemetry.file)
        }
        
        # Validate files
        is_valid, error_msg = validate_lap_csvs(files)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Merge CSVs
        merged_df = merge_lap_csvs(
            files['lap_start'],
            files['lap_end'],
            files['lap_time'],
            files['telemetry']
        )
        
        # Clean data
        cleaned_df = clean_telemetry(merged_df, max_rows=500000)
        
        # Predict mistakes if model is available
        if model:
            predicted_df, insights = predict_mistakes(model, cleaned_df, imputer)
        else:
            predicted_df = cleaned_df
            insights = []
        
        # Normalize lap distance to start from 0 for each lap
        if 'lap' in predicted_df.columns and 'Laptrigger_lapdist_dls' in predicted_df.columns:
            print("[INFO] Normalizing lap distances...")
            def normalize_distance(group):
                result = group.copy()
                result['Laptrigger_lapdist_dls'] = result['Laptrigger_lapdist_dls'] - result['Laptrigger_lapdist_dls'].min()
                return result
            
            predicted_df = predicted_df.groupby('lap', group_keys=False).apply(normalize_distance)
        
        # Get unique laps
        available_laps = sorted(predicted_df['lap'].unique())
        
        # Store each lap separately as parquet
        lap_metadata = []
        for lap_num in available_laps:
            lap_data = predicted_df[predicted_df['lap'] == lap_num].copy()
            
            # Save lap data
            lap_file = session_dir / f"lap_{lap_num}.parquet"
            lap_data.to_parquet(lap_file, index=False)
            
            # Calculate lap metadata
            metadata = {
                'lap': int(lap_num),
                'points': len(lap_data)
            }
            
            if 'timestamp' in lap_data.columns:
                start_time = lap_data['timestamp'].min()
                end_time = lap_data['timestamp'].max()
                duration = (end_time - start_time).total_seconds()
                
                metadata['start_time'] = start_time.isoformat() if pd.notna(start_time) else None
                metadata['end_time'] = end_time.isoformat() if pd.notna(end_time) else None
                metadata['duration_seconds'] = float(duration) if pd.notna(duration) else None
            
            if 'Laptrigger_lapdist_dls' in lap_data.columns:
                max_distance = lap_data['Laptrigger_lapdist_dls'].max()
                metadata['max_distance_m'] = float(max_distance) if pd.notna(max_distance) else None
            
            lap_metadata.append(metadata)
        
        # Save session metadata
        session_metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'laps': lap_metadata,
            'insights': insights,
            'total_laps': len(available_laps),
            'total_points': len(predicted_df)
        }
        
        metadata_file = session_dir / "session_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(session_metadata, f, indent=2)
        
        print(f"[INFO] ✅ Session {session_id} saved with {len(available_laps)} laps")
        
        return {
            "session_id": session_id,
            "laps": lap_metadata,
            "insights": insights,
            "total_laps": len(available_laps)
        }
        
    except Exception as e:
        print(f"[ERROR] Session upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/laps")
async def get_session_laps(session_id: str):
    """Get list of available laps for a session"""
    try:
        session_dir = DATA_DIR / session_id
        metadata_file = session_dir / "session_metadata.json"
        
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return {
            "session_id": session_id,
            "laps": metadata['laps'],
            "insights": metadata.get('insights', []),
            "total_laps": metadata['total_laps']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/lap/{lap_number}")
async def get_lap_data(session_id: str, lap_number: int):
    """Get telemetry data for a specific lap"""
    try:
        session_dir = DATA_DIR / session_id
        lap_file = session_dir / f"lap_{lap_number}.parquet"
        
        if not lap_file.exists():
            raise HTTPException(status_code=404, detail=f"Lap {lap_number} not found in session {session_id}")
        
        # Read lap data
        lap_df = pd.read_parquet(lap_file)
        
        # Convert to JSON-friendly format
        result = json.loads(lap_df.to_json(orient="records"))
        
        return {
            "session_id": session_id,
            "lap": lap_number,
            "data": result,
            "points": len(result)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/lap/{lap_number}/mistake-analysis")
async def get_mistake_analysis(session_id: str, lap_number: int):
    """Get mistake analysis data including feature importance for a specific lap"""
    try:
        session_dir = DATA_DIR / session_id
        lap_file = session_dir / f"lap_{lap_number}.parquet"
        
        if not lap_file.exists():
            raise HTTPException(status_code=404, detail=f"Lap {lap_number} not found in session {session_id}")
        
        # Read lap data
        lap_df = pd.read_parquet(lap_file)
        
        # Check if mistake analysis data exists
        required_cols = ['predicted_speed', 'speed_error', 'mistake_type']
        has_mistake_data = all(col in lap_df.columns for col in required_cols)
        
        if not has_mistake_data:
            raise HTTPException(
                status_code=400, 
                detail="Mistake analysis data not available for this lap. Please re-upload the session."
            )
        
        # Get feature importance from model
        feature_importance = []
        if model and hasattr(model, 'feature_importances_') and metadata:
            features = metadata.get('features', [])
            if features:
                importances = model.feature_importances_
                feature_importance = [
                    {"feature": feat, "importance": float(imp)}
                    for feat, imp in zip(features, importances)
                ]
                # Sort by importance descending
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Prepare mistake analysis data
        # Filter to only rows with mistakes for visualization
        mistakes_df = lap_df[lap_df['mistake_type'] != 'OK'].copy() if 'mistake_type' in lap_df.columns else pd.DataFrame()
        
        # Convert mistake points to JSON-friendly format
        mistake_points = []
        if not mistakes_df.empty:
            mistake_points = json.loads(
                mistakes_df[['Laptrigger_lapdist_dls', 'speed', 'predicted_speed', 'speed_error', 'mistake_type', 'mistake_severity']]
                .to_json(orient="records")
            )
        
        # Also send full lap data for continuous speed lines
        full_lap_data = []
        if 'Laptrigger_lapdist_dls' in lap_df.columns and 'speed' in lap_df.columns and 'predicted_speed' in lap_df.columns:
            # Downsample for frontend (max 2000 points for smooth lines)
            if len(lap_df) > 2000:
                step = len(lap_df) // 2000
                sampled_df = lap_df.iloc[::step].copy()
            else:
                sampled_df = lap_df.copy()
            
            full_lap_data = json.loads(
                sampled_df[['Laptrigger_lapdist_dls', 'speed', 'predicted_speed']]
                .to_json(orient="records")
            )
        
        # Calculate statistics
        total_points = len(lap_df)
        mistake_count = len(mistakes_df)
        too_slow_count = len(lap_df[lap_df['mistake_type'] == 'TOO_SLOW']) if 'mistake_type' in lap_df.columns else 0
        too_fast_count = len(lap_df[lap_df['mistake_type'] == 'TOO_FAST']) if 'mistake_type' in lap_df.columns else 0
        
        avg_error = float(lap_df['abs_error'].mean()) if 'abs_error' in lap_df.columns else 0
        max_error = float(lap_df['abs_error'].max()) if 'abs_error' in lap_df.columns else 0
        
        return {
            "session_id": session_id,
            "lap": lap_number,
            "statistics": {
                "total_points": total_points,
                "mistake_count": mistake_count,
                "mistake_percentage": (mistake_count / total_points * 100) if total_points > 0 else 0,
                "too_slow_count": too_slow_count,
                "too_fast_count": too_fast_count,
                "avg_error_kmh": avg_error,
                "max_error_kmh": max_error
            },
            "feature_importance": feature_importance[:10],  # Top 10 features
            "full_lap_data": full_lap_data,  # Full lap for continuous lines
            "mistake_points": mistake_points,  # Only mistakes for highlighting
            "has_data": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Gazoo Analyst Backend"}
