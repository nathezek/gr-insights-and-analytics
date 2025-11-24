from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
from data_cleanup import clean_telemetry
from ai_magic import get_model, predict_mistakes

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

@app.on_event("startup")
async def startup_event():
    global model, imputer, metadata
    try:
        # Adjust path if needed (now in root)
        model, metadata, imputer = get_model("saved/speed_model_v5.pkl")
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

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
            
            # Downsample for frontend visualization (Limit to 8k points)
            # Reduced to 8k to stay under localStorage limit (~5MB) while supporting 2-3 laps
            MAX_FRONTEND_POINTS = 8000
            if len(predicted_df) > MAX_FRONTEND_POINTS:
                predicted_df = predicted_df.sort_values(['lap', 'Laptrigger_lapdist_dls'])
                # Use systematic sampling instead of random to preserve line shape
                step = len(predicted_df) // MAX_FRONTEND_POINTS
                predicted_df = predicted_df.iloc[::step].iloc[:MAX_FRONTEND_POINTS]
            
            # Convert to JSON-friendly format (Handle NaN/Inf)
            # Use pandas to_json which handles NaNs correctly (converts to null)
            import json
            result = json.loads(predicted_df.to_json(orient="records"))
            return {
                "data": result,
                "insights": insights,
                "metadata": {
                    "rows": len(predicted_df),
                    "columns": list(predicted_df.columns)
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Gazoo Analyst Backend"}
