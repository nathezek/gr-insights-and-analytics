import joblib
import pandas as pd
import numpy as np
import os


def get_model(model_path="saved/speed_model_v5.pkl"):
    """Load the trained model and metadata"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    # Try to load metadata
    metadata = None
    for suffix in ["_metadata.pkl", "_meta.pkl", ".meta.pkl"]:
        metadata_path = model_path.replace(".pkl", suffix)
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                print(f"✅ Loaded metadata from {metadata_path}")
                break
            except Exception as e:
                print(f"⚠️  Could not load metadata from {metadata_path}: {e}")

    if metadata is None:
        print("⚠️  No metadata file found - creating default metadata")
        metadata = {
            "model_type": "Unknown",
            "test_r2_score": None,
            "mae": None,
            "features": [],
        }

    # Try to load imputer
    imputer = None
    imputer_path = "saved/imputer.pkl"
    if os.path.exists(imputer_path):
        try:
            imputer = joblib.load(imputer_path)
            print(f"✅ Loaded imputer from {imputer_path}")
        except Exception as e:
            print(f"⚠️  Could not load imputer: {e}")
    else:
        print("⚠️  No imputer found - will skip imputation")

    print("✅ Model loaded successfully")

    return model, metadata, imputer


def engineer_features(df):
    """Add engineered features (must match training)"""
    engineered = df.copy()

    # 1. Acceleration angle
    if "accx_can" in df.columns and "accy_can" in df.columns:
        engineered["acc_angle"] = np.arctan2(df["accy_can"], df["accx_can"])

    # 2. Gear-throttle interaction
    if "gear" in df.columns and "aps" in df.columns:
        engineered["gear_aps"] = (df["gear"] * df["aps"]).clip(-1000, 1000)

    # 3. Total brake
    if "pbrake_f" in df.columns and "pbrake_r" in df.columns:
        engineered["total_brake"] = df["pbrake_f"] + df["pbrake_r"]

        # Brake balance
        denominator = df["pbrake_f"] + df["pbrake_r"] + 0.01
        engineered["brake_balance"] = (df["pbrake_f"] / denominator).clip(0, 1)

    # 4. G-force per gear
    if "total_g" in df.columns and "gear" in df.columns:
        engineered["g_per_gear"] = (
            df["total_g"] / (df["gear"] + 1).clip(1, None)
        ).clip(0, 10)

    # 5. Absolute steering
    if "Steering_Angle" in df.columns:
        engineered["abs_steering"] = np.abs(df["Steering_Angle"]).clip(0, 360)

    # Replace inf with NaN
    engineered = engineered.replace([np.inf, -np.inf], np.nan)

    return engineered


def predict_chunked(model, X, imputer=None, chunksize=10000):
    """
    Predict in chunks for better memory efficiency and responsiveness

    Args:
        model: Trained model
        X: Feature matrix (DataFrame or numpy array)
        imputer: Optional imputer
        chunksize: Number of rows to process at once

    Returns:
        Array of predictions
    """

    total_rows = len(X)
    predictions = []

    print(f"[INFO] Predicting in chunks of {chunksize:,}...")

    for i in range(0, total_rows, chunksize):
        chunk = X[i: i + chunksize]

        # Apply imputer if available
        if imputer is not None:
            try:
                if isinstance(chunk, pd.DataFrame):
                    chunk = imputer.transform(chunk)
                else:
                    chunk = imputer.transform(chunk)
            except Exception as e:
                print(f"⚠️  Imputer failed on chunk: {e}")

        # Convert to numpy if needed
        if isinstance(chunk, pd.DataFrame):
            chunk = chunk.values

        # Predict
        chunk_pred = model.predict(chunk)
        predictions.append(chunk_pred)

        if (i + chunksize) % (chunksize * 5) == 0:
            print(f"  Predicted {i + chunksize:,} / {total_rows:,} rows...")

    # Combine all predictions
    all_predictions = np.concatenate(predictions)
    print(f"[INFO] Prediction complete: {len(all_predictions):,} predictions")

    return all_predictions


def predict_mistakes(
    model, telemetry_df, imputer=None, threshold_kmh=10, chunksize=10000
):
    """
    Predict expected speed and detect mistakes (CHUNKED VERSION)

    Args:
        model: Trained model
        telemetry_df: DataFrame with telemetry data
        imputer: Optional imputer for missing values
        threshold_kmh: Speed difference threshold for mistake detection
        chunksize: Process this many rows at once

    Returns:
        DataFrame with predictions and mistake flags
        List of insights/mistakes detected
    """

    print(f"\n[INFO] Starting AI analysis on {
          len(telemetry_df):,} data points...")

    # Verify speed column exists
    if "speed" not in telemetry_df.columns:
        raise ValueError("'speed' column is required in telemetry data")

    # Required features (must match training)
    required_features = [
        "gear",
        "total_g",
        "accx_can",
        "Steering_Angle",
        "aps",
        "accy_can",
        "pbrake_f",
        "pbrake_r",
        "Laptrigger_lapdist_dls",
    ]

    # Engineer features
    print("[INFO] Engineering features...")
    df = engineer_features(telemetry_df)

    # Add engineered features to required list
    engineered_features = [
        "acc_angle",
        "gear_aps",
        "total_brake",
        "brake_balance",
        "g_per_gear",
        "abs_steering",
    ]

    all_features = required_features + engineered_features

    # Check which features are available
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]

    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")

    if len(available_features) == 0:
        raise ValueError("No valid features found for prediction")

    print(f"[INFO] Using {len(available_features)} features")

    # Prepare feature matrix
    X = df[available_features].copy()

    # Predict in chunks
    predicted_speed = predict_chunked(model, X, imputer, chunksize=chunksize)

    # Add predictions to dataframe
    result = df.copy()
    result["predicted_speed"] = predicted_speed

    # Calculate errors
    if "speed" in result.columns:
        print("[INFO] Calculating errors and classifying mistakes...")
        result["speed_error"] = result["speed"] - result["predicted_speed"]
        result["abs_error"] = np.abs(result["speed_error"])

        # Classify mistakes
        result["mistake_type"] = "OK"
        result.loc[result["speed_error"] < -
                   threshold_kmh, "mistake_type"] = "TOO_SLOW"
        result.loc[result["speed_error"] >
                   threshold_kmh, "mistake_type"] = "TOO_FAST"

        # Calculate mistake severity (0-1 scale)
        result["mistake_severity"] = (result["abs_error"] / 50).clip(0, 1)

    # Generate insights
    print("[INFO] Generating insights...")
    insights = generate_insights(result, threshold_kmh)

    print("[INFO] Analysis complete!\n")
    return result, insights


def generate_insights(df, threshold_kmh):
    """Generate human-readable insights from predictions"""
    insights = []

    if "speed_error" not in df.columns:
        return ["No speed data available for comparison"]

    # Count mistakes
    mistakes = df[df["mistake_type"] != "OK"]
    mistake_count = len(mistakes)

    if mistake_count == 0:
        insights.append("Perfect lap! No significant mistakes detected.")
        return insights

    # Overall stats
    total_points = len(df)
    mistake_pct = (mistake_count / total_points) * 100
    insights.append(
        f"{mistake_count:,} mistakes detected ({mistake_pct:.1f}% of lap)"
    )

    # Breakdown by type
    too_slow = len(df[df["mistake_type"] == "TOO_SLOW"])
    too_fast = len(df[df["mistake_type"] == "TOO_FAST"])

    if too_slow > 0:
        avg_slow_loss = df[df["mistake_type"]
                           == "TOO_SLOW"]["speed_error"].mean()
        insights.append(
            f"{too_slow} points too slow (avg {abs(
                avg_slow_loss):.1f} km/h lost)"
        )

    if too_fast > 0:
        avg_fast_excess = df[df["mistake_type"]
                             == "TOO_FAST"]["speed_error"].mean()
        insights.append(
            f"{too_fast} points too fast (avg {
                avg_fast_excess:.1f} km/h over)"
        )

    # Find worst mistakes
    worst_slow = df[df["mistake_type"] ==
                    "TOO_SLOW"].nsmallest(1, "speed_error")
    worst_fast = df[df["mistake_type"] ==
                    "TOO_FAST"].nlargest(1, "speed_error")

    if len(worst_slow) > 0:
        worst_error = worst_slow.iloc[0]["speed_error"]
        insights.append(
            f"Worst slow mistake: {abs(worst_error):.1f} km/h below expected"
        )

    if len(worst_fast) > 0:
        worst_error = worst_fast.iloc[0]["speed_error"]
        insights.append(f"Worst fast mistake: {
                        worst_error:.1f} km/h above expected")

    # Corner vs straight analysis
    if "Steering_Angle" in df.columns:
        cornering = df[np.abs(df["Steering_Angle"]) > 5]
        straight = df[np.abs(df["Steering_Angle"]) <= 5]

        if len(cornering) > 0 and len(straight) > 0:
            corner_mistakes = len(cornering[cornering["mistake_type"] != "OK"])
            straight_mistakes = len(straight[straight["mistake_type"] != "OK"])

            corner_pct = (
                (corner_mistakes / len(cornering)) *
                100 if len(cornering) > 0 else 0
            )
            straight_pct = (
                (straight_mistakes / len(straight)) *
                100 if len(straight) > 0 else 0
            )

            if corner_pct > straight_pct:
                insights.append(
                    f"More mistakes in corners ({corner_pct:.1f}%) vs straights ({
                        straight_pct:.1f}%)"
                )
            elif straight_pct > corner_pct:
                insights.append(
                    f"More mistakes on straights ({straight_pct:.1f}%) vs corners ({
                        corner_pct:.1f}%)"
                )

    # Braking mistakes
    if "pbrake_f" in df.columns and "pbrake_r" in df.columns:
        braking = df[(df["pbrake_f"] > 10) | (df["pbrake_r"] > 10)]
        if len(braking) > 0:
            braking_mistakes = len(braking[braking["mistake_type"] != "OK"])
            braking_pct = (braking_mistakes / len(braking)) * 100

            total_mistake_pct = (mistake_count / total_points) * 100
            if braking_pct > total_mistake_pct:
                insights.append(
                    f"High mistake rate during braking ({braking_pct:.1f}%)"
                )

    return insights


# Import track definitions
try:
    from track_definitions import get_track_corners, BARBER_CORNERS
    BARBER_TURNS = [
        {"name": v['name'], "start": v['start'], "end": v['end']}
        for k, v in BARBER_CORNERS.items()
    ]
except ImportError:
    # Fallback to hardcoded if track_definitions not available
    BARBER_TURNS = [
        {"name": "T1", "start": 0, "end": 150},
        {"name": "T2", "start": 400, "end": 550},
        {"name": "T3", "start": 700, "end": 850},
        {"name": "T4", "start": 1000, "end": 1150},
        {"name": "T5", "start": 1300, "end": 1500},
        {"name": "T6", "start": 1600, "end": 1750},
        {"name": "T7", "start": 1900, "end": 2050},
        {"name": "T8", "start": 2200, "end": 2350},
        {"name": "T9", "start": 2500, "end": 2650},
        {"name": "T10", "start": 2800, "end": 2950},
        {"name": "T11", "start": 3050, "end": 3200},
        {"name": "T12", "start": 3300, "end": 3400},
        {"name": "T13", "start": 3450, "end": 3600},
    ]


def generate_turn_advice(df):
    """Generate specific advice for each turn based on mistakes"""
    advice_list = []
    
    if "Laptrigger_lapdist_dls" not in df.columns or "mistake_type" not in df.columns:
        return []
        
    for turn in BARBER_TURNS:
        # Filter data for this turn
        turn_data = df[
            (df["Laptrigger_lapdist_dls"] >= turn["start"]) & 
            (df["Laptrigger_lapdist_dls"] <= turn["end"])
        ]
        
        if len(turn_data) == 0:
            continue
            
        # Analyze mistakes in this turn
        mistakes = turn_data[turn_data["mistake_type"] != "OK"]
        
        if len(mistakes) > 0:
            # Check dominant mistake type
            too_slow = len(mistakes[mistakes["mistake_type"] == "TOO_SLOW"])
            too_fast = len(mistakes[mistakes["mistake_type"] == "TOO_FAST"])
            
            # Get max error
            max_error = mistakes["abs_error"].max() if "abs_error" in mistakes.columns else 0
            
            if too_slow > too_fast and too_slow > 5: # Threshold to avoid noise
                advice_list.append({
                    "turn": turn["name"],
                    "type": "Too Slow",
                    "message": f"{turn['name']}: You are overslowing. Try braking later or carrying more entry speed. ({max_error:.1f} km/h too slow)",
                    "severity": "medium" if max_error < 15 else "high"
                })
            elif too_fast > too_slow and too_fast > 5:
                advice_list.append({
                    "turn": turn["name"],
                    "type": "Too Fast",
                    "message": f"{turn['name']}: You are entering too fast. Brake earlier to hit the apex. ({max_error:.1f} km/h too fast)",
                    "severity": "medium" if max_error < 15 else "high"
                })
                
    return advice_list

