"""
Comprehensive model training script
Trains a new speed prediction model using all available track data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib
import gc
from data_cleanup import clean_telemetry
from csv_merger import merge_lap_csvs
from ai_magic import engineer_features


def load_track_data(track_name, race_name):
    """Load and merge data for a specific track/race"""
    print(f"\n[INFO] Loading {track_name} - {race_name}...")
    
    base_path = Path(f"raw_training_data/{track_name}/{race_name}")
    
    # Find the CSV files
    lap_start_file = list(base_path.glob("*lap_start.csv"))
    lap_end_file = list(base_path.glob("*lap_end.csv"))
    lap_time_file = list(base_path.glob("*lap_time.csv"))
    telemetry_file = list(base_path.glob("telemetry.csv"))
    
    if not all([lap_start_file, lap_end_file, lap_time_file, telemetry_file]):
        print(f"[WARN] Missing files for {track_name}/{race_name}")
        return None
    
    try:
        # Read CSVs
        lap_start = pd.read_csv(lap_start_file[0])
        lap_end = pd.read_csv(lap_end_file[0])
        lap_time = pd.read_csv(lap_time_file[0])
        telemetry = pd.read_csv(telemetry_file[0])
        
        print(f"  Telemetry rows: {len(telemetry):,}")
        
        # Merge
        merged = merge_lap_csvs(lap_start, lap_end, lap_time, telemetry)
        
        # Clean
        cleaned = clean_telemetry(merged, max_rows=500000, sample_frac=0.15)
        
        # Add track identifier
        cleaned['track'] = track_name
        cleaned['race'] = race_name
        
        print(f"  ✅ Loaded {len(cleaned):,} rows")
        
        return cleaned
        
    except Exception as e:
        print(f"[ERROR] Failed to load {track_name}/{race_name}: {e}")
        return None


def main():
    print("="*80)
    print("TRAINING NEW SPEED PREDICTION MODEL")
    print("="*80)
    
    # Define tracks and races to load
    tracks_to_load = [
        ("barber", "Race_1"),
        ("barber", "Race_2"),
        # Add more as available
    ]
    
    # Load all data
    all_data = []
    for track, race in tracks_to_load:
        data = load_track_data(track, race)
        if data is not None:
            all_data.append(data)
            gc.collect()
    
    if not all_data:
        print("[ERROR] No data loaded! Exiting.")
        return
    
    # Combine all data
    print(f"\n[INFO] Combining data from {len(all_data)} sources...")
    combined_df = pd.concat(all_data, ignore_index=True)
    del all_data
    gc.collect()
    
    print(f"[INFO] Total combined rows: {len(combined_df):,}")
    print(f"[INFO] Columns: {list(combined_df.columns)}")
    
    # Engineer features
    print("\n[INFO] Engineering features...")
    featured_df = engineer_features(combined_df)
    
    # Define features for training
    base_features = [
        "gear", "total_g", "accx_can", "Steering_Angle", "aps",
        "accy_can", "pbrake_f", "pbrake_r", "Laptrigger_lapdist_dls"
    ]
    
    engineered_features = [
        "acc_angle", "gear_aps", "total_brake", "brake_balance",
        "g_per_gear", "abs_steering"
    ]
    
    all_features = base_features + engineered_features
    
    # Filter to available features
    available_features = [f for f in all_features if f in featured_df.columns]
    print(f"[INFO] Using {len(available_features)} features: {available_features}")
    
    # Prepare training data
    print("\n[INFO] Preparing training data...")
    X = featured_df[available_features].copy()
    y = featured_df['speed'].copy()
    
    # Remove rows with NaN in target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"[INFO] Training samples: {len(X):,}")
    
    # Handle missing values with imputation
    print("[INFO] Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Split data
    print("[INFO] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    # Train model
    print("\n[INFO] Training Random Forest model...")
    print("  This may take several minutes...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n[INFO] Evaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'MAE (km/h)':<20} {train_mae:<15.2f} {test_mae:<15.2f}")
    print(f"{'RMSE (km/h)':<20} {train_rmse:<15.2f} {test_rmse:<15.2f}")
    
    # Feature importance
    print("\n[INFO] Top 10 Most Important Features:")
    feature_importance = sorted(
        zip(available_features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        print(f"  {i}. {feat:<25} {imp:.4f}")
    
    # Save model
    print("\n[INFO] Saving model...")
    save_dir = Path("saved")
    save_dir.mkdir(exist_ok=True)
    
    model_path = save_dir / "speed_model_v6.pkl"
    metadata_path = save_dir / "speed_model_v6_metadata.pkl"
    imputer_path = save_dir / "imputer.pkl"
    
    # Save model
    joblib.dump(model, model_path)
    print(f"  ✅ Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "RandomForestRegressor",
        "features": available_features,
        "test_r2_score": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "tracks": list(featured_df['track'].unique()) if 'track' in featured_df.columns else [],
        "feature_importance": dict(feature_importance)
    }
    joblib.dump(metadata, metadata_path)
    print(f"  ✅ Metadata saved to {metadata_path}")
    
    # Save imputer
    joblib.dump(imputer, imputer_path)
    print(f"  ✅ Imputer saved to {imputer_path}")
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTo use the new model, update main.py to load:")
    print(f"  model_path = 'saved/speed_model_v6.pkl'")
    print("\nThen restart the backend server.")


if __name__ == "__main__":
    main()
