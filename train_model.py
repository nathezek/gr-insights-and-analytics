import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import os
import numpy as np
import gc
from data_cleanup import clean_telemetry

# Load from processed_data
processed_dir = "processed_data"
race_paths = {
    "Race 1": os.path.join(processed_dir, "merged_r1.csv"),
    "Race 2": os.path.join(processed_dir, "merged_r2.csv"),
    "Race 3": os.path.join(processed_dir, "merged_r3.csv"),
    "Race 4": os.path.join(processed_dir, "merged_r4.csv"),
    "Race 5": os.path.join(processed_dir, "merged_r5.csv"),
}

# AGGRESSIVE loading - use more data
def load_merged_memory_efficient(path, sample_frac=0.35, chunksize=250000):
    """Load data in chunks and sample to reduce memory usage"""
    if not os.path.exists(path):
        return pd.DataFrame()
    
    print(f"[INFO] Loading {path} (sample_frac={sample_frac}, chunksize={chunksize})...")
    sampled_chunks = []
    chunk_count = 0
    
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        # Sample each chunk
        sampled = chunk.sample(frac=sample_frac, random_state=42)
        sampled_chunks.append(sampled)
        chunk_count += 1
        
        if chunk_count % 5 == 0:
            print(f"  Processed {chunk_count} chunks, current total: {sum(len(c) for c in sampled_chunks):,} rows")
    
    result = pd.concat(sampled_chunks, ignore_index=True)
    print(f"[INFO] Loaded {len(result):,} rows from {chunk_count} chunks")
    return result


print("=" * 60)
print("LOADING DATA (MAXIMUM PERFORMANCE MODE - ALL RACES)")
print("=" * 60)

# Load all available races
race_data = []
for race_name, race_path in race_paths.items():
    if os.path.exists(race_path):
        print(f"\n📊 Loading {race_name}...")
        merged = load_merged_memory_efficient(race_path, sample_frac=0.35, chunksize=250000)
        if not merged.empty:
            # Add race identifier
            merged['race_id'] = race_name
            race_data.append(merged)
            print(f"✅ {race_name}: {merged.shape}")
        gc.collect()
    else:
        print(f"⚠️  {race_name} not found: {race_path}")

if not race_data:
    raise ValueError("No race data found! Run merge_csvs.py first for all races.")

# Concatenate all races
all_data = pd.concat(race_data, ignore_index=True)
print(f"\n{'='*60}")
print(f"📈 Total data from {len(race_data)} races: {all_data.shape}")
print(f"{'='*60}")

# Clear individual race data
del race_data
gc.collect()

# Clean
try:
    all_data = clean_telemetry(all_data)
    print(f"[INFO] After cleaning all_data shape: {all_data.shape}")
except Exception as e:
    print(f"[WARN] Skipping clean_telemetry: {e}")

# Features - prioritize by importance
features = ["gear", "total_g", "accx_can", "Steering_Angle", "aps", 
            "accy_can", "pbrake_f", "pbrake_r", "Laptrigger_lapdist_dls"]

features = [f for f in features if f in all_data.columns]
target = "speed"

print(f"\n[INFO] Using base features: {features}")

# FEATURE ENGINEERING - add derived features WITH SAFETY CHECKS
print(f"\n[INFO] Engineering additional features...")
engineered_count = 0

if 'accx_can' in all_data.columns and 'accy_can' in all_data.columns:
    # Acceleration angle
    all_data['acc_angle'] = np.arctan2(all_data['accy_can'], all_data['accx_can'])
    features.append('acc_angle')
    engineered_count += 1
    
if 'gear' in all_data.columns and 'aps' in all_data.columns:
    # Gear-throttle interaction (clip to reasonable range)
    all_data['gear_aps'] = (all_data['gear'] * all_data['aps']).clip(-1000, 1000)
    features.append('gear_aps')
    engineered_count += 1

if 'pbrake_f' in all_data.columns and 'pbrake_r' in all_data.columns:
    # Total braking pressure
    all_data['total_brake'] = all_data['pbrake_f'] + all_data['pbrake_r']
    features.append('total_brake')
    engineered_count += 1
    
    # Brake balance - SAFE division
    denominator = all_data['pbrake_f'] + all_data['pbrake_r'] + 0.01  # Larger epsilon
    all_data['brake_balance'] = all_data['pbrake_f'] / denominator
    all_data['brake_balance'] = all_data['brake_balance'].clip(0, 1)  # Balance between 0 and 1
    features.append('brake_balance')
    engineered_count += 1

if 'total_g' in all_data.columns and 'gear' in all_data.columns:
    # G-force per gear - SAFE division
    # Only calculate when gear > 0
    all_data['g_per_gear'] = all_data['total_g'] / (all_data['gear'] + 1).clip(1, None)
    all_data['g_per_gear'] = all_data['g_per_gear'].clip(0, 10)  # Reasonable upper limit
    features.append('g_per_gear')
    engineered_count += 1

if 'Steering_Angle' in all_data.columns:
    # Absolute steering
    all_data['abs_steering'] = np.abs(all_data['Steering_Angle']).clip(0, 360)
    features.append('abs_steering')
    engineered_count += 1

print(f"[INFO] Created {engineered_count} engineered features")
print(f"[INFO] Total features ({len(features)}): {features}")

if target not in all_data.columns:
    raise ValueError(f"Missing target column: {target}")

# Keep only required columns
required_cols = features + [target]
all_data = all_data[required_cols]

# CRITICAL: Replace inf/-inf with NaN BEFORE imputation
print(f"\n[INFO] Checking for infinite values...")
for col in features:
    inf_count = np.isinf(all_data[col]).sum()
    if inf_count > 0:
        print(f"  {col}: {inf_count} infinite values → converting to NaN")
        all_data[col] = all_data[col].replace([np.inf, -np.inf], np.nan)

gc.collect()

print(f"\n[INFO] Rows before processing: {len(all_data):,}")

# Remove outliers
all_data = all_data[(all_data[target] >= 0) & (all_data[target] <= 400)]
print(f"[INFO] Rows after removing speed outliers: {len(all_data):,}")

# Use IMPUTATION
print(f"\n[INFO] Handling missing values with median imputation...")
imputer = SimpleImputer(strategy='median')

X_with_missing = all_data[features]
y = all_data[target].dropna()

valid_indices = all_data[target].notna()
X_with_missing = X_with_missing[valid_indices]
y = y[valid_indices]

# Double-check no inf values before imputation
print(f"[INFO] Pre-imputation check...")
for col in X_with_missing.columns:
    inf_count = np.isinf(X_with_missing[col]).sum()
    nan_count = X_with_missing[col].isna().sum()
    print(f"  {col}: {nan_count} NaN, {inf_count} inf")
    if inf_count > 0:
        print(f"    WARNING: Still has inf values, replacing...")
        X_with_missing[col] = X_with_missing[col].replace([np.inf, -np.inf], np.nan)

X = pd.DataFrame(
    imputer.fit_transform(X_with_missing),
    columns=features,
    index=X_with_missing.index
)

print(f"[INFO] Final dataset size: {len(X):,} rows")

# Convert to float32
X = X.astype('float32')
y = y.astype('float32')

print(f"\n[INFO] Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"[INFO] Feature matrix shape: {X.shape}")
print(f"[INFO] Target shape: {y.shape}")
print(f"[INFO] Target stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")

del all_data, X_with_missing
gc.collect()

# Convert to numpy
X = X.values
y = y.values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)
print(f"\n[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

del X, y
gc.collect()

os.makedirs("saved", exist_ok=True)

print("\n" + "=" * 60)
print("🚀 TRAINING ULTRA-PERFORMANCE GRADIENT BOOSTING 🚀")
print("=" * 60)

# MAXIMUM PERFORMANCE hyperparameters
model = GradientBoostingRegressor(
    n_estimators=10,
    max_depth=11,
    learning_rate=0.05,
    random_state=42,
    warm_start=True,
    subsample=0.97,
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    validation_fraction=0.15,
    n_iter_no_change=20,
)

n_stages = 30
trees_per_stage = 25

best_test_r2 = -np.inf
best_n_trees = 0
best_mae = np.inf
no_improvement_count = 0
patience = 7

print(f"[INFO] Will train up to {n_stages * trees_per_stage} trees with early stopping")
print(f"[INFO] Training on {len(X_train):,} samples from multiple races")

for stage in range(n_stages):
    print(f"\n[INFO] Training stage {stage + 1}/{n_stages}...")
    model.n_estimators = (stage + 1) * trees_per_stage
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train[:5000])
    y_pred_test = model.predict(X_test)
    
    train_r2_sample = r2_score(y_train[:5000], y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    improvement = ""
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_n_trees = model.n_estimators
        best_mae = test_mae
        no_improvement_count = 0
        joblib.dump(model, "saved/speed_model_best_with_r5.pkl")
        improvement = " ⭐ NEW BEST!"
    else:
        no_improvement_count += 1
    
    print(f"  Trees: {model.n_estimators:3d} | Train R²: {train_r2_sample:.4f} | Test R²: {test_r2:.4f} | MAE: {test_mae:.2f}{improvement}")
    
    if no_improvement_count >= patience:
        print(f"\n[INFO] Early stopping: No improvement for {patience} stages")
        break
    
    gc.collect()

print(f"\n[INFO] Training complete with {model.n_estimators} trees")
print(f"[INFO] 🏆 BEST → Test R²: {best_test_r2:.4f}, MAE: {best_mae:.2f} at {best_n_trees} trees")

if os.path.exists("saved/speed_model_best.pkl"):
    print(f"[INFO] Loading best model ({best_n_trees} trees)...")
    model = joblib.load("saved/speed_model_best.pkl")

print(f"\n[INFO] Final evaluation on test set...")
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

print("\n" + "=" * 60)
print("🏆 FINAL MODEL PERFORMANCE 🏆")
print("=" * 60)
print(f"TRAINING SET ({len(X_train):,} samples):")
print(f"  R² Score:  {train_r2:.4f}")
print(f"  MAE:       {train_mae:.2f} km/h")
print(f"  RMSE:      {train_rmse:.2f} km/h")
print(f"\nTEST SET ({len(X_test):,} samples):")
print(f"  R² Score:  {r2:.4f} ⭐")
print(f"  MAE:       {mae:.2f} km/h")
print(f"  RMSE:      {rmse:.2f} km/h")
print(f"\nGeneralization: {r2/train_r2:.4f}")
print(f"Overfitting:    {train_r2 - r2:.4f}")

print("\n" + "=" * 60)
print("📊 FEATURE IMPORTANCES")
print("=" * 60)
importances = sorted(
    zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True  # type: ignore
)
for i, (feature, importance) in enumerate(importances, 1):  # type: ignore
    bar = "█" * int(importance * 100)
    print(f"{i:2d}. {feature:25s}: {importance:.4f} {bar}")

model_path = "saved/speed_model.pkl"
joblib.dump(model, model_path)
print(f"\n✅ Model saved to {model_path}!")

joblib.dump(imputer, "saved/imputer.pkl")
print("✅ Imputer saved to saved/imputer.pkl")

metadata = {
    "features": features,
    "model_type": "GradientBoosting_MultiRace",
    "n_estimators": model.n_estimators,
    "races_used": len([p for p in race_paths.values() if os.path.exists(p)]),
    "train_r2_score": train_r2,
    "test_r2_score": r2,
    "mae": mae,
    "rmse": rmse,
    "training_samples": len(X_train),
}
joblib.dump(metadata, "saved/model_metadata.pkl")
print("✅ Model metadata saved")

print("\n" + "=" * 60)
print("📈 PERFORMANCE BY SPEED RANGE")
print("=" * 60)

speed_ranges = [
    (0, 50, "Very Slow"),
    (50, 100, "Slow"),
    (100, 150, "Medium"),
    (150, 200, "Fast"),
]

for low, high, label in speed_ranges:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() > 0:
        range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        range_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
        range_r2 = r2_score(y_test[mask], y_pred[mask])
        print(f"{label:12s} ({low:3d}-{high:3d} km/h): R²={range_r2:.4f}, MAE={range_mae:5.2f}, RMSE={range_rmse:5.2f}, n={mask.sum():,}")

print("\n" + "=" * 60)
print("📉 ERROR DISTRIBUTION")
print("=" * 60)
errors = np.abs(y_test - y_pred)
percentiles = [50, 75, 90, 95, 99]
for p in percentiles:
    print(f"{p:2d}th percentile: {np.percentile(errors, p):.2f} km/h")

print(f"\nErrors < 3 km/h:  {(errors < 3).sum() / len(errors) * 100:.1f}%")
print(f"Errors < 5 km/h:  {(errors < 5).sum() / len(errors) * 100:.1f}%")
print(f"Errors < 10 km/h: {(errors < 10).sum() / len(errors) * 100:.1f}%")

print("\n" + "=" * 60)
print("🎯 SAMPLE PREDICTIONS (25 examples)")
print("=" * 60)
sample_indices = np.random.choice(len(X_test), min(25, len(X_test)), replace=False)
sample_errors = []
for idx in sample_indices:
    actual = y_test[idx]
    predicted = model.predict(X_test[idx:idx+1])[0]
    error = abs(actual - predicted)
    sample_errors.append(error)
    status = "✓" if error < 3 else ("~" if error < 7 else "✗")
    print(f"{status} Actual: {actual:6.2f} | Predicted: {predicted:6.2f} | Error: {error:5.2f} km/h")

print(f"\nSample average: {np.mean(sample_errors):.2f} km/h")
print(f"Sample median:  {np.median(sample_errors):.2f} km/h")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Model: {model.n_estimators} trees, depth={model.max_depth}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.2f} km/h")
print(f"Training data: {len(X_train):,} samples from {len([p for p in race_paths.values() if os.path.exists(p)])} races")
print("=" * 60)