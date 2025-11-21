import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np
import gc
from data_cleanup import clean_telemetry

# Load from processed_data
processed_dir = "processed_data"
r1_path = os.path.join(processed_dir, "merged_r1.csv")
r2_path = os.path.join(processed_dir, "merged_r2.csv")


# Memory-efficient loading
def load_merged_memory_efficient(path, sample_frac=0.05):
    """Load data in chunks and sample to reduce memory usage"""
    if not os.path.exists(path):
        return pd.DataFrame()

    print(f"[INFO] Loading {path} in memory-efficient mode...")
    sampled_chunks = []
    chunk_count = 0

    for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
        # Sample each chunk
        sampled = chunk.sample(frac=sample_frac, random_state=42)
        sampled_chunks.append(sampled)
        chunk_count += 1

        if chunk_count % 10 == 0:
            print(f"  Processed {chunk_count} chunks...")

    result = pd.concat(sampled_chunks, ignore_index=True)
    print(f"[INFO] Loaded {len(result):,} rows from {chunk_count} chunks")
    return result


# Load data with sampling
print("=" * 60)
print("LOADING DATA (MEMORY-EFFICIENT MODE)")
print("=" * 60)

merged_r1 = load_merged_memory_efficient(r1_path, sample_frac=0.03)
print(f"[INFO] Loaded merged_r1 shape: {merged_r1.shape}")

# Force garbage collection
gc.collect()

merged_r2 = load_merged_memory_efficient(r2_path, sample_frac=0.03)
print(f"[INFO] Loaded merged_r2 shape: {merged_r2.shape}")

all_data = pd.concat([merged_r1, merged_r2], ignore_index=True)
print(f"[INFO] Concatenated all_data shape: {all_data.shape}")

# Clear merged dataframes to free memory
del merged_r1, merged_r2
gc.collect()

if all_data.empty:
    raise ValueError(
        "No merged files found in processed_data. Run merge_csvs.py first."
    )

# Clean
try:
    all_data = clean_telemetry(all_data)
    print(f"[INFO] After cleaning all_data shape: {all_data.shape}")
except Exception as e:
    print(f"[WARN] Skipping clean_telemetry: {e}")

# Features
base_features = ["aps", "pbrake_f", "pbrake_r", "Steering_Angle", "gear", "accy_can"]
extra_features = ["accx_can", "total_g", "Laptrigger_lapdist_dls"]
features = base_features.copy()
for feat in extra_features:
    if feat in all_data.columns:
        features.append(feat)

target = "speed"

print(f"\n[INFO] Using features: {features}")

# Check columns
missing_features = [f for f in features if f not in all_data.columns]
if missing_features:
    print(f"[ERROR] Missing feature columns: {missing_features}")
    raise ValueError(f"Missing feature columns: {missing_features}")

if target not in all_data.columns:
    raise ValueError(f"Missing target column: {target}")

# Keep only required columns to reduce memory
required_cols = features + [target]
all_data = all_data[required_cols]
gc.collect()

# Drop missing values
print(f"[INFO] Rows before dropna: {len(all_data)}")
all_data = all_data.dropna(subset=required_cols)
print(f"[INFO] Rows after dropna: {len(all_data)}")

# Remove outliers
all_data = all_data[(all_data[target] >= 0) & (all_data[target] <= 400)]
print(f"[INFO] Rows after removing outliers: {len(all_data)}")

if len(all_data) == 0:
    raise ValueError("No data remaining after removing missing values")

# Convert to more memory-efficient dtypes
for col in features:
    if col in all_data.columns:
        all_data[col] = all_data[col].astype("float32")
all_data[target] = all_data[target].astype("float32")

print(
    f"\n[INFO] Memory usage: {all_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
)

# Features and target
X = all_data[features].values  # Convert to numpy array (more efficient)
y = all_data[target].values

print(f"\n[INFO] Feature matrix shape: {X.shape}")
print(f"[INFO] Target shape: {y.shape}")
print(
    f"[INFO] Target stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}"
)

# Clear dataframe to free memory
del all_data
gc.collect()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"\n[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Clear original arrays
del X, y
gc.collect()

os.makedirs("saved", exist_ok=True)

print("\n" + "=" * 60)
print("TRAINING GRADIENT BOOSTING (INCREMENTAL MODE)")
print("=" * 60)

# Train incrementally with warm_start
model = GradientBoostingRegressor(
    n_estimators=10,  # Start with fewer trees
    max_depth=5,  # Reduced depth
    learning_rate=0.1,
    random_state=42,
    warm_start=True,  # Enable incremental training
    subsample=0.8,  # Use subset of data per tree
    max_features="sqrt",  # Use subset of features
)

# Train in stages
n_stages = 10
trees_per_stage = 10

for stage in range(n_stages):
    print(f"\n[INFO] Training stage {stage + 1}/{n_stages}...")
    model.n_estimators = (stage + 1) * trees_per_stage
    model.fit(X_train, y_train)

    # Evaluate progress
    y_pred_train = model.predict(X_train[:1000])  # Sample for speed
    train_score = r2_score(y_train[:1000], y_pred_train)
    print(f"  Trees: {model.n_estimators}, Train R² (sample): {train_score:.4f}")

    gc.collect()

print(f"\n[INFO] Training complete with {model.n_estimators} trees")

# Test set performance
print(f"\n[INFO] Evaluating on test set...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Mean Absolute Error (MAE):      {mae:.2f} km/h")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} km/h")
print(f"R² Score:                       {r2:.4f}")

# Feature importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCES")
print("=" * 60)
importances = sorted(
    zip(features, model.feature_importances_),  # type: ignore
    key=lambda x: x[1],  # type:ignore
    reverse=True,  # type: ignore
)
for feature, importance in importances:  # type: ignore
    print(f"{feature:25s}: {importance:.4f}")

# Save model
model_path = "saved/speed_model.pkl"
joblib.dump(model, model_path)
print(f"\n✅ Model saved to {model_path}!")

# Save metadata
metadata = {
    "features": features,
    "model_type": "GradientBoosting_Incremental",
    "n_estimators": model.n_estimators,
    "r2_score": r2,
    "mae": mae,
    "rmse": rmse,
}
joblib.dump(metadata, "saved/model_metadata.pkl")
print("✅ Model metadata saved to saved/model_metadata.pkl")

# Prediction examples
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
for idx in sample_indices:
    actual = y_test[idx]
    predicted = model.predict(X_test[idx : idx + 1])[0]
    error = abs(actual - predicted)
    print(
        f"Actual: {actual:6.2f} km/h | Predicted: {predicted:6.2f} km/h | Error: {error:5.2f} km/h"
    )

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("Peak memory saved by using incremental training and optimizations")
