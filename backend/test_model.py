import pandas as pd
from ai_magic import get_model, predict_mistakes

# Load model
model, metadata, imputer = get_model("saved/speed_model_v5.pkl")

# Load some test data
test_data = pd.read_csv("processed_data/merged_r1.csv", nrows=10000)

# Filter to one lap
lap_data = test_data[test_data['lap'] == 5]

print(f"Testing on {len(lap_data)} data points from lap 5")

# Predict
predictions, insights = predict_mistakes(model, lap_data, imputer, threshold_kmh=10)

# Show insights
print("\n=== AI Insights ===")
for insight in insights:
    print(f"  {insight}")

# Show sample predictions
print("\n=== Sample Predictions ===")
print(predictions[['speed', 'predicted_speed', 'speed_error', 'mistake_type']].head(20))

# Save results
predictions.to_csv("test_predictions.csv", index=False)
print("\n✅ Saved predictions to test_predictions.csv")