import joblib
import os


def get_model():
    """Load saved model or raise error if not exists."""
    model_path = "speed_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError("Run train_model.py first to create speed_model.pkl")


def predict_mistakes(model, new_lap_data):
    """Use model to predict expected speed and find differences (mistakes)."""
    features = [
        "aps",
        "pbrake_f",
        "pbrake_r",
        "Steering_Angle",
        "gear",
        "accy_can",
    ]  # Add 'AIR_TEMP' if available
    predicted_speed = model.predict(new_lap_data[features])

    new_lap_data["predicted_speed"] = predicted_speed
    new_lap_data["speed_diff"] = new_lap_data["speed"] - predicted_speed

    mistakes = new_lap_data[new_lap_data["speed_diff"] < -10]
    insights = []
    for _, row in mistakes.iterrows():
        insights.append(
            f"At distance {row.get('Laptrigger_lapdist_dls', 'unknown')}m: Braked too hard? Lost ~{-row['speed_diff']:.1f} km/h."
        )

    return new_lap_data, insights
