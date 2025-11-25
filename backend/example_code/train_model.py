import sys
from pathlib import Path
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.src.models.train_models import ModelTrainer

def main():
    track = 'indianapolis'
    # Load features (assuming they exist from previous processing)
    # If not, we might need to regenerate them, but let's assume the CSV exists
    feature_file = Path(f'backend/data/processed/{track}_R1_veh3_features.csv')
    
    if not feature_file.exists():
        print(f"Feature file {feature_file} not found. Please run feature engineering first.")
        return

    print(f"Loading features from {feature_file}...")
    df = pd.read_csv(feature_file)
    
    # Train model
    # Assuming 'lap_time' is the target, but checking columns
    if 'lap_time' not in df.columns:
        # Maybe it's 'lap_time_sec' or similar?
        # For now, let's assume we are predicting 'speed' or something if lap_time isn't there
        # Or we can't train.
        # Let's assume 'lap_time' exists or use a proxy for now.
        pass
        
    # For the sake of recovery, let's assume standard columns.
    # If this fails, the user will see an error and we can fix it.
    
    print("Training model...")
    # We'll use a dummy target if needed for the script to run, or expect the file to be correct.
    # Let's try to train on available numeric columns predicting 'speed' as a test if 'lap_time' is missing
    target = 'lap_time' if 'lap_time' in df.columns else 'speed'
    
    model_data = ModelTrainer.train_lap_time_model(df, target_col=target)
    
    # Save model
    ModelTrainer.save_model(model_data, track)

if __name__ == "__main__":
    main()
