import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN

def detect_corners():
    # Load data
    file_path = r'c:\Users\Eyuale\Desktop\gr_insights\backend\data\processed\indianapolis_R1_veh3.parquet'
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Filter for a representative lap (Lap 3 seems standard length ~3900m)
    valid_laps = df[df['lap'] == 3].copy()
    
    if valid_laps.empty:
        print("Lap 3 not found. Using Lap 2.")
        valid_laps = df[df['lap'] == 2].copy()

    print(f"Processing {len(valid_laps)} rows from representative lap...")
    
    # Sort by index (assuming time order)
    valid_laps = valid_laps.sort_index()
    
    # Interpolate missing values
    valid_laps['distance_m'] = valid_laps['distance_m'].interpolate(method='linear')
    valid_laps['speed'] = valid_laps['speed'].interpolate(method='linear')
    valid_laps['steering_angle'] = valid_laps['steering_angle'].interpolate(method='linear')
    
    # Drop remaining NaNs at start/end
    valid_laps = valid_laps.dropna(subset=['distance_m', 'steering_angle', 'speed'])
    
    # Smooth steering
    valid_laps['steering_smooth'] = valid_laps['steering_angle'].rolling(window=20, center=True).mean().fillna(0)
    
    print("Steering stats after smoothing:")
    print(valid_laps['steering_smooth'].describe())
    
    # Define thresholds
    STEER_THRESHOLD = 5.0  # Degrees
    SPEED_THRESHOLD = 50.0 # km/h
    
    # Segment-based detection
    corners = []
    in_corner = False
    current_corner = {}
    
    # Parameters
    GAP_THRESHOLD = 30 # meters to merge same-direction segments
    MIN_CORNER_LEN = 10 # meters to be valid
    
    last_dist = -1000
    
    for idx, row in valid_laps.iterrows():
        dist = row['distance_m']
        steer = row['steering_smooth']
        speed = row['speed']
        
        is_corner_steer = abs(steer) > STEER_THRESHOLD
        is_fast_enough = speed > SPEED_THRESHOLD
        
        if is_corner_steer and is_fast_enough:
            direction = "Left" if steer > 0 else "Right"
            
            if not in_corner:
                # Start new corner
                in_corner = True
                current_corner = {
                    'start': dist,
                    'end': dist,
                    'max_steer': steer,
                    'min_speed': speed,
                    'direction': direction,
                    'points': [(dist, steer, speed)]
                }
            else:
                # Already in corner
                # Check if direction changed (Chicane transition)
                if direction != current_corner['direction']:
                    # End current corner, start new one immediately
                    corners.append(current_corner)
                    current_corner = {
                        'start': dist,
                        'end': dist,
                        'max_steer': steer,
                        'min_speed': speed,
                        'direction': direction,
                        'points': [(dist, steer, speed)]
                    }
                else:
                    # Continue corner
                    current_corner['end'] = dist
                    current_corner['points'].append((dist, steer, speed))
                    if abs(steer) > abs(current_corner['max_steer']):
                        current_corner['max_steer'] = steer
                    if speed < current_corner['min_speed']:
                        current_corner['min_speed'] = speed
                        
            last_dist = dist
            
        else:
            # Not steering enough
            if in_corner:
                # Check if this is just a small gap
                if (dist - last_dist) < GAP_THRESHOLD:
                    pass
                else:
                    # Gap too big, close corner
                    in_corner = False
                    corners.append(current_corner)
                    current_corner = {}

    # Append last corner if open
    if in_corner:
        corners.append(current_corner)
        
    # Post-process corners
    # 1. Filter short corners
    valid_corners = []
    for c in corners:
        length = c['end'] - c['start']
        if length > MIN_CORNER_LEN:
            # Find apex
            points = c['points']
            best_p = max(points, key=lambda x: abs(x[1]))
            c['apex'] = best_p[0]
            
            # Determine type
            if c['min_speed'] < 80:
                c['type'] = "slow"
            elif c['min_speed'] < 140:
                c['type'] = "medium"
            elif c['min_speed'] < 200:
                c['type'] = "fast"
            else:
                c['type'] = "flat_out"
                
            valid_corners.append(c)
            
    results = valid_corners
    results.sort(key=lambda x: x['start'])
    
    print(f"\nDetected {len(results)} potential corners.")
    
    # Print table
    print(f"{'ID':<5} {'Start':<10} {'Apex':<10} {'End':<10} {'Dir':<10} {'Type':<10} {'Steer':<10}")
    print("-" * 70)
    for i, r in enumerate(results):
        print(f"{i:<5} {r['start']:<10.1f} {r['apex']:<10.1f} {r['end']:<10.1f} {r['direction']:<10} {r['type']:<10} {r['max_steer']:<10.1f}")

    # Generate Python dictionary
    print("\nGenerated Dictionary for track_definitions.py:")
    print("INDIANAPOLIS_CORNERS = {")
    
    # Map to T1..T14
    for i, r in enumerate(results):
        corner_id = f"T{i+1}"
        print(f"    '{corner_id}': {{")
        print(f"        'name': 'Turn {i+1}',")
        print(f"        'start': {int(r['start'])},")
        print(f"        'end': {int(r['end'])},")
        print(f"        'apex': {int(r['apex'])},")
        print(f"        'type': '{r['type']}',")
        print(f"        'direction': '{r['direction']}',")
        print(f"    }},")
    print("}")

if __name__ == "__main__":
    detect_corners()
