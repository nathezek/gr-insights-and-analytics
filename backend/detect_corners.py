"""
Corner detection utility for analyzing telemetry data
Based on steering angle and speed analysis
"""
import pandas as pd
import numpy as np


def detect_corners(df, lap_number=None, steer_threshold=5.0, speed_threshold=50.0):
    """
    Detect corners in telemetry data based on steering angle and speed
    
    Args:
        df: DataFrame with telemetry data
        lap_number: Specific lap to analyze (None for all laps)
        steer_threshold: Minimum steering angle (degrees) to consider a corner
        speed_threshold: Minimum speed (km/h) to filter out slow sections
        
    Returns:
        List of detected corners with metadata
    """
    
    # Filter for specific lap if requested
    if lap_number is not None and 'lap' in df.columns:
        valid_laps = df[df['lap'] == lap_number].copy()
    else:
        valid_laps = df.copy()
    
    if valid_laps.empty:
        print(f"No data found for lap {lap_number}")
        return []
    
    print(f"Processing {len(valid_laps)} rows...")
    
    # Sort by distance
    if 'Laptrigger_lapdist_dls' in valid_laps.columns:
        valid_laps = valid_laps.sort_values('Laptrigger_lapdist_dls')
        distance_col = 'Laptrigger_lapdist_dls'
    elif 'distance_m' in valid_laps.columns:
        valid_laps = valid_laps.sort_values('distance_m')
        distance_col = 'distance_m'
    else:
        print("No distance column found")
        return []
    
    # Interpolate missing values
    valid_laps[distance_col] = valid_laps[distance_col].interpolate(method='linear')
    valid_laps['speed'] = valid_laps['speed'].interpolate(method='linear')
    valid_laps['Steering_Angle'] = valid_laps['Steering_Angle'].interpolate(method='linear')
    
    # Drop remaining NaNs
    valid_laps = valid_laps.dropna(subset=[distance_col, 'Steering_Angle', 'speed'])
    
    # Smooth steering
    valid_laps['steering_smooth'] = valid_laps['Steering_Angle'].rolling(window=20, center=True).mean().fillna(0)
    
    # Detect corners
    corners = []
    in_corner = False
    current_corner = {}
    
    GAP_THRESHOLD = 30  # meters
    MIN_CORNER_LEN = 10  # meters
    last_dist = -1000
    
    for idx, row in valid_laps.iterrows():
        dist = row[distance_col]
        steer = row['steering_smooth']
        speed = row['speed']
        
        is_corner_steer = abs(steer) > steer_threshold
        is_fast_enough = speed > speed_threshold
        
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
                # Check if direction changed
                if direction != current_corner['direction']:
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
            if in_corner:
                if (dist - last_dist) < GAP_THRESHOLD:
                    pass
                else:
                    in_corner = False
                    corners.append(current_corner)
                    current_corner = {}
    
    # Append last corner if open
    if in_corner:
        corners.append(current_corner)
    
    # Post-process corners
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
    
    valid_corners.sort(key=lambda x: x['start'])
    
    print(f"\nDetected {len(valid_corners)} corners")
    
    return valid_corners


def print_corner_definitions(corners, track_name="TRACK"):
    """Print corners in track_definitions.py format"""
    print(f"\n{track_name.upper()}_CORNERS = {{")
    
    for i, r in enumerate(corners):
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
