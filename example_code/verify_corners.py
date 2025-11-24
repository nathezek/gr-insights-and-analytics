import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.data.reference.track_definitions import get_track_corners

try:
    corners = get_track_corners('indianapolis')
    print(f"Successfully loaded {len(corners)} corners for Indianapolis.")
    for k, v in corners.items():
        print(f"{k}: {v['name']} ({v['start']}-{v['end']})")
except Exception as e:
    print(f"Error: {e}")
