"""
Example usage of track_satellites_normalized()
"""

import sys
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sat.computer_vision.pipeline import track_satellites_normalized


def main():
    video_path = Path(__file__).parent / "data/videos/animation_01.mp4"
    
    for frame in track_satellites_normalized(str(video_path)):
        print(f"Frame {frame['frame_id']}")
        
        if not frame["satellites"]:
            print("  No satellites detected")
            continue
        
        for sat_id, (x, y) in frame["satellites"].items():
            print(f"  Satellite {sat_id}: x={x:.3f}, y={y:.3f}")


if __name__ == "__main__":
    main()
