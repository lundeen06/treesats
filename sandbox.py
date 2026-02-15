#!/usr/bin/env python3
"""
Sandbox script for testing the simulation and CV pipeline independently.

Commands:
    python sandbox.py simulate   - Run simulation, generate video
    python sandbox.py track      - Run CV tracking, print matrices
    python sandbox.py visualize  - Generate annotated output video
    python sandbox.py all        - Run all steps
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
SIMULATION_VIDEO = PROJECT_ROOT / "sat" / "data" / "star_tracker_sequence.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sandbox"


def run_simulate():
    """Run the simulation to generate star tracker video."""
    print("=" * 60)
    print("STEP 1: Running Simulation")
    print("=" * 60)
    
    # Run main.py --mode train
    cmd = [sys.executable, "main.py", "--mode", "train"]
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print("\n❌ Simulation failed!")
        return False
    
    # Check output
    if SIMULATION_VIDEO.exists():
        print(f"\n✓ Video generated: {SIMULATION_VIDEO}")
        print(f"  Size: {SIMULATION_VIDEO.stat().st_size / 1024:.1f} KB")
    else:
        print(f"\n❌ Video not found at: {SIMULATION_VIDEO}")
        return False
    
    return True


def run_track(max_frames: int = 10):
    """Run CV tracking and print the matrices."""
    print("=" * 60)
    print("STEP 2: Running CV Pipeline (Tracking)")
    print("=" * 60)
    
    if not SIMULATION_VIDEO.exists():
        print(f"❌ Video not found: {SIMULATION_VIDEO}")
        print("   Run 'python sandbox.py simulate' first.")
        return False
    
    print(f"Input video: {SIMULATION_VIDEO}\n")
    
    # Import the pipeline
    from sat.computer_vision.pipeline import SatelliteTracker
    
    tracker = SatelliteTracker()
    
    print(f"Processing video (showing first {max_frames} frames)...\n")
    print("-" * 60)
    
    for frame in tracker.stream_video(str(SIMULATION_VIDEO)):
        if frame.frame_id >= max_frames:
            print(f"... (stopped at {max_frames} frames, video has more)")
            break
        
        print(f"Frame {frame.frame_id}: {frame.num_satellites} satellites detected")
        
        if frame.num_satellites > 0:
            print("  Normalized matrix [sat_id, x, y]:")
            for row in frame.normalized:
                sat_id = int(row[0])
                x, y = row[1], row[2]
                print(f"    sat_id:{sat_id:2d}  x:{x:+.3f}  y:{y:+.3f}")
        print()
    
    print("-" * 60)
    print("✓ Tracking complete")
    return True


def run_visualize():
    """Generate annotated output video with tracking overlays."""
    print("=" * 60)
    print("STEP 3: Generating Tracked Video")
    print("=" * 60)
    
    if not SIMULATION_VIDEO.exists():
        print(f"❌ Video not found: {SIMULATION_VIDEO}")
        print("   Run 'python sandbox.py simulate' first.")
        return False
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_video = OUTPUT_DIR / "tracked_output.mp4"
    
    print(f"Input:  {SIMULATION_VIDEO}")
    print(f"Output: {output_video}\n")
    
    # Import and run
    from sat.computer_vision.inference import track_video_custom
    
    result_path = track_video_custom(
        video_path=str(SIMULATION_VIDEO),
        output_path=str(output_video),
        conf_threshold=0.25,
    )
    
    print(f"\n✓ Tracked video saved: {result_path}")
    
    # Also save tracking data to JSON
    from sat.computer_vision.inference import track_satellites_to_file
    
    json_output = OUTPUT_DIR / "tracking_data.json"
    track_satellites_to_file(
        video_path=str(SIMULATION_VIDEO),
        output_path=str(json_output),
    )
    print(f"✓ Tracking data saved: {json_output}")
    
    return True


def run_all():
    """Run all steps in sequence."""
    print("\n" + "=" * 60)
    print("SANDBOX: Running Full Pipeline")
    print("=" * 60 + "\n")
    
    # Step 1: Simulate
    if not run_simulate():
        return False
    print("\n")
    
    # Step 2: Track
    if not run_track():
        return False
    print("\n")
    
    # Step 3: Visualize
    if not run_visualize():
        return False
    
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Simulation video: {SIMULATION_VIDEO}")
    print(f"  Tracked video:    {OUTPUT_DIR / 'tracked_output.mp4'}")
    print(f"  Tracking JSON:    {OUTPUT_DIR / 'tracking_data.json'}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sandbox for testing simulation and CV pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sandbox.py simulate   # Run simulation, generate video
    python sandbox.py track      # Run CV tracking, print matrices  
    python sandbox.py visualize  # Generate annotated output video
    python sandbox.py all        # Run all steps
        """
    )
    
    parser.add_argument(
        "command",
        choices=["simulate", "track", "visualize", "all"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Max frames to display in track mode (default: 10)"
    )
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        success = run_simulate()
    elif args.command == "track":
        success = run_track(max_frames=args.max_frames)
    elif args.command == "visualize":
        success = run_visualize()
    elif args.command == "all":
        success = run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
