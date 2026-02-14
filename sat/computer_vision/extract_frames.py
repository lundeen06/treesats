"""
Extract frames from animation videos, auto-label them, and organize into train/val/test splits.

Usage:
    python extract_frames.py                    # Process all animation_*.mp4 files
    python extract_frames.py --preview          # Preview first frame detections without saving
    python extract_frames.py --threshold 180    # Custom brightness threshold
"""

import cv2
import argparse
import random
from pathlib import Path
from auto_label import auto_label_image, preview_detections

# Directories
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def extract_frames_from_video(
    video_path: str,
    output_dir: Path,
    video_prefix: str,
) -> list:
    """
    Extract all frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        video_prefix: Prefix for frame filenames (e.g., "anim01")
    
    Returns:
        List of paths to extracted frame images
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"{video_prefix}_frame{frame_num:04d}.png"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_num += 1
    
    cap.release()
    print(f"  Extracted {frame_num} frames from {Path(video_path).name}")
    return frame_paths


def process_animations(
    brightness_threshold: int = 200,
    min_area: int = 4,
    max_area: int = 500,
    seed: int = 42,
    preview_only: bool = False,
) -> dict:
    """
    Process all animation_*.mp4 files: extract frames, auto-label, and split.
    
    Args:
        brightness_threshold: Pixel intensity threshold for detection
        min_area: Minimum blob area in pixels
        max_area: Maximum blob area in pixels
        seed: Random seed for reproducible splits
        preview_only: If True, only preview first frame detections
    
    Returns:
        Dictionary with counts of frames in each split
    """
    random.seed(seed)
    
    # Find all animation videos
    video_files = sorted(VIDEOS_DIR.glob("animation_*.mp4"))
    
    if not video_files:
        print("No animation_*.mp4 files found in", VIDEOS_DIR)
        return {}
    
    print(f"Found {len(video_files)} animation videos")
    print("=" * 60)
    
    # Preview mode: just show detections on first frame of first video
    if preview_only:
        cap = cv2.VideoCapture(str(video_files[0]))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            temp_path = SCRIPT_DIR / "_preview_temp.png"
            cv2.imwrite(str(temp_path), frame)
            print(f"Previewing detections on first frame of {video_files[0].name}")
            preview_detections(
                str(temp_path),
                brightness_threshold=brightness_threshold,
                min_area=min_area,
                max_area=max_area,
                save_path=str(SCRIPT_DIR / "preview_detection.png"),
            )
            temp_path.unlink()  # Clean up temp file
        return {}
    
    # Create temporary directory for all extracted frames
    temp_frames_dir = DATA_DIR / "_temp_frames"
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames from all videos
    all_frame_paths = []
    for video_path in video_files:
        # Create prefix from video name: animation_01.mp4 -> anim01
        video_num = video_path.stem.split("_")[1]  # "01", "02", etc.
        video_prefix = f"anim{video_num}"
        
        frame_paths = extract_frames_from_video(
            video_path,
            temp_frames_dir,
            video_prefix,
        )
        all_frame_paths.extend(frame_paths)
    
    print("=" * 60)
    print(f"Total frames extracted: {len(all_frame_paths)}")
    
    # Shuffle and split
    random.shuffle(all_frame_paths)
    
    n_total = len(all_frame_paths)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_frames = all_frame_paths[:n_train]
    val_frames = all_frame_paths[n_train:n_train + n_val]
    test_frames = all_frame_paths[n_train + n_val:]
    
    splits = {
        "train": train_frames,
        "val": val_frames,
        "test": test_frames,
    }
    
    print(f"\nSplit: train={len(train_frames)}, val={len(val_frames)}, test={len(test_frames)}")
    print("=" * 60)
    
    # Process each split
    total_detections = 0
    for split_name, frame_paths in splits.items():
        print(f"\nProcessing {split_name} split ({len(frame_paths)} frames)...")
        
        images_split_dir = IMAGES_DIR / split_name
        labels_split_dir = LABELS_DIR / split_name
        images_split_dir.mkdir(parents=True, exist_ok=True)
        labels_split_dir.mkdir(parents=True, exist_ok=True)
        
        split_detections = 0
        for frame_path in frame_paths:
            # Move frame to split directory
            dest_image_path = images_split_dir / frame_path.name
            frame_path.rename(dest_image_path)
            
            # Auto-label the frame
            label_path = auto_label_image(
                str(dest_image_path),
                str(labels_split_dir),
                class_id=0,  # satellite
                brightness_threshold=brightness_threshold,
                min_area=min_area,
                max_area=max_area,
            )
            
            # Count detections
            with open(label_path, "r") as f:
                split_detections += len(f.readlines())
        
        total_detections += split_detections
        print(f"  {split_name}: {split_detections} detections")
    
    # Clean up temp directory
    if temp_frames_dir.exists():
        temp_frames_dir.rmdir()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nLabeled frames saved to:")
    print(f"  Images: {IMAGES_DIR}/{{train,val,test}}/")
    print(f"  Labels: {LABELS_DIR}/{{train,val,test}}/")
    print(f"\nTotal frames: {n_total}")
    print(f"Total detections: {total_detections}")
    print(f"\nSettings used:")
    print(f"  Brightness threshold: {brightness_threshold}")
    print(f"  Min blob area: {min_area}px")
    print(f"  Max blob area: {max_area}px")
    
    # Count existing + new data
    print("\n" + "-" * 60)
    print("DATASET SUMMARY (existing + new)")
    print("-" * 60)
    for split in ["train", "val", "test"]:
        img_count = len(list((IMAGES_DIR / split).glob("*.png"))) + len(list((IMAGES_DIR / split).glob("*.jpg")))
        lbl_count = len(list((LABELS_DIR / split).glob("*.txt")))
        print(f"  {split}: {img_count} images, {lbl_count} labels")
    
    return {
        "train": len(train_frames),
        "val": len(val_frames),
        "test": len(test_frames),
        "total_detections": total_detections,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from animation videos and auto-label them"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Brightness threshold for detection (0-255, default: 200)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=4,
        help="Minimum blob area in pixels (default: 4)",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=500,
        help="Maximum blob area in pixels (default: 500)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview detections on first frame without processing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split (default: 42)",
    )
    
    args = parser.parse_args()
    
    process_animations(
        brightness_threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        seed=args.seed,
        preview_only=args.preview,
    )
