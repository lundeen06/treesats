"""
Visualize auto-labeling detections on video.

Creates an output video showing bounding boxes from the brightness-threshold
auto-labeler (no ML model involved).
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def detect_bright_spots_frame(
    frame: np.ndarray,
    brightness_threshold: int = 200,
    min_area: int = 4,
    max_area: int = 500,
    padding: int = 2,
) -> list:
    """
    Detect bright spots in a frame using thresholding.
    
    Returns:
        List of bounding boxes as [x_min, y_min, x_max, y_max]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Threshold to find bright spots
    _, binary = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    # Find connected components (blobs)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if area < min_area or area > max_area:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(width, x + w + padding)
        y_max = min(height, y + h + padding)

        bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes


def visualize_video(
    video_path: str,
    output_path: str = None,
    brightness_threshold: int = 200,
    min_area: int = 4,
    max_area: int = 500,
    padding: int = 2,
) -> str:
    """
    Create visualization video showing auto-label detections.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (default: auto-generated)
        brightness_threshold: Pixel intensity threshold (0-255)
        min_area: Minimum blob area in pixels
        max_area: Maximum blob area in pixels
        padding: Pixels to add around each detection
    
    Returns:
        Path to output video
    """
    video_path = Path(video_path)
    
    if output_path is None:
        output_dir = Path(__file__).parent / "output" / "autolabel_viz"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_autolabel.avi"
    
    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}")
    print(f"  Settings: threshold={brightness_threshold}, min_area={min_area}, max_area={max_area}")
    print()
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_num = 0
    total_detections = 0
    frames_with_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect bright spots
        bboxes = detect_bright_spots_frame(
            frame,
            brightness_threshold=brightness_threshold,
            min_area=min_area,
            max_area=max_area,
            padding=padding,
        )
        
        # Draw bounding boxes
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            # Draw center point
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
        
        # Add info overlay
        info_text = f"Frame: {frame_num} | Detections: {len(bboxes)} | Threshold: {brightness_threshold}"
        cv2.putText(frame, info_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Write frame
        out.write(frame)
        
        # Stats
        total_detections += len(bboxes)
        if len(bboxes) > 0:
            frames_with_detections += 1
        
        frame_num += 1
        if frame_num % 50 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print()
    print("=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {output_path}")
    print(f"Total frames: {frame_num}")
    print(f"Frames with detections: {frames_with_detections} ({100*frames_with_detections/frame_num:.1f}%)")
    print(f"Total detections: {total_detections}")
    print(f"Avg detections per frame: {total_detections/frame_num:.2f}")
    
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize auto-labeling detections on video (no ML model)"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save output video (default: output/autolabel_viz/<name>_autolabel.avi)",
    )
    parser.add_argument(
        "-t", "--threshold",
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
    
    args = parser.parse_args()
    
    visualize_video(
        video_path=args.video,
        output_path=args.output,
        brightness_threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
    )
