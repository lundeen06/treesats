"""
Inference script for satellite detection.

Run predictions on new images using trained YOLO weights.
"""

from ultralytics import YOLO
from pathlib import Path
import argparse
import json
import cv2
import torch
import numpy as np
from datetime import datetime


def get_device():
    """Get the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return 0  # CUDA GPU
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


def load_model(weights_path: str = None) -> YOLO:
    """
    Load trained YOLO model.

    Args:
        weights_path: Path to weights file. If None, uses default best.pt

    Returns:
        Loaded YOLO model
    """
    if weights_path is None:
        weights_path = (
            Path(__file__).parent
            / "weights"
            / "satellite_detector"
            / "weights"
            / "best.pt"
        )

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}\n"
            "Train the model first with: python train.py"
        )

    model = YOLO(str(weights_path))
    print(f"Loaded model from: {weights_path}")
    return model


def detect(
    source: str,
    weights_path: str = None,
    conf_threshold: float = 0.25,
    save: bool = True,
    show: bool = False,
    save_centers: bool = False,
    output_dir: str = None,
) -> list:
    """
    Run satellite detection on image(s).

    Args:
        source: Path to image, directory, or video
        weights_path: Path to model weights (default: weights/satellite_detector/weights/best.pt)
        conf_threshold: Confidence threshold (0-1)
        save: Save annotated images to runs/detect/
        show: Display results (requires display)
        save_centers: Save center coordinates to JSON file
        output_dir: Directory for center coordinates output (default: output/detections/)

    Returns:
        List of detection results
    """
    model = load_model(weights_path)

    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=save,
        show=show,
        imgsz=256,  # Match training size
        device=get_device(),
    )

    # Print summary and collect centers
    all_detections = []
    
    for result in results:
        boxes = result.boxes
        image_name = Path(result.path).stem
        
        centers = []
        detections = []
        
        if len(boxes) > 0:
            print(f"\nImage: {result.path}")
            print(f"  Detections: {len(boxes)}")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                
                # Calculate center
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)
                
                centers.append([x_center, y_center])
                detections.append({
                    "x": x_center,
                    "y": y_center,
                    "confidence": round(conf, 4),
                    "class": cls_name,
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                })
                
                print(f"    - {cls_name}: {conf:.2f} @ center ({x_center}, {y_center})")
        else:
            print(f"\nImage: {result.path} - No detections")
        
        # Save centers to file if requested
        if save_centers:
            if output_dir is None:
                output_dir = Path(__file__).parent / "output" / "detections"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                "image": result.path,
                "num_detections": len(centers),
                "centers": centers,  # List of [x, y] coordinates
                "detections": detections,  # Full details
            }
            
            output_file = output_dir / f"{image_name}.json"
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  Centers saved to: {output_file}")
            
        all_detections.append({
            "image": result.path,
            "centers": centers,
            "detections": detections,
        })

    return results


def detect_single(image_path: str, weights_path: str = None, conf_threshold: float = 0.25) -> dict:
    """
    Run detection on a single image and return structured results.

    Args:
        image_path: Path to image file
        weights_path: Path to model weights
        conf_threshold: Confidence threshold

    Returns:
        Dict with 'image_path', 'detections' list, and 'count'
    """
    model = load_model(weights_path)

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        show=False,
        imgsz=256,
        device=get_device(),
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_id": cls_id,
                "class_name": result.names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox_xyxy": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                "bbox_xywh": box.xywh[0].tolist(),  # [x_center, y_center, w, h]
            })

    return {
        "image_path": image_path,
        "detections": detections,
        "count": len(detections),
    }


def track(
    source: str,
    weights_path: str = None,
    conf_threshold: float = 0.25,
    tracker: str = "botsort.yaml",
    save: bool = True,
    show: bool = False,
) -> list:
    """
    Track satellites across video frames or image sequences.

    Uses BoT-SORT or ByteTrack for multi-object tracking, assigning
    persistent IDs to satellites/debris as they move across frames.

    Args:
        source: Path to video file or directory of sequential images
        weights_path: Path to model weights
        conf_threshold: Confidence threshold (0-1)
        tracker: Tracker config - "botsort.yaml" or "bytetrack.yaml"
        save: Save annotated video/images
        show: Display results in window

    Returns:
        List of tracking results with track IDs
    """
    model = load_model(weights_path)

    results = model.track(
        source=source,
        conf=conf_threshold,
        tracker=tracker,
        save=save,
        show=show,
        imgsz=256,
        device=get_device(),
        persist=True,  # Persist tracks between frames
    )

    # Print tracking summary
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0 and boxes.id is not None:
            print(f"\nFrame: {result.path}")
            print(f"  Tracked objects: {len(boxes)}")
            for i, box in enumerate(boxes):
                track_id = int(boxes.id[i]) if boxes.id is not None else -1
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                print(f"    - ID {track_id}: {cls_name} ({conf:.2f}) @ [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

    return results


def track_and_save(
    source: str,
    output_dir: str = None,
    run_name: str = None,
    weights_path: str = None,
    conf_threshold: float = 0.25,
    tracker: str = "botsort.yaml",
) -> str:
    """
    Track satellites and save results to organized run folder.

    For each frame, outputs:
    - Annotated image with bounding boxes and track IDs
    - JSON file with N×3 tracking matrix [track_id, x_pixel, y_pixel]

    Args:
        source: Path to video file or directory of sequential images
        output_dir: Base output directory (default: output/runs/tracking)
        run_name: Name for this run (default: auto-increment run1, run2, ...)
        weights_path: Path to model weights
        conf_threshold: Confidence threshold (0-1)
        tracker: Tracker config - "botsort.yaml" or "bytetrack.yaml"

    Returns:
        Path to the run folder containing all outputs
    """
    # Set default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "runs" / "tracking"
    else:
        output_dir = Path(output_dir)

    # Auto-generate run name if not provided
    if run_name is None:
        run_num = 1
        while (output_dir / f"run{run_num}").exists():
            run_num += 1
        run_name = f"run{run_num}"

    # Create run directory
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tracking results to: {run_dir}")

    # Load model
    model = load_model(weights_path)

    # Run tracking
    results = model.track(
        source=source,
        conf=conf_threshold,
        tracker=tracker,
        save=False,  # We'll save manually
        show=False,
        imgsz=256,
        device=get_device(),
        persist=True,
        stream=True,  # Stream results for memory efficiency
    )

    # Process each frame
    frame_count = 0
    total_detections = 0
    start_time = datetime.now()

    for result in results:
        frame_id = frame_count
        frame_name = f"frame_{frame_id:04d}"

        # Get original image
        img = result.orig_img.copy()

        # Build tracking data
        tracking_matrix = []
        detections = []

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            for i, box in enumerate(boxes):
                track_id = int(boxes.id[i])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]

                # Get center position in pixels
                xywh = box.xywh[0].tolist()
                x_center = int(xywh[0])
                y_center = int(xywh[1])

                # Get bounding box for drawing
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                # Add to tracking matrix (N×3: track_id, x, y)
                tracking_matrix.append([track_id, x_center, y_center])

                # Add to detections list (full details)
                detections.append({
                    "track_id": track_id,
                    "x": x_center,
                    "y": y_center,
                    "confidence": round(conf, 4),
                    "class": cls_name,
                    "bbox": [x1, y1, x2, y2],
                })

                # Draw on image
                color = (0, 255, 0)  # Green
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                label = f"ID:{track_id}"
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save annotated frame
        frame_image_path = run_dir / f"{frame_name}.jpg"
        cv2.imwrite(str(frame_image_path), img)

        # Save frame JSON
        frame_data = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "num_satellites": len(tracking_matrix),
            "tracking_matrix": tracking_matrix,
            "detections": detections,
        }

        frame_json_path = run_dir / f"{frame_name}.json"
        with open(frame_json_path, "w") as f:
            json.dump(frame_data, f, indent=2)

        total_detections += len(tracking_matrix)
        frame_count += 1

        # Progress update
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count} frames...")

    # Save metadata
    end_time = datetime.now()
    metadata = {
        "run_name": run_name,
        "source": str(source),
        "created_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "total_frames": frame_count,
        "total_detections": total_detections,
        "settings": {
            "conf_threshold": conf_threshold,
            "tracker": tracker,
            "weights": str(weights_path) if weights_path else "default",
        },
    }

    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTracking complete!")
    print(f"  Frames: {frame_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Output: {run_dir}")

    return str(run_dir)


def track_satellites(
    video_path: str,
    weights_path: str = None,
    conf_threshold: float = 0.25,
    tracker: str = "botsort.yaml",
    verbose: bool = True,
) -> dict:
    """
    Track satellites across video frames and return N×3 matrices per frame.

    Each matrix row contains [track_id, x_center, y_center] where track_id
    persists across frames for the same satellite.

    Args:
        video_path: Path to video file
        weights_path: Path to model weights (default: uses best.pt)
        conf_threshold: Confidence threshold (0-1)
        tracker: Tracker config - "botsort.yaml" or "bytetrack.yaml"
        verbose: Print progress updates

    Returns:
        Dictionary containing:
        {
            "frames": [
                {
                    "frame_id": int,
                    "pixel_matrix": np.ndarray,      # N×3 [track_id, x_px, y_px]
                    "normalized_matrix": np.ndarray, # N×3 [track_id, x_norm, y_norm]
                },
                ...
            ],
            "image_size": (width, height),
            "total_frames": int,
            "total_tracks": int,
        }

    Example:
        >>> result = track_satellites("video.mp4")
        >>> frame_0 = result["frames"][0]
        >>> pixel_coords = frame_0["pixel_matrix"]      # [[1, 125, 80], [2, 200, 150], ...]
        >>> normalized = frame_0["normalized_matrix"]   # [[1, 0.0, -0.36], [2, 0.6, 0.2], ...]
    """
    # Load model
    model = load_model(weights_path)

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if verbose:
        print(f"Processing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {total_frames}")

    # Run tracking
    results = model.track(
        source=video_path,
        conf=conf_threshold,
        tracker=tracker,
        save=False,
        show=False,
        imgsz=256,
        device=get_device(),
        persist=True,
        stream=True,  # Stream for memory efficiency
        verbose=False,
    )

    # Process each frame
    frames_data = []
    all_track_ids = set()
    frame_count = 0

    for result in results:
        # Build matrices for this frame
        pixel_rows = []
        normalized_rows = []

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            for i, box in enumerate(boxes):
                track_id = int(boxes.id[i])
                all_track_ids.add(track_id)

                # Get center position from xywh (center_x, center_y, width, height)
                xywh = box.xywh[0].tolist()
                x_center = xywh[0]
                y_center = xywh[1]

                # Pixel coordinates
                pixel_rows.append([track_id, x_center, y_center])

                # Normalized coordinates:
                # X: -1 (left) to +1 (right)
                # Y: -1 (bottom) to +1 (top)
                x_norm = (x_center / (width / 2)) - 1.0
                y_norm = 1.0 - (y_center / (height / 2))  # Flip Y axis

                normalized_rows.append([track_id, x_norm, y_norm])

        # Convert to numpy arrays
        if pixel_rows:
            pixel_matrix = np.array(pixel_rows, dtype=np.float32)
            normalized_matrix = np.array(normalized_rows, dtype=np.float32)
        else:
            # Empty frame - return empty N×3 arrays
            pixel_matrix = np.empty((0, 3), dtype=np.float32)
            normalized_matrix = np.empty((0, 3), dtype=np.float32)

        frames_data.append({
            "frame_id": frame_count,
            "pixel_matrix": pixel_matrix,
            "normalized_matrix": normalized_matrix,
        })

        frame_count += 1

        if verbose and frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    if verbose:
        print(f"\nTracking complete!")
        print(f"  Total frames: {frame_count}")
        print(f"  Unique tracks: {len(all_track_ids)}")

    return {
        "frames": frames_data,
        "image_size": (width, height),
        "total_frames": frame_count,
        "total_tracks": len(all_track_ids),
    }


def track_video_custom(
    video_path: str,
    output_path: str = None,
    output_dir: str = None,
    weights_path: str = None,
    conf_threshold: float = 0.25,
    tracker: str = "botsort.yaml",
    font_scale: float = 0.25,
) -> str:
    """
    Track satellites and create custom annotated video with normalized coordinates.

    Labels show: sat_id:{id} x:{x_norm:.2f} y:{y_norm:.2f}
    where x,y are normalized to [-1, 1].
    X: -1 (left) to +1 (right)
    Y: -1 (bottom) to +1 (top)

    Args:
        video_path: Path to input video
        output_path: Path to save output video (default: auto-generated)
        output_dir: Directory to save output (overrides output_path auto-generation)
        weights_path: Path to model weights
        conf_threshold: Confidence threshold (0-1)
        tracker: Tracker config
        font_scale: Font size scale (default: 0.25 for small text)

    Returns:
        Path to output video
    """
    video_path = Path(video_path)

    # Set output path
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_tracked.mp4"
    else:
        # Auto-generate with run folder
        base_output_dir = Path(__file__).parent / "output" / "tracked_videos"
        
        # Find next run number
        run_num = 1
        while (base_output_dir / f"run{run_num}").exists():
            run_num += 1
        
        output_dir = base_output_dir / f"run{run_num}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_tracked.mp4"

    # Load model
    model = load_model(weights_path)

    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Processing: {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    # Create video writer (use mp4v codec for better compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Run tracking
    results = model.track(
        source=str(video_path),
        conf=conf_threshold,
        tracker=tracker,
        save=False,
        show=False,
        imgsz=256,
        device=get_device(),
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_count = 0
    for result in results:
        # Get original frame
        frame = result.orig_img.copy()

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            for i, box in enumerate(boxes):
                track_id = int(boxes.id[i])

                # Get bounding box
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                # Get center in pixels
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2

                # Convert to normalized coordinates [-1, 1]
                # X: -1 (left) to +1 (right)
                # Y: -1 (bottom) to +1 (top)
                x_norm = (x_center / (width / 2)) - 1.0
                y_norm = 1.0 - (y_center / (height / 2))  # Flip Y axis

                # Draw bounding box (thin green line)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Create label with small font
                label = f"sat_id:{track_id} x:{x_norm:.2f} y:{y_norm:.2f}"
                
                # Calculate text position (above the box)
                text_y = max(y1 - 2, 8)
                
                # Draw text with small font
                cv2.putText(
                    frame, 
                    label, 
                    (x1, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 255, 0), 
                    1,
                    cv2.LINE_AA
                )

        # Write frame
        out.write(frame)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    out.release()

    print(f"\nVideo saved to: {output_path}")
    print(f"  Total frames: {frame_count}")

    return str(output_path)


def track_satellites_to_file(
    video_path: str,
    output_path: str = None,
    weights_path: str = None,
    conf_threshold: float = 0.25,
    tracker: str = "botsort.yaml",
) -> str:
    """
    Track satellites and save matrices to a single JSON file.

    Args:
        video_path: Path to video file
        output_path: Path to save JSON output (default: auto-generated)
        weights_path: Path to model weights
        conf_threshold: Confidence threshold (0-1)
        tracker: Tracker config

    Returns:
        Path to saved JSON file
    """
    result = track_satellites(
        video_path=video_path,
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        tracker=tracker,
        verbose=True,
    )

    # Set default output path
    if output_path is None:
        output_dir = Path(__file__).parent / "output" / "tracking_matrices"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem
        output_path = output_dir / f"{video_name}_tracking.json"

    # Convert numpy arrays to lists for JSON serialization
    serializable_frames = []
    for frame in result["frames"]:
        serializable_frames.append({
            "frame_id": frame["frame_id"],
            "pixel_matrix": frame["pixel_matrix"].tolist(),
            "normalized_matrix": frame["normalized_matrix"].tolist(),
        })

    output_data = {
        "source": str(video_path),
        "image_size": result["image_size"],
        "total_frames": result["total_frames"],
        "total_tracks": result["total_tracks"],
        "frames": serializable_frames,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved tracking matrices to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run satellite detection inference")
    parser.add_argument(
        "source",
        type=str,
        help="Path to image, directory of images, or video file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: weights/satellite_detector/weights/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save annotated images",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in window",
    )
    parser.add_argument(
        "--save-centers",
        action="store_true",
        help="Save center coordinates of detections to JSON file",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable tracking mode for video/sequences",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort.yaml",
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="Tracker algorithm (default: botsort.yaml)",
    )
    parser.add_argument(
        "--save-run",
        action="store_true",
        help="Save tracking results to output/runs/tracking/ with N×3 matrices",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the tracking run (default: auto-increment run1, run2, ...)",
    )
    parser.add_argument(
        "--matrices",
        action="store_true",
        help="Output N×3 tracking matrices (pixel and normalized) to JSON",
    )

    args = parser.parse_args()

    if args.matrices:
        # Output tracking matrices
        track_satellites_to_file(
            video_path=args.source,
            weights_path=args.weights,
            conf_threshold=args.conf,
            tracker=args.tracker,
        )
    elif args.track:
        if args.save_run:
            # Use track_and_save for organized output with N×3 matrices
            track_and_save(
                source=args.source,
                run_name=args.run_name,
                weights_path=args.weights,
                conf_threshold=args.conf,
                tracker=args.tracker,
            )
        else:
            track(
                source=args.source,
                weights_path=args.weights,
                conf_threshold=args.conf,
                tracker=args.tracker,
                save=not args.no_save,
                show=args.show,
            )
    else:
        detect(
            source=args.source,
            weights_path=args.weights,
            conf_threshold=args.conf,
            save=not args.no_save,
            show=args.show,
            save_centers=args.save_centers,
        )
