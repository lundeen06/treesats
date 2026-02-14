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

    args = parser.parse_args()

    if args.track:
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
