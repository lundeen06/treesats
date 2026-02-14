"""
Inference script for satellite detection.

Run predictions on new images using trained YOLO weights.
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


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
) -> list:
    """
    Run satellite detection on image(s).

    Args:
        source: Path to image, directory, or video
        weights_path: Path to model weights (default: weights/satellite_detector/weights/best.pt)
        conf_threshold: Confidence threshold (0-1)
        save: Save annotated images to runs/detect/
        show: Display results (requires display)

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
        device=0,  # Use GPU
    )

    # Print summary
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            print(f"\nImage: {result.path}")
            print(f"  Detections: {len(boxes)}")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                print(f"    - {cls_name}: {conf:.2f} @ [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
        else:
            print(f"\nImage: {result.path} - No detections")

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
        device=0,
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
        device=0,
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

    args = parser.parse_args()

    if args.track:
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
        )
