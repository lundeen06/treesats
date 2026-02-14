"""
Computer vision algorithms for star tracking and pose estimation.

Satellite Detection Pipeline
============================
Train and run YOLO models for detecting satellites, debris, and other
orbital objects in 250x250 space images.

Usage:
    # Training
    python -m sat.computer_vision.train --epochs 150 --batch 64

    # Inference
    python -m sat.computer_vision.inference path/to/image.jpg

    # Data preparation
    python -m sat.computer_vision.prepare_data validate
"""

from .train import train
from .inference import load_model, detect, detect_single, track
from .prepare_data import (
    CLASSES,
    CLASS_NAMES,
    convert_bbox_to_yolo,
    convert_yolo_to_bbox,
    create_yolo_label,
    read_yolo_label,
    validate_dataset,
)
from .auto_label import (
    detect_bright_spots,
    auto_label_image,
    auto_label_directory,
    preview_detections,
)

__all__ = [
    # Training
    "train",
    # Inference
    "load_model",
    "detect",
    "detect_single",
    "track",
    # Data preparation
    "CLASSES",
    "CLASS_NAMES",
    "convert_bbox_to_yolo",
    "convert_yolo_to_bbox",
    "create_yolo_label",
    "read_yolo_label",
    "validate_dataset",
    # Auto-labeling
    "detect_bright_spots",
    "auto_label_image",
    "auto_label_directory",
    "preview_detections",
]
