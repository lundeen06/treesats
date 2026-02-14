"""
YOLO Training Pipeline for Satellite Detection
Optimized for DGX Spark (Grace Blackwell architecture)

Detects satellites, debris, and other orbital objects in 250x250 space images.
"""

from ultralytics import YOLO
import argparse
import torch
from pathlib import Path


def train(
    model_size: str = "yolov8m",
    epochs: int = 100,
    batch_size: int = 64,
    img_size: int = 256,
    resume: bool = False,
):
    """
    Train YOLO model for satellite detection.

    Args:
        model_size: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        batch_size: Batch size (DGX Spark can handle 64-128 with 128GB memory)
        img_size: Input image size (256 for 250x250 source images)
        resume: Resume from last checkpoint
    """
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize model
    weights_dir = Path(__file__).parent / "weights"
    if resume and (weights_dir / "satellite_detector" / "weights" / "last.pt").exists():
        model = YOLO(weights_dir / "satellite_detector" / "weights" / "last.pt")
        print("Resuming from last checkpoint...")
    else:
        model = YOLO(f"{model_size}.pt")  # Load pretrained weights
        print(f"Starting fresh with {model_size} pretrained weights...")

    # Dataset config path
    data_config = Path(__file__).parent / "config" / "dataset.yaml"

    # Train with settings optimized for small dot-like objects
    results = model.train(
        data=str(data_config),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=0,  # Use GPU 0
        workers=8,  # DGX Spark has plenty of CPU cores
        project=str(weights_dir),
        name="satellite_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Data augmentation - tuned for small objects in space images
        hsv_h=0.015,  # Slight hue variation
        hsv_s=0.7,  # Saturation variation
        hsv_v=0.4,  # Value/brightness variation
        degrees=180,  # Full rotation - satellites can be at any angle
        translate=0.1,  # Small translation
        scale=0.5,  # Scale variation
        flipud=0.5,  # Vertical flip
        fliplr=0.5,  # Horizontal flip
        mosaic=0.5,  # Reduced mosaic - helps preserve tiny dots
        mixup=0.1,  # Light mixup augmentation
        copy_paste=0.0,  # Disabled - not useful for point-like objects
    )

    print(f"\nTraining complete!")
    print(f"Best weights saved to: {weights_dir}/satellite_detector/weights/best.pt")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train satellite detection model")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m",
        help="Model size: yolov8n/s/m/l/x (default: yolov8m)",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs (default: 150)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Batch size - DGX Spark can handle 64-128 (default: 64)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Input image size (default: 256 for 250x250 images)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )

    args = parser.parse_args()

    train(
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        resume=args.resume,
    )
