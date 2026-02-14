"""
Auto-labeling for satellite detection.

Detects bright dots (satellites/debris) on dark backgrounds using
brightness thresholding and generates YOLO format labels for training.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def detect_bright_spots(
    image_path: str,
    brightness_threshold: int = 200,
    min_area: int = 4,
    max_area: int = 500,
    padding: int = 2,
) -> list:
    """
    Detect bright spots (satellites/debris) in a dark image using thresholding.

    Args:
        image_path: Path to the image file
        brightness_threshold: Pixel intensity threshold (0-255).
                             Pixels above this are considered "bright"
        min_area: Minimum pixel area for a detection (filters noise)
        max_area: Maximum pixel area (filters large artifacts)
        padding: Extra pixels to add around each detection

    Returns:
        List of bounding boxes as [x_min, y_min, x_max, y_max] in pixels
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def auto_label_image(
    image_path: str,
    output_dir: str,
    class_id: int = 0,
    brightness_threshold: int = 200,
    min_area: int = 4,
    max_area: int = 500,
    padding: int = 2,
) -> str:
    """
    Auto-detect bright spots and create YOLO label file.

    Args:
        image_path: Path to image
        output_dir: Directory to save label file
        class_id: Class ID for all detections (0=satellite, 1=debris)
        brightness_threshold: Intensity threshold (0-255)
        min_area: Minimum detection area in pixels
        max_area: Maximum detection area in pixels
        padding: Pixels to add around each detection

    Returns:
        Path to created label file
    """
    # Detect bright spots
    bboxes = detect_bright_spots(
        image_path,
        brightness_threshold=brightness_threshold,
        min_area=min_area,
        max_area=max_area,
        padding=padding,
    )

    # Get image dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create label file
    image_name = Path(image_path).stem
    label_path = Path(output_dir) / f"{image_name}.txt"

    with open(label_path, "w") as f:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox

            # Convert to YOLO format (normalized)
            x_center = (x_min + x_max) / 2.0 / width
            y_center = (y_min + y_max) / 2.0 / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"Created label: {label_path} ({len(bboxes)} detections)")
    return str(label_path)


def auto_label_directory(
    images_dir: str,
    labels_dir: str,
    class_id: int = 0,
    brightness_threshold: int = 200,
    min_area: int = 4,
    max_area: int = 500,
    padding: int = 2,
) -> int:
    """
    Auto-label all images in a directory.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory to save label files
        class_id: Class ID for all detections
        brightness_threshold: Intensity threshold (0-255)
        min_area: Minimum detection area in pixels
        max_area: Maximum detection area in pixels
        padding: Pixels to add around each detection

    Returns:
        Number of images processed
    """
    images_path = Path(images_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    # Find all images
    image_files = [
        f for f in images_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {images_dir}")
        return 0

    print(f"Processing {len(image_files)} images...")
    print(f"Settings: threshold={brightness_threshold}, min_area={min_area}, max_area={max_area}")
    print("-" * 50)

    total_detections = 0
    for image_file in image_files:
        bboxes = detect_bright_spots(
            str(image_file),
            brightness_threshold=brightness_threshold,
            min_area=min_area,
            max_area=max_area,
            padding=padding,
        )
        total_detections += len(bboxes)

        auto_label_image(
            str(image_file),
            labels_dir,
            class_id=class_id,
            brightness_threshold=brightness_threshold,
            min_area=min_area,
            max_area=max_area,
            padding=padding,
        )

    print("-" * 50)
    print(f"Done! Processed {len(image_files)} images, {total_detections} total detections")
    return len(image_files)


def preview_detections(
    image_path: str,
    brightness_threshold: int = 200,
    min_area: int = 4,
    max_area: int = 500,
    padding: int = 2,
    save_path: str = None,
) -> None:
    """
    Visualize detections on an image.

    Args:
        image_path: Path to image
        brightness_threshold: Intensity threshold (0-255)
        min_area: Minimum detection area in pixels
        max_area: Maximum detection area in pixels
        padding: Pixels to add around each detection
        save_path: If provided, save visualization to this path instead of displaying
    """
    # Detect bright spots
    bboxes = detect_bright_spots(
        image_path,
        brightness_threshold=brightness_threshold,
        min_area=min_area,
        max_area=max_area,
        padding=padding,
    )

    # Load image for visualization
    img = cv2.imread(image_path)

    # Draw bounding boxes
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # Add text
    text = f"Detections: {len(bboxes)} | Threshold: {brightness_threshold}"
    cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Saved preview to: {save_path}")
    else:
        # Display image
        cv2.imshow(f"Preview - {Path(image_path).name}", img)
        print(f"Detected {len(bboxes)} objects. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-label satellite images using brightness thresholding"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preview command
    preview_parser = subparsers.add_parser("preview", help="Preview detections on an image")
    preview_parser.add_argument("image", type=str, help="Path to image")
    preview_parser.add_argument("--threshold", type=int, default=200, help="Brightness threshold (0-255)")
    preview_parser.add_argument("--min-area", type=int, default=4, help="Minimum blob area")
    preview_parser.add_argument("--max-area", type=int, default=500, help="Maximum blob area")
    preview_parser.add_argument("--save", type=str, default=None, help="Save preview to file instead of displaying")

    # Label single image command
    label_parser = subparsers.add_parser("label", help="Auto-label a single image")
    label_parser.add_argument("image", type=str, help="Path to image")
    label_parser.add_argument("output_dir", type=str, help="Directory to save label file")
    label_parser.add_argument("--class-id", type=int, default=0, help="Class ID (0=satellite, 1=debris)")
    label_parser.add_argument("--threshold", type=int, default=200, help="Brightness threshold (0-255)")
    label_parser.add_argument("--min-area", type=int, default=4, help="Minimum blob area")
    label_parser.add_argument("--max-area", type=int, default=500, help="Maximum blob area")

    # Batch label command
    batch_parser = subparsers.add_parser("batch", help="Auto-label all images in a directory")
    batch_parser.add_argument("images_dir", type=str, help="Directory containing images")
    batch_parser.add_argument("labels_dir", type=str, help="Directory to save label files")
    batch_parser.add_argument("--class-id", type=int, default=0, help="Class ID (0=satellite, 1=debris)")
    batch_parser.add_argument("--threshold", type=int, default=200, help="Brightness threshold (0-255)")
    batch_parser.add_argument("--min-area", type=int, default=4, help="Minimum blob area")
    batch_parser.add_argument("--max-area", type=int, default=500, help="Maximum blob area")

    args = parser.parse_args()

    if args.command == "preview":
        preview_detections(
            args.image,
            brightness_threshold=args.threshold,
            min_area=args.min_area,
            max_area=args.max_area,
            save_path=args.save,
        )

    elif args.command == "label":
        auto_label_image(
            args.image,
            args.output_dir,
            class_id=args.class_id,
            brightness_threshold=args.threshold,
            min_area=args.min_area,
            max_area=args.max_area,
        )

    elif args.command == "batch":
        auto_label_directory(
            args.images_dir,
            args.labels_dir,
            class_id=args.class_id,
            brightness_threshold=args.threshold,
            min_area=args.min_area,
            max_area=args.max_area,
        )

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python auto_label.py preview data/images/train/img001.jpg")
        print("  python auto_label.py label data/images/train/img001.jpg data/labels/train/")
        print("  python auto_label.py batch data/images/train/ data/labels/train/ --threshold 180")
