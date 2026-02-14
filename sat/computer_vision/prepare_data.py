"""
Data preparation utilities for satellite detection dataset.

Helps convert annotations to YOLO format and validate dataset integrity.
Designed for 250x250 space images with small dot-like satellites.
"""

import os
from pathlib import Path
from PIL import Image


# Class mapping - matches config/dataset.yaml
CLASSES = {
    "satellite": 0,
    "debris": 1,
}

# Reverse mapping for display
CLASS_NAMES = {v: k for k, v in CLASSES.items()}


def convert_bbox_to_yolo(img_width: int, img_height: int, bbox: list) -> list:
    """
    Convert bounding box from pixel coordinates to YOLO format.

    Args:
        img_width: Image width in pixels (250 for our images)
        img_height: Image height in pixels (250 for our images)
        bbox: Bounding box as [x_min, y_min, x_max, y_max] in pixels

    Returns:
        YOLO format: [x_center, y_center, width, height] (all normalized 0-1)

    Example:
        # For a satellite at pixel (100, 80) to (108, 88) in a 250x250 image:
        >>> convert_bbox_to_yolo(250, 250, [100, 80, 108, 88])
        [0.416, 0.336, 0.032, 0.032]
    """
    x_min, y_min, x_max, y_max = bbox

    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return [x_center, y_center, width, height]


def convert_yolo_to_bbox(img_width: int, img_height: int, yolo_bbox: list) -> list:
    """
    Convert YOLO format back to pixel coordinates.

    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        yolo_bbox: [x_center, y_center, width, height] normalized 0-1

    Returns:
        Pixel coordinates: [x_min, y_min, x_max, y_max]
    """
    x_center, y_center, w, h = yolo_bbox

    x_min = int((x_center - w / 2) * img_width)
    y_min = int((y_center - h / 2) * img_height)
    x_max = int((x_center + w / 2) * img_width)
    y_max = int((y_center + h / 2) * img_height)

    return [x_min, y_min, x_max, y_max]


def create_yolo_label(image_path: str, annotations: list, output_dir: str) -> str:
    """
    Create YOLO format label file for an image.

    Args:
        image_path: Path to the image file
        annotations: List of dicts with 'class' and 'bbox' keys
                    class: one of 'satellite', 'debris', 'rocket_body', 'unknown_object'
                    bbox: [x_min, y_min, x_max, y_max] in pixels
        output_dir: Directory to save label file

    Returns:
        Path to created label file

    Example:
        >>> annotations = [
        ...     {"class": "satellite", "bbox": [100, 150, 108, 158]},
        ...     {"class": "debris", "bbox": [200, 50, 206, 56]},
        ... ]
        >>> create_yolo_label("data/images/train/img001.jpg", annotations, "data/labels/train")
    """
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Create label filename (same name as image, but .txt)
    image_name = Path(image_path).stem
    label_path = Path(output_dir) / f"{image_name}.txt"

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(label_path, "w") as f:
        for ann in annotations:
            class_name = ann["class"]
            class_id = CLASSES.get(class_name, 1)  # Default to debris if unknown
            bbox = ann["bbox"]

            yolo_bbox = convert_bbox_to_yolo(img_width, img_height, bbox)

            # Write: class_id x_center y_center width height
            line = f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_bbox)}\n"
            f.write(line)

    print(f"Created label: {label_path}")
    return str(label_path)


def read_yolo_label(label_path: str) -> list:
    """
    Read a YOLO format label file.

    Args:
        label_path: Path to .txt label file

    Returns:
        List of dicts with 'class_id', 'class_name', and 'yolo_bbox'
    """
    annotations = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]
                annotations.append({
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, "unknown"),
                    "yolo_bbox": yolo_bbox,
                })

    return annotations


def validate_dataset(data_dir: str = None):
    """
    Validate that all images have corresponding labels and vice versa.
    Reports missing labels, missing images, and dataset statistics.

    Args:
        data_dir: Path to data directory (defaults to ./data relative to this file)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)

    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    total_images = 0
    total_labels = 0
    total_objects = 0
    class_counts = {name: 0 for name in CLASSES.keys()}

    for split in ["train", "val", "test"]:
        images_dir = data_dir / "images" / split
        labels_dir = data_dir / "labels" / split

        if not images_dir.exists():
            print(f"\n[WARNING] {images_dir} does not exist")
            continue

        image_files = set(
            p.stem
            for p in images_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        )

        if not labels_dir.exists():
            print(f"\n[WARNING] {labels_dir} does not exist")
            label_files = set()
        else:
            label_files = set(p.stem for p in labels_dir.glob("*.txt"))

        missing_labels = image_files - label_files
        missing_images = label_files - image_files

        print(f"\n{split.upper()} SPLIT:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")

        total_images += len(image_files)
        total_labels += len(label_files)

        if missing_labels:
            print(f"  [!] Images without labels: {len(missing_labels)}")
            for name in list(missing_labels)[:5]:
                print(f"      - {name}")
            if len(missing_labels) > 5:
                print(f"      ... and {len(missing_labels) - 5} more")

        if missing_images:
            print(f"  [!] Labels without images: {len(missing_images)}")
            for name in list(missing_images)[:5]:
                print(f"      - {name}")

        # Count objects per class
        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                annotations = read_yolo_label(str(label_file))
                total_objects += len(annotations)
                for ann in annotations:
                    class_name = ann["class_name"]
                    if class_name in class_counts:
                        class_counts[class_name] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Total objects: {total_objects}")
    print("\nObjects per class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    if total_images == 0:
        print("\n[INFO] No images found. Add your 250x250 space images to:")
        print("       data/images/train/  (training images)")
        print("       data/images/val/    (validation images)")
        print("       data/images/test/   (test images)")


def example_annotation():
    """
    Print example of how to annotate an image for this pipeline.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    ANNOTATION EXAMPLE                            ║
    ╚══════════════════════════════════════════════════════════════════╝

    For a 250x250 space image with satellites visible as dots:

    1. MANUAL ANNOTATION (Python):
    ───────────────────────────────
    from prepare_data import create_yolo_label

    # For image: data/images/train/space_001.jpg
    # With a satellite at pixels (100, 80) to (108, 88)
    # And debris at pixels (200, 150) to (205, 155)

    annotations = [
        {"class": "satellite", "bbox": [100, 80, 108, 88]},
        {"class": "debris", "bbox": [200, 150, 205, 155]},
    ]

    create_yolo_label(
        image_path="data/images/train/space_001.jpg",
        annotations=annotations,
        output_dir="data/labels/train"
    )

    # This creates: data/labels/train/space_001.txt


    2. YOLO LABEL FORMAT (what gets saved):
    ───────────────────────────────────────
    # Each line: class_id x_center y_center width height
    # All coordinates normalized 0-1

    0 0.416000 0.336000 0.032000 0.032000
    1 0.810000 0.610000 0.020000 0.020000


    3. RECOMMENDED LABELING TOOLS:
    ──────────────────────────────
    - Roboflow (roboflow.com) - Can export YOLO format directly
    - CVAT (cvat.ai) - Free, open source
    - Label Studio (labelstud.io) - Flexible

    Export as "YOLO" or "YOLOv8" format and place files in:
    - Images -> data/images/train/ or data/images/val/
    - Labels -> data/labels/train/ or data/labels/val/


    4. CLASSES AVAILABLE:
    ─────────────────────
    0: satellite - Active/intact satellites
    1: debris    - Space debris fragments

    """)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "validate":
            validate_dataset()
        elif command == "example":
            example_annotation()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python prepare_data.py [validate|example]")
    else:
        print("Usage: python prepare_data.py [validate|example]")
        print("\nCommands:")
        print("  validate  - Check dataset integrity")
        print("  example   - Show annotation examples")
