# Auto-Labeling Guide

Automatically detect and label bright satellites/debris in dark space images using brightness thresholding.

---

## How It Works

```
1. Load 250x250 image
2. Convert to grayscale
3. Threshold: pixels > 200 brightness = "bright"
4. Find connected blobs (contours)
5. Filter by size (ignore noise and artifacts)
6. Create bounding box around each blob
7. Write YOLO format label file
```

This works because satellites appear as **bright white dots** on a **black background** — high contrast makes simple thresholding very effective.

---

## Quick Start

```bash
cd sat/computer_vision

# 1. Preview detections on a sample image first
python auto_label.py preview data/images/train/sample.jpg

# 2. If detections look good, batch label all images
python auto_label.py batch data/images/train/ data/labels/train/
python auto_label.py batch data/images/val/ data/labels/val/
python auto_label.py batch data/images/test/ data/labels/test/

# 3. Validate the dataset
python prepare_data.py validate
```

---

## Commands

### Preview Detections

Visualize what will be detected before creating labels:

```bash
python auto_label.py preview <image_path> [options]
```

**Examples:**
```bash
# Display preview window
python auto_label.py preview data/images/train/img001.jpg

# Save preview to file (no window)
python auto_label.py preview data/images/train/img001.jpg --save preview.png

# Test different threshold
python auto_label.py preview data/images/train/img001.jpg --threshold 150
```

### Label Single Image

```bash
python auto_label.py label <image_path> <output_dir> [options]
```

**Example:**
```bash
python auto_label.py label data/images/train/img001.jpg data/labels/train/
# Creates: data/labels/train/img001.txt
```

### Batch Label Directory

Label all images in a folder at once:

```bash
python auto_label.py batch <images_dir> <labels_dir> [options]
```

**Example:**
```bash
python auto_label.py batch data/images/train/ data/labels/train/
# Output:
# Processing 500 images...
# Settings: threshold=200, min_area=4, max_area=500
# --------------------------------------------------
# Created label: data/labels/train/img001.txt (3 detections)
# Created label: data/labels/train/img002.txt (1 detections)
# ...
# --------------------------------------------------
# Done! Processed 500 images, 1247 total detections
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 200 | Brightness cutoff (0-255). Pixels above this are detected. |
| `--min-area` | 4 | Minimum blob size in pixels. Filters out noise. |
| `--max-area` | 500 | Maximum blob size in pixels. Filters out large artifacts. |
| `--class-id` | 0 | Class ID for detections (0=satellite, 1=debris) |

---

## Tuning the Threshold

The `--threshold` parameter is the most important setting. It determines what counts as "bright enough" to be a satellite.

### Finding the Right Threshold

1. **Start with preview:**
   ```bash
   python auto_label.py preview image.jpg --threshold 200
   ```

2. **If missing dim satellites:** Lower the threshold
   ```bash
   python auto_label.py preview image.jpg --threshold 150
   ```

3. **If detecting noise/artifacts:** Raise the threshold
   ```bash
   python auto_label.py preview image.jpg --threshold 220
   ```

### Threshold Guidelines

| Image Type | Suggested Threshold |
|------------|---------------------|
| High contrast (bright satellites, pure black background) | 200-220 |
| Medium contrast | 150-200 |
| Low contrast (dim satellites, some background noise) | 100-150 |

---

## Filtering by Size

### `--min-area` (default: 4)

Minimum blob size in pixels. Blobs smaller than this are ignored.

- **Increase** if you're getting false positives from noise
- **Decrease** if missing very small/dim satellites

```bash
# Ignore blobs smaller than 10 pixels
python auto_label.py batch images/ labels/ --min-area 10
```

### `--max-area` (default: 500)

Maximum blob size in pixels. Blobs larger than this are ignored.

- **Increase** if satellites appear as larger blobs in your images
- **Decrease** if you're detecting large artifacts (stars, lens flares)

```bash
# Ignore blobs larger than 100 pixels
python auto_label.py batch images/ labels/ --max-area 100
```

---

## Labeling Different Classes

By default, all detections are labeled as class 0 (satellite). To label as debris:

```bash
# Label all detections as debris (class 1)
python auto_label.py batch images/ labels/ --class-id 1
```

### Strategy for Multiple Classes

If you need both satellites and debris labels, you have two options:

**Option 1: Manual separation**
- Separate your images into `satellites/` and `debris/` folders
- Label each folder with the appropriate class ID

```bash
python auto_label.py batch satellites/ labels/ --class-id 0
python auto_label.py batch debris/ labels/ --class-id 1
```

**Option 2: Label all as one class, refine later**
- Auto-label everything as "satellite"
- Manually edit specific label files to change class IDs

---

## Python API

Use auto-labeling programmatically:

```python
from sat.computer_vision import (
    detect_bright_spots,
    auto_label_image,
    auto_label_directory,
    preview_detections,
)

# Detect bright spots and get bounding boxes
bboxes = detect_bright_spots(
    "image.jpg",
    brightness_threshold=200,
    min_area=4,
    max_area=500,
)
print(f"Found {len(bboxes)} objects")
# bboxes = [[x_min, y_min, x_max, y_max], ...]

# Create YOLO label for single image
label_path = auto_label_image(
    "data/images/train/img001.jpg",
    "data/labels/train/",
    class_id=0,
    brightness_threshold=200,
)

# Batch label directory
count = auto_label_directory(
    "data/images/train/",
    "data/labels/train/",
    brightness_threshold=200,
)
print(f"Labeled {count} images")

# Preview detections (save to file)
preview_detections(
    "image.jpg",
    brightness_threshold=200,
    save_path="preview.png",
)
```

---

## Workflow: From Raw Images to Training

```bash
# 1. Place your 250x250 space images in the data folders
#    data/images/train/  (70% of images)
#    data/images/val/    (15% of images)
#    data/images/test/   (15% of images)

# 2. Preview a few images to find the right threshold
python auto_label.py preview data/images/train/sample1.jpg
python auto_label.py preview data/images/train/sample2.jpg --threshold 180

# 3. Batch label all splits
python auto_label.py batch data/images/train/ data/labels/train/ --threshold 200
python auto_label.py batch data/images/val/ data/labels/val/ --threshold 200
python auto_label.py batch data/images/test/ data/labels/test/ --threshold 200

# 4. Validate the dataset
python prepare_data.py validate

# 5. Train the YOLO model
python train.py --epochs 150 --batch 64
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Missing satellites | Lower `--threshold` (try 150) |
| Too many false detections | Raise `--threshold` (try 220) |
| Detecting noise specks | Increase `--min-area` (try 6-10) |
| Missing small satellites | Decrease `--min-area` (try 2) |
| Detecting stars/large objects | Decrease `--max-area` (try 100) |
| No detections at all | Check image format, ensure it's grayscale-compatible |

---

## Output Format

Each label file is a `.txt` with the same name as the image:

```
image001.jpg  →  image001.txt
```

**Label file contents (YOLO format):**
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized 0-1. Example:
```
0 0.416000 0.336000 0.032000 0.032000
0 0.810000 0.610000 0.024000 0.024000
```

This means:
- 2 satellites detected
- First at center (0.416, 0.336) with size (0.032, 0.032)
- Second at center (0.810, 0.610) with size (0.024, 0.024)
