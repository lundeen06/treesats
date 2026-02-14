# Satellite Detection Training Guide

Train a YOLOv8 model to detect satellites and debris in 250x250 space images.

---

## Prerequisites

### Hardware
- **Recommended:** NVIDIA DGX Spark (128GB memory) or similar GPU
- **Minimum:** Any CUDA-capable GPU with 8GB+ VRAM

### Software
```bash
pip install ultralytics torch torchvision pillow
```

---

## Step 1: Prepare Your Images

### Image Requirements
- **Size:** 250×250 pixels
- **Format:** JPG, PNG, or TIFF
- **Content:** Black space background with bright dot-like satellites/debris

### Directory Structure
Place your images in the following folders:

```
computer_vision/data/
├── images/
│   ├── train/          # ~70% of your images
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/            # ~15% of your images
│   │   └── ...
│   └── test/           # ~15% of your images (held out for final evaluation)
│       └── ...
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## Step 2: Label Your Images

You need to create a `.txt` label file for each image with bounding boxes around satellites/debris.

### Option A: Use a Labeling Tool (Recommended)

Use one of these tools to annotate your images:

| Tool | URL | Notes |
|------|-----|-------|
| **Roboflow** | [roboflow.com](https://roboflow.com) | Easiest, exports YOLO format directly |
| **CVAT** | [cvat.ai](https://cvat.ai) | Free, open source, powerful |
| **Label Studio** | [labelstud.io](https://labelstud.io) | Flexible, self-hosted |

**Export settings:**
- Format: **YOLO** or **YOLOv8**
- Place exported images in `data/images/{split}/`
- Place exported labels in `data/labels/{split}/`

### Option B: Manual Labeling with Python

```python
from sat.computer_vision import create_yolo_label

# For each image, define bounding boxes as [x_min, y_min, x_max, y_max] in pixels
annotations = [
    {"class": "satellite", "bbox": [100, 80, 108, 88]},   # 8x8 pixel box
    {"class": "debris", "bbox": [200, 150, 205, 155]},    # 5x5 pixel box
]

create_yolo_label(
    image_path="data/images/train/space_001.jpg",
    annotations=annotations,
    output_dir="data/labels/train"
)
```

### YOLO Label Format

Each label file (`.txt`) contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized 0-1. Example for a 250×250 image:
```
0 0.416 0.336 0.032 0.032
1 0.810 0.610 0.020 0.020
```

### Classes
| ID | Name | Description |
|----|------|-------------|
| 0 | satellite | Active/intact satellites |
| 1 | debris | Space debris fragments |

---

## Step 3: Validate Your Dataset

Before training, verify your data is properly set up:

```bash
cd sat/computer_vision
python prepare_data.py validate
```

**Expected output:**
```
============================================================
DATASET VALIDATION REPORT
============================================================

TRAIN SPLIT:
  Images: 800
  Labels: 800

VAL SPLIT:
  Images: 100
  Labels: 100

TEST SPLIT:
  Images: 100
  Labels: 100

============================================================
SUMMARY
============================================================
Total images: 1000
Total labels: 1000
Total objects: 2500

Objects per class:
  satellite: 1500
  debris: 1000
```

**Fix any issues before proceeding:**
- Images without labels → add labels or remove images
- Labels without images → remove orphan label files
- Class imbalance → collect more samples of underrepresented class

---

## Step 4: Train the Model

### Basic Training

```bash
cd sat/computer_vision
python train.py --epochs 150 --batch 64
```

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolov8m` | Model size: `yolov8n` (fast) → `yolov8x` (accurate) |
| `--epochs` | `150` | Training iterations over full dataset |
| `--batch` | `64` | Images per batch (increase for more GPU memory) |
| `--img-size` | `256` | Input image size |
| `--resume` | off | Continue from last checkpoint |

### Recommended Settings by Hardware

| Hardware | Command |
|----------|---------|
| **DGX Spark** | `python train.py --model yolov8l --epochs 200 --batch 128` |
| **RTX 4090** | `python train.py --model yolov8m --epochs 150 --batch 64` |
| **RTX 3080** | `python train.py --model yolov8s --epochs 150 --batch 32` |
| **Laptop GPU** | `python train.py --model yolov8n --epochs 100 --batch 16` |

### Training Output

Training creates:
```
weights/satellite_detector/
├── weights/
│   ├── best.pt          # Best model (lowest val loss)
│   └── last.pt          # Latest checkpoint
├── results.csv          # Metrics per epoch
├── confusion_matrix.png
├── results.png          # Loss/mAP curves
└── ...
```

### Resume Interrupted Training

```bash
python train.py --resume
```

---

## Step 5: Evaluate the Model

### Run on Test Set

```bash
python inference.py data/images/test/ --conf 0.25
```

### Key Metrics to Check

After training, check `weights/satellite_detector/results.png`:

| Metric | Good Value | Meaning |
|--------|------------|---------|
| **mAP50** | > 0.7 | Mean Average Precision at 50% IoU |
| **mAP50-95** | > 0.5 | Stricter mAP across IoU thresholds |
| **Precision** | > 0.8 | % of detections that are correct |
| **Recall** | > 0.8 | % of actual objects detected |

### Test on Individual Images

```bash
python inference.py path/to/image.jpg
```

Output:
```
Image: path/to/image.jpg
  Detections: 3
    - satellite: 0.92 @ [100, 80, 108, 88]
    - satellite: 0.87 @ [50, 200, 58, 208]
    - debris: 0.76 @ [180, 120, 185, 125]
```

---

## Step 6: Use the Model

### Detection (Single Images)

```python
from sat.computer_vision import detect_single

result = detect_single("new_image.jpg")
print(f"Found {result['count']} objects")
for det in result['detections']:
    print(f"  {det['class_name']}: {det['confidence']:.2f}")
```

### Tracking (Video/Sequences)

```bash
python inference.py video.mp4 --track
```

Or in Python:
```python
from sat.computer_vision import track

results = track("video.mp4", tracker="botsort.yaml")
```

---

## Troubleshooting

### Low Accuracy

| Problem | Solution |
|---------|----------|
| mAP < 0.5 | Need more training data (500+ images recommended) |
| Missing small objects | Reduce `conf` threshold to 0.1 |
| False positives | Increase `conf` threshold to 0.5 |
| Class imbalance | Collect more samples of minority class |

### Training Issues

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch` size |
| Training too slow | Reduce `--img-size` to 224 |
| Loss not decreasing | Check labels are correct, try lower learning rate |

### Validation Errors

| Error | Fix |
|-------|-----|
| "No labels found" | Check label files exist in `data/labels/` |
| "Image/label mismatch" | Ensure each image has a matching `.txt` file |
| "Invalid class ID" | Classes must be 0 or 1 (satellite/debris) |

---

## Quick Reference

```bash
# Validate dataset
python prepare_data.py validate

# Train model
python train.py --epochs 150 --batch 64

# Resume training
python train.py --resume

# Run detection
python inference.py image.jpg

# Run tracking
python inference.py video.mp4 --track

# Show annotation example
python prepare_data.py example
```
