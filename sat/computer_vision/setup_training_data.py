"""
Setup script to prepare training data for YOLO model.

1. Auto-labels all images in sat/data/
2. Splits into train/val/test (80/10/10)
3. Copies to YOLO directory structure
"""

import shutil
import random
from pathlib import Path

# Paths
SOURCE_IMAGES = Path(__file__).parent.parent / "data"  # sat/data/
SOURCE_LABELS = SOURCE_IMAGES / "labels"  # sat/data/labels/
DEST_ROOT = Path(__file__).parent / "data"  # sat/computer_vision/data/

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def setup_training_data():
    """Split and copy labeled data into YOLO training structure."""
    
    # Find all images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = [
        f for f in SOURCE_IMAGES.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {SOURCE_IMAGES}")
        return
    
    # Check that labels exist
    if not SOURCE_LABELS.exists():
        print(f"Labels directory not found: {SOURCE_LABELS}")
        print("Run the auto-labeler first:")
        print(f"  python auto_label.py batch {SOURCE_IMAGES} {SOURCE_LABELS} --threshold 30 --min-area 0")
        return
    
    # Shuffle for random split
    random.seed(42)  # Reproducible split
    random.shuffle(image_files)
    
    # Calculate split indices
    n = len(image_files)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }
    
    print(f"Found {n} images")
    print(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print("-" * 50)
    
    # Copy files to YOLO structure
    for split_name, files in splits.items():
        img_dest = DEST_ROOT / "images" / split_name
        lbl_dest = DEST_ROOT / "labels" / split_name
        
        # Create directories
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        missing_labels = 0
        
        for img_file in files:
            # Copy image
            shutil.copy2(img_file, img_dest / img_file.name)
            
            # Copy corresponding label
            label_file = SOURCE_LABELS / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, lbl_dest / label_file.name)
                copied += 1
            else:
                missing_labels += 1
                # Create empty label file (no detections)
                (lbl_dest / f"{img_file.stem}.txt").touch()
        
        print(f"{split_name}: {copied} images with labels", end="")
        if missing_labels > 0:
            print(f" ({missing_labels} without labels - created empty)")
        else:
            print()
    
    print("-" * 50)
    print("Done! Data ready for training.")
    print(f"Images: {DEST_ROOT}/images/{{train,val,test}}/")
    print(f"Labels: {DEST_ROOT}/labels/{{train,val,test}}/")


if __name__ == "__main__":
    setup_training_data()
