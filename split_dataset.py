#!/usr/bin/env python3
"""
Split annotated frames into train/val/test sets for YOLOv8 training.

This script:
1. Reads all annotated images and labels from dataset/images/all and dataset/labels/all
2. Randomly splits them: 70% train, 20% val, 10% test
3. Copies images and labels to respective directories
4. Validates label files for YOLO format correctness

Usage:
    python scripts/split_dataset.py --train 0.7 --val 0.2 --test 0.1
"""

import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm


# ===== CONFIGURATION =====
DATASET_ROOT = Path("dataset")
SOURCE_IMAGES_DIR = DATASET_ROOT / "images" / "all"
SOURCE_LABELS_DIR = DATASET_ROOT / "labels" / "all"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=TRAIN_RATIO,
        help=f"Training set ratio (default: {TRAIN_RATIO})"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=VAL_RATIO,
        help=f"Validation set ratio (default: {VAL_RATIO})"
    )
    parser.add_argument(
        "--test",
        type=float,
        default=TEST_RATIO,
        help=f"Test set ratio (default: {TEST_RATIO})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def validate_label_file(label_path, num_classes=2):
    """
    Validate YOLO format label file.

    YOLO format: class_id cx cy w h (normalized 0-1)

    Args:
        label_path: Path to label file
        num_classes: Number of expected classes

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"⚠️  {label_path.name}:{line_num} - Invalid format "
                      f"(expected 5 values, got {len(parts)})")
                return False

            class_id, cx, cy, w, h = parts

            # Validate class ID
            class_id = int(class_id)
            if class_id < 0 or class_id >= num_classes:
                print(f"⚠️  {label_path.name}:{line_num} - Invalid class_id "
                      f"{class_id} (expected 0-{num_classes-1})")
                return False

            # Validate normalized coordinates
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"⚠️  {label_path.name}:{line_num} - Coordinates out of "
                      f"range (cx={cx}, cy={cy}, w={w}, h={h})")
                return False

        return True

    except Exception as e:
        print(f"⚠️  Error reading {label_path.name}: {e}")
        return False


def get_paired_files(images_dir, labels_dir):
    """
    Find all image-label pairs.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels

    Returns:
        list: List of tuples (image_path, label_path)
    """
    paired_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))

    print(f"📷 Found {len(image_files)} image file(s) in {images_dir}")

    # Match with labels
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if label_path.exists():
            paired_files.append((img_path, label_path))
        else:
            print(f"⚠️  No label found for {img_path.name}")

    return paired_files


def copy_files(file_pairs, dest_images_dir, dest_labels_dir, desc="Copying"):
    """
    Copy image-label pairs to destination directories.

    Args:
        file_pairs: List of (image_path, label_path) tuples
        dest_images_dir: Destination directory for images
        dest_labels_dir: Destination directory for labels
        desc: Description for progress bar
    """
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in tqdm(file_pairs, desc=desc):
        # Copy image
        shutil.copy2(img_path, dest_images_dir / img_path.name)

        # Copy label
        shutil.copy2(label_path, dest_labels_dir / label_path.name)


def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate split ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Error: Split ratios must sum to 1.0 (got {total_ratio})")
        return

    print(f"🎲 Random seed: {args.seed}")
    random.seed(args.seed)

    # Check if source directories exist
    if not SOURCE_IMAGES_DIR.exists():
        print(f"❌ Error: Source images directory not found: {SOURCE_IMAGES_DIR}")
        print(f"   Run 'python scripts/extract_frames.py' first")
        return

    if not SOURCE_LABELS_DIR.exists():
        print(f"⚠️  Warning: Source labels directory not found: {SOURCE_LABELS_DIR}")
        print(f"   Please annotate your images and place labels in {SOURCE_LABELS_DIR}")
        print(f"   Label format: class_id cx cy w h (normalized 0-1)")
        return

    # Find paired files
    paired_files = get_paired_files(SOURCE_IMAGES_DIR, SOURCE_LABELS_DIR)

    if not paired_files:
        print("❌ No image-label pairs found!")
        return

    print(f"✓ Found {len(paired_files)} valid image-label pair(s)")

    # Validate label files
    print("\n🔍 Validating label files...")
    valid_pairs = []
    invalid_count = 0

    for img_path, label_path in tqdm(paired_files, desc="Validating"):
        if validate_label_file(label_path):
            valid_pairs.append((img_path, label_path))
        else:
            invalid_count += 1

    if invalid_count > 0:
        print(f"⚠️  Found {invalid_count} invalid label file(s)")
        response = input("Continue with valid files only? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    paired_files = valid_pairs

    # Shuffle files
    random.shuffle(paired_files)

    # Calculate split indices
    n_total = len(paired_files)
    n_train = int(n_total * args.train)
    n_val = int(n_total * args.val)
    n_test = n_total - n_train - n_val  # Remaining goes to test

    # Split data
    train_pairs = paired_files[:n_train]
    val_pairs = paired_files[n_train:n_train + n_val]
    test_pairs = paired_files[n_train + n_val:]

    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(train_pairs):4d} ({len(train_pairs)/n_total*100:.1f}%)")
    print(f"   Validation: {len(val_pairs):4d} ({len(val_pairs)/n_total*100:.1f}%)")
    print(f"   Test:       {len(test_pairs):4d} ({len(test_pairs)/n_total*100:.1f}%)")

    # Copy files to respective directories
    print("\n📁 Copying files to train/val/test directories...")

    copy_files(
        train_pairs,
        DATASET_ROOT / "images" / "train",
        DATASET_ROOT / "labels" / "train",
        desc="Train set"
    )

    copy_files(
        val_pairs,
        DATASET_ROOT / "images" / "val",
        DATASET_ROOT / "labels" / "val",
        desc="Val set"
    )

    copy_files(
        test_pairs,
        DATASET_ROOT / "images" / "test",
        DATASET_ROOT / "labels" / "test",
        desc="Test set"
    )

    print("\n✅ Dataset split complete!")
    print(f"\n📝 Next step: Train the model using 'python scripts/train_yolo.py'")


if __name__ == "__main__":
    main()
