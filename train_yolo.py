#!/usr/bin/env python3
"""
Train YOLOv8 model for table tennis ball and racket detection.

This script:
1. Loads YOLOv8n pretrained weights
2. Trains on the prepared dataset (data.yaml)
3. Saves best weights to /weights/best.pt
4. Optionally exports to ONNX format for optimized inference

Usage:
    python scripts/train_yolo.py --epochs 100 --batch 16 --imgsz 640
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


# ===== CONFIGURATION =====
DATA_CONFIG = Path("data.yaml")
PRETRAINED_MODEL = "yolov8n.pt"  # Nano model (fastest)
PROJECT_NAME = "tabletennis_yolo"
RUN_NAME = "ball_racket_detector"
WEIGHTS_DIR = Path("weights")

# Training hyperparameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMG_SIZE = 640
DEFAULT_PATIENCE = 50  # Early stopping patience


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for table tennis detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PRETRAINED_MODEL,
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Input image size (default: {DEFAULT_IMG_SIZE})"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (e.g., '0' for GPU, 'cpu' for CPU). Auto-detect if empty."
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export trained model to ONNX format"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)"
    )
    return parser.parse_args()


def check_dataset(data_yaml_path):
    """
    Verify that dataset is properly prepared.

    Args:
        data_yaml_path: Path to data.yaml configuration file

    Returns:
        bool: True if dataset is valid, False otherwise
    """
    if not data_yaml_path.exists():
        print(f"❌ Error: Dataset configuration not found: {data_yaml_path}")
        print(f"   Make sure data.yaml exists in the project root")
        return False

    dataset_root = Path("dataset")

    # Check for required directories
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val"
    ]

    for dir_path in required_dirs:
        if not dir_path.exists() or not any(dir_path.iterdir()):
            print(f"❌ Error: Required directory empty or missing: {dir_path}")
            print(f"   Run 'python scripts/split_dataset.py' first")
            return False

    return True


def main():
    """Main execution function."""
    args = parse_arguments()

    # Check if dataset is ready
    print("🔍 Checking dataset...")
    if not check_dataset(DATA_CONFIG):
        return

    print("✓ Dataset configuration valid")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU detected, training on CPU (this will be slow)")

    # Load pretrained model
    print(f"\n📦 Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    # Display model info
    print(f"✓ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # Training configuration
    print(f"\n⚙️  Training configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Patience: {args.patience}")
    print(f"   Workers: {args.workers}")
    print(f"   Device: {args.device if args.device else 'auto'}")

    # Create weights directory
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Start training
    print(f"\n🏋️  Starting training...")
    print(f"=" * 60)

    try:
        results = model.train(
            data=str(DATA_CONFIG),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=RUN_NAME,
            project=PROJECT_NAME,
            patience=args.patience,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            device=args.device if args.device else None,
            workers=args.workers,
            pretrained=True,
            optimizer="auto",
            verbose=True,
            seed=42,
            deterministic=False,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=args.resume,
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,
            profile=False,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        )

        print(f"\n" + "=" * 60)
        print(f"✅ Training complete!")

        # Find best weights
        best_weights = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"

        if best_weights.exists():
            # Copy to main weights directory
            import shutil
            dest_weights = WEIGHTS_DIR / "best.pt"
            shutil.copy2(best_weights, dest_weights)
            print(f"✓ Best weights saved to: {dest_weights}")

            # Display training metrics
            print(f"\n📊 Training metrics:")
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                if 'metrics/mAP50(B)' in metrics:
                    print(f"   mAP50: {metrics['metrics/mAP50(B)']:.4f}")
                if 'metrics/mAP50-95(B)' in metrics:
                    print(f"   mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")

            # Export to ONNX if requested
            if args.export_onnx:
                print(f"\n📤 Exporting model to ONNX format...")
                try:
                    onnx_model = YOLO(str(dest_weights))
                    onnx_path = onnx_model.export(format="onnx", dynamic=True)
                    print(f"✓ ONNX model exported to: {onnx_path}")
                except Exception as e:
                    print(f"⚠️  ONNX export failed: {e}")

        else:
            print(f"⚠️  Warning: Best weights not found at {best_weights}")

        print(f"\n📝 Next step: Run inference with 'python scripts/detect_hit.py'")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        print(f"   Use --resume flag to continue from last checkpoint")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
