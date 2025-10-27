#!/usr/bin/env python3
"""
Extract frames from table tennis videos for dataset preparation.

This script:
1. Scans the /videos directory for all video files
2. Extracts frames at a specified FPS (default: 30)
3. Optionally filters frames using motion detection to discard static frames
4. Saves extracted frames to /dataset/images/all/

Usage:
    python scripts/extract_frames.py --fps 30 --motion-filter
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


# ===== CONFIGURATION =====
INPUT_VIDEO_DIR = Path("videos")
OUTPUT_FRAME_DIR = Path("dataset/images/all")
DEFAULT_FPS = 30
MOTION_THRESHOLD = 500  # Minimum pixel diff to consider motion present


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from table tennis videos"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second to extract (default: {DEFAULT_FPS})"
    )
    parser.add_argument(
        "--motion-filter",
        action="store_true",
        help="Enable motion-based filtering to discard static frames"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=INPUT_VIDEO_DIR,
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_FRAME_DIR,
        help="Directory to save extracted frames"
    )
    return parser.parse_args()


def has_motion(prev_frame, curr_frame, threshold=MOTION_THRESHOLD):
    """
    Detect if there's significant motion between two consecutive frames.

    Args:
        prev_frame: Previous frame (grayscale)
        curr_frame: Current frame (grayscale)
        threshold: Minimum pixel difference to consider motion

    Returns:
        bool: True if motion detected, False otherwise
    """
    if prev_frame is None:
        return True

    # Compute absolute difference
    diff = cv2.absdiff(prev_frame, curr_frame)

    # Calculate total difference
    motion_score = np.sum(diff)

    return motion_score > threshold


def extract_frames_from_video(video_path, output_dir, target_fps=30,
                              use_motion_filter=False):
    """
    Extract frames from a single video file.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        target_fps: Target frames per second to extract
        use_motion_filter: Whether to filter out static frames

    Returns:
        int: Number of frames extracted
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval to achieve target FPS
    frame_interval = max(1, int(original_fps / target_fps))

    print(f"\n📹 Processing: {video_path.name}")
    print(f"   Original FPS: {original_fps:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Extracting every {frame_interval} frame(s)")

    frame_count = 0
    saved_count = 0
    prev_gray = None

    # Create progress bar
    pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.name}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Only process frames at the specified interval
        if frame_count % frame_interval == 0:
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply motion filter if enabled
            if use_motion_filter:
                if not has_motion(prev_gray, gray):
                    frame_count += 1
                    pbar.update(1)
                    continue

            # Generate output filename
            video_name = video_path.stem
            frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
            output_path = output_dir / frame_filename

            # Save frame
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

            # Update previous frame for motion detection
            prev_gray = gray

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"   ✓ Saved {saved_count} frames")

    return saved_count


def main():
    """Main execution function."""
    args = parse_arguments()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(args.video_dir.glob(f'*{ext}'))
        video_files.extend(args.video_dir.glob(f'*{ext.upper()}'))

    if not video_files:
        print(f"❌ No video files found in {args.video_dir}")
        print(f"   Supported formats: {', '.join(video_extensions)}")
        return

    print(f"🎬 Found {len(video_files)} video file(s)")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"⚙️  Target FPS: {args.fps}")
    print(f"🎯 Motion filter: {'Enabled' if args.motion_filter else 'Disabled'}")

    # Process each video
    total_frames_extracted = 0

    for video_path in video_files:
        frames_extracted = extract_frames_from_video(
            video_path,
            args.output_dir,
            target_fps=args.fps,
            use_motion_filter=args.motion_filter
        )
        total_frames_extracted += frames_extracted

    print(f"\n✅ Extraction complete!")
    print(f"   Total frames extracted: {total_frames_extracted}")
    print(f"   Saved to: {args.output_dir}")
    print(f"\n📝 Next step: Annotate frames and run 'python scripts/split_dataset.py'")


if __name__ == "__main__":
    main()
