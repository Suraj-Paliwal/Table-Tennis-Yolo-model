#!/usr/bin/env python3
"""
Real-time table tennis hit detection using trained YOLOv8 model.

This script:
1. Loads the trained YOLOv8 model
2. Processes video frames or live camera feed
3. Detects ball and racket using YOLO
4. Analyzes geometric proximity and velocity changes
5. Detects hit events (ball-racket collisions)
6. Displays live overlay with hit indicators
7. Logs hit events to CSV file

Usage:
    # Live camera
    python scripts/detect_hit.py --source 0

    # Video file
    python scripts/detect_hit.py --source path/to/video.mp4

    # Save output video
    python scripts/detect_hit.py --source video.mp4 --save-video
"""

import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque
from ultralytics import YOLO


# ===== CONFIGURATION VARIABLES =====
DIST_THRESHOLD = 40          # Pixel distance threshold for hit detection
VELOCITY_MIN = 2.0           # Minimum velocity to consider valid collision
SMOOTH_WINDOW = 5            # Number of frames for trajectory smoothing
CONFIDENCE_THRESHOLD = 0.25  # Minimum detection confidence
HIT_COOLDOWN = 15            # Minimum frames between consecutive hits
FLASH_DURATION = 10          # Number of frames to show hit indicator

# Model and output paths
WEIGHTS_PATH = Path("weights/best.pt")
OUTPUT_DIR = Path("outputs")
HITS_CSV = OUTPUT_DIR / "hits.csv"


class HitDetector:
    """Detects ball-racket collisions using geometric and temporal analysis."""

    def __init__(self, dist_threshold=DIST_THRESHOLD, velocity_min=VELOCITY_MIN,
                 smooth_window=SMOOTH_WINDOW, cooldown=HIT_COOLDOWN):
        """
        Initialize hit detector.

        Args:
            dist_threshold: Maximum distance for collision (pixels)
            velocity_min: Minimum velocity change for valid hit
            smooth_window: Window size for velocity smoothing
            cooldown: Minimum frames between hits
        """
        self.dist_threshold = dist_threshold
        self.velocity_min = velocity_min
        self.smooth_window = smooth_window
        self.cooldown = cooldown

        # Tracking state
        self.ball_positions = deque(maxlen=smooth_window)
        self.ball_velocities = deque(maxlen=smooth_window)
        self.frames_since_hit = cooldown + 1
        self.total_hits = 0

        # Hit event storage
        self.hit_events = []

    def update(self, ball_center, racket_center, frame_id, timestamp):
        """
        Update detector with new frame data.

        Args:
            ball_center: (x, y) tuple of ball center, or None if not detected
            racket_center: (x, y) tuple of racket center, or None if not detected
            frame_id: Current frame number
            timestamp: Current timestamp in seconds

        Returns:
            tuple: (hit_detected, contact_point, hit_info)
        """
        self.frames_since_hit += 1

        # Check if both objects are detected
        if ball_center is None or racket_center is None:
            return False, None, None

        # Track ball position
        self.ball_positions.append(ball_center)

        # Calculate velocity if we have previous positions
        velocity = None
        if len(self.ball_positions) >= 2:
            prev_pos = self.ball_positions[-2]
            curr_pos = self.ball_positions[-1]
            velocity = np.array([
                curr_pos[0] - prev_pos[0],
                curr_pos[1] - prev_pos[1]
            ])
            self.ball_velocities.append(velocity)

        # Need at least 2 velocity samples for hit detection
        if len(self.ball_velocities) < 2:
            return False, None, None

        # Compute distance between ball and racket
        distance = np.linalg.norm(
            np.array(ball_center) - np.array(racket_center)
        )

        # Check proximity condition
        if distance > self.dist_threshold:
            return False, None, None

        # Check velocity magnitude
        curr_velocity = self.ball_velocities[-1]
        velocity_magnitude = np.linalg.norm(curr_velocity)

        if velocity_magnitude < self.velocity_min:
            return False, None, None

        # Check for velocity reversal (dot product < 0)
        prev_velocity = self.ball_velocities[-2]
        velocity_dot = np.dot(curr_velocity, prev_velocity)
        velocity_angle_change = np.degrees(
            np.arccos(
                np.clip(
                    velocity_dot / (
                        np.linalg.norm(curr_velocity) *
                        np.linalg.norm(prev_velocity) + 1e-6
                    ),
                    -1.0, 1.0
                )
            )
        )

        # Detect hit: close proximity + significant velocity change
        if velocity_angle_change > 90:  # Direction changed significantly
            # Apply cooldown to avoid duplicate detections
            if self.frames_since_hit < self.cooldown:
                return False, None, None

            # Hit detected!
            self.frames_since_hit = 0
            self.total_hits += 1

            # Calculate contact point (midpoint between ball and racket)
            contact_point = (
                int((ball_center[0] + racket_center[0]) / 2),
                int((ball_center[1] + racket_center[1]) / 2)
            )

            # Store hit event
            hit_info = {
                'frame': frame_id,
                'timestamp': timestamp,
                'distance': distance,
                'velocity': velocity_magnitude,
                'angle_change': velocity_angle_change,
                'ball_pos': ball_center,
                'racket_pos': racket_center,
                'contact_point': contact_point
            }
            self.hit_events.append(hit_info)

            return True, contact_point, hit_info

        return False, None, None

    def get_hit_events_df(self):
        """
        Get hit events as pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all hit events
        """
        if not self.hit_events:
            return pd.DataFrame()

        data = []
        for event in self.hit_events:
            data.append({
                'frame': event['frame'],
                'timestamp_sec': event['timestamp'],
                'distance_px': event['distance'],
                'velocity_px_per_frame': event['velocity'],
                'angle_change_deg': event['angle_change'],
                'ball_x': event['ball_pos'][0],
                'ball_y': event['ball_pos'][1],
                'racket_x': event['racket_pos'][0],
                'racket_y': event['racket_pos'][1],
                'contact_x': event['contact_point'][0],
                'contact_y': event['contact_point'][1]
            })

        return pd.DataFrame(data)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect table tennis ball-racket hits in real-time"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (0 for webcam, or path to video file)"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_PATH,
        help=f"Path to trained model weights (default: {WEIGHTS_PATH})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Detection confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--dist-threshold",
        type=float,
        default=DIST_THRESHOLD,
        help=f"Distance threshold for hit detection in pixels (default: {DIST_THRESHOLD})"
    )
    parser.add_argument(
        "--velocity-min",
        type=float,
        default=VELOCITY_MIN,
        help=f"Minimum velocity for collision (default: {VELOCITY_MIN})"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output video with annotations"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_DIR / "demo_hit_output.mp4",
        help="Path to save output video"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live display (useful for headless processing)"
    )
    return parser.parse_args()


def draw_detections(frame, ball_box, racket_box, ball_center, racket_center):
    """
    Draw bounding boxes and centers for detected objects.

    Args:
        frame: Input frame
        ball_box: Ball bounding box (x1, y1, x2, y2) or None
        racket_box: Racket bounding box (x1, y1, x2, y2) or None
        ball_center: Ball center (x, y) or None
        racket_center: Racket center (x, y) or None
    """
    # Draw ball
    if ball_box is not None:
        x1, y1, x2, y2 = map(int, ball_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "BALL", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if ball_center is not None:
            cv2.circle(frame, ball_center, 5, (0, 255, 0), -1)

    # Draw racket
    if racket_box is not None:
        x1, y1, x2, y2 = map(int, racket_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "RACKET", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if racket_center is not None:
            cv2.circle(frame, racket_center, 5, (255, 0, 0), -1)

    # Draw line connecting centers if both exist
    if ball_center is not None and racket_center is not None:
        cv2.line(frame, ball_center, racket_center, (255, 255, 0), 1)


def draw_hit_indicator(frame, contact_point, frames_since_hit, total_hits):
    """
    Draw hit detection indicator.

    Args:
        frame: Input frame
        contact_point: (x, y) contact point
        frames_since_hit: Frames elapsed since hit
        total_hits: Total hit count
    """
    # Flash effect: larger circle that fades
    if frames_since_hit < FLASH_DURATION:
        alpha = 1.0 - (frames_since_hit / FLASH_DURATION)
        radius = int(30 + frames_since_hit * 3)
        thickness = max(1, int(5 * alpha))

        cv2.circle(frame, contact_point, radius, (0, 0, 255), thickness)
        cv2.circle(frame, contact_point, 10, (0, 0, 255), -1)

        # "HIT DETECTED!" text
        text = "HIT DETECTED!"
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 1.5
        thickness = 3

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = contact_point[0] - text_size[0] // 2
        text_y = contact_point[1] - 50

        # Draw text with background
        cv2.rectangle(frame,
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y),
                   font, font_scale, (0, 0, 255), thickness)


def main():
    """Main execution function."""
    args = parse_arguments()

    # Check if model weights exist
    if not args.weights.exists():
        print(f"❌ Error: Model weights not found at {args.weights}")
        print(f"   Run 'python scripts/train_yolo.py' first")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    print(f"📦 Loading model from {args.weights}")
    model = YOLO(str(args.weights))
    print(f"✓ Model loaded successfully")

    # Initialize video source
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"📹 Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {source}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Video opened: {width}x{height} @ {fps} FPS")

    # Initialize video writer if saving
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(args.output_path), fourcc, fps, (width, height)
        )
        print(f"💾 Saving output to: {args.output_path}")

    # Initialize hit detector
    hit_detector = HitDetector(
        dist_threshold=args.dist_threshold,
        velocity_min=args.velocity_min
    )

    print(f"\n⚙️  Detection parameters:")
    print(f"   Distance threshold: {args.dist_threshold} px")
    print(f"   Velocity minimum: {args.velocity_min} px/frame")
    print(f"   Confidence threshold: {args.conf}")

    print(f"\n🎬 Starting inference... (Press 'q' to quit)")
    print("=" * 60)

    frame_id = 0
    last_contact_point = None
    frames_since_visual_hit = FLASH_DURATION + 1

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("\n📽️  End of video reached")
                break

            frame_id += 1
            timestamp = frame_id / fps

            # Run YOLO inference
            results = model(frame, conf=args.conf, verbose=False)

            # Extract detections
            ball_box = None
            racket_box = None
            ball_center = None
            racket_center = None

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                    # Class 0: ball, Class 1: racket
                    if cls == 0 and ball_box is None:  # Ball (highest conf)
                        ball_box = (x1, y1, x2, y2)
                        ball_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                    elif cls == 1 and racket_box is None:  # Racket (highest conf)
                        racket_box = (x1, y1, x2, y2)
                        racket_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Update hit detector
            hit_detected, contact_point, hit_info = hit_detector.update(
                ball_center, racket_center, frame_id, timestamp
            )

            # Log hit event
            if hit_detected:
                frames_since_visual_hit = 0
                last_contact_point = contact_point

                print(f"💥 HIT DETECTED at frame {frame_id} (t={timestamp:.2f}s)")
                print(f"   Distance: {hit_info['distance']:.1f} px")
                print(f"   Velocity: {hit_info['velocity']:.1f} px/frame")
                print(f"   Angle change: {hit_info['angle_change']:.1f}°")
                print(f"   Total hits: {hit_detector.total_hits}")

            # Draw detections
            draw_detections(frame, ball_box, racket_box, ball_center, racket_center)

            # Draw hit indicator
            if last_contact_point is not None and frames_since_visual_hit < FLASH_DURATION:
                draw_hit_indicator(
                    frame, last_contact_point,
                    frames_since_visual_hit, hit_detector.total_hits
                )

            frames_since_visual_hit += 1

            # Draw info overlay
            info_y = 30
            cv2.putText(frame, f"Frame: {frame_id} | Time: {timestamp:.2f}s",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            info_y += 30
            cv2.putText(frame, f"Total Hits: {hit_detector.total_hits}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            info_y += 30
            cv2.putText(frame, f"FPS: {fps}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display frame
            if not args.no_display:
                cv2.imshow("Table Tennis Hit Detection", frame)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Stopped by user")
                    break

            # Save frame to video
            if video_writer is not None:
                video_writer.write(frame)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")

    finally:
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        # Save hit events to CSV
        if hit_detector.hit_events:
            df = hit_detector.get_hit_events_df()
            df.to_csv(HITS_CSV, index=False)
            print(f"\n✓ Saved {len(df)} hit event(s) to: {HITS_CSV}")

            # Display summary
            print(f"\n📊 Detection Summary:")
            print(f"   Total frames processed: {frame_id}")
            print(f"   Total hits detected: {hit_detector.total_hits}")
            print(f"   Average hits per second: {hit_detector.total_hits / (frame_id / fps):.2f}")

        else:
            print(f"\n⚠️  No hits detected")

        print(f"\n✅ Processing complete!")


if __name__ == "__main__":
    main()
