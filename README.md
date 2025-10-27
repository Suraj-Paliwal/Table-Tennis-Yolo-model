# Table Tennis Hit Detection System

A complete end-to-end computer vision pipeline for detecting table tennis ball-racket collisions using YOLOv8 object detection and geometric/temporal analysis.

## Features

- **Dataset Preparation**: Extract and split video frames for training
- **YOLOv8 Training**: Train custom object detection model for ball and racket
- **Real-time Hit Detection**: Detect ball-racket collisions using:
  - Geometric proximity analysis
  - Velocity vector tracking
  - Temporal smoothing
- **Live Visualization**: Real-time overlay with hit indicators
- **Event Logging**: CSV export of all detected hits with timestamps

## Project Structure

```
tabletennis/
├── videos/                    # Raw input videos
├── dataset/
│   ├── images/
│   │   ├── all/              # Extracted frames (before split)
│   │   ├── train/            # Training images
│   │   ├── val/              # Validation images
│   │   └── test/             # Test images
│   └── labels/
│       ├── all/              # All labels (before split)
│       ├── train/            # Training labels
│       ├── val/              # Validation labels
│       └── test/             # Test labels
├── weights/                   # Model checkpoints
│   └── best.pt               # Best trained weights
├── outputs/                   # Detection outputs
│   ├── hits.csv              # Hit event log
│   └── demo_hit_output.mp4   # Annotated video
├── scripts/
│   ├── extract_frames.py     # Frame extraction
│   ├── split_dataset.py      # Dataset splitting
│   ├── train_yolo.py         # Model training
│   └── detect_hit.py         # Hit detection inference
├── data.yaml                  # YOLO dataset config
└── requirements.txt           # Python dependencies
```

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

- **ultralytics** (YOLOv8)
- **opencv-python** (Video processing)
- **torch & torchvision** (Deep learning)
- **numpy** (Numerical operations)
- **pandas** (Data handling)
- **tqdm** (Progress bars)

## Usage

### Step 1: Extract Frames from Videos

Place your table tennis videos in the `videos/` directory, then run:

```bash
python scripts/extract_frames.py --fps 30 --motion-filter
```

**Options:**
- `--fps`: Frames per second to extract (default: 30)
- `--motion-filter`: Enable motion detection to discard static frames
- `--video-dir`: Custom video directory
- `--output-dir`: Custom output directory

**Output:** Frames saved to `dataset/images/all/`

### Step 2: Upload to Roboflow & Annotate

#### Option A: Upload to Roboflow (Recommended)

Roboflow provides an easy web-based annotation tool and automatically handles dataset splitting.

**1. Set up Roboflow:**
- Create account at https://roboflow.com
- Create a new "Object Detection" project
- Get your Private API Key from https://app.roboflow.com/settings/api
- Add classes: `ball` and `racket`

**2. Upload images:**
```bash
# Set your API key
export ROBOFLOW_API_KEY="your_private_api_key"

# Upload all images
python scripts/upload_to_roboflow.py \
  --workspace "your-workspace" \
  --project "your-project"
```

**3. Annotate in Roboflow:**
- Go to your project dashboard
- Draw bounding boxes around ball (tight fit) and racket
- Label each object correctly

**4. Generate & Download Dataset:**
- Click "Generate" → "New Version"
- Choose split ratio: 70% train, 20% valid, 10% test
- Add preprocessing/augmentation if desired
- Download in **YOLO format**
- Extract to your project root (will create organized train/val/test folders)

**5. Update data.yaml:**
```yaml
path: path/to/downloaded/dataset  # Update this line
train: train/images
val: valid/images
test: test/images
names:
  0: ball
  1: racket
```

**Advantages:**
- ✅ Web-based annotation (no local tools needed)
- ✅ Automatic train/val/test splitting
- ✅ Built-in preprocessing and augmentation
- ✅ Team collaboration support
- ✅ Version control for datasets

#### Option B: Manual Annotation (Alternative)

If you prefer local annotation tools:

**1. Annotate using:**
- [CVAT](https://www.cvat.ai/) - Powerful, self-hosted
- [LabelImg](https://github.com/heartexlabs/labelImg) - Simple desktop tool

**2. Label Format:** YOLO format (one `.txt` file per image)
```
class_id center_x center_y width height
```

**Classes:**
- `0`: ball
- `1`: racket

**Coordinates:** Normalized (0-1)

**3. Save labels to** `dataset/labels/all/`

**4. Split dataset:**
```bash
python scripts/split_dataset.py --train 0.7 --val 0.2 --test 0.1
```

**Options:**
- `--train`: Training set ratio (default: 0.7)
- `--val`: Validation set ratio (default: 0.2)
- `--test`: Test set ratio (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)

**Output:** Organized dataset in `dataset/images/{train,val,test}/` and `dataset/labels/{train,val,test}/`

### Step 3: Train YOLOv8 Model

**Note:** If using Roboflow (Option A), the dataset is already split when you download it. Skip to training directly!

Train the object detection model:

```bash
python scripts/train_yolo.py --epochs 100 --batch 16 --imgsz 640
```

**Options:**
- `--model`: YOLOv8 variant (`yolov8n.pt`, `yolov8s.pt`, etc.)
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--patience`: Early stopping patience (default: 50)
- `--resume`: Resume from last checkpoint
- `--device`: Device to use (`0` for GPU, `cpu` for CPU)
- `--export-onnx`: Export to ONNX format after training

**Output:**
- Best weights saved to `weights/best.pt`
- Training logs in `tabletennis_yolo/ball_racket_detector/`

### Step 4: Run Hit Detection

Run inference on live camera or video:

```bash
# Live camera
python scripts/detect_hit.py --source 0

# Video file
python scripts/detect_hit.py --source path/to/video.mp4

# Save output video
python scripts/detect_hit.py --source video.mp4 --save-video
```

**Options:**
- `--source`: Video source (0 for webcam, or path to video)
- `--weights`: Path to trained weights (default: `weights/best.pt`)
- `--conf`: Detection confidence threshold (default: 0.25)
- `--dist-threshold`: Distance threshold for hit detection in pixels (default: 40)
- `--velocity-min`: Minimum velocity for collision (default: 2.0)
- `--save-video`: Save annotated output video
- `--output-path`: Output video path (default: `outputs/demo_hit_output.mp4`)
- `--no-display`: Disable live display window

**Output:**
- Live visualization with hit indicators
- Hit events logged to `outputs/hits.csv`
- Optional annotated video

## Hit Detection Algorithm

The system detects ball-racket collisions using:

### 1. Object Detection
- YOLOv8 detects ball and racket in each frame
- Extracts bounding boxes and confidence scores

### 2. Geometric Analysis
- Computes Euclidean distance between ball and racket centers
- Hit candidate when distance < `DIST_THRESHOLD` (default: 40px)

### 3. Temporal Analysis
- Tracks ball velocity between consecutive frames
- Detects velocity direction changes (dot product < 0)
- Requires minimum velocity magnitude (> `VELOCITY_MIN`)

### 4. Trajectory Smoothing
- Uses sliding window of last N frames (default: 5)
- Reduces false positives from detection jitter

### 5. Cooldown Period
- Prevents duplicate detections
- Minimum frames between hits (default: 15)

### Hit Criteria
A hit is detected when ALL conditions are met:
1. Ball and racket detected with sufficient confidence
2. Distance between centers < threshold
3. Ball velocity > minimum threshold
4. Velocity direction changed > 90 degrees
5. Cooldown period elapsed

## Configuration

Key parameters in `scripts/detect_hit.py`:

```python
DIST_THRESHOLD = 40          # Pixel distance for hit detection
VELOCITY_MIN = 2.0           # Minimum speed to consider collision
SMOOTH_WINDOW = 5            # Frames for trajectory smoothing
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold
HIT_COOLDOWN = 15            # Minimum frames between hits
FLASH_DURATION = 10          # Hit indicator display duration
```

Tune these values based on:
- Video resolution and FPS
- Camera distance from table
- Ball size in pixels
- Detection accuracy

## Output Examples

### Console Output

```
💥 HIT DETECTED at frame 232 (t=7.73s)
   Distance: 28.4 px
   Velocity: 15.2 px/frame
   Angle change: 145.8°
   Total hits: 1

💥 HIT DETECTED at frame 487 (t=16.23s)
   Distance: 35.1 px
   Velocity: 12.7 px/frame
   Angle change: 132.4°
   Total hits: 2
```

### CSV Output (`outputs/hits.csv`)

| frame | timestamp_sec | distance_px | velocity_px_per_frame | angle_change_deg | ball_x | ball_y | racket_x | racket_y | contact_x | contact_y |
|-------|---------------|-------------|----------------------|------------------|--------|--------|----------|----------|-----------|-----------|
| 232   | 7.73          | 28.4        | 15.2                 | 145.8            | 450    | 320    | 425      | 310      | 437       | 315       |
| 487   | 16.23         | 35.1        | 12.7                 | 132.4            | 580    | 280    | 555      | 295      | 567       | 287       |

### Visual Output

- **Ball**: Green bounding box with center dot
- **Racket**: Blue bounding box with center dot
- **Connection**: Yellow line between centers
- **Hit Event**:
  - Red flashing circle at contact point
  - "HIT DETECTED!" text overlay
  - Fades over 10 frames

## Troubleshooting

### No videos found
- Ensure videos are in `videos/` directory
- Check supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

### No labels found
- Annotate frames using external tools
- Ensure labels are in YOLO format
- Place `.txt` files in `dataset/labels/all/`
- Filename must match image (e.g., `frame_001.jpg` → `frame_001.txt`)

### Model weights not found
- Run training first: `python scripts/train_yolo.py`
- Check `weights/best.pt` exists

### Low detection accuracy
- Collect more training data (500+ images recommended)
- Increase training epochs
- Use larger model variant (`yolov8s.pt` or `yolov8m.pt`)
- Improve annotation quality

### False positive hits
- Increase `DIST_THRESHOLD`
- Increase `VELOCITY_MIN`
- Increase `HIT_COOLDOWN`
- Increase `CONFIDENCE_THRESHOLD`

### Missed hits
- Decrease `DIST_THRESHOLD`
- Decrease `VELOCITY_MIN`
- Decrease `CONFIDENCE_THRESHOLD`

## Optional Extensions

### 1. Stereo 3D Distance
Use two calibrated cameras for precise 3D distance:
```python
# Triangulate ball and racket in 3D space
distance_3d = compute_stereo_distance(left_cam, right_cam)
```

### 2. Audio Fusion
Combine visual detection with audio spike detection:
```python
# Detect impact sound
audio_spike = detect_audio_spike(audio_stream, threshold=0.8)
confirmed_hit = visual_hit and audio_spike
```

### 3. Web Dashboard
Create a Flask/Streamlit dashboard:
- Live hit counter
- Timeline visualization
- Statistics (hits/minute, velocity distribution)

### 4. Multi-Player Tracking
Track multiple players and rackets:
- Assign hits to specific players
- Track rally length
- Compute player-specific statistics

## Performance

### Training
- **GPU**: ~2 hours for 100 epochs (RTX 3080)
- **CPU**: ~10-15 hours for 100 epochs

### Inference
- **GPU**: 60-100 FPS (YOLOv8n)
- **CPU**: 10-20 FPS (YOLOv8n)

## License

This project is provided for educational purposes.

## Citation

If you use this project, please cite:
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV: [OpenCV](https://opencv.org/)

## Author

Created as a complete computer vision pipeline for table tennis analytics.
