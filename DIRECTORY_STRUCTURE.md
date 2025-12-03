# Directory Structure

```
sn44-training-pod/
├── README.md                          # Quick start guide
├── SETUP.md                           # Detailed setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── download_and_prepare_dataset.py    # Download SoccerNet-v3 and convert to YOLO
├── train_all_models.py                # Train all three models
├── run_training_pipeline.sh           # Shell script wrapper
└── training/                          # Training scripts
    ├── train_player_detection.py      # Player model training
    ├── train_pitch_keypoints.py       # Pitch keypoint model training
    ├── train_ball_detection.py        # Ball model training
    ├── prepare_dataset.py             # Dataset utilities
    ├── download_soccernet.py          # SoccerNet download helper
    ├── extract_frames.py              # Frame extraction utility
    └── upload_to_huggingface.py      # (Not used - models saved locally)
```

## File Descriptions

### Main Scripts

- **`download_and_prepare_dataset.py`**: Main script to download SoccerNet-v3 dataset and convert to YOLO format
- **`train_all_models.py`**: Main script to train all three models sequentially
- **`run_training_pipeline.sh`**: Shell script to run everything in one command

### Training Scripts

- **`training/train_player_detection.py`**: Trains player/goalkeeper/referee detection model
- **`training/train_pitch_keypoints.py`**: Trains pitch keypoint detection model (32 keypoints)
- **`training/train_ball_detection.py`**: Trains ball detection model

### Utility Scripts

- **`training/prepare_dataset.py`**: Dataset preparation utilities (convert, verify, split)
- **`training/download_soccernet.py`**: Helper for downloading SoccerNet
- **`training/extract_frames.py`**: Extract frames from videos

## What Gets Created (Not in Git)

These directories are created during execution and are in `.gitignore`:

- `soccernet_data/` - Downloaded SoccerNet-v3 dataset
- `soccernet_yolo/` - Converted YOLO format dataset
- `models/` - Trained model files (.pt)
- `sn44-training/` - Training logs and checkpoints

## Usage

1. Clone this repository on RunPod
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python download_and_prepare_dataset.py --output-dir soccernet_yolo`
4. Run: `python train_all_models.py --dataset soccernet_yolo --output-dir models --gpus 0`

See `README.md` for detailed instructions.

