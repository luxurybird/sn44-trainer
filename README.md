# Subnet 44 Model Training - RunPod Deployment

## Quick Start

### 1. Install Dependencies

```bash
# PyTorch is pre-installed on RunPod, just install other packages
pip install ultralytics SoccerNet opencv-python tqdm numpy pyyaml
```

### 2. Download and Prepare Dataset

```bash
python download_and_prepare_dataset.py \
    --output-dir soccernet_yolo \
    --frame-interval 30
```

**Note**: This downloads ~60GB and takes several hours.

### 3. Train All Models

```bash
python train_all_models.py \
    --dataset soccernet_yolo \
    --output-dir models \
    --model-size l \
    --image-size 1280 \
    --epochs 300 \
    --gpus 0
```

## Estimated Training Time

With RTX 5090 (32GB VRAM, single GPU):
- **Player Detection**: 3-4 days
- **Pitch Keypoints**: 4-5 days
- **Ball Detection**: 1-2 days
- **Total**: 8-11 days

## Disk Space

- Dataset: ~180GB peak usage
- You have 230GB - should be sufficient

## Output

After training, models will be in `models/`:
- `football-player-detection.pt`
- `football-pitch-detection.pt`
- `football-ball-detection.pt`

## Using Screen (Recommended)

```bash
# Install screen
apt-get update && apt-get install -y screen

# Start session
screen -S training

# Run training
python train_all_models.py --dataset soccernet_yolo --output-dir models --model-size l --epochs 300 --gpus 0

# Detach: Ctrl+A then D
# Reattach: screen -r training
```

## Files

- `download_and_prepare_dataset.py` - Downloads SoccerNet-v3 and converts to YOLO format
- `train_all_models.py` - Trains all three models sequentially
- `run_training_pipeline.sh` - Shell script to run everything
- `training/` - Training scripts for individual models

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python train_all_models.py --dataset soccernet_yolo --batch-size 16 --ball-batch-size 32
```

### Disk Space Full
Delete extracted zip files after conversion:
```bash
find soccernet_data -name "*.zip" -type f -delete
```

### Check GPU
```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

