# Setup Instructions for RunPod

## Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd sn44-training-pod
```

## Step 2: Install Dependencies

```bash
pip install ultralytics SoccerNet opencv-python tqdm numpy pyyaml
```

## Step 3: Verify GPU

```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

## Step 4: Run Training Pipeline

### Option A: Use Shell Script

```bash
bash run_training_pipeline.sh
```

### Option B: Run Manually

```bash
# Download dataset
python download_and_prepare_dataset.py --output-dir soccernet_yolo --frame-interval 30

# Train models
python train_all_models.py --dataset soccernet_yolo --output-dir models --model-size l --epochs 300 --gpus 0
```

## Step 5: Download Models

After training completes, download models from `models/` directory.

## Notes

- Training takes 8-11 days on RTX 5090
- Use `screen` or `tmux` to keep session alive
- Monitor with `watch -n 1 nvidia-smi`
- Check disk space with `df -h`

