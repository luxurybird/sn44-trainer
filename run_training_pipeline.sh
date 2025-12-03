#!/bin/bash
# Automated training pipeline for Subnet 44 models
# Downloads SoccerNet-v3, prepares dataset, and trains all models

set -e  # Exit on error

echo "=========================================="
echo "Subnet 44 Model Training Pipeline"
echo "=========================================="

# Configuration
DATASET_DIR="soccernet_yolo"
SOCCERNET_DIR="soccernet_data"
MODELS_DIR="models"
MODEL_SIZE="l"
IMAGE_SIZE=1280
EPOCHS=300
FRAME_INTERVAL=30
GPUS="0"  # Single GPU (RTX 5090)
BATCH_SIZE=24  # Optimized for 32GB VRAM
BALL_BATCH_SIZE=48  # Optimized for 32GB VRAM

# Parse command line arguments
SKIP_DOWNLOAD=false
SKIP_TRAINING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-download      Skip dataset download (use existing)"
            echo "  --skip-training      Skip training (only download/prepare)"
            echo "  --dataset-dir DIR    Dataset directory (default: soccernet_yolo)"
            echo "  --models-dir DIR     Models output directory (default: models)"
            echo "  --gpus GPUS         GPU IDs (default: 0)"
            echo "  --epochs N           Number of epochs (default: 300)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Step 1: Download and Prepare Dataset
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "=========================================="
    echo "Step 1: Downloading and Preparing Dataset"
    echo "=========================================="
    echo "This may take several hours and requires ~60GB of storage"
    echo ""
    
    python download_and_prepare_dataset.py \
        --output-dir "$DATASET_DIR" \
        --soccernet-dir "$SOCCERNET_DIR" \
        --splits train valid test \
        --frame-interval $FRAME_INTERVAL
    
    if [ $? -ne 0 ]; then
        echo "Error: Dataset preparation failed"
        exit 1
    fi
    
    echo ""
    echo "✓ Dataset prepared successfully"
else
    echo "Skipping dataset download (using existing dataset)"
fi

# Step 2: Train All Models
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo "=========================================="
    echo "Step 2: Training All Models"
    echo "=========================================="
    echo "This will take several days depending on your hardware"
    echo ""
    
    # Convert GPU string to space-separated list for Python
    GPU_LIST=$(echo $GPUS | tr ',' ' ')
    
    python train_all_models.py \
        --dataset "$DATASET_DIR" \
        --output-dir "$MODELS_DIR" \
        --model-size "$MODEL_SIZE" \
        --image-size $IMAGE_SIZE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --ball-batch-size $BALL_BATCH_SIZE \
        --gpus $GPU_LIST
    
    if [ $? -ne 0 ]; then
        echo "Error: Training failed"
        exit 1
    fi
    
    echo ""
    echo "✓ Training completed successfully"
    echo ""
    echo "Models saved to: $MODELS_DIR"
    echo "  - football-player-detection.pt"
    echo "  - football-pitch-detection.pt"
    echo "  - football-ball-detection.pt"
else
    echo "Skipping training"
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="

