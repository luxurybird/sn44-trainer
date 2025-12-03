"""
Automated training script for all Subnet 44 models
Trains player detection, pitch keypoints, and ball detection models sequentially
Exports models locally (not to Hugging Face)
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add training directory to path
training_dir = Path(__file__).parent / "training"
sys.path.insert(0, str(training_dir))

def train_player_model(dataset_path: str, output_dir: str = "models", **kwargs):
    """Train player detection model."""
    print("\n" + "=" * 60)
    print("Training Player Detection Model")
    print("=" * 60)
    
    from train_player_detection import train_player_detection, export_model
    
    model_size = kwargs.get('model_size', 'l')
    image_size = kwargs.get('image_size', 1280)
    epochs = kwargs.get('epochs', 300)
    batch_size = kwargs.get('batch_size', 16)
    gpus = kwargs.get('gpus', [0])
    
    # Train
    model, results = train_player_detection(
        dataset_path=dataset_path,
        model_size=model_size,
        image_size=image_size,
        epochs=epochs,
        batch_size=batch_size,
        device=gpus,
        project_name='sn44-training',
        experiment_name='player-detection',
        use_wandb=kwargs.get('use_wandb', False),
    )
    
    # Export
    best_model_path = f"{results.save_dir}/weights/best.pt"
    output_path = Path(output_dir) / "football-player-detection.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting player model to: {output_path}")
    export_model(best_model_path, str(output_path), format='torchscript')
    
    return str(output_path)

def train_pitch_keypoint_model(dataset_path: str, output_dir: str = "models", **kwargs):
    """Train pitch keypoint detection model."""
    print("\n" + "=" * 60)
    print("Training Pitch Keypoint Detection Model")
    print("=" * 60)
    
    from train_pitch_keypoints import train_pitch_keypoints, export_keypoint_model
    
    model_size = kwargs.get('model_size', 'l')
    image_size = kwargs.get('image_size', 1280)
    epochs = kwargs.get('epochs', 400)
    batch_size = kwargs.get('batch_size', 16)
    gpus = kwargs.get('gpus', [0])
    
    # Note: This requires keypoint dataset in YOLO pose format
    # You may need to prepare keypoint annotations separately
    
    try:
        model, results = train_pitch_keypoints(
            dataset_path=dataset_path,
            model_size=model_size,
            image_size=image_size,
            epochs=epochs,
            batch_size=batch_size,
            device=gpus,
            project_name='sn44-training',
            experiment_name='pitch-keypoints',
            use_wandb=kwargs.get('use_wandb', False),
        )
        
        # Export
        best_model_path = f"{results.save_dir}/weights/best.pt"
        output_path = Path(output_dir) / "football-pitch-detection.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting pitch keypoint model to: {output_path}")
        export_keypoint_model(best_model_path, str(output_path))
        
        return str(output_path)
    except Exception as e:
        print(f"⚠️  Error training pitch keypoint model: {e}")
        print("Note: Pitch keypoint training may require separate keypoint annotations")
        return None

def train_ball_model(dataset_path: str, output_dir: str = "models", **kwargs):
    """Train ball detection model."""
    print("\n" + "=" * 60)
    print("Training Ball Detection Model")
    print("=" * 60)
    
    from train_ball_detection import train_ball_detection, export_ball_model
    
    model_size = kwargs.get('model_size', 'n')
    image_size = kwargs.get('image_size', 1280)
    epochs = kwargs.get('epochs', 300)
    batch_size = kwargs.get('batch_size', 32)
    gpus = kwargs.get('gpus', [0])
    
    # Train
    model, results = train_ball_detection(
        dataset_path=dataset_path,
        model_size=model_size,
        image_size=image_size,
        epochs=epochs,
        batch_size=batch_size,
        device=gpus,
        project_name='sn44-training',
        experiment_name='ball-detection',
        use_wandb=kwargs.get('use_wandb', False),
    )
    
    # Export
    best_model_path = f"{results.save_dir}/weights/best.pt"
    output_path = Path(output_dir) / "football-ball-detection.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting ball model to: {output_path}")
    export_ball_model(best_model_path, str(output_path))
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Train all Subnet 44 models')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to YOLO format dataset')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained models')
    parser.add_argument('--model-size', type=str, default='l',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size for player/pitch models')
    parser.add_argument('--ball-model-size', type=str, default='n',
                        choices=['n', 's', 'm', 'l'],
                        help='Model size for ball detection')
    parser.add_argument('--image-size', type=int, default=1280,
                        choices=[640, 896, 1280, 1920],
                        help='Training image size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--pitch-epochs', type=int, default=400,
                        help='Number of epochs for pitch keypoint model')
    parser.add_argument('--batch-size', type=int, default=24,
                        help='Batch size for player/pitch models (optimized for 32GB VRAM)')
    parser.add_argument('--ball-batch-size', type=int, default=48,
                        help='Batch size for ball model (optimized for 32GB VRAM)')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='GPU device IDs (single GPU: [0])')
    parser.add_argument('--skip-player', action='store_true',
                        help='Skip player model training')
    parser.add_argument('--skip-pitch', action='store_true',
                        help='Skip pitch keypoint model training')
    parser.add_argument('--skip-ball', action='store_true',
                        help='Skip ball model training')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Verify dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {args.dataset}")
        sys.exit(1)
    
    # Check for dataset.yaml
    dataset_yaml = dataset_path / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"Warning: dataset.yaml not found at {dataset_yaml}")
        print("Training may fail. Ensure dataset is in YOLO format.")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("=" * 60)
        print("Subnet 44 Model Training Pipeline")
        print("=" * 60)
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f} GB")
        print(f"Dataset: {args.dataset}")
        print(f"Output directory: {output_dir}")
        print(f"GPUs: {args.gpus}")
        print(f"Model size: {args.model_size}")
        print(f"Image size: {args.image_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size} (player/pitch), {args.ball_batch_size} (ball)")
        print("=" * 60)
    else:
        print("Warning: CUDA not available! Training will be very slow on CPU.")
        print("=" * 60)
    
    start_time = datetime.now()
    trained_models = {}
    
    # Training parameters
    train_kwargs = {
        'model_size': args.model_size,
        'image_size': args.image_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gpus': args.gpus,
        'use_wandb': args.use_wandb,
    }
    
    # Train player model
    if not args.skip_player:
        try:
            player_model = train_player_model(
                dataset_path=str(args.dataset),
                output_dir=str(output_dir),
                **train_kwargs
            )
            trained_models['player'] = player_model
            print(f"\n✓ Player model saved to: {player_model}")
        except Exception as e:
            print(f"\n✗ Error training player model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⏭️  Skipping player model training")
    
    # Train pitch keypoint model
    if not args.skip_pitch:
        try:
            pitch_model = train_pitch_keypoint_model(
                dataset_path=str(args.dataset),
                output_dir=str(output_dir),
                epochs=args.pitch_epochs,
                **{k: v for k, v in train_kwargs.items() if k != 'epochs'}
            )
            if pitch_model:
                trained_models['pitch'] = pitch_model
                print(f"\n✓ Pitch keypoint model saved to: {pitch_model}")
        except Exception as e:
            print(f"\n✗ Error training pitch keypoint model: {e}")
            print("Note: Pitch keypoint training requires keypoint annotations")
            import traceback
            traceback.print_exc()
    else:
        print("\n⏭️  Skipping pitch keypoint model training")
    
    # Train ball model
    if not args.skip_ball:
        try:
            ball_model = train_ball_model(
                dataset_path=str(args.dataset),
                output_dir=str(output_dir),
                model_size=args.ball_model_size,
                batch_size=args.ball_batch_size,
                **{k: v for k, v in train_kwargs.items() if k not in ['model_size', 'batch_size']}
            )
            trained_models['ball'] = ball_model
            print(f"\n✓ Ball model saved to: {ball_model}")
        except Exception as e:
            print(f"\n✗ Error training ball model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⏭️  Skipping ball model training")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {duration}")
    print(f"\nTrained models:")
    for model_type, model_path in trained_models.items():
        if model_path:
            print(f"  {model_type}: {model_path}")
    print(f"\nAll models saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

