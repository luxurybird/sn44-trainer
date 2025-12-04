"""
Training script for Subnet 44 Ball Detection Model
Specialized for small object detection
"""

import warnings
import os
import torch
from ultralytics import YOLO
import wandb
from pathlib import Path
import yaml

# Suppress NNPACK warning (harmless - we use GPU, not CPU optimizations)
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore', message='.*NNPACK.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

def create_ball_dataset_yaml(dataset_path: str, output_path: str = "ball_detection.yaml"):
    """Create YOLO dataset configuration for ball detection."""
    dataset_path_obj = Path(dataset_path)
    
    # Resolve the path - handle both absolute and relative paths
    dataset_path_obj = dataset_path_obj.resolve()
    
    # Verify the dataset directory exists
    if not dataset_path_obj.exists() or not (dataset_path_obj / 'images').exists():
        # Try common alternative locations
        dataset_name = dataset_path_obj.name if dataset_path_obj.name else 'soccernet_yolo'
        possible_paths = [
            dataset_path_obj,
            Path('/workspace/sn44-trainer') / dataset_name,
            Path('/workspace/sn44-trainer') / 'soccernet_yolo',
            Path('/workspace/sn44-trainer'),
            Path.cwd() / dataset_name,
            Path.cwd() / 'soccernet_yolo',
            Path.cwd(),
        ]
        
        found_path = None
        for path in possible_paths:
            if path.exists() and (path / 'images').exists():
                # Verify it has the required subdirectories
                if (path / 'images' / 'train').exists() or (path / 'images' / 'val').exists():
                    found_path = path
                    print(f"Found dataset at alternative location: {found_path}")
                    break
        
        if found_path is None:
            error_msg = (
                f"Dataset directory not found: {dataset_path_obj}\n"
                f"Checked paths:\n"
            )
            for p in possible_paths:
                exists = "✓" if p.exists() else "✗"
                has_images = "✓" if (p.exists() and (p / 'images').exists()) else "✗"
                error_msg += f"  {exists} {has_images} {p}\n"
            error_msg += (
                f"\nPlease ensure the dataset directory exists and contains 'images/train' subdirectory.\n"
                f"Expected location: /workspace/sn44-trainer/soccernet_yolo or current directory"
            )
            raise FileNotFoundError(error_msg)
        dataset_path_obj = found_path
    
    # Verify required subdirectories exist
    train_dir = dataset_path_obj / 'images' / 'train'
    val_dir = dataset_path_obj / 'images' / 'val'
    test_dir = dataset_path_obj / 'images' / 'test'
    
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Required dataset directory not found: {train_dir}\n"
            f"Dataset path: {dataset_path_obj}\n"
            f"Please ensure the dataset contains 'images/train' subdirectory."
        )
    
    # Handle missing val directory - use train for validation if val doesn't exist
    val_path = 'images/val'
    if not val_dir.exists():
        print(f"⚠️  Warning: Validation directory not found: {val_dir}")
        print(f"   Using training set for validation (not ideal but will work)")
        val_path = 'images/train'  # Use train for val if val doesn't exist
    
    # Handle missing test directory - use val (or train) for test if test doesn't exist
    test_path = 'images/test'
    if not test_dir.exists():
        test_path = val_path if val_dir.exists() else 'images/train'
    
    # Create config with resolved absolute path
    config = {
        'path': str(dataset_path_obj),
        'train': 'images/train',
        'val': val_path,
        'test': test_path,
        'nc': 1,
        'names': {
            0: 'ball'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created ball dataset config: {output_path}")
    print(f"Dataset path: {dataset_path_obj}")
    print(f"Train images: {dataset_path_obj / 'images/train'}")
    print(f"Val images: {dataset_path_obj / val_path}")
    if test_path != 'images/test' or not test_dir.exists():
        print(f"Test images: {dataset_path_obj / test_path} (using {'val' if val_dir.exists() else 'train'} set)")
    return output_path

def train_ball_detection(
    dataset_path: str,
    model_size: str = 'n',  # Use smaller model for speed, or 's' for accuracy
    image_size: int = 1280,  # Larger size helps with small ball detection
    epochs: int = 300,
    batch_size: int = 32,  # Can use larger batch for smaller model
    device: list = [0],
    project_name: str = 'sn44-models',
    experiment_name: str = 'ball-detection',
    use_wandb: bool = True,
):
    """
    Train ball detection model optimized for small object detection.
    """
    
    # Check GPU availability and configure device
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"GPU Configuration")
        print(f"{'='*60}")
        print(f"CUDA available: ✓")
        print(f"GPU count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Ensure device is properly formatted
        if isinstance(device, list):
            # Validate GPU IDs
            device = [d for d in device if d < gpu_count]
            if not device:
                print(f"⚠️  Warning: No valid GPU IDs in {device}, using GPU 0")
                device = 0
            elif len(device) == 1:
                device = device[0]  # YOLO accepts single int or list
        elif isinstance(device, int):
            if device >= gpu_count:
                print(f"⚠️  Warning: GPU {device} not available, using GPU 0")
                device = 0
        print(f"Using device: {device}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: CUDA not available! Training will use CPU (very slow)")
        print(f"{'='*60}\n")
        device = 'cpu'
    
    if use_wandb:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                'model_size': model_size,
                'image_size': image_size,
                'epochs': epochs,
                'batch_size': batch_size,
            }
        )
    
    dataset_yaml = create_ball_dataset_yaml(dataset_path)
    
    model_name = f'yolov10{model_size}.pt'  # or yolov8n.pt
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Optimize for small object detection
    training_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'imgsz': image_size,
        'batch': batch_size,
        'device': device,
        'workers': 8,
        'patience': 50,
        'save': True,
        'save_period': 10,
        'project': project_name,
        'name': experiment_name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        
        # Loss weights (focus on detection, less on classification)
        'box': 10.0,  # Higher box loss for small objects
        'cls': 0.3,  # Lower cls loss (only one class)
        'dfl': 2.0,
        
        # Augmentation (preserve small ball visibility)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,  # Less rotation
        'translate': 0.05,  # Less translation
        'scale': 0.3,  # Less scaling to keep ball visible
        'shear': 2.0,
        'perspective': 0.0,  # No perspective (can hide small ball)
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,  # Less mosaic (can hide small ball)
        'mixup': 0.0,  # No mixup (can confuse small ball)
        'copy_paste': 0.0,
        
        'close_mosaic': 20,  # Disable mosaic earlier
        'amp': True,
        'val': True,
        
        # Small object detection optimizations
        'multi_scale': False,  # Can use True for better small object detection
    }
    
    print("Starting ball detection model training...")
    print(f"Training configuration:")
    print(f"  Device: {device} (type: {type(device).__name__})")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Current CUDA device: {torch.cuda.current_device()}")
    for key, value in training_args.items():
        if key != 'device':  # Already printed above
            print(f"  {key}: {value}")
    
    results = model.train(**training_args)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = model.val(data=dataset_yaml, split='test')
    print(f"Ball mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Ball mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    if use_wandb:
        wandb.log({
            'test/mAP50': metrics.box.map50,
            'test/mAP50-95': metrics.box.map,
        })
        wandb.finish()
    
    return model, results

def export_ball_model(model_path: str, output_path: str = None):
    """Export ball detection model."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    if output_path is None:
        output_path = str(Path(model_path).parent / "exported.torchscript")
    
    print(f"Exporting ball detection model...")
    model.export(
        format='torchscript',
        imgsz=1280,
        optimize=True,
        half=True,
    )
    
    print(f"Model exported to: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ball detection model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to ball dataset')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l'])
    parser.add_argument('--image-size', type=int, default=1280)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    
    args = parser.parse_args()
    
    model, results = train_ball_detection(
        dataset_path=args.dataset,
        model_size=args.model_size,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.gpus,
        use_wandb=not args.no_wandb,
    )
    
    if args.export:
        best_model_path = f"{results.save_dir}/weights/best.pt"
        export_ball_model(best_model_path)

