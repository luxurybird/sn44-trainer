"""
Training script for Subnet 44 Player/Goalkeeper/Referee Detection Model
Optimized for maximum accuracy with GPU resources
"""

import torch
from ultralytics import YOLO
import wandb
from pathlib import Path
import yaml

def create_dataset_yaml(dataset_path: str, output_path: str = "dataset.yaml"):
    """Create YOLO dataset configuration file."""
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,
        'names': {
            0: 'ball',
            1: 'goalkeeper',
            2: 'player',
            3: 'referee'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created dataset config: {output_path}")
    return output_path

def train_player_detection(
    dataset_path: str,
    model_size: str = 'l',  # 'n', 's', 'm', 'l', 'x'
    image_size: int = 1280,
    epochs: int = 300,
    batch_size: int = 16,
    device: list = [0],
    project_name: str = 'sn44-models',
    experiment_name: str = 'player-detection',
    use_wandb: bool = True,
):
    """
    Train player detection model with optimal settings.
    
    Args:
        dataset_path: Path to dataset directory
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        image_size: Training image size (1280 or 1920 for best accuracy)
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        device: List of GPU device IDs
        project_name: Weights & Biases project name
        experiment_name: Experiment name
        use_wandb: Whether to use Weights & Biases logging
    """
    
    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                'model_size': model_size,
                'image_size': image_size,
                'epochs': epochs,
                'batch_size': batch_size,
                'device': device,
            }
        )
    
    # Create dataset config
    dataset_yaml = create_dataset_yaml(dataset_path)
    
    # Load model
    model_name = f'yolov10{model_size}.pt'  # or 'yolov9-e.pt' for maximum accuracy
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Training parameters optimized for accuracy
    training_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'imgsz': image_size,
        'batch': batch_size,
        'device': device,
        'workers': 8,
        'patience': 50,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'project': project_name,
        'name': experiment_name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # Better than SGD for this task
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        
        # Augmentation parameters (optimized for football)
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,  # HSV-Saturation augmentation
        'hsv_v': 0.4,  # HSV-Value augmentation
        'degrees': 10.0,  # Rotation (+/- deg)
        'translate': 0.1,  # Translation (+/- fraction)
        'scale': 0.5,  # Scale (+/- gain)
        'shear': 5.0,  # Shear (+/- deg)
        'perspective': 0.0001,  # Perspective (+/- fraction)
        'flipud': 0.0,  # Don't flip vertically (players should be upright)
        'fliplr': 0.5,  # Flip left-right (50% probability)
        'mosaic': 1.0,  # Mosaic augmentation (100% probability)
        'mixup': 0.1,  # Mixup augmentation (10% probability)
        'copy_paste': 0.1,  # Copy-paste augmentation (10% probability)
        
        # Advanced settings
        'close_mosaic': 10,  # Disable mosaic for last N epochs
        'resume': False,
        'amp': True,  # Automatic Mixed Precision (FP16)
        'fraction': 1.0,  # Dataset fraction to use
        'profile': False,  # Profile ONNX and TensorRT speeds
        'freeze': None,  # Freeze layers: backbone=10, first3=0 1 2
        'multi_scale': False,  # Multi-scale training
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,  # Validate during training
    }
    
    print("Starting training...")
    print(f"Training configuration:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    
    # Train model
    results = model.train(**training_args)
    
    # Print results
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Last model saved at: {results.save_dir}/weights/last.pt")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.val(data=dataset_yaml, split='test')
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    if use_wandb:
        wandb.log({
            'test/mAP50': metrics.box.map50,
            'test/mAP50-95': metrics.box.map,
        })
        wandb.finish()
    
    return model, results

def export_model(model_path: str, output_path: str = None, format: str = 'torchscript'):
    """
    Export trained model to .pt format for deployment.
    
    Args:
        model_path: Path to trained model weights
        output_path: Output path for exported model
        format: Export format ('torchscript', 'onnx', 'engine')
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    if output_path is None:
        output_path = str(Path(model_path).parent / f"exported.{format}")
    
    print(f"Exporting to {format} format...")
    model.export(
        format=format,
        imgsz=1280,
        optimize=True,
        half=True,  # FP16 quantization
        int8=False,  # Set to True for INT8 quantization (slower export, faster inference)
    )
    
    print(f"Model exported to: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train player detection model for Subnet 44')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model-size', type=str, default='l', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size')
    parser.add_argument('--image-size', type=int, default=1280, choices=[640, 896, 1280, 1920],
                        help='Training image size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU device IDs')
    parser.add_argument('--export', action='store_true', help='Export model after training')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Train model
    model, results = train_player_detection(
        dataset_path=args.dataset,
        model_size=args.model_size,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.gpus,
        use_wandb=not args.no_wandb,
    )
    
    # Export if requested
    if args.export:
        best_model_path = f"{results.save_dir}/weights/best.pt"
        export_model(best_model_path, format='torchscript')

