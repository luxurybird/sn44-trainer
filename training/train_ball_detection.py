"""
Training script for Subnet 44 Ball Detection Model
Specialized for small object detection
"""

import torch
from ultralytics import YOLO
import wandb
from pathlib import Path
import yaml

def create_ball_dataset_yaml(dataset_path: str, output_path: str = "ball_detection.yaml"):
    """Create YOLO dataset configuration for ball detection."""
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': {
            0: 'ball'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created ball dataset config: {output_path}")
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
    for key, value in training_args.items():
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

