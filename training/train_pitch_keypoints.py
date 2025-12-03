"""
Training script for Subnet 44 Pitch Keypoint Detection Model
Detects 32 keypoints representing pitch lines
"""

import torch
from ultralytics import YOLO
import wandb
from pathlib import Path
import yaml

def create_keypoint_dataset_yaml(dataset_path: str, output_path: str = "pitch_keypoints.yaml"):
    """Create YOLO pose/keypoint dataset configuration file."""
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # Number of classes (pitch is single class)
        'names': {
            0: 'pitch'
        },
        'kpt_shape': [32, 2],  # 32 keypoints, 2 coordinates (x, y) each
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created keypoint dataset config: {output_path}")
    return output_path

def train_pitch_keypoints(
    dataset_path: str,
    model_size: str = 'l',
    image_size: int = 1280,
    epochs: int = 400,
    batch_size: int = 16,
    device: list = [0],
    project_name: str = 'sn44-models',
    experiment_name: str = 'pitch-keypoints',
    use_wandb: bool = True,
):
    """
    Train pitch keypoint detection model.
    
    Note: This requires a custom YOLO pose model modified for 32 keypoints.
    You may need to modify ultralytics code or use a custom architecture.
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
                'keypoints': 32,
            }
        )
    
    # Create dataset config
    dataset_yaml = create_keypoint_dataset_yaml(dataset_path)
    
    # Load pose model (will need modification for 32 keypoints)
    # Option 1: Use YOLOv8 pose and modify
    model_name = f'yolov8{model_size}-pose.pt'
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Modify model for 32 keypoints (this may require custom code)
    # The default pose model has 17 keypoints, we need 32
    # You'll need to modify the model architecture or use a custom model
    
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
        
        # Keypoint-specific parameters
        'kpt_shape': [32, 2],  # 32 keypoints, 2 coordinates
        'flip_idx': None,  # Keypoint flip indices (if applicable)
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,  # Pose/keypoint loss gain (important!)
        'kobj': 2.0,  # Keypoint object loss gain
        
        # Augmentation (more conservative for keypoints)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,  # Less rotation to preserve geometry
        'translate': 0.1,
        'scale': 0.3,  # Less scaling
        'shear': 2.0,  # Less shear
        'perspective': 0.0001,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.8,  # Slightly less mosaic
        'mixup': 0.05,  # Less mixup
        'copy_paste': 0.0,  # No copy-paste for keypoints
        
        'close_mosaic': 10,
        'amp': True,
        'val': True,
    }
    
    print("Starting keypoint model training...")
    print(f"Training configuration:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    
    # Train model
    results = model.train(**training_args)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = model.val(data=dataset_yaml, split='test')
    print(f"Keypoint mAP: {metrics.pose.map:.4f}")
    print(f"Keypoint mAP@0.5: {metrics.pose.map50:.4f}")
    
    if use_wandb:
        wandb.log({
            'test/keypoint_mAP': metrics.pose.map,
            'test/keypoint_mAP50': metrics.pose.map50,
        })
        wandb.finish()
    
    return model, results

def export_keypoint_model(model_path: str, output_path: str = None):
    """Export keypoint model to .pt format."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    if output_path is None:
        output_path = str(Path(model_path).parent / "exported.torchscript")
    
    print(f"Exporting keypoint model...")
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
    
    parser = argparse.ArgumentParser(description='Train pitch keypoint detection model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to keypoint dataset')
    parser.add_argument('--model-size', type=str, default='l', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--image-size', type=int, default=1280)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    
    args = parser.parse_args()
    
    model, results = train_pitch_keypoints(
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
        export_keypoint_model(best_model_path)

