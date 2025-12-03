"""
Helper script to prepare dataset for YOLO training
Converts various formats to YOLO format and organizes data
"""

import os
import shutil
from pathlib import Path
import json
from typing import List, Tuple
import cv2
import numpy as np

def create_yolo_structure(base_path: str):
    """Create YOLO dataset directory structure."""
    base = Path(base_path)
    
    dirs = [
        base / 'images' / 'train',
        base / 'images' / 'val',
        base / 'images' / 'test',
        base / 'labels' / 'train',
        base / 'labels' / 'val',
        base / 'labels' / 'test',
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created YOLO structure at: {base_path}")
    return base

def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bbox from (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height) normalized.
    
    Args:
        bbox: [x1, y1, x2, y2] in pixels
        img_width: Image width
        img_height: Image height
    
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height

def convert_coco_to_yolo(coco_json_path: str, output_dir: str, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Output directory for YOLO format
        split_ratio: (train, val, test) ratios
    """
    import random
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output structure
    base = create_yolo_structure(output_dir)
    
    # Map COCO image IDs to filenames
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # Map category names to class IDs (adjust based on your classes)
    class_mapping = {
        'ball': 0,
        'goalkeeper': 1,
        'player': 2,
        'referee': 3,
    }
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Split images
    image_ids = list(images.keys())
    random.shuffle(image_ids)
    
    n_total = len(image_ids)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
    }
    
    # Process each split
    for split_name, img_ids in splits.items():
        print(f"Processing {split_name} split: {len(img_ids)} images")
        
        for img_id in img_ids:
            img_info = images[img_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            src_img = Path(coco_json_path).parent / img_filename
            dst_img = base / 'images' / split_name / img_filename
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Create label file
            label_filename = Path(img_filename).stem + '.txt'
            label_path = base / 'labels' / split_name / label_filename
            
            with open(label_path, 'w') as f:
                if img_id in annotations_by_image:
                    for ann in annotations_by_image[img_id]:
                        category_name = categories[ann['category_id']]['name']
                        class_id = class_mapping.get(category_name.lower(), -1)
                        
                        if class_id == -1:
                            continue
                        
                        # Convert bbox
                        bbox = ann['bbox']  # COCO: [x, y, width, height]
                        x1, y1 = bbox[0], bbox[1]
                        x2, y2 = x1 + bbox[2], y1 + bbox[3]
                        
                        x_center, y_center, width, height = convert_bbox_to_yolo(
                            [x1, y1, x2, y2], img_width, img_height
                        )
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Conversion complete! Dataset saved to: {output_dir}")

def verify_dataset(dataset_path: str):
    """Verify YOLO dataset structure and annotations."""
    base = Path(dataset_path)
    
    print("Verifying dataset structure...")
    
    # Check structure
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test',
    ]
    
    for dir_path in required_dirs:
        full_path = base / dir_path
        if not full_path.exists():
            print(f"❌ Missing: {dir_path}")
            return False
        print(f"✓ Found: {dir_path}")
    
    # Check images and labels match
    for split in ['train', 'val', 'test']:
        img_dir = base / 'images' / split
        label_dir = base / 'labels' / split
        
        images = set(f.stem for f in img_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
        labels = set(f.stem for f in label_dir.glob('*.txt'))
        
        missing_labels = images - labels
        missing_images = labels - images
        
        if missing_labels:
            print(f"⚠️  {split}: {len(missing_labels)} images without labels")
        if missing_images:
            print(f"⚠️  {split}: {len(missing_images)} labels without images")
        
        print(f"✓ {split}: {len(images)} images, {len(labels)} labels")
    
    # Sample label check
    print("\nSample label format:")
    sample_label = list((base / 'labels' / 'train').glob('*.txt'))[0]
    with open(sample_label, 'r') as f:
        print(f.read().strip())
    
    print("\n✓ Dataset verification complete!")
    return True

def split_dataset(source_dir: str, output_dir: str, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Split a dataset into train/val/test.
    
    Args:
        source_dir: Directory with images and labels
        output_dir: Output directory
        split_ratio: (train, val, test) ratios
    """
    import random
    
    source = Path(source_dir)
    base = create_yolo_structure(output_dir)
    
    # Find all images
    images = list((source / 'images').glob('*'))
    if not images:
        images = list(source.glob('*.jpg')) + list(source.glob('*.png'))
    
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:],
    }
    
    for split_name, split_images in splits.items():
        print(f"Processing {split_name}: {len(split_images)} images")
        
        for img_path in split_images:
            # Copy image
            dst_img = base / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Copy corresponding label
            label_path = source / 'labels' / (img_path.stem + '.txt')
            if not label_path.exists():
                label_path = source / (img_path.stem + '.txt')
            
            if label_path.exists():
                dst_label = base / 'labels' / split_name / label_path.name
                shutil.copy2(label_path, dst_label)
    
    print(f"Dataset split complete! Output: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    parser.add_argument('--action', type=str, required=True,
                        choices=['convert-coco', 'verify', 'split'],
                        help='Action to perform')
    parser.add_argument('--input', type=str, required=True,
                        help='Input path (COCO JSON or dataset directory)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--split-ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Train/val/test split ratio')
    
    args = parser.parse_args()
    
    if args.action == 'convert-coco':
        convert_coco_to_yolo(args.input, args.output, tuple(args.split_ratio))
    elif args.action == 'verify':
        verify_dataset(args.input)
    elif args.action == 'split':
        split_dataset(args.input, args.output, tuple(args.split_ratio))

