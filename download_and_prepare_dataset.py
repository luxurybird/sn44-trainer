"""
Automated script to download SoccerNet-v3 dataset and prepare it for YOLO training
This script handles:
1. Downloading SoccerNet-v3 dataset
2. Extracting frames from zipped folders
3. Converting annotations to YOLO format
4. Organizing into train/val/test splits
"""

import os
import json
import zipfile
from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def install_dependencies():
    """Install required packages."""
    import subprocess
    import sys
    
    # Check if PyTorch is already installed (RunPod has it pre-installed)
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} already installed")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("PyTorch not found - RunPod should have it pre-installed")
    
    packages = ['SoccerNet', 'opencv-python', 'tqdm', 'numpy', 'ultralytics', 'pyyaml']
    
    for package in packages:
        try:
            __import__(package.replace('-', '_').replace('opencv-python', 'cv2'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed")

def download_soccernet_v3(output_dir: str = "soccernet_data", splits: list = ["train", "valid", "test"]):
    """
    Download SoccerNet-v3 dataset.
    
    Args:
        output_dir: Directory to save dataset
        splits: Which splits to download
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        
        print("=" * 60)
        print("Downloading SoccerNet-v3 Dataset")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Splits: {splits}")
        print("\nNote: This requires ~60GB of storage for frames + ~1GB for labels")
        print("This may take several hours depending on your internet connection.")
        print("=" * 60)
        
        downloader = SoccerNetDownloader(LocalDirectory=output_dir)
        
        # Download frames and labels for SoccerNet-v3
        print("\nDownloading frames and labels...")
        print("Files: Labels-v3.json, Frames-v3.zip")
        
        downloader.downloadGames(
            files=["Labels-v3.json", "Frames-v3.zip"],
            split=splits,
            task="frames"
        )
        
        print(f"\n✓ Dataset downloaded to: {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"Error downloading SoccerNet-v3: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have enough disk space (~60GB)")
        print("2. Check your internet connection")
        print("3. Verify SoccerNet package is installed: pip install SoccerNet")
        raise

def extract_zip_frames(zip_path: Path, output_dir: Path):
    """Extract frames from a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def convert_bbox_to_yolo(bbox: list, img_width: int, img_height: int) -> tuple:
    """
    Convert bbox from SoccerNet format to YOLO format.
    SoccerNet: [x_top, y_top, width, height]
    YOLO: [x_center, y_center, width, height] (normalized)
    """
    x_top, y_top, width, height = bbox
    
    # Calculate center
    x_center = (x_top + width / 2.0) / img_width
    y_center = (y_top + height / 2.0) / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    return x_center, y_center, width_norm, height_norm

def map_soccernet_class_to_yolo(class_index: int, jersey_number: int = None) -> int:
    """
    Map SoccerNet class to YOLO class.
    SoccerNet classes: 0=player, 1=goalkeeper, 2=referee, 3=ball, etc.
    YOLO classes: 0=ball, 1=goalkeeper, 2=player, 3=referee
    """
    # SoccerNet class mapping (based on typical SoccerNet-v3 structure)
    # Adjust based on actual SoccerNet-v3 class definitions
    class_mapping = {
        0: 2,  # player -> player (class 2)
        1: 1,  # goalkeeper -> goalkeeper (class 1)
        2: 3,  # referee -> referee (class 3)
        3: 0,  # ball -> ball (class 0)
    }
    
    return class_mapping.get(class_index, 2)  # Default to player if unknown

def convert_soccernet_to_yolo(
    soccernet_dir: str,
    output_dir: str,
    extract_frames: bool = True,
    frame_interval: int = 1  # Extract all frames by default
):
    """
    Convert SoccerNet-v3 dataset to YOLO format.
    
    Args:
        soccernet_dir: Path to SoccerNet-v3 dataset
        output_dir: Output directory for YOLO dataset
        extract_frames: Whether to extract frames from zip files
        frame_interval: Extract every Nth frame (1 = all frames)
    """
    soccernet_path = Path(soccernet_dir)
    output_path = Path(output_dir)
    
    # Create YOLO structure
    for split in ['train', 'valid', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting SoccerNet-v3 to YOLO Format")
    print("=" * 60)
    
    # Map SoccerNet splits to YOLO splits
    split_mapping = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }
    
    total_images = 0
    total_labels = 0
    
    # Process each championship/season/game
    for championship_dir in soccernet_path.iterdir():
        if not championship_dir.is_dir():
            continue
        
        print(f"\nProcessing championship: {championship_dir.name}")
        
        for season_dir in championship_dir.iterdir():
            if not season_dir.is_dir():
                continue
            
            for game_dir in season_dir.iterdir():
                if not game_dir.is_dir():
                    continue
                
                # Find Labels-v3.json
                labels_file = game_dir / "Labels-v3.json"
                if not labels_file.exists():
                    continue
                
                # Find Frames-v3.zip
                frames_zip = game_dir / "Frames-v3.zip"
                
                # Determine split based on directory structure
                # SoccerNet organizes by split in the directory structure
                # Check parent directories to determine split
                split = 'train'  # Default
                path_str = str(game_dir)
                if '/valid/' in path_str or '/validation/' in path_str:
                    split = 'valid'
                elif '/test/' in path_str:
                    split = 'test'
                elif '/train/' in path_str:
                    split = 'train'
                
                # Load annotations
                try:
                    with open(labels_file, 'r') as f:
                        annotations = json.load(f)
                except Exception as e:
                    print(f"Error loading {labels_file}: {e}")
                    continue
                
                # Extract frames if needed
                frames_dir = None
                if extract_frames and frames_zip.exists():
                    # Try different possible extracted folder names
                    possible_dirs = [
                        game_dir / "Frames-v3",
                        game_dir / "Frames",
                        game_dir / frames_zip.stem
                    ]
                    
                    # Check if already extracted
                    for possible_dir in possible_dirs:
                        if possible_dir.exists() and possible_dir.is_dir():
                            frames_dir = possible_dir
                            break
                    
                    # Extract if not found
                    if frames_dir is None or not frames_dir.exists():
                        print(f"  Extracting frames from {frames_zip.name}...")
                        if extract_zip_frames(frames_zip, game_dir):
                            # Check which directory was created
                            for possible_dir in possible_dirs:
                                if possible_dir.exists() and possible_dir.is_dir():
                                    frames_dir = possible_dir
                                    break
                elif not extract_frames:
                    # Try to find existing extracted frames
                    for possible_dir in [game_dir / "Frames-v3", game_dir / "Frames"]:
                        if possible_dir.exists():
                            frames_dir = possible_dir
                            break
                
                # Process annotations
                # SoccerNet-v3 format: annotations is a dict with 'actions' key
                # Each action has 'action' and 'replays' with images
                images_data = []
                
                if isinstance(annotations, dict):
                    # SoccerNet-v3 format
                    actions = annotations.get('actions', [])
                    for action in actions:
                        # Action frame
                        if 'action' in action:
                            images_data.append(action['action'])
                        # Replay frames
                        if 'replays' in action:
                            images_data.extend(action['replays'])
                elif isinstance(annotations, list):
                    # Alternative format: list of images
                    images_data = annotations
                
                for img_data in tqdm(images_data, desc=f"  {game_dir.name}"):
                    # Get image filename
                    img_filename = img_data.get('file_name', '')
                    if not img_filename:
                        continue
                    
                    # Find image file
                    img_path = None
                    if frames_dir and frames_dir.exists():
                        img_path = frames_dir / img_filename
                    else:
                        # Try in game directory
                        img_path = game_dir / img_filename
                    
                    if not img_path.exists():
                        continue
                    
                    # Get image dimensions
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        img_height, img_width = img.shape[:2]
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
                        continue
                    
                    # Get bounding boxes
                    # SoccerNet-v3 format: bboxes is a list of [x, y, w, h, class, jersey]
                    bboxes = img_data.get('bboxes', [])
                    if not bboxes:
                        # Skip images without bboxes
                        continue
                    
                    # Create YOLO label file
                    label_filename = Path(img_filename).stem + '.txt'
                    label_path = output_path / 'labels' / split_mapping[split] / label_filename
                    
                    # Skip if we're using frame interval and this frame doesn't match
                    frame_num = int(Path(img_filename).stem) if Path(img_filename).stem.isdigit() else 0
                    if frame_interval > 1 and frame_num % frame_interval != 0:
                        continue
                    
                    with open(label_path, 'w') as f:
                        for bbox_data in bboxes:
                            # SoccerNet-v3 bbox format: [x_top, y_top, width, height, class_index, jersey_number]
                            # Or could be dict: {'x': x, 'y': y, 'w': w, 'h': h, 'class': class, ...}
                            if isinstance(bbox_data, dict):
                                x_top = bbox_data.get('x', 0)
                                y_top = bbox_data.get('y', 0)
                                width = bbox_data.get('w', 0)
                                height = bbox_data.get('h', 0)
                                class_index = bbox_data.get('class', 0)
                            elif isinstance(bbox_data, list) and len(bbox_data) >= 5:
                                x_top, y_top, width, height = bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]
                                class_index = int(bbox_data[4]) if len(bbox_data) > 4 else 0
                            else:
                                continue
                            
                            # Skip invalid bboxes
                            if width <= 0 or height <= 0:
                                continue
                            
                            # Convert to YOLO format
                            x_center, y_center, width_norm, height_norm = convert_bbox_to_yolo(
                                [x_top, y_top, width, height], img_width, img_height
                            )
                            
                            # Map class
                            yolo_class = map_soccernet_class_to_yolo(class_index)
                            
                            # Write to label file
                            f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                    
                    # Copy image to output directory
                    output_img_path = output_path / 'images' / split_mapping[split] / img_filename
                    shutil.copy2(img_path, output_img_path)
                    
                    total_images += 1
                    total_labels += 1
                    
                    if total_images % 100 == 0:
                        print(f"  Processed {total_images} images...")
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Total images processed: {total_images}")
    print(f"Total labels created: {total_labels}")
    print(f"Output directory: {output_dir}")
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir)
    
    return output_dir

def create_dataset_yaml(dataset_dir: str):
    """Create YOLO dataset.yaml file."""
    dataset_path = Path(dataset_dir)
    yaml_path = dataset_path / "dataset.yaml"
    
    yaml_content = f"""# SoccerNet-v3 Dataset Configuration
path: {dataset_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 4
names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created dataset.yaml at: {yaml_path}")

def verify_dataset(dataset_dir: str):
    """Verify the prepared dataset."""
    dataset_path = Path(dataset_dir)
    
    print("\n" + "=" * 60)
    print("Verifying Dataset")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        if not img_dir.exists() or not label_dir.exists():
            print(f"⚠️  {split}: Missing directories")
            continue
        
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        labels = list(label_dir.glob('*.txt'))
        
        print(f"\n{split}:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # Check matching
        img_stems = {img.stem for img in images}
        label_stems = {label.stem for label in labels}
        
        missing_labels = img_stems - label_stems
        missing_images = label_stems - img_stems
        
        if missing_labels:
            print(f"  ⚠️  {len(missing_labels)} images without labels")
        if missing_images:
            print(f"  ⚠️  {len(missing_images)} labels without images")
        
        if not missing_labels and not missing_images:
            print(f"  ✓ All images have matching labels")
    
    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Download and prepare SoccerNet-v3 dataset for YOLO training')
    parser.add_argument('--output-dir', type=str, default='soccernet_yolo',
                        help='Output directory for prepared dataset')
    parser.add_argument('--soccernet-dir', type=str, default='soccernet_data',
                        help='Directory where SoccerNet-v3 will be downloaded')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'valid', 'test'],
                        help='Dataset splits to download')
    parser.add_argument('--frame-interval', type=int, default=30,
                        help='Extract every Nth frame (default: 30, use 1 for all frames)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download if dataset already exists')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip frame extraction (use if already extracted)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing dataset, skip download/conversion')
    
    args = parser.parse_args()
    
    # Install dependencies
    print("Checking dependencies...")
    install_dependencies()
    
    if args.verify_only:
        verify_dataset(args.output_dir)
        return
    
    # Download dataset
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("Step 1: Downloading SoccerNet-v3")
        print("=" * 60)
        download_soccernet_v3(args.soccernet_dir, args.splits)
    else:
        print("Skipping download (using existing dataset)")
    
    # Convert to YOLO format
    print("\n" + "=" * 60)
    print("Step 2: Converting to YOLO Format")
    print("=" * 60)
    convert_soccernet_to_yolo(
        soccernet_dir=args.soccernet_dir,
        output_dir=args.output_dir,
        extract_frames=not args.skip_extract,
        frame_interval=args.frame_interval
    )
    
    # Verify dataset
    verify_dataset(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nDataset ready at: {args.output_dir}")
    print(f"Use this path for training: --dataset {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

