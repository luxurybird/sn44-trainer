"""
Script to upload trained models to Hugging Face Hub
"""

from huggingface_hub import HfApi, upload_file, create_repo
from pathlib import Path
import argparse
import json

def upload_model(
    model_path: str,
    repo_id: str,
    model_filename: str = None,
    readme_path: str = None,
    create_readme: bool = True,
):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_path: Path to model .pt file
        repo_id: Hugging Face repository ID (e.g., 'username/sn44-player-detection')
        model_filename: Filename in repository (default: same as model_path)
        readme_path: Path to README.md (optional)
        create_readme: Create default README if none provided
    """
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Repository {repo_id} already exists or error: {e}")
    
    # Determine model filename
    if model_filename is None:
        model_filename = Path(model_path).name
    
    # Upload model file
    print(f"Uploading {model_path} to {repo_id}/{model_filename}...")
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✓ Model uploaded successfully!")
    
    # Create and upload README if needed
    if create_readme and readme_path is None:
        readme_content = create_default_readme(repo_id, model_filename)
        readme_path = "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"Created default README.md")
    
    if readme_path and Path(readme_path).exists():
        print(f"Uploading README.md...")
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"✓ README uploaded successfully!")
    
    print(f"\n✓ Model available at: https://huggingface.co/{repo_id}")

def create_default_readme(repo_id: str, model_filename: str) -> str:
    """Create a default README for the model."""
    
    # Determine model type from repo_id
    if 'player' in repo_id.lower():
        model_type = "Player/Goalkeeper/Referee Detection"
        classes = "ball, goalkeeper, player, referee"
        usage_example = """
```python
from ultralytics import YOLO

# Load model
model = YOLO('your-username/sn44-player-detection', model='football-player-detection.pt')

# Run inference
results = model('path/to/video.mp4')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {cls}, Confidence: {conf}")
```
"""
    elif 'keypoint' in repo_id.lower() or 'pitch' in repo_id.lower():
        model_type = "Pitch Keypoint Detection"
        classes = "32 pitch keypoints"
        usage_example = """
```python
from ultralytics import YOLO

# Load model
model = YOLO('your-username/sn44-pitch-keypoints', model='football-pitch-detection.pt')

# Run inference
results = model('path/to/video.mp4')

# Process keypoints
for result in results:
    keypoints = result.keypoints
    if keypoints is not None:
        kp = keypoints.xy[0]  # 32 keypoints
        print(f"Detected {len(kp)} keypoints")
```
"""
    elif 'ball' in repo_id.lower():
        model_type = "Ball Detection"
        classes = "ball"
        usage_example = """
```python
from ultralytics import YOLO

# Load model
model = YOLO('your-username/sn44-ball-detection', model='football-ball-detection.pt')

# Run inference
results = model('path/to/video.mp4')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        conf = float(box.conf[0])
        print(f"Ball detected with confidence: {conf}")
```
"""
    else:
        model_type = "Football Detection"
        classes = "Unknown"
        usage_example = "# See model documentation"
    
    readme = f"""---
tags:
- computer-vision
- object-detection
- soccer
- football
- yolo
- bittensor
- subnet44
license: mit
datasets:
- soccernet
- custom-football-dataset
metrics:
- mAP@0.5
- mAP@0.5:0.95
---

# Subnet 44 {model_type} Model

## Model Description

High-performance YOLO model for {model_type.lower()} in football videos, optimized for Subnet 44 (Bittensor).

**Model Type:** {model_type}  
**Classes:** {classes}  
**Format:** PyTorch (.pt)

## Performance

- **mAP@0.5:** TBD (update after training)
- **mAP@0.5:0.95:** TBD
- **FPS:** TBD (on RTX 4090)

## Usage

{usage_example}

## Model Details

- **Architecture:** YOLOv10-L (or as specified)
- **Input Size:** 1280x1280
- **Output Format:** YOLO format
- **Device:** CUDA/CPU compatible

## Training Details

- **Dataset:** SoccerNet + Custom annotations
- **Epochs:** 300-400
- **Batch Size:** 16-32
- **Optimizer:** AdamW
- **Learning Rate:** 0.001 (with cosine decay)

## Integration with Subnet 44 Miner

This model is designed to work with the Subnet 44 miner. Place the model file in the `miner/data/` directory:

```bash
cp {model_filename} miner/data/
```

The miner will automatically load and use this model.

## Citation

If you use this model, please cite:

```bibtex
@misc{{sn44-models,
  title={{Subnet 44 Football Detection Models}},
  author={{Your Name}},
  year={{2025}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## License

MIT License - See LICENSE file for details.
"""
    return readme

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload model to Hugging Face')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pt file')
    parser.add_argument('--repo-id', type=str, required=True, help='Hugging Face repo ID (username/repo-name)')
    parser.add_argument('--filename', type=str, default=None, help='Filename in repository')
    parser.add_argument('--readme', type=str, default=None, help='Path to README.md')
    parser.add_argument('--no-readme', action='store_true', help='Skip README creation')
    
    args = parser.parse_args()
    
    upload_model(
        model_path=args.model,
        repo_id=args.repo_id,
        model_filename=args.filename,
        readme_path=args.readme,
        create_readme=not args.no_readme,
    )

