"""
Script to download and prepare SoccerNet dataset for training
"""

import os
from pathlib import Path
import subprocess
import json

def install_soccernet():
    """Install SoccerNet package if not available."""
    try:
        import SoccerNet
        print("✓ SoccerNet already installed")
    except ImportError:
        print("Installing SoccerNet...")
        subprocess.check_call(["pip", "install", "SoccerNet"])
        print("✓ SoccerNet installed")

def download_soccernet_dataset(
    output_dir: str = "soccernet_data",
    splits: list = ["train", "valid", "test"],
    num_games: int = None
):
    """
    Download SoccerNet dataset.
    
    Args:
        output_dir: Directory to save dataset
        splits: Which splits to download ['train', 'valid', 'test']
        num_games: Number of games per split (None = all)
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        
        downloader = SoccerNetDownloader(LocalDirectory=output_dir)
        
        print(f"Downloading SoccerNet dataset to: {output_dir}")
        print(f"Splits: {splits}")
        
        # Download games
        files_to_download = ["1_720p.mkv", "2_720p.mkv"]  # Main camera views
        
        for split in splits:
            print(f"\nDownloading {split} split...")
            downloader.downloadGames(
                files=files_to_download,
                split=[split],
                task="tracking",  # For player/ball tracking annotations
                numGames=num_games
            )
        
        print(f"\n✓ Dataset downloaded to: {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"Error downloading SoccerNet: {e}")
        print("\nAlternative: Download manually from https://www.soccer-net.org/")
        return None

def extract_frames_from_videos(
    video_dir: str,
    output_dir: str,
    frame_interval: int = 30,
    image_format: str = "jpg"
):
    """
    Extract frames from videos.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (30 = 1 frame per second at 30fps)
        image_format: Image format ('jpg' or 'png')
    """
    import cv2
    from pathlib import Path
    
    video_path = Path(video_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_path.glob("*.mkv")) + \
                  list(video_path.glob("*.mp4")) + \
                  list(video_path.glob("*.avi"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    total_frames = 0
    
    for video_file in video_files:
        print(f"Processing: {video_file.name}")
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            print(f"  ⚠️  Could not open {video_file.name}")
            continue
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_file.stem}_frame_{saved_count:06d}.{image_format}"
                frame_path = output_path / frame_filename
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        total_frames += saved_count
        print(f"  ✓ Extracted {saved_count} frames")
    
    print(f"\n✓ Total frames extracted: {total_frames}")

def get_soccernet_info():
    """Print information about SoccerNet dataset."""
    print("=" * 60)
    print("SoccerNet Dataset Information")
    print("=" * 60)
    print("\nWebsite: https://www.soccer-net.org/")
    print("GitHub: https://github.com/SoccerNet/soccer-net")
    print("\nDataset includes:")
    print("  - Professional football match videos")
    print("  - Player/ball tracking annotations")
    print("  - Multiple camera angles")
    print("  - High-quality broadcast footage")
    print("\nLicense: CC BY-NC-SA 4.0 (Non-commercial use)")
    print("\nTo download:")
    print("  1. Install: pip install SoccerNet")
    print("  2. Use SoccerNet API to download")
    print("  3. Or download manually from website")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare SoccerNet dataset')
    parser.add_argument('--action', type=str, required=True,
                        choices=['info', 'download', 'extract-frames'],
                        help='Action to perform')
    parser.add_argument('--output', type=str, default='soccernet_data',
                        help='Output directory')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'valid', 'test'],
                        help='Dataset splits to download')
    parser.add_argument('--num-games', type=int, default=None,
                        help='Number of games per split (None = all)')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Directory with videos (for extract-frames)')
    parser.add_argument('--frame-interval', type=int, default=30,
                        help='Extract every Nth frame')
    
    args = parser.parse_args()
    
    if args.action == 'info':
        get_soccernet_info()
    elif args.action == 'download':
        install_soccernet()
        download_soccernet_dataset(
            output_dir=args.output,
            splits=args.splits,
            num_games=args.num_games
        )
    elif args.action == 'extract-frames':
        if args.video_dir is None:
            args.video_dir = args.output
        extract_frames_from_videos(
            video_dir=args.video_dir,
            output_dir=args.output + "_frames",
            frame_interval=args.frame_interval
        )

