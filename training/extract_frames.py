"""
Simple script to extract frames from videos
"""

import cv2
from pathlib import Path
import argparse

def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    image_format: str = "jpg",
    start_frame: int = 0,
    max_frames: int = None
):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame
        image_format: Image format ('jpg' or 'png')
        start_frame: Frame to start from
        max_frames: Maximum number of frames to extract (None = all)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Opening video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0
    
    print(f"\nExtracting frames (every {frame_interval} frames)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and saved_count >= max_frames:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_path.stem}_frame_{saved_count:06d}.{image_format}"
            frame_path = output_dir / frame_filename
            
            if image_format.lower() == 'jpg':
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(frame_path), frame)
            
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n✓ Extraction complete!")
    print(f"  Total frames extracted: {saved_count}")
    print(f"  Saved to: {output_dir}")

def extract_from_directory(
    video_dir: str,
    output_dir: str,
    frame_interval: int = 30,
    image_format: str = "jpg"
):
    """Extract frames from all videos in a directory."""
    video_dir = Path(video_dir)
    
    video_extensions = ['.mkv', '.mp4', '.avi', '.mov', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    for video_file in video_files:
        video_output_dir = Path(output_dir) / video_file.stem
        extract_frames(
            video_path=str(video_file),
            output_dir=str(video_output_dir),
            frame_interval=frame_interval,
            image_format=image_format
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--video-dir', type=str, help='Path to directory with videos')
    parser.add_argument('--output', type=str, required=True, help='Output directory for frames')
    parser.add_argument('--interval', type=int, default=30,
                        help='Extract every Nth frame (default: 30)')
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'png'],
                        help='Image format (default: jpg)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Frame to start from (default: 0)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to extract (default: all)')
    
    args = parser.parse_args()
    
    if args.video:
        extract_frames(
            video_path=args.video,
            output_dir=args.output,
            frame_interval=args.interval,
            image_format=args.format,
            start_frame=args.start_frame,
            max_frames=args.max_frames
        )
    elif args.video_dir:
        extract_from_directory(
            video_dir=args.video_dir,
            output_dir=args.output,
            frame_interval=args.interval,
            image_format=args.format
        )
    else:
        print("Error: Must specify either --video or --video-dir")
        parser.print_help()

