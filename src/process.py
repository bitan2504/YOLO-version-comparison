import os
from rich import print
from src.frame_extraction import frame_extraction
from src.video_metadata import video_metadata
from src.run_yolo import run_YOLO_pose_v8, run_YOLOv8


def process(video_path: str, models_v8: list[str], models_pose: list[str], yolo_configs: dict) -> None:
    """
    Orchestrates the video processing pipeline: extracts frames, retrieves metadata, 
    and runs both standard YOLOv8 and YOLOv8-pose models on the extracted frames.
    
    Args:
        video_path (str): Path to the input video file.
        models_v8 (list[str]): List of YOLOv8 model filenames to run.
        models_pose (list[str]): List of YOLOv8 pose model filenames to run.
        yolo_configs (dict): Configuration dictionary containing YOLO parameters.
    """
    # Validate that the video file exists
    if not os.path.exists(video_path):
        print(f"[bold red]Error: Video file '{video_path}' does not exist.[/bold red]")
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

    # Extract the interview ID safely using os.path.basename to avoid OS-specific path issues
    video_filename = os.path.basename(video_path)
    interview_id = video_filename.replace("_final_camera.mp4", "").strip()
    print(f"[bold blue]Processing interview with ID: {interview_id}[/bold blue]")

    # Create output directories for the specific interview
    interview_path = os.path.join("output", interview_id)
    frames_dir = os.path.join(interview_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Retrieve video metadata
    duration, fps, fps_rounded, total_frames = video_metadata(video_path)
    
    # Calculate the total number of frames that will be extracted based on the sample interval
    sample_interval = yolo_configs.get("FRAME_SAMPLE_INTERVAL", 1)
    total_frames_extracted = total_frames // (sample_interval * fps_rounded)

    # Compile metadata dictionary for downstream YOLO processing
    interview_metadata = {
        "interview_id": interview_id,
        "video_path": video_path,
        "duration_seconds": duration,
        "fps": fps,
        "fps_rounded": fps_rounded,
        "total_frames": total_frames,
        "total_frames_extracted": int(total_frames_extracted),
    }

    # Extract frames from the video using FFmpeg
    print("[cyan]Extracting frames from video...[/cyan]")
    frame_extraction(video_path, frames_dir, interview_id, fps_rounded)

    # Run standard YOLOv8 object detection models
    for model in models_v8:
        model_path = os.path.join("models", model)
        print(f"[cyan]Running YOLOv8 model: {model}[/cyan]")
        run_YOLOv8(interview_metadata, yolo_configs, model_path)

    # Run YOLOv8 pose estimation models
    for model in models_pose:
        model_path = os.path.join("models", model)
        print(f"[cyan]Running YOLOv8 Pose model: {model}[/cyan]")
        run_YOLO_pose_v8(interview_metadata, yolo_configs, model_path)
        
    print(f"[bold green]Processing complete for interview ID: {interview_id}![/bold green]")