import os
from rich import print
from src.process import process

# --- Configuration & Assets ---

# Standard YOLOv8 object detection models to execute
MODELS_V8 = [
    "yolov8n.pt",
    # "yolov8s.pt",
    # "yolov8m.pt",
    # "yolov8l.pt",
    # "yolov8x.pt",
]

# YOLOv8 pose estimation models to execute
MODELS_POSE = [
    "yolov8n-pose.pt",
    # "yolov8s-pose.pt",
    # "yolov8m-pose.pt",
    # "yolov8l-pose.pt",
    # "yolov8x-pose.pt",
]

# Target video files to process (uncomment to activate)
VIDEO_PATHS = [
    # "input/694121e9f7e54c4adf74ade7_final_camera.mp4",
    # "input/69422f3af3ff46e9da4a8f49_final_camera.mp4",
    # "input/6944fba2f3ff46e9da4b5d12_final_camera.mp4",
    # "input/694562cd0bb70fa3a440721e_final_camera.mp4",
    # "input/694a8baf524205756b53b515_final_camera.mp4",
    # "input/694e3f763f538ba966a75877_final_camera.mp4",
    # "input/695d0b8a45375bf805a59a22_final_camera.mp4",
    # "input/69620cc245375bf805a6f488_final_camera.mp4",
    "input/6936fce7df612f1f12a9697b_final_camera.mp4",
    # "input/6a17e5b58f4cd056c6b449fc_final_camera.mp4",
    # "input/6a27f96c00ceb75bfffa5dd7_final_camera.mp4",
    # "input/6a21ae7ab97dc11b8ea61007_final_camera.mp4",
    # "input/tour1001.mp4",
    # "input/tour1002.mp4",
    # "input/6a34d2ab647c87729fd3344e_final_camera.mp4",
    # "input/6a37afce647c87729fd7b659_final_camera.mp4",
    # "input/6a284c0600ceb75bfffb06b0_final_camera.mp4",
    # "input/6a284ce600ceb75bfffb0be5_final_camera.mp4",
    # "input/6a351d88647c87729fd43001_final_camera.mp4",
    # "input/6a354ddb647c87729fd4b85c_final_camera.mp4",
    # "input/6a350382647c87729fd3c832_final_camera.mp4",
    # "input/6a353929647c87729fd47461_final_camera.mp4",
]

# Hyperparameters for YOLO inference and pipeline logic
YOLO_CONFIGS = {
    "YOLO_CONFIDENCE": 0.0,             # Confidence threshold for valid detections
    "PERSON_CLASS_ID": 0,               # COCO class ID for 'person'
    "FRAME_SAMPLE_INTERVAL": 1,         # Interval at which frames are sampled
    "BATCH_SIZE_GPU": 16,               # Batch size for CUDA-enabled devices
    "BATCH_SIZE_CPU": 4,                # Batch size for CPU fallback
    "MULTIPLE_PEOPLE_THRESHOLD": 1,     # Detection threshold for triggering multiple people logic
    "ABSENCE_THRESHOLD": 0,             # Detection threshold for triggering absence logic
}


def main() -> None:
    """
    Main execution pipeline. Iterates through enabled video paths, 
    running the configured YOLO object detection and pose estimation models.
    """
    print("[bold cyan]=== Starting Video Processing Pipeline ===[/bold cyan]")
    print("[bold]Active Configurations:[/bold]")
    print(YOLO_CONFIGS)
    print("[bold cyan]============================================[/bold cyan]\n")

    # Guardrails: Ensure there is actually work to do before initializing 
    if not VIDEO_PATHS:
        print("[bold yellow]WARNING: No video paths are currently enabled in VIDEO_PATHS.[/bold yellow]")
        return
    
    if not MODELS_V8 and not MODELS_POSE:
        print("[bold yellow]WARNING: No models (V8 or Pose) are currently enabled.[/bold yellow]")
        return

    # Iterate through the batch of videos
    for index, video_path in enumerate(VIDEO_PATHS, start=1):
        print(f"\n[bold magenta]Processing Video {index}/{len(VIDEO_PATHS)}:[/bold magenta] {video_path}")
        
        try:
            # Delegate to the processing pipeline
            process(
                video_path=video_path, 
                models_v8=MODELS_V8, 
                models_pose=MODELS_POSE, 
                yolo_configs=YOLO_CONFIGS
            )
        except Exception as e:
            # Catch errors to prevent a single bad video from breaking the entire batch loop
            print(f"[bold red]Failed to process {video_path}: {e}[/bold red]")

    print("\n[bold green]Pipeline execution finished![/bold green]")


if __name__ == "__main__":
    main()