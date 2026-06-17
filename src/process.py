import os
import pandas as pd
from src.frame_extraction import frame_extraction
from src.video_metadata import video_metadata
from src.run_yolo import run_YOLO_pose_v8, run_YOLOv8, run_YOLOv26
from src.csv_utils import save_to_csv


def process(video_path: str, models_v8: list[str], models_pose: list[str], models_v26: list[str]) -> None:
    # basic configuration and setup
    if not os.path.exists(video_path):
        print(f"[Error]: Video file {video_path} does not exist.")
        return

    # configure YOLO models
    yolo_configs = {
        "YOLO_CONFIDENCE_CONFIRMED": 0.9,
        "YOLO_CONFIDENCE": 0.0,
        "PERSON_CLASS_ID": 0,
        "FRAME_SAMPLE_INTERVAL": 1,
        "BATCH_SIZE_GPU": 16,
        "BATCH_SIZE_CPU": 4,
        "MULTIPLE_PEOPLE_THRESHOLD": 1,
        "ABSENCE_THRESHOLD": 0,
    }

    interview_id = video_path.strip().split("/")[-1].replace("_final_camera.mp4", "")
    interview_path = os.path.join("output", interview_id)
    os.makedirs(interview_path, exist_ok=True)
    frames_dir = os.path.join(interview_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"\nProcessing interview with ID: {interview_id}")

    duration, fps, fps_rounded, total_frames = video_metadata(video_path)
    interview_metadata = {
        "interview_id": interview_id,
        "video_path": video_path,
        "duration_seconds": duration,
        "fps": fps,
        "fps_rounded": fps_rounded,
        "total_frames": total_frames,
        "total_frames_extracted": total_frames // (yolo_configs["FRAME_SAMPLE_INTERVAL"] * fps_rounded),
    }

    # extract video metadata and frames
    frame_extraction(video_path, frames_dir, interview_id, fps_rounded)

    for model in models_v8:
        model_path = os.path.join("models", model)
        run_YOLOv8(interview_metadata, yolo_configs, model_path)

    for model in models_pose:
        model_path = os.path.join("models", model)
        run_YOLO_pose_v8(interview_metadata, yolo_configs, model_path)

    # for model in models_v26:
    #     model_path = os.path.join("models", model)
    #     print(f"Running YOLO model: {model} on frames...")

    #     data = run_YOLOv26(frames_dir, frame_files, yolo_configs, model_path, fps_rounded)
    #     save_to_csv(data, (f"output/{interview_id}/{model}_results.csv").replace(".pt", ""))
