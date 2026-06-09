import os
from src.frame_extraction import frame_extraction
from src.video_metadata import video_metadata
from src.run_yolo import run_YOLO_pose_v8, run_YOLOv8, run_YOLOv26


def process(video_path: str, models_v8: list[str], models_pose: list[str], models_v26: list[str]) -> None:
    # basic configuration and setup
    if not os.path.exists(video_path):
        print(f"[Error]: Video file {video_path} does not exist.")
        return

    interview_id = video_path.strip().split("/")[-1].split(".")[0]
    interview_path = os.path.join("output", interview_id)
    os.makedirs(interview_path, exist_ok=True)
    frames_dir = os.path.join(interview_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Processing video: {video_path}")

    # extract video metadata and frames
    duration, fps, fps_rounded, total_frames = video_metadata(video_path)
    frame_extraction(video_path, frames_dir, interview_id, fps_rounded)

    # configure YOLO models
    yolo_configs = {
        "YOLO_CONFIDENCE_CONFIRMED": 0.9,
        "YOLO_CONFIDENCE": 0.6,
        "PERSON_CLASS_ID": 0,
        "FRAME_SAMPLE_INTERVAL": 1,
        "BATCH_SIZE_GPU": 16,
        "BATCH_SIZE_CPU": 4,
        "MULTIPLE_PEOPLE_THRESHOLD": 1,
        "ABSENCE_THRESHOLD": 0,
    }

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not frame_files:
        raise RuntimeError(f"No frame images found in: {frames_dir}")

    print(f"Total frames extracted: {len(frame_files)}. Running YOLO models...")

    # run YOLO models and collect results
    json_data = {
        "interview_id": interview_id,
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
    }

    for model in models_v8:
        model_path = os.path.join("models", model)
        print(f"Running YOLO model: {model} on frames...")

        data = run_YOLOv8(frames_dir, frame_files, yolo_configs, model_path, fps_rounded)
        json_data[model] = data
    
    for model in models_pose:
        model_path = os.path.join("models", model)
        print(f"Running YOLO pose model: {model} on frames...")

        data = run_YOLO_pose_v8(frames_dir, frame_files, yolo_configs, model_path, fps_rounded)
        json_data[model] = data

    for model in models_v26:
        model_path = os.path.join("models", model)
        print(f"Running YOLO model: {model} on frames...")

        data = run_YOLOv26(frames_dir, frame_files, yolo_configs, model_path, fps_rounded)

        json_data[model] = data
    


    # save results to JSON
    output_json_path = os.path.join(interview_path, f"{interview_id}_results.json")
    with open(output_json_path, "w") as json_file:
        import json

        json.dump(json_data, json_file, indent=4)
