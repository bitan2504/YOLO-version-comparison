import torch
import os
import pandas as pd
from ultralytics import YOLO
from src.av_utils import load_image_with_av


def run_YOLOv8(interview_metadata: dict, yolo_configs: dict, model_path: str) -> None:
    print(f"\n#{'-' * 40} Running YOLOv8: {os.path.basename(model_path)} {'-' * 40}#")

    interview_id = interview_metadata["interview_id"]
    frames_dir = os.path.join("output", interview_id, "frames")
    total_frames_extracted = interview_metadata["total_frames_extracted"]
    fps_rounded = interview_metadata["fps_rounded"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]
    batch = []

    results_data = []

    def process():
        frames_batch = [item["frame"] for item in batch]
        results = model(
            frames_batch,
            conf=yolo_configs["YOLO_CONFIDENCE"],
            device=device,
            verbose=False,
        )

        for index, result in enumerate(results):
            confidences = [float(box.conf[0]) for box in result.boxes[:2]]
            time = batch[index]["time"]

            max_conf = confidences[0] if len(confidences) > 0 else 0.0
            max_conf_2 = confidences[1] if len(confidences) > 1 else 0.0

            results_data.append({"time": time, "max_conf": max_conf, "max_conf_2": max_conf_2})

    for index in range(total_frames_extracted):
        frame_name = f"{index:08d}.jpg"
        frame_processing = (100 * index) // total_frames_extracted + 1
        print(f"Processing: [{'#' * frame_processing + ' ' * (100 - frame_processing)}]", end="\r", flush=True)

        frame_path = os.path.join(frames_dir, frame_name)
        frame = load_image_with_av(frame_path)
        if frame is None:
            print(f"\nWARNING: Could not read frame: {frame_path} — skipping")
            continue

        time = index * yolo_configs["FRAME_SAMPLE_INTERVAL"]
        frame_number = int(time * fps_rounded)
        batch.append(
            {
                "frame_number": frame_number,
                "frame": frame,
                "time": time,
            }
        )

        if len(batch) >= batch_size or index == total_frames_extracted - 1:
            process()
            batch.clear()

    print("\nProcessing complete.")

    dataframe = pd.DataFrame(results_data)
    output_csv_path = f"output/{interview_id}/{os.path.basename(model_path).replace('.pt', '')}_results.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    dataframe.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")


def run_YOLO_pose_v8(interview_metadata: dict, yolo_configs: dict, model_path: str) -> None:
    keypoints = [
        "Nose",
        "L_Eye",
        "R_Eye",
        "L_Ear",
        "R_Ear",
        "L_Shoulder",
        "R_Shoulder",
        "L_Elbow",
        "R_Elbow",
        "L_Wrist",
        "R_Wrist",
        "L_Hip",
        "R_Hip",
        "L_Knee",
        "R_Knee",
        "L_Ankle",
        "R_Ankle",
    ]

    print(f"\n#{"-" * 40} Running YOLOv8: {os.path.basename(model_path)} {"-" * 40}#")

    interview_id = interview_metadata["interview_id"]
    frames_dir = os.path.join("output", interview_id, "frames")
    total_frames_extracted = interview_metadata["total_frames_extracted"]
    fps_rounded = interview_metadata["fps_rounded"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]
    batch = []

    results_data = []

    def process():
        frames_batch = [item["frame"] for item in batch]
        results = model(
            frames_batch,
            conf=yolo_configs["YOLO_CONFIDENCE"],
            device=device,
            verbose=False,
        )

        for index, result in enumerate(results):
            max_conf = float(result.boxes[0].conf[0]) if len(result.boxes) > 0 else 0.0
            keypoints_conf = [float(conf) for conf in result.keypoints.conf[0]]
            results_data.append(
                {
                    "time": batch[index]["time"],
                    "max_conf": max_conf,
                    **{f"{keypoint}_conf": keypoints_conf[i] for i, keypoint in enumerate(keypoints)},
                }
            )

    for index in range(total_frames_extracted):
        frame_name = f"{index:08d}.jpg"
        frame_processing = (100 * index) // total_frames_extracted + 1
        print(f"Processing: [{'#' * frame_processing + ' ' * (100 - frame_processing)}]", end="\r", flush=True)

        frame_path = os.path.join(frames_dir, frame_name)
        frame = load_image_with_av(frame_path)
        if frame is None:
            print(f"\nWARNING: Could not read frame: {frame_path} — skipping")
            continue

        time = index * yolo_configs["FRAME_SAMPLE_INTERVAL"]
        frame_number = int(time * fps_rounded)
        batch.append(
            {
                "frame_number": frame_number,
                "frame": frame,
                "time": time,
            }
        )

        if len(batch) >= batch_size or index == total_frames_extracted - 1:
            process()
            batch.clear()

    print("\nProcessing complete.")

    dataframe = pd.DataFrame(results_data)
    output_csv_path = f"output/{interview_id}/{os.path.basename(model_path).replace('.pt', '')}_results.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    dataframe.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")


def run_YOLOv26(frames_dir: str, frame_files: list, yolo_configs: dict, model_path: str, fps_rounded: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for YOLO26 inference")

    # Loads the YOLO26 model (e.g., 'yolo26n.pt')
    model = YOLO(model_path)
    model.to(device)

    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]

    # Batch tracking lists
    frames_batch = []
    seconds_batch = []
    frame_numbers_batch = []

    data = []

    def process():
        # YOLO26 is natively NMS-free, ensuring predictable inference times
        results = model(
            frames_batch,
            conf=yolo_configs["YOLO_CONFIDENCE"],
            device=device,
            half=(device == "cuda"),  # Leverage YOLO26 FP16 optimization on GPU
            verbose=False,
        )

        for i, r in enumerate(results):
            confidences = [float(box.conf[0]) for box in (r.boxes or []) if int(box.cls[0]) == yolo_configs["PERSON_CLASS_ID"]]

            num_people = len(confidences)
            sec_val = seconds_batch[i]
            frame_num = frame_numbers_batch[i]

            verdict = "safe"

            confidences.sort(reverse=True)
            if len(confidences) > 1 and confidences[1] >= yolo_configs["YOLO_CONFIDENCE"]:
                verdict = "multiple_people"
            elif len(confidences) == 0 or confidences[0] < yolo_configs["YOLO_CONFIDENCE"]:
                verdict = "absence"

            data.append(
                {
                    "time_seconds": sec_val,
                    # "time_minutes": round(sec_val / 60, 2),
                    # "frame_number": frame_num,
                    # "num_people": num_people,
                    "max_confidence": confidences[0] if confidences else 0.0,
                    "max_confidence_2": confidences[1] if len(confidences) > 1 else 0.0,
                    "verdict": verdict,
                }
            )

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        print(f"Processing frame {idx + 1}/{len(frame_files)}", end="\r")

        frame = load_image_with_av(frame_path)
        if frame is None:
            print(f"\nWARNING: Could not read frame: {frame_path} — skipping")
            continue

        sec_val = idx * yolo_configs["FRAME_SAMPLE_INTERVAL"]
        frame_number = int(sec_val * fps_rounded)

        # Append to batches
        frames_batch.append(frame)
        seconds_batch.append(sec_val)
        frame_numbers_batch.append(frame_number)

        # Process the batch if full, or if it's the last frame
        if len(frames_batch) >= batch_size or idx == len(frame_files) - 1:
            process()

            # Clear batches for the next sequence
            frames_batch.clear()
            seconds_batch.clear()
            frame_numbers_batch.clear()

    print()  # Clear the loading line

    return data
