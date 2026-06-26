import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from src.av_utils import load_image_with_av
from rich import print

# --- Constants & Configuration ---
BOX_COLORS = [(255, 0, 0), (0, 255, 0)]  # Blue for first box, Green for second
NOSE_COLOR = (0, 0, 255)
EYE_COLOR = (0, 0, 255)
EAR_COLOR = (0, 255, 255)

POINT_RADIUS = 4
POINT_THICKNESS = -1
KEYPOINT_NAMES = ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear"]
PROGRESS_BAR_LENGTH = 60


def run_YOLOv8(interview_metadata: dict, yolo_configs: dict, model_path: str) -> None:
    """
    Runs standard YOLOv8 object detection on extracted interview frames in batches,
    annotates the frames, and saves the detection metadata to a CSV.
    """
    interview_id = interview_metadata["interview_id"]
    frames_dir = os.path.join("output", interview_id, "frames")
    total_frames = interview_metadata["total_frames_extracted"]
    fps_rounded = interview_metadata["fps_rounded"]
    model_name = os.path.basename(model_path).replace(".pt", "")

    # Hardware setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]
    
    batch = []
    results_data = []

    def process_batch():
        """Processes the current batch of frames through the YOLO model."""
        frames_batch = [item["frame"] for item in batch]
        time_batch = [item["time"] for item in batch]
        index_batch = [item["original_index"] for item in batch]

        # Run inference
        results = model(
            frames_batch,
            conf=yolo_configs["YOLO_CONFIDENCE"],
            classes=[yolo_configs["PERSON_CLASS_ID"]],
            device=device,
            verbose=False,
        )

        for i, result in enumerate(results):
            # Limit to top 2 detected boxes
            boxes = result.boxes[:2]
            confidences = [float(box.conf[0]) for box in boxes]
            annotated_frame = frames_batch[i].copy()
            frame_area = frames_batch[i].shape[0] * frames_batch[i].shape[1]

            # Draw bounding boxes on the frame
            for box_index, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLORS[box_index], 2)

            # Ensure output directory exists and save the annotated frame
            annotated_frame_dir = os.path.join(frames_dir, model_name)
            os.makedirs(annotated_frame_dir, exist_ok=True)
            annotated_frame_path = os.path.join(annotated_frame_dir, f"{index_batch[i]:08d}.jpg")
            cv2.imwrite(annotated_frame_path, annotated_frame)

            # Extract metrics safely
            max_conf = confidences[0] if len(confidences) > 0 else 0.0
            max_conf_2 = confidences[1] if len(confidences) > 1 else 0.0

            area_1 = float(boxes[0].xywh[0][2] * boxes[0].xywh[0][3]) if len(boxes) > 0 else 0.0
            area_2 = float(boxes[1].xywh[0][2] * boxes[1].xywh[0][3]) if len(boxes) > 1 else 0.0

            # Store metrics for CSV
            results_data.append({
                "time": time_batch[i],
                "max_conf": max_conf,
                "max_conf_2": max_conf_2,
                "area_1": area_1,
                "area_2": area_2,
                "frame_area": frame_area,
                "area_ratio_1": (area_1 / frame_area) if frame_area > 0 else 0.0,
                "area_ratio_2": (area_2 / frame_area) if frame_area > 0 else 0.0,
            })

    # Main loop: Iterate through all frames and build batches
    for index in range(total_frames):
        # Console progress bar
        progress_pct = int((index + 1) / total_frames * PROGRESS_BAR_LENGTH)
        bar = ('#' * progress_pct) + ('-' * (PROGRESS_BAR_LENGTH - progress_pct))
        ending = '\n' if index == total_frames - 1 else '\r'
        print(f"{model_name}:\t[bold]{bar}[/bold]", end=ending, flush=True)

        frame_path = os.path.join(frames_dir, f"{index:08d}.jpg")
        frame = load_image_with_av(frame_path)

        if frame is None:
            print(f"\n[bold yellow]WARNING: Could not read frame: {frame_path} — skipping")
            continue

        time = index * yolo_configs["FRAME_SAMPLE_INTERVAL"]
        
        batch.append({
            "original_index": index,
            "frame_number": int(time * fps_rounded),
            "frame": frame,
            "time": time,
        })

        # Process when batch is full
        if len(batch) >= batch_size:
            process_batch()
            batch.clear()

    # Process any remaining frames in the final batch
    if batch:
        process_batch()
        batch.clear()

    # Save results to CSV
    output_csv_path = os.path.join("output", interview_id, f"{model_name}_results.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    pd.DataFrame(results_data).to_csv(output_csv_path, index=False)


def run_YOLO_pose_v8(interview_metadata: dict, yolo_configs: dict, model_path: str) -> None:
    """
    Runs YOLOv8 pose estimation to detect bounding boxes and facial/body keypoints,
    annotates the frames, and saves the structural metadata to a CSV.
    """
    interview_id = interview_metadata["interview_id"]
    frames_dir = os.path.join("output", interview_id, "frames")
    total_frames = interview_metadata["total_frames_extracted"]
    fps_rounded = interview_metadata["fps_rounded"]
    model_name = os.path.basename(model_path).replace(".pt", "")

    # Hardware setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]
    
    batch = []
    results_data = []

    def process_batch():
        """Processes the current batch of frames through the YOLO Pose model."""
        frames_batch = [item["frame"] for item in batch]
        time_batch = [item["time"] for item in batch]
        
        # Run inference
        results = model(
            frames_batch,
            conf=yolo_configs["YOLO_CONFIDENCE"],
            classes=[yolo_configs["PERSON_CLASS_ID"]],
            device=device,
            verbose=False,
        )

        for index, result in enumerate(results):
            boxes = result.boxes[:2]
            annotated_frame = frames_batch[index].copy()
            frame_area = frames_batch[index].shape[0] * frames_batch[index].shape[1]

            # Iterate through up to 2 detected persons
            for box_index, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLORS[box_index], 2)

                # Ensure keypoints exist before trying to draw them
                if result.keypoints is not None and len(result.keypoints.xy) > box_index:
                    current_keypoints = result.keypoints.xy[box_index]
                    
                    # Extract (x, y) for the 5 facial keypoints
                    points = {
                        "nose": (int(current_keypoints[0][0]), int(current_keypoints[0][1])),
                        "leye": (int(current_keypoints[1][0]), int(current_keypoints[1][1])),
                        "reye": (int(current_keypoints[2][0]), int(current_keypoints[2][1])),
                        "lear": (int(current_keypoints[3][0]), int(current_keypoints[3][1])),
                        "rear": (int(current_keypoints[4][0]), int(current_keypoints[4][1])),
                    }

                    # Draw keypoints
                    cv2.circle(annotated_frame, points["nose"], POINT_RADIUS, NOSE_COLOR, POINT_THICKNESS)
                    cv2.circle(annotated_frame, points["leye"], POINT_RADIUS, EYE_COLOR, POINT_THICKNESS)
                    cv2.circle(annotated_frame, points["reye"], POINT_RADIUS, EYE_COLOR, POINT_THICKNESS)
                    cv2.circle(annotated_frame, points["lear"], POINT_RADIUS, EAR_COLOR, POINT_THICKNESS)
                    cv2.circle(annotated_frame, points["rear"], POINT_RADIUS, EAR_COLOR, POINT_THICKNESS)

            # Save annotated frame
            annotated_frame_dir = os.path.join(frames_dir, model_name)
            os.makedirs(annotated_frame_dir, exist_ok=True)
            annotated_frame_path = os.path.join(annotated_frame_dir, f"{time_batch[index]:08d}.jpg")
            cv2.imwrite(annotated_frame_path, annotated_frame)

            # Gather data for CSV export
            max_conf = float(boxes[0].conf[0]) if len(boxes) > 0 else 0.0
            area = float(boxes[0].xywh[0][2] * boxes[0].xywh[0][3]) if len(boxes) > 0 else 0.0
            
            # Dictionary for the current row
            row_data = {
                "time": time_batch[index],
                "max_conf": max_conf,
                "area": area,
                "frame_area": frame_area,
                "area_ratio": (area / frame_area) if frame_area > 0 else 0.0,
            }

            # Safely append keypoint confidences if they exist
            if len(boxes) > 0 and result.keypoints is not None and len(result.keypoints.conf) > 0:
                keypoints_conf = [float(conf) for conf in result.keypoints.conf[0]]
                # Map available keypoints to names safely to avoid IndexError
                for i, name in enumerate(KEYPOINT_NAMES):
                    row_data[f"{name}_conf"] = keypoints_conf[i] if i < len(keypoints_conf) else 0.0
            else:
                for name in KEYPOINT_NAMES:
                    row_data[f"{name}_conf"] = 0.0

            results_data.append(row_data)

    # Main loop: Iterate through frames and process batches
    for index in range(total_frames):
        # Console progress bar
        progress_pct = int((index + 1) / total_frames * PROGRESS_BAR_LENGTH)
        bar = '#' * progress_pct + '-' * (PROGRESS_BAR_LENGTH - progress_pct)
        ending = '\n' if index == total_frames - 1 else '\r'
        print(f"{model_name}:\t[bold]{bar}[/bold]", end=ending, flush=True)

        frame_path = os.path.join(frames_dir, f"{index:08d}.jpg")
        frame = load_image_with_av(frame_path)
        
        if frame is None:
            print(f"\n[bold yellow]WARNING: Could not read frame: {frame_path} — skipping")
            continue

        time = index * yolo_configs["FRAME_SAMPLE_INTERVAL"]
        
        batch.append({
            "frame_number": int(time * fps_rounded),
            "frame": frame,
            "time": time,
        })

        # Process when batch is full or on the very last frame
        if len(batch) >= batch_size or index == total_frames - 1:
            process_batch()
            batch.clear()

    # Save results to CSV
    output_csv_path = os.path.join("output", interview_id, f"{model_name}_results.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    pd.DataFrame(results_data).to_csv(output_csv_path, index=False)