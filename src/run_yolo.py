import cv2
import torch
import os
import pandas as pd
from ultralytics import YOLO
from src.av_utils import load_image_with_av

box_color = [(255, 0, 0), (0, 255, 0)]
nose_color = (0, 0, 255)
eye_color = (0, 0, 255)
ear_color = (0, 255, 255)
point_radius = 4
point_thickness = -1


def run_YOLOv8(interview_metadata: dict, yolo_configs: dict, model_path: str) -> None:
    interview_id = interview_metadata["interview_id"]
    frames_dir = os.path.join("output", interview_id, "frames")
    total_frames_extracted = interview_metadata["total_frames_extracted"]
    fps_rounded = interview_metadata["fps_rounded"]
    model_name = os.path.basename(model_path).replace(".pt", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]
    batch = []
    results_data = []

    def process():
        frames_batch = [item["frame"] for item in batch]
        time_batch = [item["time"] for item in batch]
        index_batch = [item["original_index"] for item in batch]

        results = model(
            frames_batch,
            conf=yolo_configs["YOLO_CONFIDENCE"],
            classes=[yolo_configs["PERSON_CLASS_ID"]],
            device=device,
            verbose=False,
        )

        for i, result in enumerate(results):
            boxes = result.boxes[:2]
            confidences = [float(box.conf[0]) for box in boxes]
            annotated_frame = frames_batch[i].copy()

            for box_index, box in enumerate(boxes):
                x1, y1, x2, y2 = (
                    int(box.xyxy[0][0]),
                    int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]),
                    int(box.xyxy[0][3]),
                )
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color[box_index], 2)

            annotated_frame_dir = os.path.join(frames_dir, model_name)
            os.makedirs(annotated_frame_dir, exist_ok=True)

            annotated_frame_path = os.path.join(annotated_frame_dir, f"{index_batch[i]:08d}.jpg")
            cv2.imwrite(annotated_frame_path, annotated_frame)

            max_conf = confidences[0] if len(confidences) > 0 else 0.0
            max_conf_2 = confidences[1] if len(confidences) > 1 else 0.0

            area_1 = float(boxes[0].xywh[0][2] * boxes[0].xywh[0][3]) if len(boxes) > 0 else 0.0
            area_2 = float(boxes[1].xywh[0][2] * boxes[1].xywh[0][3]) if len(boxes) > 1 else 0.0

            frame_area = frames_batch[i].shape[0] * frames_batch[i].shape[1]
            area_ratio_1 = area_1 / frame_area if frame_area > 0 else 0.0
            area_ratio_2 = area_2 / frame_area if frame_area > 0 else 0.0

            results_data.append(
                {
                    "time": time_batch[i],
                    "max_conf": max_conf,
                    "max_conf_2": max_conf_2,
                    "area_1": area_1,
                    "area_2": area_2,
                    "frame_area": frame_area,
                    "area_ratio_1": area_ratio_1,
                    "area_ratio_2": area_ratio_2,
                }
            )

            if time_batch[i] == 1195 or time_batch[i] == 1196:
                print(int(result.boxes.cls[0]))

    for index in range(total_frames_extracted):
        frame_name = f"{index:08d}.jpg"
        frame_processing = (100 * index) // total_frames_extracted + 1
        print(f"{model_name}:\t[{'#' * frame_processing + ' ' * (100 - frame_processing)}]", end=f"{'\n' if index == total_frames_extracted - 1 else '\r'}", flush=True)

        frame_path = os.path.join(frames_dir, frame_name)
        frame = load_image_with_av(frame_path)

        if frame is None:
            print(f"\nWARNING: Could not read frame: {frame_path} — skipping")
            continue

        time = index * yolo_configs["FRAME_SAMPLE_INTERVAL"]
        frame_number = int(time * fps_rounded)

        batch.append(
            {
                "original_index": index,
                "frame_number": frame_number,
                "frame": frame,
                "time": time,
            }
        )

        if len(batch) >= batch_size:
            process()
            batch.clear()

    if len(batch) > 0:
        process()
        batch.clear()

    dataframe = pd.DataFrame(results_data)
    output_csv_path = f"output/{interview_id}/{model_name}_results.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    dataframe.to_csv(output_csv_path, index=False)


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

    interview_id = interview_metadata["interview_id"]
    frames_dir = os.path.join("output", interview_id, "frames")
    total_frames_extracted = interview_metadata["total_frames_extracted"]
    fps_rounded = interview_metadata["fps_rounded"]
    model_name = os.path.basename(model_path).replace(".pt", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]
    batch = []

    results_data = []

    def process():
        frames_batch = [item["frame"] for item in batch]
        time_batch = [item["time"] for item in batch]
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

            for box_index, box in enumerate(boxes):
                x1, y1, x2, y2 = (
                    int(box.xyxy[0][0]),
                    int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]),
                    int(box.xyxy[0][3]),
                )

                current_keypoints = result.keypoints.xy[box_index]

                x_nose, y_nose = int(current_keypoints[0][0]), int(current_keypoints[0][1])
                x_leye, y_leye = int(current_keypoints[1][0]), int(current_keypoints[1][1])
                x_reye, y_reye = int(current_keypoints[2][0]), int(current_keypoints[2][1])
                x_lear, y_lear = int(current_keypoints[3][0]), int(current_keypoints[3][1])
                x_rear, y_rear = int(current_keypoints[4][0]), int(current_keypoints[4][1])

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color[box_index], 2)
                cv2.circle(annotated_frame, (x_nose, y_nose), point_radius, nose_color, point_thickness)
                cv2.circle(annotated_frame, (x_leye, y_leye), point_radius, eye_color, point_thickness)
                cv2.circle(annotated_frame, (x_reye, y_reye), point_radius, eye_color, point_thickness)
                cv2.circle(annotated_frame, (x_lear, y_lear), point_radius, ear_color, point_thickness)
                cv2.circle(annotated_frame, (x_rear, y_rear), point_radius, ear_color, point_thickness)

                if time_batch[index] == 67:
                    print(
                        f"\nFrame {time_batch[index]}: Conf({box.conf[0]:.2f}) Rectangle ({x1}, {y1}, {x2}, {y2}), Nose ({x_nose}, {y_nose}), L_Eye ({x_leye}, {y_leye}), R_Eye ({x_reye}, {y_reye}), L_Ear ({x_lear}, {y_lear}), R_Ear ({x_rear}, {y_rear})"
                    )

            annotated_frame_dir = os.path.join(frames_dir, model_name)
            os.makedirs(annotated_frame_dir, exist_ok=True)
            annotated_frame_path = os.path.join(annotated_frame_dir, f"{time_batch[index]:08d}.jpg")
            cv2.imwrite(annotated_frame_path, annotated_frame)

            max_conf = float(boxes[0].conf[0]) if len(boxes) > 0 else 0.0

            area = (float(boxes[0].xyxy[0][2]) - float(boxes[0].xyxy[0][0])) * (float(boxes[0].xyxy[0][3]) - float(boxes[0].xyxy[0][1])) if len(boxes) > 0 else 0.0
            frame_area = frames_batch[index].shape[0] * frames_batch[index].shape[1]
            area_ratio = area / frame_area if frame_area > 0 else 0.0

            keypoints_conf = [float(conf) for conf in result.keypoints.conf[0]]
            results_data.append(
                {
                    "time": time_batch[index],
                    "max_conf": max_conf,
                    "area": area,
                    "frame_area": frame_area,
                    "area_ratio": area_ratio,
                    **{f"{keypoint}_conf": keypoints_conf[i] for i, keypoint in enumerate(keypoints)},
                }
            )

    for index in range(total_frames_extracted):
        frame_name = f"{index:08d}.jpg"
        frame_processing = (100 * index) // total_frames_extracted + 1
        print(f"{model_name}:\t[{'#' * frame_processing + ' ' * (100 - frame_processing)}]", end=f"{'\n' if index == total_frames_extracted - 1 else '\r'}", flush=True)

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

    dataframe = pd.DataFrame(results_data)
    output_csv_path = f"output/{interview_id}/{model_name}_results.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    dataframe.to_csv(output_csv_path, index=False)


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
