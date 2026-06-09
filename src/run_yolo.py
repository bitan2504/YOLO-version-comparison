import torch
from ultralytics import YOLO
import os
from src.av_utils import load_image_with_av, save_image_with_av


def run_YOLOv8(frames_dir: str, frame_files: list, yolo_configs: dict, model_path: str, fps_rounded: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for YOLO inference")
    model = YOLO(model_path)
    model.to(device)

    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]
    frames_batch = []
    seconds_batch = []
    frame_numbers_batch = []
    data = []

    def process():
        results = model(
            frames_batch,
            conf=yolo_configs["YOLO_CONFIDENCE"],
            device=device,
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
                    "time_minutes": round(sec_val / 60, 2),
                    "frame_number": frame_num,
                    "num_people": num_people,
                    "max_confidence": confidences[0] if confidences else 0.0,
                    "verdict": verdict,
                }
            )

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        print(f"Processing frame {idx + 1}/{len(frame_files)}", end="\r")

        frame = load_image_with_av(frame_path)
        if frame is None:
            print(f"WARNING: Could not read frame: {frame_path} — skipping")
            continue

        sec_val = idx * yolo_configs["FRAME_SAMPLE_INTERVAL"]
        frame_number = int(sec_val * fps_rounded)
        frames_batch.append(frame)
        seconds_batch.append(sec_val)
        frame_numbers_batch.append(frame_number)

        if len(frames_batch) >= batch_size or idx == len(frame_files) - 1:
            process()
            frames_batch.clear()
            seconds_batch.clear()
            frame_numbers_batch.clear()

    print()  # Clear the loading line

    return data


map = dict(
    {
        0: "Nose",
        1: "Left Eye",
        2: "Right Eye",
        3: "Left Ear",
        4: "Right Ear",
        5: "Left Shoulder",
        6: "Right Shoulder",
        7: "Left Elbow",
        8: "Right Elbow",
        9: "Left Wrist",
        10: "Right Wrist",
        11: "Left Hip",
        12: "Right Hip",
        13: "Left Knee",
        14: "Right Knee",
        15: "Left Ankle",
        16: "Right Ankle",
    }
)


def run_YOLO_pose_v8(frames_dir: str, frame_files: list, yolo_configs: dict, model_path: str, fps_rounded: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for YOLO inference")
    model = YOLO(model_path)
    model.to(device)

    batch_size = yolo_configs["BATCH_SIZE_GPU"] if device == "cuda" else yolo_configs["BATCH_SIZE_CPU"]
    frames_batch = []
    seconds_batch = []
    frame_numbers_batch = []
    data = []

    def process():
        results = model(
            frames_batch,
            conf=yolo_configs["YOLO_CONFIDENCE"],
            device=device,
            verbose=False,
        )


        for i, r in enumerate(results):
            person_dict = {}
            if r.keypoints is not None and len(r.keypoints) > 0:
                kpts_list = r.keypoints.data.cpu().numpy().tolist()
                kpts = kpts_list[0] if len(kpts_list) > 0 else []
                for idx, kpt in enumerate(kpts):
                    kpt_name = map.get(idx, map[idx])
                    person_dict[kpt_name] = float(kpt[2])
        
            data.append({
                "time_seconds": seconds_batch[i],
                "keypoints": person_dict,
            })

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        print(f"Processing frame {idx + 1}/{len(frame_files)}", end="\r")

        frame = load_image_with_av(frame_path)
        if frame is None:
            print(f"WARNING: Could not read frame: {frame_path} — skipping")
            continue

        sec_val = idx * yolo_configs["FRAME_SAMPLE_INTERVAL"]
        frame_number = int(sec_val * fps_rounded)
        frames_batch.append(frame)
        seconds_batch.append(sec_val)
        frame_numbers_batch.append(frame_number)

        if len(frames_batch) >= batch_size or idx == len(frame_files) - 1:
            process()
            frames_batch.clear()
            seconds_batch.clear()
            frame_numbers_batch.clear()

    print()  # Clear the loading line

    return data


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
                    "time_minutes": round(sec_val / 60, 2),
                    "frame_number": frame_num,
                    "num_people": num_people,
                    "max_confidence": confidences[0] if confidences else 0.0,
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
