from src.process import process

models = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    # "yolov26n.pt",
    # "yolov26s.pt",
    # "yolov26m.pt",
    # "yolov26l.pt",
    # "yolov26x.pt",
]

if __name__ == "__main__":
    print("Starting video processing with YOLO models...")
    video_path = "input/6936fce7df612f1f12a9697b_final_camera.mp4"
    process(video_path, models)
