from src.process import process

models_v8 = [
    # "yolov8n.pt",
    # "yolov8s.pt",
    # "yolov8m.pt",
    "yolov8l.pt",
    # "yolov8x.pt",
]

models_pose = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt",
]

models_v26 = [
    # "yolo26n.pt",
    # "yolo26s.pt",
    # "yolo26m.pt",
    # "yolo26l.pt",
    # "yolo26x.pt",
]

video_paths = [
    # "input/694121e9f7e54c4adf74ade7_final_camera.mp4",
    # "input/69422f3af3ff46e9da4a8f49_final_camera.mp4",
    # "input/6944fba2f3ff46e9da4b5d12_final_camera.mp4",
    # "input/694562cd0bb70fa3a440721e_final_camera.mp4",
    # "input/694a8baf524205756b53b515_final_camera.mp4",
    # "input/694e3f763f538ba966a75877_final_camera.mp4",
    # "input/695d0b8a45375bf805a59a22_final_camera.mp4",
    # "input/69620cc245375bf805a6f488_final_camera.mp4",
    # "input/6936fce7df612f1f12a9697b_final_camera.mp4",
    # "input/6a17e5b58f4cd056c6b449fc_final_camera.mp4",
    # "input/6a27f96c00ceb75bfffa5dd7_final_camera.mp4",
    # "input/6a21ae7ab97dc11b8ea61007_final_camera.mp4",
    # "input/tour1001.mp4",
    # "input/tour1002.mp4",
    "input/6a34d2ab647c87729fd3344e_final_camera.mp4",
    "input/6a37afce647c87729fd7b659_final_camera.mp4",
    "input/6a284c0600ceb75bfffb06b0_final_camera.mp4",
    "input/6a284ce600ceb75bfffb0be5_final_camera.mp4",
    "input/6a351d88647c87729fd43001_final_camera.mp4",
    "input/6a354ddb647c87729fd4b85c_final_camera.mp4",
    "input/6a350382647c87729fd3c832_final_camera.mp4",
    "input/6a353929647c87729fd47461_final_camera.mp4",

]

if __name__ == "__main__":
    print("Starting video processing with YOLO models...")
    # video_path = "input/6936fce7df612f1f12a9697b_final_camera.mp4"
    for video_path in video_paths:
        process(video_path, models_v8, models_pose, models_v26)
