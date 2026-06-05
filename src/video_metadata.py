import os
import ffmpeg


def video_metadata(video_path: str) -> tuple[int, float, int, int]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Extracting metadata from video: {video_path}")

    # ── extract video metadata ────────────────────────────────────────────────────────
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")

    num, den = video_info["r_frame_rate"].split("/")
    fps = round(float(num) / float(den))
    fps_rounded = round(fps)
    total_frames = int(video_info["nb_frames"])

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    duration = int(total_frames / fps)

    return (
        duration,
        fps,
        fps_rounded,
        total_frames,
    )
