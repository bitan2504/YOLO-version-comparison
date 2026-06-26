import os
import ffmpeg


def frame_extraction(video_path: str, frames_dir: str, interview_id: str, fps_rounded: int) -> None:
    try:
        (
            ffmpeg.input(video_path)
            .output(
                os.path.join(frames_dir, f"%08d.jpg"),
                vf=f"select='not(mod(n,{fps_rounded}))'",
                fps_mode="passthrough",
                start_number="0",
                **{
                    "qscale:v": 1,
                },
                qmin=1,
                qmax=1,
            )
            .global_args("-loglevel", "error")
            .run(overwrite_output=True)
        )

    except ffmpeg.Error as e:
        print(f"ffmpeg error during frame extraction: {e.stderr.decode()}")
        raise RuntimeError(f"ffmpeg failed to extract frames from video: {video_path}")
