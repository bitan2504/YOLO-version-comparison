import os
import ffmpeg
from rich import print


def frame_extraction(video_path: str, frames_dir: str, interview_id: str, fps_rounded: int) -> None:
    """
    Extracts frames from a video file using FFmpeg, saving one frame per second.
    
    Args:
        video_path (str): Path to the input video file.
        frames_dir (str): Directory where the extracted frames will be saved.
        interview_id (str): Unique identifier for the interview.
        fps_rounded (int): Rounded frames per second of the video (used to calculate step size).
        
    Raises:
        RuntimeError: If FFmpeg fails to extract the frames.
    """
    print(f"[cyan]Extracting frames from video:[/cyan] {video_path} [cyan]at 1 frame per {fps_rounded} frames...[/cyan]")
    
    try:
        # Build and run the FFmpeg pipeline
        (
            ffmpeg.input(video_path)
            .output(
                # Output file pattern (e.g., 00000000.jpg, 00000001.jpg)
                os.path.join(frames_dir, "%08d.jpg"),
                
                # Video filter: Extract exactly 1 frame every 'fps_rounded' frames (i.e., 1 FPS)
                vf=f"select='not(mod(n,{fps_rounded}))'",
                fps_mode="passthrough",
                start_number="0",
                
                # JPEG Quality settings (qscale:v 1, qmin 1, qmax 1 forces the highest possible quality)
                **{"qscale:v": 1},
                qmin=1,
                qmax=1,
            )
            # Suppress standard output to keep the console clean; only show errors
            .global_args("-loglevel", "error")
            .run(overwrite_output=True)
        )

    except ffmpeg.Error as e:
        # Safely decode the FFmpeg error stream, falling back to a generic message if empty
        error_msg = e.stderr.decode('utf-8') if e.stderr else "Unknown FFmpeg error."
        print(f"[bold red]Error during frame extraction for {interview_id}:[/bold red]\n{error_msg}")
        raise RuntimeError(f"Frame extraction failed for {interview_id}. See error message above.")

    print(f"[bold green]Frame extraction completed for {interview_id}.[/bold green] Frames saved in: {frames_dir}")