import ffmpeg
from rich import print


def video_metadata(video_path: str) -> tuple[int, float, int, int]:
    """
    Extracts metadata from a video file using FFmpeg.
    
    Args:
        video_path (str): Path to the input video file.
        
    Returns:
        tuple: A tuple containing the following metadata:
            - duration (int): Duration of the video in seconds.
            - fps (float): Exact frames per second of the video (e.g., 29.97).
            - fps_rounded (int): Rounded frames per second (e.g., 30).
            - total_frames (int): Total number of frames in the video.
            
    Raises:
        RuntimeError: If FFmpeg fails to extract metadata from the video.
        ValueError: If an invalid FPS value or no video stream is detected.
    """
    print(f"[cyan]Extracting metadata from video:[/cyan] {video_path}")

    try:
        # Extract raw metadata using ffprobe
        probe = ffmpeg.probe(video_path)
        
        # Isolate the primary video stream
        video_info = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
        
        if not video_info:
            raise ValueError(f"No video stream found in the file: {video_path}")

        # Calculate exact FPS
        num, den = video_info["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        fps_rounded = round(fps)

        if fps <= 0:
            print(f"[bold red]Error: Invalid FPS value extracted ({fps}). Please check the video file.[/bold red]")
            raise ValueError(f"Invalid FPS value extracted: {fps}")

        # Safely extract total frames
        if "nb_frames" in video_info:
            total_frames = int(video_info["nb_frames"])
        else:
            # Retrieve exact duration from either the video stream or the overall container format
            exact_duration = float(video_info.get("duration", probe["format"].get("duration", 0)))
            total_frames = int(exact_duration * fps)

        # Calculate integer duration based on final frame count
        duration_sec = int(total_frames / fps)

        return (
            duration_sec,
            fps,
            fps_rounded,
            total_frames,
        )

    except ffmpeg.Error as e:
        # Decode and print the exact standard error thrown by FFprobe for easier debugging
        error_message = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"[bold red]FFmpeg Error while processing '{video_path}':[/bold red]\n{error_message}")
        raise RuntimeError(f"FFmpeg failed to extract metadata: {error_message}")
    except Exception as e:
        print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise