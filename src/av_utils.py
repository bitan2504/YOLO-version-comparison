import av
import numpy as np
from PIL import Image
from rich import print


def load_image_with_av(image_path: str) -> np.ndarray | None:
    """
    Loads an image file using PyAV and converts it to a NumPy array in BGR format.
    Useful for feeding frames into OpenCV or YOLO models that expect BGR.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        np.ndarray | None: The image as a NumPy array (BGR format), or None if loading fails.
    """
    try:
        # Use a context manager to ensure the container is properly closed after use
        with av.open(image_path) as container:
            # Decode the first frame of the video/image stream
            for frame in container.decode(video=0):
                # Extract the frame as an RGB numpy array
                rgb_array = frame.to_ndarray(format="rgb24")
                
                # Convert RGB to BGR by reversing the last axis, and copy to avoid memory issues
                bgr_array = rgb_array[:, :, ::-1].copy()
                return bgr_array
                
        # Return None if no frames were found in the container
        return None

    except Exception as e:
        print(f"[bold yellow]WARNING: Failed to load image: {image_path} | Error: {e}[/bold yellow]")
        return None


def save_image_with_av(image_array: np.ndarray, save_path: str) -> bool:
    """
    Saves a BGR NumPy array to disk as a JPEG image using the PIL (Pillow) library.
    
    Args:
        image_array (np.ndarray): The image as a NumPy array in BGR format.
        save_path (str): Path where the image will be saved (e.g., 'output.jpg').
        
    Returns:
        bool: True if the image is saved successfully, False otherwise.
    """
    try:
        # Convert BGR back to RGB before saving (PIL expects RGB)
        rgb_array = image_array[:, :, ::-1].copy()
        
        # Convert the NumPy array to a PIL Image object
        image = Image.fromarray(rgb_array.astype(np.uint8))
        
        # Save the image with high quality
        image.save(save_path, format="JPEG", quality=95)
        return True
        
    except Exception as e:
        print(f"[bold yellow]WARNING: Failed to save image: {save_path} | Error: {e}[/bold yellow]")
        return False