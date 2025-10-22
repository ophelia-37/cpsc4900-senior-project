"""
Utility functions for image processing and manipulation.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union


def load_image(image_path: str, color_mode: str = 'BGR') -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        color_mode: 'BGR', 'RGB', or 'GRAY'
        
    Returns:
        Image as numpy array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    if color_mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode == 'GRAY':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image


def save_image(image: np.ndarray, output_path: str, color_mode: str = 'BGR') -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save the image
        color_mode: Current color mode of the image ('BGR', 'RGB', or 'GRAY')
    """
    if color_mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, image)


def resize_image(image: np.ndarray, 
                max_size: Tuple[int, int] = (1024, 1024),
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize an image to fit within max_size while optionally maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum width and height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size
    
    if maintain_aspect:
        # Calculate scale factor
        scale = min(max_w / w, max_h / h)
        if scale >= 1:
            return image  # Don't upscale
        
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = max_w, max_h
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def create_comparison(original: np.ndarray,
                     restored: np.ndarray,
                     mode: str = 'side-by-side') -> np.ndarray:
    """
    Create a comparison image showing original and restored versions.
    
    Args:
        original: Original image
        restored: Restored image
        mode: 'side-by-side', 'vertical', or 'split'
        
    Returns:
        Comparison image
    """
    if mode == 'side-by-side':
        # Place images side by side horizontally
        comparison = np.hstack([original, restored])
        
    elif mode == 'vertical':
        # Stack images vertically
        comparison = np.vstack([original, restored])
        
    elif mode == 'split':
        # Split view: left half original, right half restored
        h, w = original.shape[:2]
        comparison = original.copy()
        comparison[:, w//2:] = restored[:, w//2:]
        
        # Draw a line in the middle
        cv2.line(comparison, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")
    
    return comparison


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend two images together.
    
    Args:
        img1: First image
        img2: Second image
        alpha: Blend factor (0.0 = all img1, 1.0 = all img2)
        
    Returns:
        Blended image
    """
    return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)


def get_image_stats(image: np.ndarray) -> dict:
    """
    Get statistics about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image))
    }
    
    if len(image.shape) == 3:
        stats['channels'] = image.shape[2]
        stats['channel_means'] = [float(np.mean(image[:, :, i])) for i in range(image.shape[2])]
    else:
        stats['channels'] = 1
    
    return stats


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image
        
    Returns:
        OpenCV image (BGR)
    """
    # Convert PIL to numpy array (RGB)
    numpy_image = np.array(pil_image)
    
    # Convert RGB to BGR if color image
    if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    else:
        opencv_image = numpy_image
    
    return opencv_image


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV format to PIL Image.
    
    Args:
        cv2_image: OpenCV image (BGR)
        
    Returns:
        PIL Image
    """
    # Convert BGR to RGB if color image
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2_image
    
    return Image.fromarray(rgb_image)


def add_text_overlay(image: np.ndarray, 
                     text: str,
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 1.0,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to an image.
    
    Args:
        image: Input image
        text: Text to add
        position: (x, y) position of text
        font_scale: Font size scale
        color: Text color (BGR)
        thickness: Text thickness
        
    Returns:
        Image with text overlay
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Add black outline for better visibility
    cv2.putText(result, text, position, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(result, text, position, font, font_scale, color, thickness)
    
    return result


