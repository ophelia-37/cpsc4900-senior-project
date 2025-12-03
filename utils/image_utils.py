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
    # Ensure both images have the same number of channels
    # Remove alpha channel if present (4 channels -> 3 channels)
    if len(original.shape) == 3 and original.shape[2] == 4:
        original = original[:, :, :3]
    if len(restored.shape) == 3 and restored.shape[2] == 4:
        restored = restored[:, :, :3]
    
    # Ensure both images are same height for side-by-side
    if mode == 'side-by-side':
        h1, w1 = original.shape[:2]
        h2, w2 = restored.shape[:2]
        if h1 != h2:
            # Resize to match height
            scale = h1 / h2
            new_w2 = int(w2 * scale)
            restored = cv2.resize(restored, (new_w2, h1), interpolation=cv2.INTER_AREA)
        # Place images side by side horizontally
        comparison = np.hstack([original, restored])
        
    elif mode == 'vertical':
        # Ensure both images are same width for vertical stacking
        h1, w1 = original.shape[:2]
        h2, w2 = restored.shape[:2]
        if w1 != w2:
            # Resize to match width
            scale = w1 / w2
            new_h2 = int(h2 * scale)
            restored = cv2.resize(restored, (w1, new_h2), interpolation=cv2.INTER_AREA)
        # Stack images vertically
        comparison = np.vstack([original, restored])
        
    elif mode == 'split':
        # Split view: left half original, right half restored
        h, w = original.shape[:2]
        # Resize restored to match original dimensions
        if restored.shape[:2] != (h, w):
            restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_AREA)
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


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by specified angle.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    return rotated


def flip_image(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Flip image horizontally or vertically.
    
    Args:
        image: Input image
        direction: 'horizontal' or 'vertical'
        
    Returns:
        Flipped image
    """
    if direction == 'horizontal':
        return cv2.flip(image, 1)
    elif direction == 'vertical':
        return cv2.flip(image, 0)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        PSNR value in dB
    """
    # Ensure images are same size
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    
    # Calculate MSE
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf')  # Images are identical
    
    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index between two images.
    Simplified SSIM calculation.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        SSIM value (0-1, higher is better)
    """
    # Ensure images are same size
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calculate means
    mu1 = cv2.GaussianBlur(img1.astype(np.float64), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2.astype(np.float64), (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate sigma
    sigma1_sq = cv2.GaussianBlur((img1.astype(np.float64) ** 2), (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur((img2.astype(np.float64) ** 2), (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur((img1.astype(np.float64) * img2.astype(np.float64)), (11, 11), 1.5) - mu1_mu2
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / denominator
    
    return float(np.mean(ssim))


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop image to specified region.
    
    Args:
        image: Input image
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Crop width
        height: Crop height
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    width = min(width, w - x)
    height = min(height, h - y)
    
    return image[y:y+height, x:x+width]


