"""
Color correction and enhancement algorithms for faded or discolored paintings.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ColorCorrector:
    """
    Engine for performing color correction and enhancement on artwork images.
    """
    
    def __init__(self):
        pass
    
    def histogram_equalization(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """
        Perform histogram equalization to enhance contrast.
        
        Args:
            image: Input image (BGR)
            method: 'standard' for global equalization, 'adaptive' for CLAHE
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space for better color preservation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        if method == 'adaptive':
            # Adaptive histogram equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
        else:
            # Standard histogram equalization
            l = cv2.equalizeHist(l)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def color_balance(self, image: np.ndarray, percent: float = 1.0) -> np.ndarray:
        """
        Automatic color balance using the gray world assumption.
        
        Args:
            image: Input image (BGR)
            percent: Percentage of extreme values to clip (0-100)
            
        Returns:
            Color balanced image
        """
        result = image.copy().astype(np.float32)
        
        for channel in range(3):
            hist, bins = np.histogram(image[:, :, channel], 256, [0, 256])
            
            # Calculate cumulative distribution
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1]  # Normalize
            
            # Find percentiles
            low_val = np.searchsorted(cdf, percent / 100.0)
            high_val = np.searchsorted(cdf, 1.0 - percent / 100.0)
            
            # Stretch the histogram
            result[:, :, channel] = np.clip(
                (result[:, :, channel] - low_val) * 255.0 / (high_val - low_val),
                0, 255
            )
        
        return result.astype(np.uint8)
    
    def enhance_faded_colors(self, image: np.ndarray, saturation_factor: float = 1.5) -> np.ndarray:
        """
        Enhance faded colors by increasing saturation.
        
        Args:
            image: Input image (BGR)
            saturation_factor: Factor to multiply saturation by (> 1.0 increases saturation)
            
        Returns:
            Enhanced image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Increase saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        
        # Convert back
        hsv = hsv.astype(np.uint8)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    def adjust_brightness_contrast(self,
                                   image: np.ndarray,
                                   brightness: int = 0,
                                   contrast: float = 1.0) -> np.ndarray:
        """
        Adjust brightness and contrast of the image.
        
        Args:
            image: Input image (BGR)
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast adjustment (0.5 to 3.0)
            
        Returns:
            Adjusted image
        """
        # Apply contrast and brightness
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        return adjusted
    
    def white_balance(self, image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
        """
        Perform white balance correction.
        
        Args:
            image: Input image (BGR)
            method: 'gray_world' or 'white_patch'
            
        Returns:
            White balanced image
        """
        result = image.astype(np.float32)
        
        if method == 'gray_world':
            # Gray world assumption: average color should be gray
            avg_b = np.mean(result[:, :, 0])
            avg_g = np.mean(result[:, :, 1])
            avg_r = np.mean(result[:, :, 2])
            
            avg_gray = (avg_b + avg_g + avg_r) / 3
            
            result[:, :, 0] = np.clip(result[:, :, 0] * avg_gray / avg_b, 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] * avg_gray / avg_g, 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * avg_gray / avg_r, 0, 255)
            
        elif method == 'white_patch':
            # White patch assumption: brightest point should be white
            max_b = np.percentile(result[:, :, 0], 99)
            max_g = np.percentile(result[:, :, 1], 99)
            max_r = np.percentile(result[:, :, 2], 99)
            
            result[:, :, 0] = np.clip(result[:, :, 0] * 255 / max_b, 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] * 255 / max_g, 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * 255 / max_r, 0, 255)
        
        return result.astype(np.uint8)
    
    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Remove noise while preserving details.
        
        Args:
            image: Input image (BGR)
            strength: Denoising strength (higher = more smoothing)
            
        Returns:
            Denoised image
        """
        # Use Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        
        return denoised
    
    def unsharp_mask(self, image: np.ndarray, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
        """
        Apply unsharp masking to enhance details.
        
        Args:
            image: Input image (BGR)
            sigma: Gaussian blur sigma
            amount: Strength of sharpening
            
        Returns:
            Sharpened image
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Calculate difference
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        
        return sharpened
    
    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically enhance the image using multiple techniques.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Auto-enhanced image
        """
        # Step 1: Denoise
        enhanced = self.denoise(image, strength=7)
        
        # Step 2: White balance
        enhanced = self.white_balance(enhanced, method='gray_world')
        
        # Step 3: Histogram equalization
        enhanced = self.histogram_equalization(enhanced, method='adaptive')
        
        # Step 4: Enhance saturation slightly
        enhanced = self.enhance_faded_colors(enhanced, saturation_factor=1.2)
        
        # Step 5: Sharpen slightly
        enhanced = self.unsharp_mask(enhanced, sigma=1.0, amount=0.5)
        
        return enhanced


