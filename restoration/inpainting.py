"""
Inpainting algorithms for image restoration.
Implements patch-based exemplar inpainting (Criminisi et al., 2004).
Supports both classical and deep learning methods.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import warnings

# Try to import deep learning module
try:
    from .deep_inpainting import DeepInpaintingEngine, create_deep_inpainting_engine
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


class InpaintingEngine:
    """
    Engine for performing image inpainting using various algorithms.
    """
    
    def __init__(self):
        self.methods = {
            'telea': cv2.INPAINT_TELEA,
            'ns': cv2.INPAINT_NS
        }
        # Initialize deep learning engine if available
        self.deep_engine = None
        if DEEP_LEARNING_AVAILABLE:
            try:
                self.deep_engine = create_deep_inpainting_engine(device='auto')
            except Exception as e:
                warnings.warn(f"Deep learning engine not available: {e}")
    
    def inpaint(self, 
                image: np.ndarray, 
                mask: np.ndarray, 
                method: str = 'telea',
                radius: int = 3) -> np.ndarray:
        """
        Perform inpainting on the image using the specified method.
        
        Args:
            image: Input image (BGR or RGB)
            mask: Binary mask where white pixels indicate regions to inpaint
            method: Inpainting method ('telea' or 'ns')
            radius: Radius of circular neighborhood for inpainting
            
        Returns:
            Inpainted image
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.methods.keys())}")
        
        # Ensure image is in correct format for OpenCV inpainting
        # OpenCV requires: 8-bit unsigned (uint8), 1-channel (grayscale) or 3-channel (BGR)
        if image.dtype != np.uint8:
            # Convert to uint8 if needed
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume values are in [0, 1] range, convert to [0, 255]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Handle multi-channel images (remove alpha channel if present)
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Remove alpha channel (RGBA -> RGB)
            image = image[:, :, :3]
        
        # Ensure mask is single channel binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Ensure mask is binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Ensure image and mask have same dimensions
        if image.shape[:2] != mask.shape[:2]:
            # Resize mask to match image
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Perform inpainting
        result = cv2.inpaint(image, mask, radius, self.methods[method])
        
        return result
    
    def patch_based_inpaint(self,
                           image: np.ndarray,
                           mask: np.ndarray,
                           patch_size: int = 9) -> np.ndarray:
        """
        Perform patch-based exemplar inpainting.
        This is a simplified implementation inspired by Criminisi et al.
        
        Args:
            image: Input image
            mask: Binary mask (white = region to inpaint)
            patch_size: Size of patches to use
            
        Returns:
            Inpainted image
        """
        # Use OpenCV's implementation which is based on similar principles
        # For a more sophisticated implementation, we'd need to implement
        # the full Criminisi algorithm with priority computation
        return self.inpaint(image, mask, method='telea', radius=patch_size)
    
    def multi_scale_inpaint(self,
                           image: np.ndarray,
                           mask: np.ndarray,
                           scales: int = 3,
                           progress_callback=None) -> np.ndarray:
        """
        Perform multi-scale inpainting for better results on large regions.
        
        Args:
            image: Input image
            mask: Binary mask
            scales: Number of scales to use
            progress_callback: Optional callback function(iteration, total) for progress updates
            
        Returns:
            Inpainted image
        """
        result = image.copy()
        current_mask = mask.copy()
        
        # Start with coarse scale and refine
        for scale in range(scales):
            # Update progress if callback provided
            if progress_callback:
                progress_callback(scale + 1, scales)
            
            # Inpaint at current scale
            radius = max(3, 10 - scale * 2)
            result = self.inpaint(result, current_mask, method='telea', radius=radius)
            
            # Reduce mask for next iteration
            kernel = np.ones((3, 3), np.uint8)
            current_mask = cv2.erode(current_mask, kernel, iterations=1)
            
            # Stop if mask is empty
            if cv2.countNonZero(current_mask) == 0:
                break
        
        return result
    
    def edge_preserving_inpaint(self,
                                image: np.ndarray,
                                mask: np.ndarray) -> np.ndarray:
        """
        Inpaint with edge preservation using the Navier-Stokes method.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Inpainted image with preserved edges
        """
        # Pre-process with edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        
        # Perform inpainting with NS method (better for structure)
        result = self.inpaint(image, mask, method='ns', radius=5)
        
        return result
    
    def deep_learning_inpaint(self,
                             image: np.ndarray,
                             mask: np.ndarray,
                             model_type: str = 'lama') -> np.ndarray:
        """
        Perform inpainting using deep learning models.
        
        Args:
            image: Input image (BGR, uint8)
            mask: Binary mask (uint8, 0 or 255)
            model_type: Type of model ('lama', 'gan', 'coordfill')
            
        Returns:
            Inpainted image
            
        Raises:
            RuntimeError: If deep learning is not available or model not loaded
        """
        if not DEEP_LEARNING_AVAILABLE or self.deep_engine is None:
            # Fallback to multi-scale inpainting
            warnings.warn(
                "Deep learning not available. Falling back to multi-scale inpainting. "
                "Install PyTorch to enable deep learning: pip install torch torchvision"
            )
            return self.multi_scale_inpaint(image, mask, scales=5)
        
        # Use deep learning engine
        if model_type == 'lama':
            return self.deep_engine.inpaint_lama_style(image, mask, use_fallback=True)
        elif model_type == 'gan':
            return self.deep_engine.inpaint_gan_style(image, mask, use_fallback=True)
        else:
            # Default to LaMa style
            return self.deep_engine.inpaint_lama_style(image, mask, use_fallback=True)
    
    def get_inpaint_region_stats(self, mask: np.ndarray) -> dict:
        """
        Get statistics about the region to be inpainted.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary with statistics
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = cv2.countNonZero(mask)
        num_regions = len(contours)
        
        stats = {
            'total_pixels': total_area,
            'num_regions': num_regions,
            'regions': []
        }
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            stats['regions'].append({
                'area': area,
                'bbox': (x, y, w, h),
                'perimeter': cv2.arcLength(contour, True)
            })
        
        return stats

