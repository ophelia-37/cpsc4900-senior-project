"""
Test script for restoration algorithms.
Run this to verify that all algorithms work correctly.
"""

import cv2
import numpy as np
from restoration.inpainting import InpaintingEngine
from restoration.color_correction import ColorCorrector
from utils.image_utils import save_image, create_comparison
import os


def create_test_image():
    """Create a simple test image with a gradient and patterns."""
    # Create a 512x512 test image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(512):
        for j in range(512):
            img[i, j] = [i // 2, j // 2, (i + j) // 4]
    
    # Add some patterns
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(img, (350, 350), 50, (0, 255, 0), -1)
    cv2.line(img, (0, 256), (512, 256), (255, 255, 255), 3)
    cv2.line(img, (256, 0), (256, 512), (255, 255, 255), 3)
    
    return img


def create_test_mask(shape):
    """Create a test mask with damaged regions."""
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    
    # Add some "damaged" regions
    cv2.rectangle(mask, (150, 150), (180, 180), 255, -1)
    cv2.circle(mask, (350, 350), 20, 255, -1)
    cv2.line(mask, (100, 256), (200, 256), 255, 10)
    
    return mask


def test_inpainting():
    """Test all inpainting methods."""
    print("Testing Inpainting Algorithms...")
    print("-" * 50)
    
    # Create test image and mask
    img = create_test_image()
    mask = create_test_mask(img.shape)
    
    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)
    
    # Save original and mask
    save_image(img, "test_outputs/original.png", color_mode='BGR')
    save_image(mask, "test_outputs/mask.png", color_mode='GRAY')
    print("✓ Created test image and mask")
    
    engine = InpaintingEngine()
    
    # Test Telea method
    try:
        result_telea = engine.inpaint(img, mask, method='telea', radius=5)
        save_image(result_telea, "test_outputs/inpaint_telea.png", color_mode='BGR')
        print("✓ Telea inpainting successful")
    except Exception as e:
        print(f"✗ Telea inpainting failed: {e}")
    
    # Test NS method
    try:
        result_ns = engine.inpaint(img, mask, method='ns', radius=5)
        save_image(result_ns, "test_outputs/inpaint_ns.png", color_mode='BGR')
        print("✓ Navier-Stokes inpainting successful")
    except Exception as e:
        print(f"✗ Navier-Stokes inpainting failed: {e}")
    
    # Test multi-scale
    try:
        result_multi = engine.multi_scale_inpaint(img, mask, scales=3)
        save_image(result_multi, "test_outputs/inpaint_multiscale.png", color_mode='BGR')
        print("✓ Multi-scale inpainting successful")
    except Exception as e:
        print(f"✗ Multi-scale inpainting failed: {e}")
    
    # Test edge-preserving
    try:
        result_edge = engine.edge_preserving_inpaint(img, mask)
        save_image(result_edge, "test_outputs/inpaint_edge.png", color_mode='BGR')
        print("✓ Edge-preserving inpainting successful")
    except Exception as e:
        print(f"✗ Edge-preserving inpainting failed: {e}")
    
    # Get stats
    try:
        stats = engine.get_inpaint_region_stats(mask)
        print(f"✓ Mask statistics: {stats['num_regions']} regions, {stats['total_pixels']} pixels")
    except Exception as e:
        print(f"✗ Statistics computation failed: {e}")
    
    print()


def test_color_correction():
    """Test all color correction methods."""
    print("Testing Color Correction Algorithms...")
    print("-" * 50)
    
    # Create test image
    img = create_test_image()
    
    # Create faded version
    faded = cv2.convertScaleAbs(img, alpha=0.6, beta=-30)
    save_image(faded, "test_outputs/faded_original.png", color_mode='BGR')
    print("✓ Created faded test image")
    
    corrector = ColorCorrector()
    
    # Test histogram equalization
    try:
        result_hist = corrector.histogram_equalization(faded, method='adaptive')
        save_image(result_hist, "test_outputs/corrected_histogram.png", color_mode='BGR')
        print("✓ Histogram equalization successful")
    except Exception as e:
        print(f"✗ Histogram equalization failed: {e}")
    
    # Test color balance
    try:
        result_balance = corrector.color_balance(faded, percent=1.0)
        save_image(result_balance, "test_outputs/corrected_balance.png", color_mode='BGR')
        print("✓ Color balance successful")
    except Exception as e:
        print(f"✗ Color balance failed: {e}")
    
    # Test saturation enhancement
    try:
        result_sat = corrector.enhance_faded_colors(faded, saturation_factor=1.5)
        save_image(result_sat, "test_outputs/corrected_saturation.png", color_mode='BGR')
        print("✓ Saturation enhancement successful")
    except Exception as e:
        print(f"✗ Saturation enhancement failed: {e}")
    
    # Test brightness/contrast
    try:
        result_bc = corrector.adjust_brightness_contrast(faded, brightness=20, contrast=1.3)
        save_image(result_bc, "test_outputs/corrected_brightness_contrast.png", color_mode='BGR')
        print("✓ Brightness/contrast adjustment successful")
    except Exception as e:
        print(f"✗ Brightness/contrast adjustment failed: {e}")
    
    # Test white balance
    try:
        result_wb = corrector.white_balance(faded, method='gray_world')
        save_image(result_wb, "test_outputs/corrected_white_balance.png", color_mode='BGR')
        print("✓ White balance successful")
    except Exception as e:
        print(f"✗ White balance failed: {e}")
    
    # Test denoising
    try:
        # Add noise first
        noisy = faded.copy().astype(np.float32)
        noise = np.random.normal(0, 15, noisy.shape)
        noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
        
        result_denoise = corrector.denoise(noisy, strength=10)
        save_image(result_denoise, "test_outputs/corrected_denoised.png", color_mode='BGR')
        print("✓ Denoising successful")
    except Exception as e:
        print(f"✗ Denoising failed: {e}")
    
    # Test sharpening
    try:
        result_sharp = corrector.unsharp_mask(faded, sigma=1.0, amount=1.0)
        save_image(result_sharp, "test_outputs/corrected_sharpened.png", color_mode='BGR')
        print("✓ Sharpening successful")
    except Exception as e:
        print(f"✗ Sharpening failed: {e}")
    
    # Test auto enhance
    try:
        result_auto = corrector.auto_enhance(faded)
        save_image(result_auto, "test_outputs/corrected_auto.png", color_mode='BGR')
        print("✓ Auto enhance successful")
    except Exception as e:
        print(f"✗ Auto enhance failed: {e}")
    
    print()


def test_utils():
    """Test utility functions."""
    print("Testing Utility Functions...")
    print("-" * 50)
    
    from utils.image_utils import resize_image, blend_images, get_image_stats
    
    img1 = create_test_image()
    img2 = create_test_image()
    
    # Test resize
    try:
        resized = resize_image(img1, max_size=(256, 256))
        assert resized.shape[0] <= 256 and resized.shape[1] <= 256
        print(f"✓ Resize successful: {img1.shape} -> {resized.shape}")
    except Exception as e:
        print(f"✗ Resize failed: {e}")
    
    # Test comparison
    try:
        comp_side = create_comparison(img1, img2, mode='side-by-side')
        save_image(comp_side, "test_outputs/comparison_side.png", color_mode='BGR')
        print("✓ Side-by-side comparison successful")
    except Exception as e:
        print(f"✗ Side-by-side comparison failed: {e}")
    
    try:
        comp_vert = create_comparison(img1, img2, mode='vertical')
        save_image(comp_vert, "test_outputs/comparison_vertical.png", color_mode='BGR')
        print("✓ Vertical comparison successful")
    except Exception as e:
        print(f"✗ Vertical comparison failed: {e}")
    
    try:
        comp_split = create_comparison(img1, img2, mode='split')
        save_image(comp_split, "test_outputs/comparison_split.png", color_mode='BGR')
        print("✓ Split comparison successful")
    except Exception as e:
        print(f"✗ Split comparison failed: {e}")
    
    # Test blending
    try:
        blended = blend_images(img1, img2, alpha=0.5)
        save_image(blended, "test_outputs/blended.png", color_mode='BGR')
        print("✓ Image blending successful")
    except Exception as e:
        print(f"✗ Image blending failed: {e}")
    
    # Test statistics
    try:
        stats = get_image_stats(img1)
        print(f"✓ Image statistics: shape={stats['shape']}, mean={stats['mean']:.2f}")
    except Exception as e:
        print(f"✗ Statistics computation failed: {e}")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Digital Restoration - Algorithm Test Suite")
    print("=" * 50 + "\n")
    
    test_inpainting()
    test_color_correction()
    test_utils()
    
    print("=" * 50)
    print("Testing complete! Check test_outputs/ for results.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()


