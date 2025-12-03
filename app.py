"""
Digital Restoration of Paintings - Main Application
Interactive web-based tool for artwork restoration using computer vision.
Enhanced version with undo/redo, save/load, quality metrics, and more.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import pickle
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

from restoration.inpainting import InpaintingEngine
from restoration.color_correction import ColorCorrector
from utils.image_utils import (
    pil_to_cv2, cv2_to_pil, resize_image, 
    create_comparison, blend_images, add_text_overlay,
    rotate_image, flip_image, calculate_psnr, calculate_ssim,
    crop_image, get_image_stats
)


# Page configuration
st.set_page_config(
    page_title="Digital Restoration of Paintings",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'working_image' not in st.session_state:
        st.session_state.working_image = None
    if 'mask' not in st.session_state:
        st.session_state.mask = None
    if 'restored_image' not in st.session_state:
        st.session_state.restored_image = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'history_index' not in st.session_state:
        st.session_state.history_index = -1
    if 'metadata' not in st.session_state:
        st.session_state.metadata = {}
    if 'inpaint_radius' not in st.session_state:
        st.session_state.inpaint_radius = 5


def save_to_history(image: np.ndarray, action: str = "Edit"):
    """Save current state to history for undo/redo."""
    if st.session_state.working_image is not None:
        # Only save if different from last state
        if (len(st.session_state.history) == 0 or 
            not np.array_equal(st.session_state.history[st.session_state.history_index], image)):
            # Remove any states after current index (when undoing then making new change)
            st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
            # Add new state
            st.session_state.history.append(image.copy())
            st.session_state.history_index = len(st.session_state.history) - 1
            # Limit history size to prevent memory issues
            if len(st.session_state.history) > 20:
                st.session_state.history.pop(0)
                st.session_state.history_index -= 1


def undo():
    """Undo last change."""
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.working_image = st.session_state.history[st.session_state.history_index].copy()
        st.session_state.restored_image = None
        return True
    return False


def redo():
    """Redo last undone change."""
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history_index += 1
        st.session_state.working_image = st.session_state.history[st.session_state.history_index].copy()
        st.session_state.restored_image = None
        return True
    return False


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎨 Digital Restoration of Paintings</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An interactive tool for art historians and conservators</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/painting-palette.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select a tool:",
            ["Upload & Prepare", "Inpainting", "Color Correction", "Transform", "Compare & Export"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick actions
        if st.session_state.working_image is not None:
            st.markdown("### Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("↶ Undo", use_container_width=True):
                    if undo():
                        st.success("Undone!")
                        st.rerun()
            with col2:
                if st.button("↷ Redo", use_container_width=True):
                    if redo():
                        st.success("Redone!")
                        st.rerun()
            
            # History info
            if len(st.session_state.history) > 0:
                st.caption(f"History: {st.session_state.history_index + 1}/{len(st.session_state.history)}")
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This tool combines computer vision techniques with intuitive design "
            "to help restore damaged or faded artwork digitally.\n\n"
            "**Features:**\n"
            "- Interactive mask drawing\n"
            "- Multiple inpainting algorithms\n"
            "- Color enhancement\n"
            "- Image transformations\n"
            "- Quality metrics\n"
            "- Undo/redo support\n"
            "- Save/load projects"
        )
        
        st.markdown("---")
        st.markdown("**CPSC 4900 Senior Project**")
        st.markdown("*Ophelia Chuning He*")
    
    # Route to appropriate page
    if page == "Upload & Prepare":
        page_upload()
    elif page == "Inpainting":
        page_inpainting()
    elif page == "Color Correction":
        page_color_correction()
    elif page == "Transform":
        page_transform()
    elif page == "Compare & Export":
        page_compare_export()


def page_upload():
    """Upload and prepare artwork images."""
    st.markdown('<h2 class="section-header">📤 Upload & Prepare Artwork</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(
        "**Step 1:** Upload a digital image of the artwork you want to restore. "
        "The image will be automatically resized if it's too large."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
        help="Upload a high-quality image of the artwork"
    )
    
    # Load project button
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📁 Load Project")
        loaded_project = st.file_uploader(
            "Load saved project",
            type=['pkl'],
            help="Load a previously saved restoration project",
            key="load_project"
        )
        if loaded_project is not None:
            try:
                project_data = pickle.load(loaded_project)
                st.session_state.original_image = project_data.get('original_image')
                st.session_state.working_image = project_data.get('working_image')
                st.session_state.metadata = project_data.get('metadata', {})
                st.session_state.history = project_data.get('history', [])
                st.session_state.history_index = len(st.session_state.history) - 1 if st.session_state.history else -1
                st.success("✅ Project loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading project: {str(e)}")
    
    with col2:
        st.markdown("### 💾 Save Project")
        if st.session_state.working_image is not None:
            project_data = {
                'original_image': st.session_state.original_image,
                'working_image': st.session_state.working_image,
                'metadata': st.session_state.metadata,
                'history': st.session_state.history,
                'timestamp': datetime.now().isoformat()
            }
            buf = io.BytesIO()
            pickle.dump(project_data, buf)
            buf.seek(0)
            st.download_button(
                label="💾 Save Project",
                data=buf.getvalue(),
                file_name=f"restoration_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
    
    if uploaded_file is not None:
        # Load image
        try:
            image = Image.open(uploaded_file)
            cv2_image = pil_to_cv2(image)
            
            # Resize if too large
            max_size = st.slider("Maximum image size (pixels)", 512, 2048, 1024, 128)
            cv2_image = resize_image(cv2_image, max_size=(max_size, max_size))
            
            # Store in session state
            st.session_state.original_image = cv2_image.copy()
            st.session_state.working_image = cv2_image.copy()
            save_to_history(cv2_image, "Upload")
            
            # Display image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(cv2_to_pil(cv2_image), use_column_width=True)
                
            with col2:
                st.subheader("Image Information")
                height, width = cv2_image.shape[:2]
                st.metric("Dimensions", f"{width} × {height} px")
                st.metric("Channels", cv2_image.shape[2] if len(cv2_image.shape) == 3 else 1)
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Image statistics
                stats = get_image_stats(cv2_image)
                with st.expander("📊 Detailed Statistics"):
                    st.json(stats)
                
                # Sample artwork info
                with st.expander("📝 Add Artwork Metadata (Optional)"):
                    st.session_state.metadata['title'] = st.text_input("Title", value=st.session_state.metadata.get('title', ''))
                    st.session_state.metadata['artist'] = st.text_input("Artist", value=st.session_state.metadata.get('artist', ''))
                    st.session_state.metadata['date'] = st.text_input("Date", value=st.session_state.metadata.get('date', ''))
                    st.session_state.metadata['notes'] = st.text_area("Restoration Notes", value=st.session_state.metadata.get('notes', ''))
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success("✅ Image loaded successfully! Proceed to **Inpainting** or **Color Correction** in the sidebar.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.info("Please try a different image file.")
        
    else:
        st.info("👆 Please upload an image to begin.")
        
        # Sample images suggestion
        with st.expander("💡 Don't have an image? Try these sample sources"):
            st.markdown("""
            **Free High-Quality Artwork Repositories:**
            - [Metropolitan Museum of Art Open Access](https://www.metmuseum.org/art/collection/search)
            - [Rijksmuseum](https://www.rijksmuseum.nl/en/rijksstudio)
            - [National Gallery of Art](https://www.nga.gov/open-access-images.html)
            - [The Getty Museum](https://www.getty.edu/art/collection/)
            """)


def page_inpainting():
    """Inpainting tool for damaged regions."""
    st.markdown('<h2 class="section-header">🖌️ Inpainting - Repair Damaged Regions</h2>', unsafe_allow_html=True)
    
    if st.session_state.working_image is None:
        st.warning("⚠️ Please upload an image first in the **Upload & Prepare** section.")
        return
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(
        "**Step 2:** Draw on the canvas below to mark damaged or missing regions (shown in white). "
        "The algorithm will fill these areas by sampling from surrounding undamaged regions."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Inpainting settings
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Drawing Tools")
        
        drawing_mode = st.selectbox(
            "Tool",
            ["freedraw", "line", "rect", "circle", "transform"],
            help="Select drawing tool to mark damaged regions",
            key="inpaint_drawing_mode"
        )
        
        stroke_width = st.slider("Brush size", 1, 50, 15, key="inpaint_stroke_width")
        
        if st.button("🗑️ Clear Mask", use_container_width=True):
            st.session_state.mask = None
            st.rerun()
        
        st.markdown("---")
        st.subheader("Inpainting Method")
        
        inpaint_method = st.selectbox(
            "Algorithm",
            ["Telea (Fast)", "Navier-Stokes (Structure)", "Multi-Scale", "Edge-Preserving"],
            help="Choose inpainting algorithm",
            key="inpaint_method_select"
        )
        
        # Store radius in session state
        if "Telea" in inpaint_method or "Navier-Stokes" in inpaint_method:
            if 'inpaint_radius' not in st.session_state:
                st.session_state.inpaint_radius = 3
            st.session_state.inpaint_radius = st.slider("Inpainting radius", 1, 20, st.session_state.inpaint_radius, key="inpaint_radius_slider")
        
        # Preset configurations
        with st.expander("⚙️ Presets"):
            if st.button("Small Scratches", use_container_width=True):
                inpaint_method = "Telea (Fast)"
                st.session_state.inpaint_radius = 3
                st.info("Preset applied: Telea with radius 3")
            if st.button("Large Damaged Areas", use_container_width=True):
                inpaint_method = "Multi-Scale"
                st.info("Preset applied: Multi-Scale")
            if st.button("Preserve Edges", use_container_width=True):
                inpaint_method = "Edge-Preserving"
                st.info("Preset applied: Edge-Preserving")
        
        if st.button("✨ Apply Inpainting", type="primary", use_container_width=True, key="apply_inpaint_btn"):
            apply_inpainting(inpaint_method)
            # Don't rerun here - let apply_inpainting handle it
    
    with col2:
        st.subheader("Draw Damaged Regions")
        
        # Prepare canvas
        canvas_image = cv2_to_pil(st.session_state.working_image)
        height, width = st.session_state.working_image.shape[:2]
        
        # Resize image for canvas display if too large
        display_height = min(height, 600)
        display_width = min(width, 800)
        scale = min(display_width / width, display_height / height)
        canvas_display = canvas_image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
        
        # Create drawable canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_image=canvas_display,
            drawing_mode=drawing_mode,
            height=display_height,
            width=display_width,
            key="inpainting_canvas",
        )
        
        # Store mask
        if canvas_result.image_data is not None:
            # Extract alpha channel as mask
            mask_display = canvas_result.image_data[:, :, 3]
            if mask_display.max() > 0:
                # Resize mask back to original image size if canvas was scaled
                if mask_display.shape[:2] != (height, width):
                    mask = cv2.resize(mask_display, (width, height), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = mask_display
                st.session_state.mask = mask
        
        # Show mask statistics
        if st.session_state.mask is not None and st.session_state.mask.max() > 0:
            mask_binary = (st.session_state.mask > 127).astype(np.uint8) * 255
            mask_area = np.sum(mask_binary > 0)
            total_pixels = height * width
            mask_percent = (mask_area / total_pixels) * 100
            
            st.markdown("---")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Mask Area", f"{mask_area:,} px")
            with col_stat2:
                st.metric("Coverage", f"{mask_percent:.1f}%")
            with col_stat3:
                engine = InpaintingEngine()
                try:
                    stats = engine.get_inpaint_region_stats(mask_binary)
                    st.metric("Regions", stats['num_regions'])
                except:
                    st.metric("Regions", "N/A")
    
    # Show current restoration if available - Always show this section when result exists
    if st.session_state.restored_image is not None:
        st.markdown("---")
        st.subheader("✨ Restoration Result Preview")
        
        # Verify restored image is valid
        try:
            if st.session_state.restored_image.size == 0:
                st.error("Restored image is empty. Please try inpainting again.")
                st.session_state.restored_image = None
            else:
                # Show side-by-side comparison - use original image if available, otherwise working image
                st.markdown("**Before & After Comparison**")
                col1, col2 = st.columns(2)
                
                # ALWAYS use original image for comparison - make a copy to ensure no modification
                if st.session_state.original_image is not None:
                    compare_before = st.session_state.original_image.copy()  # Make a copy to ensure no modification
                else:
                    compare_before = st.session_state.working_image.copy()
                
                # Both images should be in BGR format (OpenCV standard)
                # cv2_to_pil will convert BGR to RGB for display consistently
                with col1:
                    before_pil = cv2_to_pil(compare_before)
                    st.image(before_pil, caption="Before (Original)", use_column_width=True)
                with col2:
                    after_pil = cv2_to_pil(st.session_state.restored_image)
                    st.image(after_pil, caption="After (Restored)", use_column_width=True)
                
                # Quality metrics - compare against original if available
                try:
                    if (compare_before is not None and 
                        st.session_state.restored_image is not None and
                        len(compare_before.shape) == len(st.session_state.restored_image.shape) and
                        compare_before.shape[:2] == st.session_state.restored_image.shape[:2]):
                        psnr = calculate_psnr(compare_before, st.session_state.restored_image)
                        ssim = calculate_ssim(compare_before, st.session_state.restored_image)
                        col_met1, col_met2 = st.columns(2)
                        with col_met1:
                            st.metric("PSNR", f"{psnr:.2f} dB" if psnr != float('inf') else "∞")
                        with col_met2:
                            st.metric("SSIM", f"{ssim:.3f}")
                except Exception:
                    pass  # Silently skip metrics if calculation fails - don't cause reruns
                
                # Action buttons
                st.markdown("---")
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    if st.button("✅ Accept Restoration", type="primary", use_container_width=True, key="accept_inpaint_btn"):
                        save_to_history(st.session_state.restored_image, "Inpainting")
                        st.session_state.working_image = st.session_state.restored_image.copy()
                        st.session_state.mask = None
                        st.session_state.restored_image = None
                        st.success("✅ Restoration accepted! You can continue with more edits or proceed to Color Correction.")
                        st.rerun()
                with col_btn2:
                    if st.button("❌ Discard", use_container_width=True, key="discard_inpaint_btn"):
                        st.session_state.restored_image = None
                        st.rerun()
        except Exception as e:
            st.error(f"Error displaying result: {str(e)}")
            st.session_state.restored_image = None


def apply_inpainting(method: str):
    """Apply inpainting to the masked region."""
    if st.session_state.mask is None or st.session_state.mask.max() == 0:
        st.warning("Please draw a mask on the damaged regions first.")
        return
    
    # Use spinner instead of progress bar to avoid reruns and flashing
    with st.spinner("🔄 Processing inpainting... This may take a moment."):
        try:
            engine = InpaintingEngine()
            
            # Prepare mask
            mask = (st.session_state.mask > 127).astype(np.uint8) * 255
            
            # Process based on method
            if "Telea" in method:
                restored = engine.inpaint(
                    st.session_state.working_image,
                    mask,
                    method='telea',
                    radius=st.session_state.inpaint_radius
                )
            elif "Navier-Stokes" in method:
                restored = engine.inpaint(
                    st.session_state.working_image,
                    mask,
                    method='ns',
                    radius=st.session_state.inpaint_radius
                )
            elif "Multi-Scale" in method:
                restored = engine.multi_scale_inpaint(
                    st.session_state.working_image,
                    mask
                )
            elif "Edge-Preserving" in method:
                restored = engine.edge_preserving_inpaint(
                    st.session_state.working_image,
                    mask
                )
            else:
                restored = engine.inpaint(st.session_state.working_image, mask)
            
            # Ensure result is in correct format and save
            if restored is not None and restored.size > 0:
                # Ensure it's uint8
                if restored.dtype != np.uint8:
                    restored = np.clip(restored, 0, 255).astype(np.uint8)
                
                # Ensure same number of channels as original
                if len(restored.shape) != len(st.session_state.working_image.shape):
                    if len(st.session_state.working_image.shape) == 3 and len(restored.shape) == 2:
                        restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
                
                # Ensure same color space as working image (BGR)
                if len(restored.shape) == 3 and len(st.session_state.working_image.shape) == 3:
                    # Both are color images - ensure they're both BGR
                    if restored.shape[2] == 3:
                        # Already BGR, keep it
                        pass
                
                # Save to session state
                st.session_state.restored_image = restored.copy()
                
                # Show success message - Streamlit will rerun automatically after button click
                st.success("✅ Inpainting completed! View the result below.")
            else:
                raise ValueError("Inpainting returned empty result")
            
        except Exception as e:
            st.error(f"❌ Error during inpainting: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.info("💡 Try reducing the image size or using a different algorithm.")


def page_color_correction():
    """Color correction and enhancement tools."""
    st.markdown('<h2 class="section-header">🎨 Color Correction & Enhancement</h2>', unsafe_allow_html=True)
    
    if st.session_state.working_image is None:
        st.warning("⚠️ Please upload an image first in the **Upload & Prepare** section.")
        return
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(
        "**Step 3:** Enhance faded colors, adjust brightness/contrast, and apply color corrections "
        "to restore the artwork's original appearance."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    corrector = ColorCorrector()
    
    # Create tabs for different correction types
    tab1, tab2, tab3, tab4 = st.tabs(["Quick Enhance", "Manual Adjustments", "Color Balance", "Advanced"])
    
    with tab1:
        st.subheader("Automatic Enhancement")
        st.write("Apply automatic enhancement using multiple algorithms in sequence.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2_to_pil(st.session_state.working_image), caption="Current Image", use_column_width=True)
        
        with col2:
            if st.button("✨ Auto Enhance", type="primary", use_container_width=True):
                with st.spinner("Enhancing image..."):
                    enhanced = corrector.auto_enhance(st.session_state.working_image)
                    st.session_state.restored_image = enhanced
                    st.rerun()
            
            st.write("This will apply:")
            st.markdown("""
            - 🔇 Noise reduction
            - ⚖️ White balance
            - 📊 Histogram equalization
            - 🌈 Saturation enhancement
            - ✨ Sharpening
            """)
            
            # Presets
            with st.expander("🎨 Enhancement Presets"):
                if st.button("Subtle Enhancement", use_container_width=True):
                    result = st.session_state.working_image.copy()
                    result = corrector.denoise(result, strength=5)
                    result = corrector.white_balance(result, method='gray_world')
                    result = corrector.enhance_faded_colors(result, saturation_factor=1.1)
                    st.session_state.restored_image = result
                    st.rerun()
                
                if st.button("Strong Enhancement", use_container_width=True):
                    result = st.session_state.working_image.copy()
                    result = corrector.denoise(result, strength=15)
                    result = corrector.white_balance(result, method='gray_world')
                    result = corrector.histogram_equalization(result, method='adaptive')
                    result = corrector.enhance_faded_colors(result, saturation_factor=1.5)
                    result = corrector.unsharp_mask(result, sigma=1.0, amount=1.0)
                    st.session_state.restored_image = result
                    st.rerun()
    
    with tab2:
        st.subheader("Manual Adjustments")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            brightness = st.slider("Brightness", -100, 100, 0, key="brightness_slider")
            contrast = st.slider("Contrast", 0.5, 3.0, 1.0, 0.1, key="contrast_slider")
            saturation = st.slider("Saturation", 0.5, 2.5, 1.0, 0.1, key="saturation_slider")
            
            show_preview = st.checkbox("Show Live Preview", value=False, key="preview_checkbox")
            
            if st.button("Apply Manual Adjustments", type="primary", use_container_width=True):
                with st.spinner("Applying adjustments..."):
                    result = st.session_state.working_image.copy()
                    
                    # Brightness & contrast
                    result = corrector.adjust_brightness_contrast(result, brightness, contrast)
                    
                    # Saturation
                    if saturation != 1.0:
                        result = corrector.enhance_faded_colors(result, saturation)
                    
                    st.session_state.restored_image = result
                    st.rerun()
        
        with col2:
            # Live preview - only compute if enabled and image exists
            if show_preview and st.session_state.working_image is not None:
                try:
                    preview = st.session_state.working_image.copy()
                    preview = corrector.adjust_brightness_contrast(preview, brightness, contrast)
                    if saturation != 1.0:
                        preview = corrector.enhance_faded_colors(preview, saturation)
                    st.image(cv2_to_pil(preview), caption="Preview", use_column_width=True)
                except Exception as e:
                    st.warning(f"Preview unavailable: {str(e)}")
                    st.image(cv2_to_pil(st.session_state.working_image), caption="Current Image", use_column_width=True)
            else:
                st.image(cv2_to_pil(st.session_state.working_image), caption="Current Image", use_column_width=True)
    
    with tab3:
        st.subheader("Color Balance & White Balance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wb_method = st.selectbox("White Balance Method", ["Gray World", "White Patch"], key="wb_method_select")
            
            if st.button("Apply White Balance", type="primary", use_container_width=True):
                with st.spinner("Applying white balance..."):
                    method = 'gray_world' if 'Gray' in wb_method else 'white_patch'
                    result = corrector.white_balance(st.session_state.working_image, method=method)
                    st.session_state.restored_image = result
                    st.rerun()
            
            st.markdown("---")
            
            balance_percent = st.slider("Color Balance (%)", 0.1, 5.0, 1.0, 0.1, key="balance_slider")
            
            if st.button("Apply Color Balance", type="primary", use_container_width=True):
                with st.spinner("Applying color balance..."):
                    result = corrector.color_balance(st.session_state.working_image, balance_percent)
                    st.session_state.restored_image = result
                    st.rerun()
        
        with col2:
            st.image(cv2_to_pil(st.session_state.working_image), caption="Current Image", use_column_width=True)
    
    with tab4:
        st.subheader("Advanced Corrections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram equalization
            st.write("**Histogram Equalization**")
            hist_method = st.radio("Method", ["Adaptive (CLAHE)", "Standard"], key="hist_method_radio")
            
            if st.button("Apply Histogram Equalization", type="primary", use_container_width=True):
                with st.spinner("Applying histogram equalization..."):
                    method = 'adaptive' if 'Adaptive' in hist_method else 'standard'
                    result = corrector.histogram_equalization(st.session_state.working_image, method=method)
                    st.session_state.restored_image = result
                    st.rerun()
            
            st.markdown("---")
            
            # Denoising
            st.write("**Denoising**")
            denoise_strength = st.slider("Strength", 1, 20, 10, key="denoise_slider")
            
            if st.button("Apply Denoising", type="primary", use_container_width=True):
                with st.spinner("Denoising image..."):
                    result = corrector.denoise(st.session_state.working_image, denoise_strength)
                    st.session_state.restored_image = result
                    st.rerun()
        
        with col2:
            # Sharpening
            st.write("**Unsharp Masking (Sharpening)**")
            sharpen_sigma = st.slider("Blur Sigma", 0.5, 3.0, 1.0, 0.1, key="sharpen_sigma_slider")
            sharpen_amount = st.slider("Sharpen Amount", 0.0, 2.0, 1.0, 0.1, key="sharpen_amount_slider")
            
            if st.button("Apply Sharpening", type="primary", use_container_width=True):
                with st.spinner("Sharpening image..."):
                    result = corrector.unsharp_mask(
                        st.session_state.working_image,
                        sharpen_sigma,
                        sharpen_amount
                    )
                    st.session_state.restored_image = result
                    st.rerun()
    
    # Show result if available
    if st.session_state.restored_image is not None:
        st.markdown("---")
        st.subheader("Result Preview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2_to_pil(st.session_state.working_image), caption="Before", use_column_width=True)
        with col2:
            st.image(cv2_to_pil(st.session_state.restored_image), caption="After", use_column_width=True)
        
        # Quality metrics - only compute if both images exist and are same size
        try:
            if (st.session_state.working_image is not None and 
                st.session_state.restored_image is not None and
                st.session_state.working_image.shape == st.session_state.restored_image.shape):
                psnr = calculate_psnr(st.session_state.working_image, st.session_state.restored_image)
                ssim = calculate_ssim(st.session_state.working_image, st.session_state.restored_image)
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("PSNR", f"{psnr:.2f} dB" if psnr != float('inf') else "∞")
                with col_met2:
                    st.metric("SSIM", f"{ssim:.3f}")
        except Exception:
            pass
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Accept Changes", type="primary", use_container_width=True):
                save_to_history(st.session_state.restored_image, "Color Correction")
                st.session_state.working_image = st.session_state.restored_image.copy()
                st.session_state.restored_image = None
                st.success("Changes accepted!")
                st.rerun()
        with col2:
            if st.button("❌ Discard Changes", use_container_width=True):
                st.session_state.restored_image = None
                st.rerun()


def page_transform():
    """Image transformation tools."""
    st.markdown('<h2 class="section-header">🔄 Image Transformations</h2>', unsafe_allow_html=True)
    
    if st.session_state.working_image is None:
        st.warning("⚠️ Please upload an image first in the **Upload & Prepare** section.")
        return
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("Transform your image: rotate, flip, or crop to prepare for restoration.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Image")
        st.image(cv2_to_pil(st.session_state.working_image), use_column_width=True)
    
    with col2:
        st.subheader("Transformations")
        
        # Rotation
        st.write("**Rotation**")
        rotation_angle = st.slider("Rotation Angle (degrees)", -180, 180, 0, 15)
        col_rot1, col_rot2 = st.columns(2)
        with col_rot1:
            if st.button("Rotate 90° CW", use_container_width=True):
                result = rotate_image(st.session_state.working_image, -90)
                st.session_state.restored_image = result
                st.rerun()
        with col_rot2:
            if st.button("Rotate 90° CCW", use_container_width=True):
                result = rotate_image(st.session_state.working_image, 90)
                st.session_state.restored_image = result
                st.rerun()
        
        if st.button("Apply Custom Rotation", use_container_width=True):
            result = rotate_image(st.session_state.working_image, rotation_angle)
            st.session_state.restored_image = result
            st.rerun()
        
        st.markdown("---")
        
        # Flip
        st.write("**Flip**")
        col_flip1, col_flip2 = st.columns(2)
        with col_flip1:
            if st.button("Flip Horizontal", use_container_width=True):
                result = flip_image(st.session_state.working_image, 'horizontal')
                st.session_state.restored_image = result
                st.rerun()
        with col_flip2:
            if st.button("Flip Vertical", use_container_width=True):
                result = flip_image(st.session_state.working_image, 'vertical')
                st.session_state.restored_image = result
                st.rerun()
    
    # Show result if available
    if st.session_state.restored_image is not None:
        st.markdown("---")
        st.subheader("Transformed Image")
        st.image(cv2_to_pil(st.session_state.restored_image), use_column_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Accept Transformation", type="primary", use_container_width=True):
                save_to_history(st.session_state.restored_image, "Transform")
                st.session_state.working_image = st.session_state.restored_image.copy()
                st.session_state.restored_image = None
                st.success("Transformation accepted!")
                st.rerun()
        with col2:
            if st.button("❌ Discard", use_container_width=True):
                st.session_state.restored_image = None
                st.rerun()


def page_compare_export():
    """Compare results and export images."""
    st.markdown('<h2 class="section-header">📊 Compare & Export Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.original_image is None:
        st.warning("⚠️ Please upload and process an image first.")
        return
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(
        "**Step 4:** Compare the original and restored images side-by-side, "
        "and export your results in various formats."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison view selection
    st.subheader("Comparison View")
    
    comparison_mode = st.radio(
        "View Mode",
        ["Side-by-Side", "Vertical Stack", "Split View", "Overlay Blend", "Original Only", "Restored Only"],
        horizontal=True
    )
    
    original = st.session_state.original_image
    restored = st.session_state.working_image
    
    if comparison_mode == "Side-by-Side":
        comparison = create_comparison(original, restored, mode='side-by-side')
        comparison = add_text_overlay(comparison, "Original", (10, 40))
        w = original.shape[1]
        comparison = add_text_overlay(comparison, "Restored", (w + 10, 40))
        st.image(cv2_to_pil(comparison), use_column_width=True)
        
    elif comparison_mode == "Vertical Stack":
        comparison = create_comparison(original, restored, mode='vertical')
        st.image(cv2_to_pil(comparison), use_column_width=True)
        
    elif comparison_mode == "Split View":
        comparison = create_comparison(original, restored, mode='split')
        st.image(cv2_to_pil(comparison), caption="Swipe left/right comparison", use_column_width=True)
        
    elif comparison_mode == "Overlay Blend":
        blend_amount = st.slider("Blend Amount", 0.0, 1.0, 0.5, 0.05)
        comparison = blend_images(original, restored, alpha=blend_amount)
        st.image(cv2_to_pil(comparison), use_column_width=True)
        
    elif comparison_mode == "Original Only":
        st.image(cv2_to_pil(original), caption="Original Image", use_column_width=True)
        
    elif comparison_mode == "Restored Only":
        st.image(cv2_to_pil(restored), caption="Restored Image", use_column_width=True)
    
    # Quality metrics
    st.markdown("---")
    st.subheader("📈 Quality Metrics")
    
    try:
        psnr = calculate_psnr(original, restored)
        ssim = calculate_ssim(original, restored)
        
        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.metric("PSNR", f"{psnr:.2f} dB" if psnr != float('inf') else "∞", 
                     help="Peak Signal-to-Noise Ratio (higher is better)")
        with col_met2:
            st.metric("SSIM", f"{ssim:.3f}", 
                     help="Structural Similarity Index (0-1, higher is better)")
        with col_met3:
            diff = np.mean(np.abs(original.astype(np.float32) - restored.astype(np.float32)))
            st.metric("Mean Difference", f"{diff:.2f}", 
                     help="Average pixel difference")
    except Exception as e:
        st.info("Quality metrics could not be calculated.")
    
    # Export section
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    export_format = st.radio("Export Format", ["PNG", "JPEG", "TIFF"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Original Image**")
        original_pil = cv2_to_pil(original)
        buf_orig = io.BytesIO()
        
        if export_format == "PNG":
            original_pil.save(buf_orig, format='PNG')
            mime = "image/png"
            ext = "png"
        elif export_format == "JPEG":
            original_pil.save(buf_orig, format='JPEG', quality=95)
            mime = "image/jpeg"
            ext = "jpg"
        else:  # TIFF
            original_pil.save(buf_orig, format='TIFF')
            mime = "image/tiff"
            ext = "tiff"
        
        st.download_button(
            label="Download Original",
            data=buf_orig.getvalue(),
            file_name=f"original_image.{ext}",
            mime=mime
        )
    
    with col2:
        st.write("**Restored Image**")
        restored_pil = cv2_to_pil(restored)
        buf_rest = io.BytesIO()
        
        if export_format == "PNG":
            restored_pil.save(buf_rest, format='PNG')
            mime = "image/png"
            ext = "png"
        elif export_format == "JPEG":
            restored_pil.save(buf_rest, format='JPEG', quality=95)
            mime = "image/jpeg"
            ext = "jpg"
        else:  # TIFF
            restored_pil.save(buf_rest, format='TIFF')
            mime = "image/tiff"
            ext = "tiff"
        
        st.download_button(
            label="Download Restored",
            data=buf_rest.getvalue(),
            file_name=f"restored_image.{ext}",
            mime=mime
        )
    
    with col3:
        st.write("**Side-by-Side Comparison**")
        comparison_img = create_comparison(original, restored, mode='side-by-side')
        comparison_pil = cv2_to_pil(comparison_img)
        buf_comp = io.BytesIO()
        
        if export_format == "PNG":
            comparison_pil.save(buf_comp, format='PNG')
            mime = "image/png"
            ext = "png"
        elif export_format == "JPEG":
            comparison_pil.save(buf_comp, format='JPEG', quality=95)
            mime = "image/jpeg"
            ext = "jpg"
        else:  # TIFF
            comparison_pil.save(buf_comp, format='TIFF')
            mime = "image/tiff"
            ext = "tiff"
        
        st.download_button(
            label="Download Comparison",
            data=buf_comp.getvalue(),
            file_name=f"comparison_image.{ext}",
            mime=mime
        )
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Image Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Image**")
        orig_stats = {
            "Dimensions": f"{original.shape[1]} × {original.shape[0]} px",
            "Mean Brightness": f"{np.mean(original):.1f}",
            "Std Deviation": f"{np.std(original):.1f}"
        }
        for key, value in orig_stats.items():
            st.metric(key, value)
    
    with col2:
        st.write("**Restored Image**")
        rest_stats = {
            "Dimensions": f"{restored.shape[1]} × {restored.shape[0]} px",
            "Mean Brightness": f"{np.mean(restored):.1f}",
            "Std Deviation": f"{np.std(restored):.1f}"
        }
        for key, value in rest_stats.items():
            st.metric(key, value)
    
    # Metadata display
    if st.session_state.metadata:
        st.markdown("---")
        st.subheader("📝 Artwork Metadata")
        st.json(st.session_state.metadata)


if __name__ == "__main__":
    main()
