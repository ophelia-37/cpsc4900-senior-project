"""
Digital Restoration of Paintings - Main Application
Interactive web-based tool for artwork restoration using computer vision.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

from restoration.inpainting import InpaintingEngine
from restoration.color_correction import ColorCorrector
from utils.image_utils import (
    pil_to_cv2, cv2_to_pil, resize_image, 
    create_comparison, blend_images, add_text_overlay
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
            ["Upload & Prepare", "Inpainting", "Color Correction", "Compare & Export"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This tool combines computer vision techniques with intuitive design "
            "to help restore damaged or faded artwork digitally.\n\n"
            "**Features:**\n"
            "- Interactive mask drawing\n"
            "- Patch-based inpainting\n"
            "- Color enhancement\n"
            "- Side-by-side comparison\n"
            "- Export results"
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
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        cv2_image = pil_to_cv2(image)
        
        # Resize if too large
        max_size = st.slider("Maximum image size (pixels)", 512, 2048, 1024, 128)
        cv2_image = resize_image(cv2_image, max_size=(max_size, max_size))
        
        # Store in session state
        st.session_state.original_image = cv2_image.copy()
        st.session_state.working_image = cv2_image.copy()
        
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(cv2_to_pil(cv2_image), use_container_width=True)
            
        with col2:
            st.subheader("Image Information")
            height, width = cv2_image.shape[:2]
            st.metric("Dimensions", f"{width} × {height} px")
            st.metric("Channels", cv2_image.shape[2] if len(cv2_image.shape) == 3 else 1)
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Sample artwork info
            with st.expander("📝 Add Artwork Metadata (Optional)"):
                artwork_title = st.text_input("Title")
                artist_name = st.text_input("Artist")
                creation_date = st.text_input("Date")
                notes = st.text_area("Restoration Notes")
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success("✅ Image loaded successfully! Proceed to **Inpainting** or **Color Correction** in the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)
        
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
            help="Select drawing tool to mark damaged regions"
        )
        
        stroke_width = st.slider("Brush size", 1, 50, 15)
        
        if st.button("🗑️ Clear Mask"):
            st.session_state.mask = None
            st.rerun()
        
        st.markdown("---")
        st.subheader("Inpainting Method")
        
        inpaint_method = st.selectbox(
            "Algorithm",
            ["Telea (Fast)", "Navier-Stokes (Structure)", "Multi-Scale", "Edge-Preserving"],
            help="Choose inpainting algorithm"
        )
        
        if "Telea" in inpaint_method or "Navier-Stokes" in inpaint_method:
            inpaint_radius = st.slider("Inpainting radius", 1, 20, 5)
        
        if st.button("✨ Apply Inpainting", type="primary"):
            apply_inpainting(inpaint_method)
    
    with col2:
        st.subheader("Draw Damaged Regions")
        
        # Prepare canvas
        canvas_image = cv2_to_pil(st.session_state.working_image)
        height, width = st.session_state.working_image.shape[:2]
        
        # Create drawable canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_image=canvas_image,
            drawing_mode=drawing_mode,
            height=height,
            width=width,
            key="inpainting_canvas",
        )
        
        # Store mask
        if canvas_result.image_data is not None:
            # Extract alpha channel as mask
            mask = canvas_result.image_data[:, :, 3]
            if mask.max() > 0:
                st.session_state.mask = mask
    
    # Show current restoration if available
    if st.session_state.restored_image is not None:
        st.markdown("---")
        st.subheader("Restoration Result")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2_to_pil(st.session_state.working_image), caption="Before", use_container_width=True)
        with col2:
            st.image(cv2_to_pil(st.session_state.restored_image), caption="After", use_container_width=True)
        
        if st.button("✅ Accept Restoration"):
            st.session_state.working_image = st.session_state.restored_image.copy()
            st.session_state.mask = None
            st.success("Restoration accepted! You can continue with more edits or proceed to Color Correction.")
            st.rerun()


def apply_inpainting(method: str):
    """Apply inpainting to the masked region."""
    if st.session_state.mask is None or st.session_state.mask.max() == 0:
        st.warning("Please draw a mask on the damaged regions first.")
        return
    
    with st.spinner("Applying inpainting... This may take a moment."):
        engine = InpaintingEngine()
        
        # Convert mask to binary
        mask = (st.session_state.mask > 127).astype(np.uint8) * 255
        
        try:
            if "Telea" in method:
                restored = engine.inpaint(
                    st.session_state.working_image,
                    mask,
                    method='telea',
                    radius=st.session_state.get('inpaint_radius', 5)
                )
            elif "Navier-Stokes" in method:
                restored = engine.inpaint(
                    st.session_state.working_image,
                    mask,
                    method='ns',
                    radius=st.session_state.get('inpaint_radius', 5)
                )
            elif "Multi-Scale" in method:
                restored = engine.multi_scale_inpaint(st.session_state.working_image, mask)
            elif "Edge-Preserving" in method:
                restored = engine.edge_preserving_inpaint(st.session_state.working_image, mask)
            else:
                restored = engine.inpaint(st.session_state.working_image, mask)
            
            st.session_state.restored_image = restored
            st.success("✅ Inpainting completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during inpainting: {str(e)}")


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
            st.image(cv2_to_pil(st.session_state.working_image), caption="Current Image", use_container_width=True)
        
        with col2:
            if st.button("✨ Auto Enhance", type="primary"):
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
    
    with tab2:
        st.subheader("Manual Adjustments")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            brightness = st.slider("Brightness", -100, 100, 0)
            contrast = st.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
            saturation = st.slider("Saturation", 0.5, 2.5, 1.0, 0.1)
            
            if st.button("Apply Manual Adjustments", type="primary"):
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
            # Live preview
            preview = st.session_state.working_image.copy()
            preview = corrector.adjust_brightness_contrast(preview, brightness, contrast)
            if saturation != 1.0:
                preview = corrector.enhance_faded_colors(preview, saturation)
            
            st.image(cv2_to_pil(preview), caption="Preview", use_container_width=True)
    
    with tab3:
        st.subheader("Color Balance & White Balance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wb_method = st.selectbox("White Balance Method", ["Gray World", "White Patch"])
            
            if st.button("Apply White Balance", type="primary"):
                with st.spinner("Applying white balance..."):
                    method = 'gray_world' if 'Gray' in wb_method else 'white_patch'
                    result = corrector.white_balance(st.session_state.working_image, method=method)
                    st.session_state.restored_image = result
                    st.rerun()
            
            st.markdown("---")
            
            balance_percent = st.slider("Color Balance (%)", 0.1, 5.0, 1.0, 0.1)
            
            if st.button("Apply Color Balance", type="primary"):
                with st.spinner("Applying color balance..."):
                    result = corrector.color_balance(st.session_state.working_image, balance_percent)
                    st.session_state.restored_image = result
                    st.rerun()
        
        with col2:
            st.image(cv2_to_pil(st.session_state.working_image), caption="Current Image", use_container_width=True)
    
    with tab4:
        st.subheader("Advanced Corrections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram equalization
            st.write("**Histogram Equalization**")
            hist_method = st.radio("Method", ["Adaptive (CLAHE)", "Standard"])
            
            if st.button("Apply Histogram Equalization", type="primary"):
                with st.spinner("Applying histogram equalization..."):
                    method = 'adaptive' if 'Adaptive' in hist_method else 'standard'
                    result = corrector.histogram_equalization(st.session_state.working_image, method=method)
                    st.session_state.restored_image = result
                    st.rerun()
            
            st.markdown("---")
            
            # Denoising
            st.write("**Denoising**")
            denoise_strength = st.slider("Strength", 1, 20, 10)
            
            if st.button("Apply Denoising", type="primary"):
                with st.spinner("Denoising image..."):
                    result = corrector.denoise(st.session_state.working_image, denoise_strength)
                    st.session_state.restored_image = result
                    st.rerun()
        
        with col2:
            # Sharpening
            st.write("**Unsharp Masking (Sharpening)**")
            sharpen_sigma = st.slider("Blur Sigma", 0.5, 3.0, 1.0, 0.1)
            sharpen_amount = st.slider("Sharpen Amount", 0.0, 2.0, 1.0, 0.1)
            
            if st.button("Apply Sharpening", type="primary"):
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
            st.image(cv2_to_pil(st.session_state.working_image), caption="Before", use_container_width=True)
        with col2:
            st.image(cv2_to_pil(st.session_state.restored_image), caption="After", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Accept Changes"):
                st.session_state.working_image = st.session_state.restored_image.copy()
                st.session_state.restored_image = None
                st.success("Changes accepted!")
                st.rerun()
        with col2:
            if st.button("❌ Discard Changes"):
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
        st.image(cv2_to_pil(comparison), use_container_width=True)
        
    elif comparison_mode == "Vertical Stack":
        comparison = create_comparison(original, restored, mode='vertical')
        st.image(cv2_to_pil(comparison), use_container_width=True)
        
    elif comparison_mode == "Split View":
        comparison = create_comparison(original, restored, mode='split')
        st.image(cv2_to_pil(comparison), caption="Swipe left/right comparison", use_container_width=True)
        
    elif comparison_mode == "Overlay Blend":
        blend_amount = st.slider("Blend Amount", 0.0, 1.0, 0.5, 0.05)
        comparison = blend_images(original, restored, alpha=blend_amount)
        st.image(cv2_to_pil(comparison), use_container_width=True)
        
    elif comparison_mode == "Original Only":
        st.image(cv2_to_pil(original), caption="Original Image", use_container_width=True)
        
    elif comparison_mode == "Restored Only":
        st.image(cv2_to_pil(restored), caption="Restored Image", use_container_width=True)
    
    # Export section
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Original Image**")
        original_pil = cv2_to_pil(original)
        buf_orig = io.BytesIO()
        original_pil.save(buf_orig, format='PNG')
        st.download_button(
            label="Download Original",
            data=buf_orig.getvalue(),
            file_name="original_image.png",
            mime="image/png"
        )
    
    with col2:
        st.write("**Restored Image**")
        restored_pil = cv2_to_pil(restored)
        buf_rest = io.BytesIO()
        restored_pil.save(buf_rest, format='PNG')
        st.download_button(
            label="Download Restored",
            data=buf_rest.getvalue(),
            file_name="restored_image.png",
            mime="image/png"
        )
    
    with col3:
        st.write("**Side-by-Side Comparison**")
        comparison_img = create_comparison(original, restored, mode='side-by-side')
        comparison_pil = cv2_to_pil(comparison_img)
        buf_comp = io.BytesIO()
        comparison_pil.save(buf_comp, format='PNG')
        st.download_button(
            label="Download Comparison",
            data=buf_comp.getvalue(),
            file_name="comparison_image.png",
            mime="image/png"
        )
    
    # Statistics
    st.markdown("---")
    st.subheader("📈 Image Statistics")
    
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


if __name__ == "__main__":
    main()


