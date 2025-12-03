# 🎨 Digital Restoration of Paintings

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-5C3EE8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)

An interactive web-based application for digital restoration of artwork, designed for art historians and conservators. This tool combines advanced computer vision techniques with an intuitive interface to restore damaged or faded paintings.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Project Structure](#-project-structure)
- [Algorithms](#-algorithms)
- [Testing](#-testing)
- [Requirements](#-requirements)
- [References](#-references)
- [Author](#-author)

---

## 🎯 Overview

This application enables users to digitally restore damaged or faded artwork through an intuitive web interface. Users can upload images, mark damaged regions interactively, apply sophisticated inpainting algorithms to fill missing areas, enhance faded colors, and export professional-quality restored images.

**Key Capabilities:**
- Interactive mask drawing for marking damaged regions
- Multiple inpainting algorithms for different restoration scenarios
- Comprehensive color correction and enhancement tools
- Side-by-side comparison and visualization
- Export functionality for all image versions

---

## ✨ Features

### 🖌️ Interactive Inpainting
- **Multiple Drawing Tools**: Freedraw, line, rectangle, circle, and transform tools
- **Adjustable Brush Size**: 1-50 pixels for precise mask creation
- **4 Inpainting Algorithms**:
  - Telea Fast Marching Method (fast, efficient)
  - Navier-Stokes based (structure-preserving)
  - Multi-scale progressive (for large regions)
  - Edge-preserving (maintains structural integrity)

### 🎨 Color Correction & Enhancement
- **Automatic Enhancement**: One-click auto-enhancement pipeline
- **Manual Adjustments**: Brightness, contrast, and saturation controls
- **White Balance**: Gray World and White Patch methods
- **Color Balance**: Automatic color correction with percentile adjustment
- **Histogram Equalization**: Adaptive (CLAHE) and standard methods
- **Advanced Tools**: Denoising, sharpening (unsharp mask), and more

### 📊 Comparison & Export
- **6 Comparison Modes**: Side-by-side, vertical stack, split view, overlay blend, and individual views
- **Export Options**: Download original, restored, and comparison images
- **Image Statistics**: Detailed metrics for analysis
- **PNG Format**: High-quality export in PNG format

### 🎛️ User Interface
- **Intuitive Navigation**: 4-page workflow (Upload → Inpaint → Enhance → Export)
- **Real-time Previews**: Live preview of adjustments before applying
- **Accept/Reject Workflow**: Non-destructive editing with undo capability
- **Responsive Design**: Works on various screen sizes
- **Professional Styling**: Custom CSS for polished appearance

---

## 📸 Screenshots

*Note: Add screenshots of your application here to showcase the interface*

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Method 1: Automated Setup (Recommended)

**macOS/Linux:**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

**Windows:**
```batch
quick_start.bat
```

### Method 2: Manual Setup

1. **Clone the repository:**
```bash
git clone https://github.com/ophelia-37/cpsc4900-senior-project.git
cd cpsc4900-senior-project
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

1. **Activate your virtual environment** (if not already active):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Run the application:**
```bash
streamlit run app.py
```

3. **Open your browser:**
   - The application will automatically open at `http://localhost:8501`
   - If it doesn't, navigate to that URL manually

4. **Start restoring:**
   - Upload an image in the "Upload & Prepare" section
   - Draw masks on damaged regions in "Inpainting"
   - Apply color corrections in "Color Correction"
   - Compare and export results in "Compare & Export"

---

## 📖 Usage

### Step 1: Upload & Prepare
1. Click "Upload & Prepare" in the sidebar
2. Upload an image file (JPG, PNG, TIFF, or BMP)
3. Adjust maximum image size if needed (default: 1024px)
4. Optionally add artwork metadata (title, artist, date, notes)

### Step 2: Inpainting
1. Navigate to "Inpainting" in the sidebar
2. Select a drawing tool (freedraw recommended for most cases)
3. Adjust brush size as needed
4. Draw on the canvas to mark damaged regions (white = damaged)
5. Choose an inpainting algorithm:
   - **Telea**: Fast, good for small-medium regions
   - **Navier-Stokes**: Better for preserving structure
   - **Multi-Scale**: Best for large damaged areas
   - **Edge-Preserving**: Maintains edges and patterns
6. Click "Apply Inpainting"
7. Review the result and click "Accept Restoration" if satisfied

### Step 3: Color Correction
1. Go to "Color Correction" in the sidebar
2. Choose from four tabs:
   - **Quick Enhance**: One-click automatic enhancement
   - **Manual Adjustments**: Fine-tune brightness, contrast, saturation
   - **Color Balance**: White balance and color correction
   - **Advanced**: Histogram equalization, denoising, sharpening
3. Adjust parameters and preview results
4. Click "Accept Changes" when satisfied

### Step 4: Compare & Export
1. Navigate to "Compare & Export"
2. Select a comparison mode to view results
3. Review image statistics
4. Download original, restored, or comparison images

---

## 🔬 Technical Details

### Architecture

The application is built with a modular architecture:

- **Frontend**: Streamlit web framework
- **Image Processing**: OpenCV and NumPy
- **UI Components**: Streamlit Drawable Canvas for interactive drawing
- **Session Management**: Streamlit session state for workflow continuity

### Design Principles

- **Modularity**: Separate modules for inpainting, color correction, and utilities
- **Non-destructive Editing**: Original image preserved, changes applied incrementally
- **User-Friendly**: No technical knowledge required
- **Performance**: Efficient algorithms with progress indicators for long operations

---

## 📁 Project Structure

```
cpsc4900-senior-project/
├── app.py                      # Main Streamlit application (639 lines)
├── restoration/
│   ├── __init__.py
│   ├── inpainting.py          # Inpainting algorithms (167 lines)
│   └── color_correction.py     # Color enhancement methods (223 lines)
├── utils/
│   ├── __init__.py
│   └── image_utils.py         # Image processing utilities (232 lines)
├── test_algorithms.py          # Automated testing suite (270 lines)
├── requirements.txt            # Python dependencies
├── quick_start.sh              # macOS/Linux setup script
├── quick_start.bat             # Windows setup script
└── README.md                   # This file
```

---

## 🧮 Algorithms

### Inpainting Methods

1. **Telea Fast Marching Method**
   - Based on: Telea, A. (2004). "An Image Inpainting Technique Based on the Fast Marching Method"
   - Fast, efficient algorithm using fast marching method
   - Good for: Small to medium damaged regions
   - Parameters: Radius (1-20 pixels)

2. **Navier-Stokes Based Inpainting**
   - Based on: Bertalmio et al. (2000). "Image Inpainting"
   - Structure-preserving algorithm using fluid dynamics
   - Good for: Maintaining edges and patterns
   - Parameters: Radius (1-20 pixels)

3. **Multi-Scale Progressive Inpainting**
   - Custom implementation for large damaged regions
   - Processes at multiple scales (coarse to fine)
   - Progressive mask erosion for refinement
   - Good for: Large damaged areas

4. **Edge-Preserving Inpainting**
   - Combines Canny edge detection with inpainting
   - Maintains structural integrity
   - Good for: Images with strong edges and patterns

### Color Correction Methods

1. **Histogram Equalization**
   - Adaptive (CLAHE): Prevents over-amplification, grid-based
   - Standard: Global histogram equalization
   - Works in LAB color space for color preservation

2. **Color Balance**
   - Automatic stretching using percentile clipping
   - Per-channel normalization
   - Adjustable clip percentage

3. **White Balance**
   - Gray World: Assumes average scene color is neutral
   - White Patch: Assumes brightest point should be white

4. **Saturation Enhancement**
   - HSV color space manipulation
   - Factor-based multiplication (0.5-2.5×)

5. **Brightness & Contrast**
   - Linear adjustments with alpha/beta parameters
   - Real-time preview support

6. **Noise Reduction**
   - Non-local Means Denoising
   - Color-aware processing
   - Adjustable strength

7. **Unsharp Masking**
   - Gaussian blur-based sharpening
   - Enhances edge details without artifacts

8. **Automatic Enhancement Pipeline**
   - 5-step process: Denoise → White Balance → Histogram Equalization → Saturation → Sharpening

---

## 🧪 Testing

Run the automated test suite to verify all algorithms:

```bash
python test_algorithms.py
```

This will:
- Test all 4 inpainting methods
- Test all 8+ color correction techniques
- Test utility functions
- Generate test output images in `test_outputs/` directory
- Display pass/fail status for each test

---

## 📦 Requirements

### Core Dependencies

- `opencv-python==4.8.1.78` - Image processing and inpainting
- `opencv-contrib-python==4.8.1.78` - Additional OpenCV features
- `numpy==1.24.3` - Numerical operations
- `Pillow==10.1.0` - Image I/O and format conversion
- `streamlit==1.28.0` - Web application framework
- `streamlit-drawable-canvas==0.9.3` - Interactive canvas component

### Optional Dependencies

- `scipy==1.11.3` - Scientific computing
- `matplotlib==3.8.0` - Plotting and visualization
- `scikit-image==0.22.0` - Additional image processing tools

---

## 📚 References

### Academic Papers

- **Criminisi, A., Pérez, P., & Toyama, K.** (2004). "Region Filling and Object Removal by Exemplar-Based Image Inpainting." *IEEE Transactions on Image Processing*, 13(9), 1200-1212.

- **Bertalmio, M., Sapiro, G., Caselles, V., & Ballester, C.** (2000). "Image Inpainting." *Proceedings of SIGGRAPH 2000*, 417-424.

- **Telea, A.** (2004). "An Image Inpainting Technique Based on the Fast Marching Method." *Journal of Graphics Tools*, 9(1), 23-34.

### Libraries & Frameworks

- [OpenCV](https://opencv.org/) - Computer vision library
- [Streamlit](https://streamlit.io/) - Web application framework
- [NumPy](https://numpy.org/) - Numerical computing
- [Pillow](https://python-pillow.org/) - Image processing

---

## 👤 Author

**Ophelia Chuning He**

- **Advisor**: Alex Wong
- **Course**: CPSC 4900 - Senior Project
- **Institution**: [Your University]
- **Year**: 2024-2025

---

## 📄 License

This project is part of an academic senior project. Please respect academic integrity and cite appropriately if using this code.

---

## 🙏 Acknowledgments

- Professor Alex Wong for guidance and supervision
- OpenCV community for excellent documentation
- Streamlit team for the web framework
- Art conservation community for inspiration

---

## 🔮 Future Enhancements

Potential improvements for future versions:

- [ ] Deep learning-based inpainting (GANs, diffusion models)
- [ ] Batch processing for multiple images
- [ ] Save/load project sessions
- [ ] Undo/redo functionality
- [ ] Additional export formats (TIFF, BMP)
- [ ] Region-specific color corrections
- [ ] User authentication and cloud storage
- [ ] Mobile-optimized interface

---

## 📞 Support

For questions or issues:

1. Check the code comments and docstrings
2. Review the test suite for usage examples
3. Open an issue on GitHub (if applicable)

---

**Made with ❤️ for art conservation and digital humanities**
