"""
Restoration module for digital painting restoration.
Contains inpainting and color correction algorithms.
"""

from .inpainting import InpaintingEngine
from .color_correction import ColorCorrector

# Try to import deep learning module (optional)
try:
    from .deep_inpainting import DeepInpaintingEngine, create_deep_inpainting_engine
    __all__ = ['InpaintingEngine', 'ColorCorrector', 'DeepInpaintingEngine', 'create_deep_inpainting_engine']
except ImportError:
    __all__ = ['InpaintingEngine', 'ColorCorrector']

