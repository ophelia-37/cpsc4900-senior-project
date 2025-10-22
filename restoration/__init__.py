"""
Restoration module for digital painting restoration.
Contains inpainting and color correction algorithms.
"""

from .inpainting import InpaintingEngine
from .color_correction import ColorCorrector

__all__ = ['InpaintingEngine', 'ColorCorrector']

