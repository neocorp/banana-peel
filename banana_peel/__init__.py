"""
Banana Peel - Gemini Watermark Remover CLI

Remove Gemini AI watermarks from images using
LaMa (Large Mask Inpainting) AI model.

https://github.com/neocorp/banana-peel
"""

__version__ = "0.1.1"
__author__ = "Niyazi Erdogan"
__license__ = "MIT"

from .remover import remove_gemini_watermark

__all__ = ["remove_gemini_watermark"]
