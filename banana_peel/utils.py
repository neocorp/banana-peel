"""
Utility functions for file handling and validation.
"""

import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image


# Supported image formats
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'}


def is_supported_image(path: str | Path) -> bool:
    """
    Check if a file is a supported image format.
    
    Args:
        path: Path to the file
        
    Returns:
        True if the file has a supported extension
    """
    path = Path(path)
    return path.suffix.lower() in SUPPORTED_FORMATS


def get_image_files(directory: str | Path) -> List[Path]:
    """
    Get all supported image files in a directory (non-recursive).
    
    Args:
        directory: Path to the directory
        
    Returns:
        List of Path objects for supported image files
    """
    directory = Path(directory)
    
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    image_files = []
    
    for item in directory.iterdir():
        if item.is_file() and is_supported_image(item):
            image_files.append(item)
    
    # Sort for consistent ordering
    image_files.sort()
    
    return image_files


def validate_input_path(path: str | Path) -> Tuple[bool, str]:
    """
    Validate that the input path exists and is either a file or directory.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(path)
    
    if not path.exists():
        return False, f"Path does not exist: {path}"
    
    if path.is_file():
        if not is_supported_image(path):
            return False, f"Unsupported file format: {path.suffix}"
        return True, ""
    
    if path.is_dir():
        return True, ""
    
    return False, f"Path is neither a file nor a directory: {path}"


def get_output_path(
    input_path: str | Path,
    suffix: str = "_clean",
    overwrite: bool = False
) -> Path:
    """
    Generate the output path for a processed image.
    
    Args:
        input_path: Original image path
        suffix: Suffix to add to filename (default: _clean)
        overwrite: If True, return original path
        
    Returns:
        Path for the output file
    """
    input_path = Path(input_path)
    
    if overwrite:
        return input_path
    
    # Generate new filename with suffix
    stem = input_path.stem
    output_name = f"{stem}{suffix}{input_path.suffix}"
    output_path = input_path.parent / output_name
    
    return output_path


def save_image_with_exif(
    image: Image.Image,
    output_path: str | Path,
    original_image: Image.Image
) -> None:
    """
    Save an image while preserving EXIF metadata from the original.
    
    Args:
        image: Processed image to save
        output_path: Path to save to
        original_image: Original image to extract EXIF from
    """
    output_path = Path(output_path)
    
    # Extract EXIF from original image
    exif_data = original_image.info.get('exif')
    
    # Save with EXIF if available
    if exif_data:
        image.save(output_path, exif=exif_data)
    else:
        image.save(output_path)
