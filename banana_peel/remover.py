"""
Core watermark removal using LaMa AI inpainting.

Uses the LaMa (Resolution-robust Large Mask Inpainting) model
via ONNX Runtime to seamlessly remove Gemini watermarks.

The model is auto-downloaded on first run (~198 MB, one-time).
"""

import os
import sys
import ssl
import re
import warnings
from pathlib import Path
from typing import Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn(
        "onnxruntime not installed. Install it with: pip install onnxruntime"
    )

# Constants
ALPHA_THRESHOLD = 0.002
MAX_ALPHA = 0.99
LOGO_VALUE = 255

# Model download config
MODEL_FILENAME = "lama_fp32.onnx"
MODEL_URL = "https://github.com/neocorp/banana-peel/releases/download/v0.1.0/lama_fp32.onnx"
MODEL_MIN_SIZE = 100 * 1024 * 1024  # 100 MB sanity check


def get_model_dir() -> Path:
    """Get the model cache directory (~/.cache/banana-peel/)."""
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    model_dir = cache_root / "banana-peel"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_model_path() -> Path:
    """Get the full path to the ONNX model file."""
    return get_model_dir() / MODEL_FILENAME


def ensure_model_downloaded() -> Path:
    """
    Ensure the LaMa ONNX model is downloaded.
    
    Downloads from GitHub Releases on first run with a progress bar.
    Returns the path to the model file.
    """
    model_path = get_model_path()
    
    # Already downloaded?
    if model_path.exists() and model_path.stat().st_size > MODEL_MIN_SIZE:
        return model_path
    
    # Also check legacy location (banana_peel/models/)
    legacy_path = Path(__file__).parent / "models" / MODEL_FILENAME
    if legacy_path.exists() and legacy_path.stat().st_size > MODEL_MIN_SIZE:
        return legacy_path
    
    print()
    print("🍌 Banana Peel - First Run Setup")
    print("━" * 35)
    print("Downloading AI model (198 MB)... this only happens once.")
    print()
    
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    req = Request(MODEL_URL, headers={"User-Agent": "banana-peel/0.1.0"})
    
    try:
        response = urlopen(req, context=ssl_context, timeout=600)
        total_size = int(response.headers.get("Content-Length", 0))
        
        # If we got an HTML page (GitHub redirect), handle it
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            html = response.read().decode("utf-8", errors="ignore")
            # Handle Google Drive-style confirmation
            confirm_match = re.search(r'name="confirm" value="([^"]+)"', html)
            if confirm_match:
                print("  Handling download confirmation...")
                # This shouldn't happen with GitHub Releases, but just in case
        
        # Download with progress
        chunk_size = 1024 * 1024  # 1 MB chunks
        downloaded = 0
        
        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if tqdm and total_size > 0:
            pbar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {MODEL_FILENAME}",
                bar_format="{desc} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {rate_fmt}",
            )
        else:
            pbar = None
        
        with open(model_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if pbar:
                    pbar.update(len(chunk))
                elif total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"\r  Downloading... {pct:.0f}%", end="", flush=True)
        
        if pbar:
            pbar.close()
        else:
            print()
        
        # Validate download
        actual_size = model_path.stat().st_size
        if actual_size < MODEL_MIN_SIZE:
            model_path.unlink(missing_ok=True)
            print(f"\n✗ Download failed (file too small: {actual_size} bytes)")
            print(f"  Please download manually from: {MODEL_URL}")
            print(f"  Place the file at: {model_path}")
            sys.exit(1)
        
        print(f"\n✓ Model ready! ({actual_size / 1024 / 1024:.1f} MB)")
        print()
        return model_path
        
    except Exception as e:
        model_path.unlink(missing_ok=True)
        print(f"\n✗ Download failed: {e}")
        print(f"  Please download manually from: {MODEL_URL}")
        print(f"  Place the file at: {model_path}")
        sys.exit(1)


# ─── Watermark Detection ────────────────────────────────────────────


def calculate_alpha_map(mask_image: Image.Image) -> np.ndarray:
    """Calculate alpha map from mask image."""
    if mask_image.mode != "RGB":
        mask_image = mask_image.convert("RGB")
    mask_array = np.array(mask_image, dtype=np.float32)
    return np.max(mask_array[:, :, :3], axis=2) / 255.0


def get_watermark_size(image_width: int, image_height: int) -> int:
    """Determine watermark size based on image dimensions."""
    if image_width > 1024 and image_height > 1024:
        return 96
    return 48


def get_watermark_position(
    image_width: int,
    image_height: int,
    watermark_size: int,
) -> Tuple[int, int, int, int]:
    """Calculate watermark position in image."""
    margin = 64 if watermark_size == 96 else 32
    x = image_width - margin - watermark_size
    y = image_height - margin - watermark_size
    return (x, y, watermark_size, watermark_size)


# ─── LaMa Inpainting ────────────────────────────────────────────────


class LaMaInpainter:
    """LaMa inpainting engine using ONNX Runtime."""

    def __init__(self, model_path: str | Path):
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is required. Install: pip install onnxruntime"
            )
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def preprocess(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Prepare image and mask tensors for the model (512×512)."""
        h, w = image.shape[:2]
        target = 512

        pad_h = max(0, target - h)
        pad_w = max(0, target - w)

        if pad_h > 0 or pad_w > 0:
            padded_img = np.pad(
                image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
            )
            padded_mask = np.pad(
                mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )
        else:
            padded_img = image[:target, :target]
            padded_mask = mask[:target, :target]

        # HWC → CHW, normalise to [0, 1]
        img_tensor = padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = img_tensor[np.newaxis, ...]

        mask_tensor = (padded_mask > 127).astype(np.float32)
        mask_tensor = mask_tensor[np.newaxis, np.newaxis, ...]

        return img_tensor, mask_tensor, h, w

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run LaMa inpainting with automatic output range detection."""
        img_t, mask_t, orig_h, orig_w = self.preprocess(image, mask)

        outputs = self.session.run(None, {"image": img_t, "mask": mask_t})

        output = outputs[0][0].transpose(1, 2, 0)  # CHW → HWC

        # Auto-detect output range ([0,1] vs [0,255])
        if np.max(np.abs(output)) <= 2.0:
            output = output * 255.0

        output = np.clip(output, 0, 255).astype(np.uint8)
        return output[:orig_h, :orig_w]


def remove_watermark(
    image: Image.Image,
    alpha_map: np.ndarray,
    position: Tuple[int, int, int, int],
    mask_dilation: int = 25,
) -> Image.Image:
    """
    Remove watermark using LaMa AI inpainting.

    Crops the watermark area to 512×512, runs the model,
    then feather-blends the result back into the original image.
    """
    x, y, wm_width, wm_height = position
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Build dilated mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    alpha_binary = (alpha_map > 0.1).astype(np.uint8) * 255
    full_mask[y : y + wm_height, x : x + wm_width] = alpha_binary

    kernel = np.ones((mask_dilation, mask_dilation), np.uint8)
    full_mask_dilated = cv2.dilate(full_mask, kernel, iterations=1)

    # Crop bottom-right 512×512 region
    crop_size = 512
    cy1 = max(0, h - crop_size)
    cx1 = max(0, w - crop_size)
    img_crop = img_array[cy1:h, cx1:w]
    mask_crop = full_mask_dilated[cy1:h, cx1:w]

    # Run model
    model_path = ensure_model_downloaded()
    inpainter = LaMaInpainter(model_path)
    result_raw = inpainter.inpaint(img_crop, mask_crop)

    # Feathered blending: only modify the watermark area
    feather_radius = 15
    mask_f = mask_crop.astype(np.float32)
    mask_f = (
        cv2.GaussianBlur(
            mask_f, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0
        )
        / 255.0
    )
    mask_f = np.expand_dims(mask_f, axis=-1)

    blended = (
        result_raw.astype(np.float32) * mask_f
        + img_crop.astype(np.float32) * (1.0 - mask_f)
    )
    result_crop = np.clip(blended, 0, 255).astype(np.uint8)

    img_array[cy1:h, cx1:w] = result_crop
    return Image.fromarray(img_array)


# ─── Public API ──────────────────────────────────────────────────────


def remove_gemini_watermark(
    image_path: str | Path,
    mask_48_path: str | Path | None = None,
    mask_96_path: str | Path | None = None,
) -> Image.Image:
    """
    Remove Gemini watermark from an image.

    Args:
        image_path: Path to image file
        mask_48_path: Path to 48×48 mask (uses bundled default)
        mask_96_path: Path to 96×96 mask (uses bundled default)

    Returns:
        Processed PIL Image with watermark removed
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    watermark_size = get_watermark_size(width, height)

    if mask_48_path is None:
        mask_48_path = Path(__file__).parent / "assets" / "bg_48.png"
    if mask_96_path is None:
        mask_96_path = Path(__file__).parent / "assets" / "bg_96.png"

    mask_image = Image.open(mask_96_path if watermark_size == 96 else mask_48_path)
    alpha_map = calculate_alpha_map(mask_image)
    position = get_watermark_position(width, height, watermark_size)

    return remove_watermark(image, alpha_map, position)
