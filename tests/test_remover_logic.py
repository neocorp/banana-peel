import pytest
from banana_peel.remover import get_watermark_size, get_watermark_position

# -------------------------------------------------------------------------
# get_watermark_size
# -------------------------------------------------------------------------

def test_watermark_size_large_image():
    # > 1024x1024
    assert get_watermark_size(2000, 2000) == 96
    assert get_watermark_size(1025, 1025) == 96

def test_watermark_size_small_image():
    # <= 1024x1024
    assert get_watermark_size(1024, 1024) == 48
    assert get_watermark_size(800, 600) == 48

def test_watermark_size_boundary():
    # Edge case mixed dimensions
    assert get_watermark_size(2000, 500) == 48  # One dim is small -> 48
    assert get_watermark_size(500, 2000) == 48

# -------------------------------------------------------------------------
# get_watermark_position
# -------------------------------------------------------------------------

def test_watermark_position_large():
    # 96px watermark has 64px margin
    w, h = 2000, 2000
    wm_size = 96
    margin = 64
    
    x, y, wa, hb = get_watermark_position(w, h, wm_size)
    
    expected_x = w - margin - wm_size
    expected_y = h - margin - wm_size
    
    assert x == expected_x
    assert y == expected_y
    assert wa == wm_size
    assert hb == wm_size

def test_watermark_position_small():
    # 48px watermark has 32px margin
    w, h = 800, 600
    wm_size = 48
    margin = 32
    
    x, y, wa, hb = get_watermark_position(w, h, wm_size)
    
    expected_x = w - margin - wm_size
    expected_y = h - margin - wm_size
    
    assert x == expected_x
    assert y == expected_y
