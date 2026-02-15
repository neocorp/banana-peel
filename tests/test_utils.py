import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from banana_peel.utils import (
    validate_input_path, 
    get_output_path, 
    is_supported_image, 
    get_image_files
)

# -------------------------------------------------------------------------
# validate_input_path
# -------------------------------------------------------------------------

def test_validate_directory_exists():
    with TemporaryDirectory() as tmpdir:
        is_valid, msg = validate_input_path(tmpdir)
        assert is_valid is True
        assert msg == ""

def test_validate_file_exists_and_supported():
    with TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.png"
        p.touch()
        is_valid, msg = validate_input_path(p)
        assert is_valid is True
        assert msg == ""

def test_validate_file_unsupported_extension():
    with TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.txt"
        p.touch()
        is_valid, msg = validate_input_path(p)
        assert is_valid is False
        assert "Unsupported file format" in msg

def test_validate_non_existent():
    is_valid, msg = validate_input_path("non_existent_file_12345.png")
    assert is_valid is False
    assert "Path does not exist" in msg

# -------------------------------------------------------------------------
# get_output_path
# -------------------------------------------------------------------------

def test_get_output_path_default():
    input_path = Path("/path/to/image.png")
    output = get_output_path(input_path)
    assert output == Path("/path/to/image_clean.png")

def test_get_output_path_custom_suffix():
    input_path = Path("/path/to/image.jpg")
    output = get_output_path(input_path, suffix="_fixed")
    assert output == Path("/path/to/image_fixed.jpg")

def test_get_output_path_overwrite():
    input_path = Path("/path/to/image.png")
    output = get_output_path(input_path, overwrite=True)
    assert output == input_path

# -------------------------------------------------------------------------
# is_supported_image
# -------------------------------------------------------------------------

def test_is_supported_image():
    assert is_supported_image("test.png") is True
    assert is_supported_image("test.JPG") is True  # Case insensitive check usually handled by Path, but utils implementation uses .lower()
    assert is_supported_image("test.webp") is True
    assert is_supported_image("test.txt") is False

# -------------------------------------------------------------------------
# get_image_files
# -------------------------------------------------------------------------

def test_get_image_files():
    with TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "a.png").touch()
        (d / "b.JPG").touch()
        (d / "c.txt").touch()
        (d / "subfolder").mkdir()
        
        files = get_image_files(d)
        filenames = [f.name for f in files]
        
        assert "a.png" in filenames
        assert "b.JPG" in filenames
        assert "c.txt" not in filenames
        assert len(files) == 2

def test_get_image_files_not_dir():
    with pytest.raises(ValueError):
        get_image_files("non_existent_dir_123")
