"""
CLI interface for banana-peel using Click.
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from PIL import Image
from tqdm import tqdm

from .remover import remove_gemini_watermark
from .utils import (
    get_image_files,
    get_output_path,
    save_image_with_exif,
    validate_input_path,
)


def process_single_image(
    input_path: Path,
    suffix: str,
    overwrite: bool,
    pbar: tqdm | None = None,
) -> tuple[bool, str]:
    """
    Process a single image file.

    Args:
        input_path: Path to input image
        suffix: Suffix for output filename
        overwrite: Whether to overwrite original
        pbar: Optional progress bar to update

    Returns:
        Tuple of (success, message)
    """
    try:
        original_image = Image.open(input_path)
        result_image = remove_gemini_watermark(input_path)

        output_path = get_output_path(input_path, suffix, overwrite)
        save_image_with_exif(result_image, output_path, original_image)

        if pbar:
            pbar.update(1)

        return True, f"✓ {input_path.name}"

    except Exception as e:
        if pbar:
            pbar.update(1)
        return False, f"✗ {input_path.name}: {str(e)}"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", required=False)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite original files instead of creating new ones with suffix",
)
@click.option(
    "--suffix",
    default="_clean",
    show_default=True,
    help="Suffix to add to output filenames",
)
@click.version_option(version="0.1.0", prog_name="banana-peel")
@click.pass_context
def main(ctx: click.Context, path: str | None, overwrite: bool, suffix: str) -> None:
    """
    Remove Gemini AI watermarks from images using LaMa AI inpainting.

    PATH can be a single image file or a directory containing images.
    By default, processed images are saved with '_clean' suffix.

    \b
    Examples:
        banana-peel image.png
        banana-peel image.png --overwrite
        banana-peel ./photos/
        banana-peel ./photos/ --suffix "_nowm"
    """
    # Handle "banana-peel" (no args) -> Show help
    if not path:
        click.echo(ctx.get_help())
        return

    # Handle "banana-peel help" -> Show help
    if path == "help":
        click.echo(ctx.get_help())
        return

    # Handle "banana-peel version" -> Show version
    if path == "version":
        click.echo(f"{ctx.find_root().info_name}, version 0.1.0")
        return

    input_path = Path(path)

    is_valid, error_msg = validate_input_path(path)
    if not is_valid:
        click.echo(f"Error: {error_msg}", err=True)
        sys.exit(1)

    if input_path.is_file():
        click.echo(f"Processing: {input_path.name}")

        success, message = process_single_image(input_path, suffix, overwrite)

        if success:
            click.echo(message)
            if not overwrite:
                output_path = get_output_path(input_path, suffix, overwrite)
                click.echo(f"Saved to: {output_path}")
        else:
            click.echo(message, err=True)
            sys.exit(1)

    else:
        try:
            image_files = get_image_files(input_path)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        if not image_files:
            click.echo("No supported image files found in directory.")
            sys.exit(0)

        click.echo(f"Found {len(image_files)} image(s) in {input_path}")
        click.echo()

        success_count = 0
        fail_count = 0

        with tqdm(total=len(image_files), desc="Processing", unit="img") as pbar:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(
                        process_single_image,
                        img_path,
                        suffix,
                        overwrite,
                        None,
                    ): img_path
                    for img_path in image_files
                }

                for future in as_completed(future_to_file):
                    success, message = future.result()
                    pbar.update(1)

                    if success:
                        success_count += 1
                        click.echo(message)
                    else:
                        fail_count += 1
                        click.echo(message, err=True)

        click.echo()
        click.echo(f"Completed: {success_count} successful, {fail_count} failed")

        if fail_count > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
