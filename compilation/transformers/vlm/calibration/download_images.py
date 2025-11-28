#!/usr/bin/env python
"""
Download 100 images from COCO validation set and resize to 224x224.
Uses the Hugging Face datasets library to access COCO images.
"""

from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_coco_samples(output_dir="images", num_images=100, target_size=(224, 224)):
    """
    Download COCO validation samples using Hugging Face datasets.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to download
        target_size: Tuple of (width, height) for resizing
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Loading COCO dataset...")
    print(f"Downloading {num_images} images to {output_dir}/")

    try:
        # Load COCO 2017 validation set (it's smaller and faster)
        dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)

        downloaded = 0
        failed = 0

        # Iterate through the dataset
        for idx, example in enumerate(
            tqdm(dataset, desc="Downloading images", total=num_images)
        ):
            if downloaded >= num_images:
                break

            try:
                # Get the image from the dataset
                img = example["image"]

                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to target size
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                # Save image
                output_file = output_path / f"image_{downloaded:04d}.jpg"
                img.save(output_file, "JPEG", quality=95)

                downloaded += 1

            except Exception as e:
                failed += 1
                if failed % 10 == 0:
                    print(f"\nWarning: {failed} downloads failed so far. Continuing...")
                continue

        print(f"\n✓ Successfully downloaded {downloaded} images")
        print(f"✗ Failed to download {failed} images")
        print(f"Images saved to: {output_path.absolute()}")

        return downloaded

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative approach with sample images...")
        return download_sample_images(output_dir, num_images, target_size)


def download_sample_images(output_dir="images", num_images=100, target_size=(224, 224)):
    """
    Generate sample images if dataset download fails.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
        target_size: Tuple of (width, height)
    """
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Generating {num_images} sample images...")

    for i in tqdm(range(num_images), desc="Generating images"):
        # Create a random image with some structure
        img_array = np.random.randint(
            0, 256, (target_size[1], target_size[0], 3), dtype=np.uint8
        )

        # Add some patterns to make it look less random
        for channel in range(3):
            gradient = np.linspace(0, 255, target_size[0], dtype=np.uint8)
            img_array[:, :, channel] = (
                img_array[:, :, channel] * 0.7 + gradient * 0.3
            ).astype(np.uint8)

        img = Image.fromarray(img_array)
        output_file = output_path / f"image_{i:04d}.jpg"
        img.save(output_file, "JPEG", quality=95)

    print(f"\n✓ Generated {num_images} sample images")
    print(f"Images saved to: {output_path.absolute()}")
    return num_images


if __name__ == "__main__":
    # Download 100 images at 224x224 resolution
    download_coco_samples(output_dir="images", num_images=100, target_size=(224, 224))
