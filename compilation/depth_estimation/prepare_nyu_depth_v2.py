"""Download the Ultralytics NYU Depth V2 archive for calibration."""

import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from ultralytics.utils.downloads import safe_download

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Download and extract NYU Depth V2 for calibration")
    parser.add_argument(
        "--output-dir",
        default="./nyu-depth-selected",
        help="Directory for the selected calibration images",
    )
    parser.add_argument("--num-images", type=int, default=100, help="Number of validation images to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to select images")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing calibration image directory",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be greater than zero")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; use --overwrite to replace it")

    with TemporaryDirectory(prefix="mblt-nyu-depth-") as temp_dir:
        temp_path = Path(temp_dir)
        safe_download(DATASET_URL, dir=temp_path, unzip=True, delete=True)
        validation_images = temp_path / "nyu-depth" / "images" / "val"
        if not validation_images.is_dir():
            raise FileNotFoundError(f"Expected validation images at {validation_images}")

        image_paths = sorted(
            path for path in validation_images.iterdir() if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png"}
        )
        if args.num_images > len(image_paths):
            raise ValueError(f"--num-images cannot exceed the {len(image_paths)} validation images")

        selected_images = random.Random(args.seed).sample(image_paths, args.num_images)
        selected_dir = temp_path / "nyu-depth-selected"
        selected_dir.mkdir()
        for image_path in selected_images:
            shutil.copy2(image_path, selected_dir / image_path.name)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(selected_dir, output_dir)

    print(f"Selected {len(selected_images)} calibration images in {output_dir}")


if __name__ == "__main__":
    main()
