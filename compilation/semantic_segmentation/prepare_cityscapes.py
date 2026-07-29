"""Select calibration images from the Cityscapes validation split."""

import json
import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

from datasets import Dataset, load_dataset

DATASET_NAME = "Chris1/cityscapes_segmentation"
DATASET_CONFIG = "default"
DATASET_SPLIT = "validation"
PARQUET_API_URL = "https://datasets-server.huggingface.co/parquet"


def get_validation_parquet_urls() -> list[str]:
    """Return only the validation Parquet URLs reported by the Dataset Viewer."""
    request_url = f"{PARQUET_API_URL}?{urlencode({'dataset': DATASET_NAME})}"
    with urlopen(request_url, timeout=30) as response:  # noqa: S310
        payload = json.load(response)

    parquet_urls = [
        parquet_file["url"]
        for parquet_file in payload.get("parquet_files", [])
        if parquet_file.get("config") == DATASET_CONFIG and parquet_file.get("split") == DATASET_SPLIT
    ]
    if not parquet_urls:
        raise RuntimeError(f"No {DATASET_SPLIT} Parquet files found for {DATASET_NAME}/{DATASET_CONFIG}")
    return parquet_urls


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Cityscapes validation images for calibration")
    parser.add_argument(
        "--output-dir",
        default="./cityscapes-selected",
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
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    validation_parquet_urls = get_validation_parquet_urls()
    print(f"Downloading {len(validation_parquet_urls)} validation Parquet file(s)")
    dataset = load_dataset(
        "parquet",
        data_files={DATASET_SPLIT: validation_parquet_urls},
        split=DATASET_SPLIT,
        columns=["image"],
    )
    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a map-style Dataset when loading the validation split")
    if args.num_images > len(dataset):
        raise ValueError(f"--num-images cannot exceed the {len(dataset)} validation images")
    selected_indices = sorted(random.Random(args.seed).sample(range(len(dataset)), args.num_images))

    with TemporaryDirectory(prefix=".cityscapes-selected-", dir=output_dir.parent) as temp_dir:
        selected_dir = Path(temp_dir) / "images"
        selected_dir.mkdir()
        for index in selected_indices:
            image = dataset[index]["image"].convert("RGB")
            image.save(selected_dir / f"cityscapes_validation_{index:04d}.png")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(selected_dir, output_dir)

    print(f"Selected {len(selected_indices)} validation images in {output_dir}")


if __name__ == "__main__":
    main()
