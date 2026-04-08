from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_REPO_ID = "CUHK-CSE/wider_face"
DATASET_FILENAME = "data/WIDER_train.zip"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "widerface-selected"
IMAGE_ROOT_IN_ZIP = "WIDER_train/images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download WIDER FACE train data and select one random image from each sub-category."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where the selected images will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible image selection. Default: 42",
    )
    return parser.parse_args()


def download_widerface_zip() -> Path:
    print("Downloading WIDER FACE training archive from Hugging Face...")
    zip_path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename=DATASET_FILENAME,
        )
    )
    print(f"Archive is ready at: {zip_path}")
    return zip_path


def collect_images_by_category(zip_path: Path) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}

    with zipfile.ZipFile(zip_path) as zip_file:
        for member in zip_file.namelist():
            if not member.startswith(f"{IMAGE_ROOT_IN_ZIP}/") or member.endswith("/"):
                continue

            relative_path = member[len(f"{IMAGE_ROOT_IN_ZIP}/") :]
            parts = relative_path.split("/")
            if len(parts) != 2:
                continue

            category_name = parts[0]
            categories.setdefault(category_name, []).append(member)

    if not categories:
        raise RuntimeError("No WIDER FACE training images were found in the downloaded archive.")

    return categories


def select_and_save_images(zip_path: Path, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    categories = collect_images_by_category(zip_path)

    with zipfile.ZipFile(zip_path) as zip_file:
        for category_name in sorted(categories):
            selected_member = rng.choice(categories[category_name])
            destination_path = output_dir / Path(selected_member).name

            with zip_file.open(selected_member) as source, destination_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

            print(f"Selected {selected_member} -> {destination_path}")

    print(f"Saved {len(categories)} images to: {output_dir}")


def main() -> None:
    args = parse_args()
    zip_path = download_widerface_zip()
    select_and_save_images(zip_path, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
