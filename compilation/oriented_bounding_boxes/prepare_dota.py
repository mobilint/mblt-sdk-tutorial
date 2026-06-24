from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP_PATH = SCRIPT_DIR / "DOTAv1.zip"
DEFAULT_EXTRACT_DIR = SCRIPT_DIR / "DOTAv1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dota-selected"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OUTPUT_MARKER = ".prepare_dota_output"
OUTPUT_MANIFEST = ".prepare_dota_manifest.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DOTAv1, extract it locally, and select a small image subset for calibration."
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=DEFAULT_DOWNLOAD_URL,
        help=f"Dataset archive URL. Default: {DEFAULT_DOWNLOAD_URL}",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help=f"Local path for the dataset archive. Default: {DEFAULT_ZIP_PATH}",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=DEFAULT_EXTRACT_DIR,
        help=f"Directory where the archive will be extracted. Default: {DEFAULT_EXTRACT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where selected calibration images will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of calibration images to select. Default: 100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible image selection. Default: 42",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and reuse an existing archive or extracted dataset.",
    )
    return parser.parse_args()


def download_archive(download_url: str, zip_path: Path) -> Path:
    if zip_path.exists():
        print(f"Using existing archive: {zip_path}")
        return zip_path

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DOTAv1 archive from {download_url}...")
    urlretrieve(download_url, zip_path)
    print(f"Saved archive to: {zip_path}")
    return zip_path


def extract_archive(zip_path: Path, extract_dir: Path) -> Path:
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"Using existing extracted dataset: {extract_dir}")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_dir)
    marker.write_text("ok\n", encoding="ascii")
    print(f"Extraction complete: {extract_dir}")
    return extract_dir


def collect_candidate_images(extract_dir: Path) -> list[Path]:
    images = [
        path
        for path in extract_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and "__MACOSX" not in path.parts
    ]

    if not images:
        raise RuntimeError(f"No images found under {extract_dir}. Check the archive contents or --extract-dir path.")

    preferred = [
        path
        for path in images
        if "train" in {part.lower() for part in path.parts} or "images" in {part.lower() for part in path.parts}
    ]
    return preferred or images


def select_images(images: list[Path], num_images: int, seed: int) -> list[Path]:
    if num_images <= 0:
        raise ValueError("--num-images must be greater than 0.")
    if len(images) < num_images:
        raise ValueError(f"Requested {num_images} images, but only found {len(images)} candidate images.")

    rng = random.Random(seed)
    selected = rng.sample(images, num_images)
    selected.sort()
    return selected


def prepare_output_dir(output_dir: Path) -> None:
    marker_path = output_dir / OUTPUT_MARKER
    manifest_path = output_dir / OUTPUT_MANIFEST

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir must be a directory path, got existing file: {output_dir}")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("generated\n", encoding="ascii")
        return

    if marker_path.exists():
        if manifest_path.exists():
            for relative_path in manifest_path.read_text(encoding="utf-8").splitlines():
                owned_path = output_dir / relative_path
                if owned_path.exists() and owned_path.is_file():
                    owned_path.unlink()
        return

    if any(output_dir.iterdir()):
        raise ValueError(
            f"Refusing to clear non-generated existing directory: {output_dir}. "
            "Use a new directory or one previously created by this script."
        )

    marker_path.write_text("generated\n", encoding="ascii")


def copy_selected_images(selected_images: list[Path], output_dir: Path) -> None:
    prepare_output_dir(output_dir)
    written_files: list[str] = []

    for image_path in selected_images:
        destination = output_dir / image_path.name
        if destination.exists():
            destination = output_dir / f"{image_path.parent.name}_{image_path.name}"
        shutil.copy2(image_path, destination)
        written_files.append(destination.name)
        print(f"Copied {image_path} -> {destination}")

    (output_dir / OUTPUT_MANIFEST).write_text("\n".join(written_files) + "\n", encoding="utf-8")
    print(f"Saved {len(selected_images)} calibration images to: {output_dir}")


def main() -> None:
    args = parse_args()
    extract_marker = args.extract_dir / ".extracted"

    if args.skip_download and not args.zip_path.exists() and not args.extract_dir.exists():
        raise FileNotFoundError(
            "--skip-download was set, but neither the archive nor the extracted dataset exists locally."
        )

    if extract_marker.exists():
        print(f"Using existing extracted dataset: {args.extract_dir}")
    else:
        zip_path = args.zip_path if args.skip_download else download_archive(args.download_url, args.zip_path)
        extract_archive(zip_path, args.extract_dir)

    candidate_images = collect_candidate_images(args.extract_dir)
    selected_images = select_images(candidate_images, args.num_images, args.seed)
    copy_selected_images(selected_images, args.output_dir)


if __name__ == "__main__":
    main()
