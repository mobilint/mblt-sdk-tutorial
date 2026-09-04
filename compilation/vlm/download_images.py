from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_ID = "detection-datasets/coco"
DATASET_REVISION = "cf0b22332314a937e9dc8a1957b21725430bb41d"


if __name__ == "__main__":
    parser = ArgumentParser(description="Download COCO validation images for VLM calibration")
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("images"))
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.size <= 0:
        raise ValueError("--size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="val", streaming=True)

    saved = 0
    for example in tqdm(dataset, desc="Downloading images", total=args.num_images):
        if saved == args.num_images:
            break
        image = example["image"].convert("RGB")
        image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
        image.save(args.output_dir / f"image_{saved:04d}.jpg", "JPEG", quality=95)
        saved += 1

    if saved != args.num_images:
        raise RuntimeError(f"Downloaded {saved}/{args.num_images} COCO images")
    print(f"Saved {saved} images to {args.output_dir.resolve()}")
