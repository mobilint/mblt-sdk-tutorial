"""Convert NYU Depth V2 RGB images to 768x768 letterboxed calibration tensors."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from qbcompiler.calibration import make_calib_man

IMAGE_SIZE = 768
PAD_VALUE = 114


def letterbox_rgb(image: np.ndarray) -> np.ndarray:
    """Return an RGB HWC float32 image letterboxed to the model input size."""
    height, width = image.shape[:2]
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    pad_height, pad_width = IMAGE_SIZE - resized_height, IMAGE_SIZE - resized_width

    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_height / 2 - 0.1)), int(round(pad_height / 2 + 0.1))
    left, right = int(round(pad_width / 2 - 0.1)), int(round(pad_width / 2 + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    return (image / 255).astype(np.float32)


def preprocess(image_path: str) -> np.ndarray:
    """Load an image and apply the calibration preprocessing in HWC format."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return letterbox_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Convert depth-estimation images to calibration tensors")
    parser.add_argument("--source-path", default="./nyu-depth-selected", help="Path to RGB images")
    parser.add_argument("--npy-path", default="./calib_data_tensor", help="Output tensor directory")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    npy_path = Path(args.npy_path)
    make_calib_man(
        pre_ftn=preprocess,
        data_dir=args.source_path,
        save_dir=str(npy_path.parent),
        save_name=npy_path.name,
        remove_npy=True,
    )


if __name__ == "__main__":
    main()
