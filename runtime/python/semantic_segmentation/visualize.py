"""Visualize Cityscapes semantic-segmentation class maps."""

from pathlib import Path

import cv2
import numpy as np

# Official Cityscapes train-ID colors converted from RGB to OpenCV BGR order.
CITYSCAPES_PALETTE_BGR = np.array(
    [
        (128, 64, 128),  # road
        (232, 35, 244),  # sidewalk
        (70, 70, 70),  # building
        (156, 102, 102),  # wall
        (153, 153, 190),  # fence
        (153, 153, 153),  # pole
        (30, 170, 250),  # traffic light
        (0, 220, 220),  # traffic sign
        (35, 142, 107),  # vegetation
        (152, 251, 152),  # terrain
        (180, 130, 70),  # sky
        (60, 20, 220),  # person
        (0, 0, 255),  # rider
        (142, 0, 0),  # car
        (70, 0, 0),  # truck
        (100, 60, 0),  # bus
        (100, 80, 0),  # train
        (230, 0, 0),  # motorcycle
        (32, 11, 119),  # bicycle
    ],
    dtype=np.uint8,
)


def save_semantic_overlay(
    image_bgr: np.ndarray,
    class_map: np.ndarray,
    output_path: str,
    alpha: float = 0.7,
) -> np.ndarray:
    """Colorize a Cityscapes class map, blend it over an image, and save it."""
    if class_map.ndim != 2:
        raise ValueError(f"Expected a two-dimensional class map, got {class_map.shape}")
    if class_map.shape != image_bgr.shape[:2]:
        raise ValueError(f"Image and class-map shapes must match, got {image_bgr.shape[:2]} and {class_map.shape}")
    if class_map.size and int(class_map.max()) >= len(CITYSCAPES_PALETTE_BGR):
        raise ValueError(f"Cityscapes class IDs must be between 0 and {len(CITYSCAPES_PALETTE_BGR) - 1}")

    overlay = CITYSCAPES_PALETTE_BGR[class_map]
    result = cv2.addWeighted(image_bgr, 1.0 - alpha, overlay, alpha, 0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), result):
        raise OSError(f"Failed to write output image: {path}")
    return result
