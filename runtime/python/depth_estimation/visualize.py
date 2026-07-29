"""Visualize a depth map as a color overlay."""

from pathlib import Path

import cv2
import numpy as np
import torch


def colorize_depth(depth: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Map inverse depth to JET colors, with nearer pixels rendered red."""
    depth_array = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional depth map, got {depth_array.shape}")

    valid = np.isfinite(depth_array) & (depth_array > 0)
    if not valid.any():
        raise ValueError("Depth output contains no positive finite values")

    disparity = np.zeros(depth_array.shape, dtype=np.float32)
    disparity[valid] = 1.0 / depth_array[valid]
    lower, upper = np.percentile(disparity[valid], (2, 98))
    if upper <= lower:
        upper = lower + 1e-6

    normalized = np.zeros(depth_array.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (disparity[valid] - lower) * 255.0 / (upper - lower),
        0,
        255,
    ).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colorized[~valid] = 0
    return colorized, valid


def save_depth_overlay(
    image_bgr: np.ndarray,
    depth: np.ndarray | torch.Tensor,
    output_path: str,
    alpha: float = 0.7,
) -> np.ndarray:
    """Blend a colorized depth map over an image and save it."""
    colorized, valid = colorize_depth(depth)
    if colorized.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(f"Image and depth shapes must match, got {image_bgr.shape[:2]} and {colorized.shape[:2]}")

    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, colorized, alpha, 0)
    result = image_bgr.copy()
    result[valid] = blended[valid]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), result):
        raise OSError(f"Failed to write output image: {path}")
    return result
