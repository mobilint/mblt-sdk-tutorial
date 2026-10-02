"""Postprocess the MXQ depth output."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _to_bchw(output: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Normalize one depth output to a float32 BCHW tensor."""
    depth = torch.as_tensor(output, dtype=torch.float32)
    if depth.ndim == 2:
        return depth[None, None]
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            return depth[None]
        if depth.shape[-1] == 1:
            return depth.permute(2, 0, 1)[None]
    if depth.ndim == 4:
        if depth.shape[1] == 1:
            return depth
        if depth.shape[-1] == 1:
            return depth.permute(0, 3, 1, 2)
    raise ValueError(f"Expected a single-channel depth output, got shape {tuple(depth.shape)}")


def postprocess_depth(
    outputs: Sequence[np.ndarray | torch.Tensor],
    input_shape: tuple[int, ...],
    original_shape: tuple[int, int],
    letterbox_borders: tuple[int, int, int, int],
) -> torch.Tensor:
    """Upsample MXQ depth, remove letterbox padding, and restore source size."""
    if len(outputs) != 1:
        raise ValueError(f"Depth estimation expects one output tensor, received {len(outputs)}")

    if input_shape[-1] == 3:
        input_height, input_width = input_shape[:2]
    elif input_shape[0] == 3:
        input_height, input_width = input_shape[1:]
    else:
        raise ValueError(f"Could not determine the channel axis from input shape {input_shape}")

    depth = _to_bchw(outputs[0])

    # Depending on the qbcompiler version, the MXQ graph returns either the
    # full-resolution 768x768 depth map (the final Resize is compiled into the
    # MXQ) or a quarter-resolution 192x192 map. Restore the 768x768 ONNX output
    # shape before undoing letterbox preprocessing.
    output_shape = tuple(depth.shape[-2:])
    if output_shape == (input_height // 4, input_width // 4):
        depth = F.interpolate(depth, scale_factor=4.0, mode="bilinear", align_corners=False)
    elif output_shape != (input_height, input_width):
        raise ValueError(
            "The MXQ depth output must match the model input shape or a quarter of it: "
            f"got {output_shape}, expected {(input_height, input_width)} "
            f"or {(input_height // 4, input_width // 4)}"
        )

    top, bottom, left, right = letterbox_borders
    height_end = input_height - bottom if bottom else input_height
    width_end = input_width - right if right else input_width
    depth = depth[:, :, top:height_end, left:width_end]
    if depth.numel() == 0:
        raise ValueError("Removing letterbox padding produced an empty depth map")

    depth = F.interpolate(depth, size=original_shape, mode="bilinear", align_corners=False)
    return depth[0, 0]
