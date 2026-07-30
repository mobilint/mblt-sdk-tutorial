"""Postprocess YOLO26 Cityscapes semantic-segmentation logits."""

from collections.abc import Sequence

import cv2
import numpy as np

NUM_CITYSCAPES_CLASSES = 19


def _logits_to_class_map(output: np.ndarray) -> np.ndarray:
    """Apply argmax to channel-last or channel-first semantic logits."""
    logits = np.asarray(output)
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError(f"Expected a batch size of one, got output shape {logits.shape}")
        logits = logits[0]
    if logits.ndim != 3:
        raise ValueError(f"Expected three-dimensional semantic logits, got shape {logits.shape}")

    if logits.shape[-1] == NUM_CITYSCAPES_CLASSES:
        class_map = np.argmax(logits, axis=-1)
    elif logits.shape[0] == NUM_CITYSCAPES_CLASSES:
        class_map = np.argmax(logits, axis=0)
    else:
        raise ValueError(
            f"Could not find the 19-class channel in semantic output shape {logits.shape}; expected HWC or CHW logits"
        )
    return class_map.astype(np.uint8)


def postprocess_semantic(
    outputs: Sequence[np.ndarray],
    original_shape: tuple[int, int],
    letterbox_borders: tuple[int, int, int, int],
) -> np.ndarray:
    """Convert MXQ logits to a class map and restore the source-image shape."""
    if len(outputs) != 1:
        raise ValueError(f"Semantic segmentation expects one output tensor, received {len(outputs)}")

    class_map = _logits_to_class_map(outputs[0])
    top, bottom, left, right = letterbox_borders
    height, width = class_map.shape
    height_end = height - bottom if bottom else height
    width_end = width - right if right else width
    class_map = class_map[top:height_end, left:width_end]
    if class_map.size == 0:
        raise ValueError("Removing letterbox padding produced an empty class map")

    if class_map.shape != original_shape:
        class_map = cv2.resize(
            class_map,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return class_map
