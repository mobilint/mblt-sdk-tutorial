from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# The three sequence inputs share one shape, so preserve the semantic order from
# the decoder calibration manifest.
DEFAULT_DECODER_RUNTIME_ORDER = (
    "tokens",
    "src_plus_pos",
    "src",
    "pos_src",
    "hrf1_nhwc",
    "hrf0_nhwc",
)

DECODER_ROLES = frozenset(DEFAULT_DECODER_RUNTIME_ORDER)

MASK_SIZE = 256


def parse_runtime_order(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_DECODER_RUNTIME_ORDER
    roles = tuple(value.split(",")) if isinstance(value, str) else tuple(value)
    if len(roles) != len(set(roles)) or set(roles) != DECODER_ROLES:
        raise ValueError(
            f"decoder runtime order must contain each role exactly once: {sorted(DECODER_ROLES)}; got {roles}"
        )
    return roles


def strip_runtime_batch(value: np.ndarray) -> np.ndarray:
    """Remove the outer model batch omitted by qbruntime buffer shapes."""
    value = np.asarray(value)
    if value.ndim >= 4 and value.shape[0] == 1:
        value = value[0]
    return np.ascontiguousarray(value, dtype=np.float32)


def build_decoder_runtime_feed(
    tensors: Mapping[str, np.ndarray], order: Sequence[str] | None = None
) -> list[np.ndarray]:
    """Order the six decoder tensors by semantic role for the runtime call."""
    order = parse_runtime_order(order)
    missing = [role for role in order if role not in tensors]
    if missing:
        raise ValueError(f"decoder tensors are missing roles: {missing}")
    return [strip_runtime_batch(tensors[role]) for role in order]


def validate_runtime_shapes(actual: Sequence[np.ndarray], expected: Sequence[Sequence[int]], label: str) -> None:
    """Compare each feed against the runtime shape, allowing dynamic `-1` axes."""
    if len(actual) != len(expected):
        raise ValueError(f"{label} input count mismatch: feeds={len(actual)}, runtime={len(expected)}")
    for index, (array, shape) in enumerate(zip(actual, expected)):
        shape = tuple(int(x) for x in shape)
        got = tuple(int(x) for x in array.shape)
        if len(got) != len(shape) or any(want != -1 and have != want for have, want in zip(got, shape)):
            raise ValueError(f"{label} input {index} shape mismatch: feed={got}, runtime={shape}")


def classify_decoder_outputs(outputs: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Identify decoder outputs by element count instead of runtime order."""
    arrays = [np.ascontiguousarray(np.asarray(value), dtype=np.float32) for value in outputs]
    for index, array in enumerate(arrays):
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"decoder output {index} with shape {array.shape} contains NaN or infinity")
    mask_area = MASK_SIZE * MASK_SIZE
    mask_matches = [a for a in arrays if a.size >= mask_area and a.size % mask_area == 0]
    if len(mask_matches) != 1:
        raise ValueError(f"expected one mask output, found {len(mask_matches)} in {[a.shape for a in arrays]}")
    masks = mask_matches[0].reshape(-1, MASK_SIZE, MASK_SIZE)
    num_masks = masks.shape[0]

    def unique(label: str, size: int, required: bool) -> np.ndarray | None:
        matches = [a for a in arrays if a is not mask_matches[0] and a.size == size]
        if len(matches) > 1 or (required and not matches):
            raise ValueError(f"expected one {label} output of size {size}, found {len(matches)}")
        return matches[0] if matches else None

    iou = unique("iou", num_masks, required=True)
    sam_tokens = unique("sam_tokens", num_masks * MASK_SIZE, required=False)
    object_score = unique("object_score", 1, required=False)
    result: dict[str, np.ndarray] = {"masks": masks, "iou": iou.reshape(num_masks)}
    if sam_tokens is not None:
        result["sam_tokens"] = sam_tokens.reshape(num_masks, MASK_SIZE)
    if object_score is not None:
        result["object_score"] = object_score.reshape(1)
    return result
