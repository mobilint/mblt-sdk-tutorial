"""Input and output contracts for the compiled SAM2 encoder and decoder.

The decoder has six inputs and three of them share the shape `(1, 256, 64, 64)`.
Feeding those in the wrong order produces plausible but wrong masks rather than
an error, so every feed is built by semantic role and then shape-checked against
the runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# Positional input order reported by the compiled decoder MXQ. It matches the
# MBLT input-name order used during calibration and compilation, which the
# earlier wrapper-traced decoder did not. Confirm it against your own artifact
# with `qbruntime.get_model_summary(<path>.mxq)`, which prints:
#
#   Input - Shapes: [(256, 64, 64), (256, 64, 64), (256, 64, 64), (1, -1, 256),
#                    (256, 256, 32), (128, 128, 64)]
#
# The `-1` is the prompt axis, so one decoder serves any point count.
DEFAULT_DECODER_RUNTIME_ORDER = (
    "image_embeddings",
    "dense_prompt_embeddings",
    "image_pe",
    "sparse_prompt_embeddings",
    "hrf0_nhwc",
    "hrf1_nhwc",
)

DECODER_ROLES = frozenset(DEFAULT_DECODER_RUNTIME_ORDER)

# The decoder always emits three mask candidates at 256x256.
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
    """Remove the outer model batch that qbruntime omits from buffer shapes.

    This is the opposite of the convention used by the single-input vision
    tutorials, which add a batch dimension before calling `infer`.
    """
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
    """Name the decoder outputs by their unambiguous element counts.

    qbruntime does not guarantee that the runtime output order matches the
    compiled graph's declared order, so each output is identified by its size
    rather than its position: `masks` is the only output whose size is a
    multiple of `256*256`, and `iou` has one entry per mask.

    The decoder is parsed with `output_meta=lambda x: x[0][:2]`, so it exposes
    exactly these two. Older wrapper-traced decoders also emitted SAM tokens and
    an object score; those are accepted when present and omitted when not.
    """
    arrays = [np.ascontiguousarray(np.asarray(value), dtype=np.float32) for value in outputs]
    # A NaN would otherwise reach argmax over the IoU scores, silently selecting the
    # wrong candidate, and the `> 0` mask threshold, turning non-finite logits into
    # plausible booleans. Fail instead of corrupting the prediction.
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
