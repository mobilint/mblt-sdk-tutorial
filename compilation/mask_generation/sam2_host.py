"""Host-side SAM2 helpers shared by calibration generation and compilation.

The compiled encoder and decoder cover only part of SAM2. The image transform,
the prompt encoder, and the token/embedding assembly in front of the mask
decoder still run on the host with official `facebookresearch/sam2` code, so
calibration tensors must be produced by exactly the same host path that the
runtime tutorial uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_ID = "facebook/sam2-hiera-large"
DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "runtime" / "python" / "rc" / "bus.jpg"


def resolve_device(name: str = "cuda") -> str:
    """Fall back to the CPU when CUDA was requested but is not available."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name


def load_image_np(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def prompt_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Three-point prompt used to trace the decoder.

    Two positive points on the bus body and one negative point on the building
    behind it. Only the shapes matter for tracing, but a prompt that actually
    selects an object keeps the captured tensors representative.
    """
    points = np.asarray([[500, 580], [620, 560], [400, 120]], dtype=np.float32)
    labels = np.asarray([1, 1, 0], dtype=np.int64)
    return points, labels


def build_predictor(model_id: str, sam2_root: str | Path | None, device: str):
    """Load the official SAM2 image predictor from a local `sam2` checkout."""
    if sam2_root:
        root = str(Path(sam2_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    # Imported lazily so `--help` works without the sam2 package installed.
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    predictor.model.to(torch.device(device)).eval()
    return predictor


def preprocess_encoder_input(predictor, image: np.ndarray) -> np.ndarray:
    """Apply the official SAM2 transform and return NHWC float32 [1, 1024, 1024, 3]."""
    tensor = predictor._transforms(np.ascontiguousarray(image))[None, ...]
    return np.ascontiguousarray(tensor.permute(0, 2, 3, 1).float().cpu().numpy(), dtype=np.float32)



def prepare_decoder_tensors(predictor, points: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Run the host prompt encoder and build the six compiled decoder inputs.

    The compiled decoder is parsed from `sam_mask_decoder` itself, so the output-token
    concat and the `image_embeddings + dense_prompt_embeddings` sum live inside the
    graph (its host-bridge subgraph) rather than being assembled here. What the host
    still owns is the prompt encoder, so these are its raw outputs plus the image
    features, laid out in the shapes the MBLT reports.

    The dictionary is keyed by semantic role, not by position: `image_embeddings`,
    `dense_prompt_embeddings`, and `image_pe` all have shape `(1, 256, 64, 64)`, so a
    positional guess would silently swap them.
    """
    mask_input, coords, point_labels, boxes = predictor._prep_prompts(
        np.asarray(points, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        None,
        None,
        True,
    )
    if boxes is not None:
        raise ValueError("box prompts are not supported by this tutorial")
    prompt_encoder = predictor.model.sam_prompt_encoder
    sparse, dense = prompt_encoder(points=(coords, point_labels), boxes=None, masks=mask_input)
    features = predictor._features
    image_embeddings = features["image_embed"][-1].unsqueeze(0)
    high_res = [value[-1].unsqueeze(0) for value in features["high_res_feats"]]

    tensors = {
        "image_embeddings": image_embeddings.float(),
        "dense_prompt_embeddings": dense.float(),
        "image_pe": prompt_encoder.get_dense_pe().float(),
        # (1, N, 256) -> (1, 1, N, 256); axis 2 is the prompt axis the graph keeps dynamic.
        "sparse_prompt_embeddings": sparse.float().unsqueeze(1).contiguous(),
        "hrf0_nhwc": high_res[0].permute(0, 2, 3, 1).contiguous(),
        "hrf1_nhwc": high_res[1].permute(0, 2, 3, 1).contiguous(),
    }
    return {
        name: np.ascontiguousarray(value.detach().float().cpu().numpy(), dtype=np.float32)
        for name, value in tensors.items()
    }
