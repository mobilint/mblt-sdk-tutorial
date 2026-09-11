from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEFAULT_MODEL_ID = "facebook/sam2-hiera-large"
DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "runtime" / "python" / "rc" / "bus.jpg"


def resolve_device(name: str = "cuda") -> str:
    """Fall back to the CPU when CUDA was requested but is not available."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name


def load_image_np(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), copy=True)


def prompt_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Three-point prompt used to trace the decoder.

    Two positive points on the bus body and one negative point on the building
    behind it. Only the shapes matter for tracing, but a prompt that actually
    selects an object keeps the captured tensors representative.
    """
    points = np.asarray([[500, 580], [620, 560], [400, 120]], dtype=np.float32)
    labels = np.asarray([1, 1, 0], dtype=np.int64)
    return points, labels


def build_predictor(model_id: str, device: str):
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    predictor.model.to(torch.device(device)).eval()
    return predictor


def preprocess_encoder_input(predictor, image: np.ndarray) -> np.ndarray:
    """Apply the official SAM2 transform and return NHWC float32 [1, 1024, 1024, 3]."""
    tensor = predictor._transforms(np.ascontiguousarray(image))[None, ...]
    return np.ascontiguousarray(tensor.permute(0, 2, 3, 1).float().cpu().numpy(), dtype=np.float32)


def prepare_decoder_tensors(predictor, points: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Run the prompt encoder and build the six decoder MBLT inputs."""
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
    decoder = predictor.model.sam_mask_decoder

    if decoder.pred_obj_scores:
        output_tokens = torch.cat(
            [decoder.obj_score_token.weight, decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0
        )
    else:
        output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse), dim=1)
    src = image_embeddings + dense
    pos_src = prompt_encoder.get_dense_pe()

    def sequence(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(2).transpose(1, 2).reshape(1, 1, -1, value.shape[1]).contiguous()

    tensors = {
        "tokens": tokens.float().unsqueeze(1).contiguous(),
        "src_plus_pos": sequence(src + pos_src).float(),
        "src": sequence(src).float(),
        "pos_src": sequence(pos_src).float(),
        "hrf0_nhwc": high_res[0].permute(0, 2, 3, 1).contiguous(),
        "hrf1_nhwc": high_res[1].permute(0, 2, 3, 1).contiguous(),
    }
    return {
        name: np.ascontiguousarray(value.detach().float().cpu().numpy(), dtype=np.float32)
        for name, value in tensors.items()
    }
