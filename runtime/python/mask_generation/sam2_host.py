"""Host-side SAM2 helpers used around the compiled encoder and decoder MXQ models.

The compiled models cover the image encoder and the mask decoder body. The
image transform, the prompt encoder, and the final mask upscaling still run on
the host with official `facebookresearch/sam2` code.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def build_predictor(model_id: str, sam2_root: str | Path | None, device: str):
    """Load the official SAM2 image predictor from a local `sam2` checkout."""
    if sam2_root:
        root = str(Path(sam2_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    # Imported lazily so `--help` works without the sam2 package installed.
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    predictor.model.to(torch.device(device)).eval()
    return predictor


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def preprocess_encoder_input(predictor, image: np.ndarray) -> np.ndarray:
    """Apply the official SAM2 transform and return NHWC float32 [1, 1024, 1024, 3]."""
    tensor = predictor._transforms(np.ascontiguousarray(image))[None, ...]
    return np.ascontiguousarray(tensor.permute(0, 2, 3, 1).float().cpu().numpy(), dtype=np.float32)


def fpn_from_runtime(outputs: Sequence[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    """Convert the three encoder FPN outputs to NCHW tensors ordered 32/64/256.

    The runtime may report either NHWC or NCHW, so the channel count is used to
    identify each level rather than the axis position.
    """
    features: dict[int, torch.Tensor] = {}
    for output in outputs:
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            continue
        if array.shape[-1] in (32, 64, 256):
            channel = int(array.shape[-1])
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]
        elif array.shape[0] in (32, 64, 256):
            channel = int(array.shape[0])
            tensor = torch.from_numpy(np.ascontiguousarray(array))[None]
        else:
            continue
        if channel in features:
            raise ValueError(f"duplicate encoder output with {channel} channels")
        features[channel] = tensor.to(device)
    missing = [channel for channel in (32, 64, 256) if channel not in features]
    if missing:
        raise ValueError(
            f"encoder outputs are missing FPN channel counts {missing}; got {[np.asarray(o).shape for o in outputs]}"
        )
    return [features[32], features[64], features[256]]


def build_predictor_features(predictor, feature_maps: Sequence[torch.Tensor]) -> dict:
    """Reshape the FPN levels into the feature dictionary the predictor expects."""
    model = predictor.model
    vision_features = [value.flatten(2).permute(2, 0, 1) for value in feature_maps]
    if model.directly_add_no_mem_embed:
        vision_features[-1] = vision_features[-1] + model.no_mem_embed
    features = [
        feature.permute(1, 2, 0).view(1, -1, *feature_size)
        for feature, feature_size in zip(vision_features[::-1], predictor._bb_feat_sizes[::-1])
    ][::-1]
    return {"image_embed": features[-1], "high_res_feats": features[:-1]}


def install_runtime_features(predictor, feature_maps: Sequence[torch.Tensor], original_hw: Sequence[int]) -> None:
    """Install NPU-computed features so the host predictor skips its own encoder."""
    predictor.reset_predictor()
    predictor._orig_hw = [tuple(int(x) for x in original_hw)]
    predictor._features = build_predictor_features(predictor, feature_maps)
    predictor._is_image_set = True



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


def postprocess_masks(predictor, low_resolution_masks: np.ndarray, original_hw: Sequence[int]) -> np.ndarray:
    """Upscale the 256x256 decoder logits back to the original image size."""
    tensor = torch.from_numpy(np.ascontiguousarray(low_resolution_masks, dtype=np.float32)).to(predictor.model.device)[
        None
    ]
    masks = predictor._transforms.postprocess_masks(tensor, tuple(original_hw))[0]
    return masks.detach().float().cpu().numpy()
