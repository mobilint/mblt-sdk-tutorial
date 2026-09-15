from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_predictor(model_id: str, device: str):
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    predictor.model.to(torch.device(device)).eval()
    return predictor


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), copy=True)


def preprocess_encoder_input(predictor, image: np.ndarray) -> np.ndarray:
    """Apply the official SAM2 transform and return NHWC float32 [1, 1024, 1024, 3]."""
    tensor = predictor._transforms(np.ascontiguousarray(image))[None, ...]
    return np.ascontiguousarray(tensor.permute(0, 2, 3, 1).float().cpu().numpy(), dtype=np.float32)


FPN_LEVELS_CHW: tuple[tuple[int, int, int], ...] = ((32, 256, 256), (64, 128, 128), (256, 64, 64))


def fpn_from_runtime(outputs: Sequence[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    """Convert the three encoder FPN outputs to NCHW tensors ordered 32/64/256.

    The runtime may report either NHWC or NCHW, so each level is identified by
    its complete shape rather than by one axis.
    """
    nhwc = {(h, w, c): c for c, h, w in FPN_LEVELS_CHW}
    nchw = {(c, h, w): c for c, h, w in FPN_LEVELS_CHW}
    features: dict[int, torch.Tensor] = {}
    for output in outputs:
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            continue
        shape = tuple(int(dim) for dim in array.shape)
        if shape in nchw:
            channel = nchw[shape]
            tensor = torch.from_numpy(np.ascontiguousarray(array))[None]
        elif shape in nhwc:
            channel = nhwc[shape]
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]
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
    """Run the prompt encoder and build the six decoder MXQ inputs."""
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


def postprocess_masks(predictor, low_resolution_masks: np.ndarray, original_hw: Sequence[int]) -> np.ndarray:
    """Upscale the 256x256 decoder logits back to the original image size."""
    tensor = torch.from_numpy(np.ascontiguousarray(low_resolution_masks, dtype=np.float32)).to(predictor.model.device)[
        None
    ]
    masks = predictor._transforms.postprocess_masks(tensor, tuple(original_hw))[0]
    return masks.detach().float().cpu().numpy()
