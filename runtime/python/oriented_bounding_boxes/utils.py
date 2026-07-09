import math
import os
from typing import overload

import numpy as np
import torch
import torch.nn as nn

NUM_THREADS = min(16, max(1, (os.cpu_count() or 1) - 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def invsigmoid(x: float) -> float:
    return -math.log(1.0 / x - 1.0)


def make_anchors(
    image_height: int,
    image_width: int,
    strides: list[int],
    grid_cell_offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_points, stride_tensor = [], []
    for stride in strides:
        height, width = image_height // stride, image_width // stride
        sx = torch.arange(end=width, dtype=torch.float32, device=DEVICE) + grid_cell_offset
        sy = torch.arange(end=height, dtype=torch.float32, device=DEVICE) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((height * width, 1), stride, dtype=torch.float32, device=DEVICE))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class DFL(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1, bias=False).requires_grad_(False)
        weights = torch.arange(channels, dtype=torch.float32, device=DEVICE)
        self.conv.weight.data = nn.Parameter(weights.view(1, channels, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, anchors = x.shape
        x = x.to(DEVICE)
        x = x.view(batch, 4, self.channels, anchors).transpose(2, 1).softmax(1)
        return self.conv(x).view(batch, 4, anchors)


def dist2rbox(distance: torch.Tensor, angle: torch.Tensor, anchor_points: torch.Tensor, dim: int = -1) -> torch.Tensor:
    lt, rb = distance.split(2, dim=dim)
    cos_value = torch.cos(angle)
    sin_value = torch.sin(angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x = xf * cos_value - yf * sin_value
    y = xf * sin_value + yf * cos_value
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def compute_ratio_pad(
    input_shape: tuple[int, int],
    image_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[float, tuple[float, float]]:
    if ratio_pad is None:
        gain = min(input_shape[0] / image_shape[0], input_shape[1] / image_shape[1])
        pad = (
            round((input_shape[1] - round(image_shape[1] * gain)) / 2 - 0.1),
            round((input_shape[0] - round(image_shape[0] * gain)) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    return gain, pad


def scale_rboxes(
    input_shape: tuple[int, int],
    rboxes: torch.Tensor,
    image_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> torch.Tensor:
    gain, pad = compute_ratio_pad(input_shape, image_shape, ratio_pad)
    scaled = rboxes.clone()
    scaled[..., 0] -= pad[0]
    scaled[..., 1] -= pad[1]
    scaled[..., :4] /= gain
    return scaled


@overload
def xywhr2xyxyxyxy(x: torch.Tensor) -> torch.Tensor: ...


@overload
def xywhr2xyxyxyxy(x: np.ndarray) -> np.ndarray: ...


def xywhr2xyxyxyxy(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(x, torch.Tensor):
        ctr = x[..., :2]
        w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
        cos_value = torch.cos(angle)
        sin_value = torch.sin(angle)
        vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
        vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
        return torch.stack([ctr + vec1 + vec2, ctr + vec1 - vec2, ctr - vec1 - vec2, ctr - vec1 + vec2], dim=-2)

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value = np.cos(angle)
    sin_value = np.sin(angle)
    vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
    vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)
    return np.stack([ctr + vec1 + vec2, ctr + vec1 - vec2, ctr - vec1 - vec2, ctr - vec1 + vec2], axis=-2)


def yolo_multilabel_candidates(
    detections: torch.Tensor,
    class_count: int,
    extra_count: int,
    conf_thres: float,
) -> torch.Tensor:
    if detections.numel() == 0:
        return torch.zeros((0, 6 + extra_count), dtype=torch.float32, device=detections.device)

    boxes = detections[:, :4]
    scores = detections[:, 4 : 4 + class_count]
    extra = detections[:, 4 + class_count :]
    box_index, class_index = torch.where(scores > conf_thres)
    if box_index.numel() == 0:
        return torch.zeros((0, 6 + extra_count), dtype=torch.float32, device=detections.device)

    output = torch.empty((box_index.numel(), 6 + extra_count), dtype=detections.dtype, device=detections.device)
    output[:, :4] = boxes[box_index]
    output[:, 4] = scores[box_index, class_index]
    output[:, 5] = class_index.to(detections.dtype)
    if extra_count > 0:
        output[:, 6:] = extra[box_index]
    return output


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos_value = c.cos()
    sin_value = c.sin()
    cos2 = cos_value.pow(2)
    sin2 = sin_value.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos_value * sin_value


def batch_probiou(obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    obb2 = obb2.to(device=obb1.device, dtype=obb1.dtype)
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    denominator = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denominator) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denominator) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def rotated_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    sorted_idx = torch.argsort(scores, descending=True)
    remaining = boxes[sorted_idx]
    keep: list[torch.Tensor] = []

    while remaining.shape[0] > 0:
        keep.append(sorted_idx[0])
        if remaining.shape[0] == 1:
            break

        ious = batch_probiou(remaining[:1], remaining[1:]).squeeze(0)
        keep_mask = ious < iou_threshold
        remaining = remaining[1:][keep_mask]
        sorted_idx = sorted_idx[1:][keep_mask]

    return torch.stack(keep)
