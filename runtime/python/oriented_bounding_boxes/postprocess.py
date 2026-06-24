from multiprocessing.pool import ThreadPool
from typing import TypeGuard

import numpy as np
import torch
from utils import (
    DEVICE,
    DFL,
    NUM_THREADS,
    dist2rbox,
    invsigmoid,
    make_anchors,
    rotated_nms,
    yolo_multilabel_candidates,
)


def _is_numpy_array_list(x: object) -> TypeGuard[list[np.ndarray]]:
    return isinstance(x, list) and all(isinstance(xi, np.ndarray) for xi in x)


def _is_tensor_list(x: object) -> TypeGuard[list[torch.Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, torch.Tensor) for xi in x)


class YoloObbPostProcess:
    def __init__(self, conf_thres: float = 0.25, iou_thres: float = 0.45) -> None:
        self.image_size = 1024
        self.nc = 15
        self.nl = 3
        self.reg_max = 16
        self.n_extra = 1
        self.stride = [2 ** (3 + i) for i in range(self.nl)]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = DEVICE
        self.dfl = DFL(self.reg_max)
        self.anchors, self.strides = (
            x.transpose(0, 1) for x in make_anchors(self.image_size, self.image_size, self.stride)
        )
        self.invconf_thres = invsigmoid(self.conf_thres)

    def check_input(self, x: list[np.ndarray] | list[torch.Tensor] | np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        if isinstance(x, np.ndarray):
            return [torch.from_numpy(x).to(self.device)]
        if _is_numpy_array_list(x):
            return [torch.from_numpy(xi).to(self.device) for xi in x]
        if isinstance(x, torch.Tensor):
            return [x.to(self.device)]
        if _is_tensor_list(x):
            return [xi.to(self.device) for xi in x]
        raise NotImplementedError(f"Input type {type(x)} not supported.")

    def __call__(
        self, x: list[np.ndarray] | list[torch.Tensor] | np.ndarray | torch.Tensor
    ) -> list[torch.Tensor] | None:
        x = self.check_input(x)
        x = self.rearrange_npu_out(x)
        x = self.decode(x)
        x = self.nms(x)
        if not x or x[0].numel() == 0:
            return None
        return x

    def rearrange_npu_out(self, x: list[torch.Tensor]) -> torch.Tensor:
        y_det, y_cls, y_angle = [], [], []
        for xi in x:
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)
            if xi.ndim != 4:
                raise ValueError(f"Expected 3D or 4D output, got {tuple(xi.shape)}.")

            if xi.shape[1] == self.reg_max * 4:
                y_det.append(xi)
            elif xi.shape[1] == self.nc:
                y_cls.append(xi)
            elif xi.shape[1] == self.n_extra:
                y_angle.append(xi)
            else:
                raise ValueError(f"Wrong shape of input: {tuple(xi.shape)}")

        y_det = sorted(y_det, key=lambda item: item.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda item: item.numel(), reverse=True)
        y_angle = sorted(y_angle, key=lambda item: item.numel(), reverse=True)
        if not (len(y_det) == len(y_cls) == len(y_angle) == self.nl):
            raise ValueError("OBB outputs are not in the expected three-level format.")

        return torch.cat(
            [torch.cat((det, cls, angle), dim=1).flatten(2) for det, cls, angle in zip(y_det, y_cls, y_angle)],
            dim=-1,
        )

    def decode(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.device.type == "cpu":
            with ThreadPool(NUM_THREADS) as pool:
                return pool.map(self.process_box_cls, x)
        return [self.process_box_cls(box_cls) for box_cls in x]

    def process_box_cls(self, box_cls: torch.Tensor) -> torch.Tensor:
        class_logits = box_cls[-self.nc - self.n_extra : -self.n_extra, :]
        keep = torch.amax(class_logits, dim=0) > self.invconf_thres
        box_cls = box_cls[:, keep]
        if box_cls.numel() == 0:
            return torch.zeros((0, 4 + self.nc + self.n_extra), dtype=torch.float32, device=self.device)

        box, scores, angle = torch.split(box_cls.unsqueeze(0), [self.reg_max * 4, self.nc, self.n_extra], dim=1)
        angle = (angle.sigmoid() - 0.25) * torch.pi
        rbox = dist2rbox(self.dfl(box), angle, self.anchors[:, keep], dim=1) * self.strides[:, keep]
        return torch.cat([rbox, scores.sigmoid(), angle], dim=1).squeeze(0).transpose(0, 1)

    def nms(
        self,
        prediction: list[torch.Tensor],
        max_det: int = 300,
        max_nms: int = 30000,
        max_wh: int = 7680,
    ) -> list[torch.Tensor]:
        assert 0 <= self.conf_thres <= 1, f"Invalid confidence threshold {self.conf_thres}"
        assert 0 <= self.iou_thres <= 1, f"Invalid IoU threshold {self.iou_thres}"

        def nms_single(x: torch.Tensor) -> torch.Tensor:
            if x.numel() == 0:
                return torch.zeros((0, 7), dtype=torch.float32, device=self.device)

            x = yolo_multilabel_candidates(x, self.nc, self.n_extra, self.conf_thres)
            if x.numel() == 0:
                return torch.zeros((0, 7), dtype=torch.float32, device=self.device)

            x = x[torch.argsort(x[:, 4], descending=True)[:max_nms]]
            class_offsets = x[:, 5:6] * max_wh
            boxes = torch.cat([x[:, :2] + class_offsets, x[:, 2:4], x[:, 6:7]], dim=-1)
            keep = rotated_nms(boxes, x[:, 4], self.iou_thres)[:max_det]
            return x[keep]

        if self.device.type == "cpu":
            with ThreadPool(NUM_THREADS) as pool:
                return pool.map(nms_single, prediction)
        return [nms_single(item) for item in prediction]
