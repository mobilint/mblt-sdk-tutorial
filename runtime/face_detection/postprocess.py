import numpy as np
import torch
from utils import DEVICE, DFL, dist2bbox, invsigmoid, make_anchors, non_max_suppression


class YoloPostProcessAnchorless:
    """Single-class YOLO face detection postprocess for anchorless models."""

    def __init__(self, conf_thres: float = 0.5, iou_thres: float = 0.5, img_size: int = 640, reg_max: int = 16):
        self.imh = self.imw = img_size
        self.nc = 1
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.reg_max = reg_max
        self.device = DEVICE
        self.dfl = DFL(self.reg_max)

        stride = [8, 16, 32]
        anchors, strides = make_anchors(self.imh, self.imw, stride, 0.5)
        self.anchors = anchors.transpose(0, 1)
        self.strides = strides.transpose(0, 1)
        self.invconf_thres = invsigmoid(self.conf_thres)

    def check_input(self, x) -> list[torch.Tensor]:
        """Convert qbruntime outputs into a list of BCHW tensors on the postprocess device."""
        if isinstance(x, np.ndarray):
            return [torch.from_numpy(x).to(self.device)]
        if isinstance(x, list) and all(isinstance(xi, np.ndarray) for xi in x):
            return [torch.from_numpy(xi).to(self.device) for xi in x]
        if isinstance(x, torch.Tensor):
            return [x.to(self.device)]
        if isinstance(x, list) and all(isinstance(xi, torch.Tensor) for xi in x):
            return [xi.to(self.device) for xi in x]

        raise NotImplementedError(f"Input type {type(x)} not supported.")

    @torch.inference_mode()
    def __call__(self, x) -> list[torch.Tensor] | None:
        outputs = self.check_input(x)
        outputs = self.rearrange_npu_out(outputs)
        decoded = self.decode(outputs)
        detections = self.nms(decoded)
        if not detections or detections[0].numel() == 0:
            return None
        return detections

    def rearrange_npu_out(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]:
        box_heads = []
        cls_heads = []

        for output in outputs:
            if output.ndim == 3:
                output = output.unsqueeze(0)
            elif output.ndim != 4:
                raise NotImplementedError(f"Got unsupported ndim for input: {output.ndim}.")

            if output.shape[1] == self.reg_max * 4:
                box_heads.append(output)
            elif output.shape[1] == self.nc:
                cls_heads.append(output)
            else:
                raise ValueError(f"Wrong shape of input: {tuple(output.shape)}")

        box_heads.sort(key=lambda tensor: tensor.shape[-1] * tensor.shape[-2], reverse=True)
        cls_heads.sort(key=lambda tensor: tensor.shape[-1] * tensor.shape[-2], reverse=True)

        if len(box_heads) != len(cls_heads):
            raise ValueError("The number of box heads and classification heads must match.")

        return [torch.cat((box_head, cls_head), dim=1).flatten(2) for box_head, cls_head in zip(box_heads, cls_heads)]

    def decode(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]:
        batch_box_cls = torch.cat(outputs, dim=-1)
        return [self.process_box_cls(box_cls) for box_cls in batch_box_cls]

    def process_box_cls(self, box_cls: torch.Tensor) -> torch.Tensor:
        score_logits = box_cls[self.reg_max * 4]
        keep = score_logits > self.invconf_thres
        if not torch.any(keep):
            return torch.zeros((0, 6), dtype=torch.float32, device=self.device)

        box = box_cls[: self.reg_max * 4, keep].unsqueeze(0)
        scores = score_logits[keep].sigmoid()
        boxes = dist2bbox(self.dfl(box), self.anchors[:, keep], xywh=False, dim=1) * self.strides[:, keep]
        class_ids = torch.zeros((scores.shape[0], 1), dtype=torch.float32, device=self.device)

        return torch.cat((boxes.squeeze(0).transpose(0, 1), scores.unsqueeze(1), class_ids), dim=1)

    def nms(self, predictions: list[torch.Tensor], max_det: int = 300, max_nms: int = 30000) -> list[torch.Tensor]:
        assert 0 <= self.conf_thres <= 1, (
            f"Invalid confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0."
        )
        assert 0 <= self.iou_thres <= 1, f"Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0."

        detections = []
        for prediction in predictions:
            if prediction.numel() == 0:
                detections.append(torch.zeros((0, 6), dtype=torch.float32, device=self.device))
                continue

            prediction = prediction[prediction[:, 4] > self.conf_thres]
            if prediction.numel() == 0:
                detections.append(torch.zeros((0, 6), dtype=torch.float32, device=self.device))
                continue

            prediction = prediction[prediction[:, 4].argsort(descending=True)[:max_nms]]
            keep = non_max_suppression(prediction[:, :4], prediction[:, 4], self.iou_thres, max_det)
            keep = torch.as_tensor(keep, dtype=torch.long, device=prediction.device)
            detections.append(prediction[keep])

        return detections
