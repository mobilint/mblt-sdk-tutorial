from multiprocessing.pool import ThreadPool
from typing import List

import numpy as np
import torch
from utils import (
    DEVICE,
    DFL,
    NUM_THREADS,
    dist2bbox,
    invsigmoid,
    make_anchors,
    non_max_suppression,
)


class YoloPostProcessAnchorless:
    """YOLO object detection postprocess for anchorless models."""

    def __init__(self, conf_thres=0.5, iou_thres=0.5):
        self.imh = self.imw = 640
        self.nc = 80
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.reg_max = (
            16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        )
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor
        self.dfl = DFL(self.reg_max)

        self.nl = 3
        self.stride = [2 ** (3 + i) for i in range(self.nl)]
        self.anchors, self.strides = (
            xx.transpose(0, 1)
            for xx in make_anchors(self.imh, self.imw, self.stride, 0.5)
        )
        self.n_extra = 0
        self.device = DEVICE
        self.invconf_thres = invsigmoid(self.conf_thres)

    def check_input(
        self,
        x,
    ) -> List[torch.Tensor]:
        """Check input type and convert to ListTensor

        Args:
            x (TensorLike): Input

        Raises:
            NotImplementedError: Input type not supported.

        Returns:
            List[torch.Tensor]
        """
        if isinstance(x, np.ndarray):
            return [torch.from_numpy(x).to(self.device)]
        if isinstance(x, list) and all(isinstance(xi, np.ndarray) for xi in x):
            return [torch.from_numpy(xi).to(self.device) for xi in x]
        if isinstance(x, torch.Tensor):
            return [x.to(self.device)]
        if isinstance(x, list) and all(isinstance(xi, torch.Tensor) for xi in x):
            return [xi.to(self.device) for xi in x]

        raise NotImplementedError(f"Input type {type(x)} not supported.")

    def __call__(self, x):
        x = self.check_input(x)
        x = self.rearrange_npu_out(x)
        x = self.decode(x)
        x = self.nms(x)
        if x[0].nelement() == 0:
            return None
        return x

    def rearrange_npu_out(self, x):
        y_det = []
        y_cls = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")

            if xi.shape[1] == self.reg_max * 4:
                y_det.append(xi)  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[1] == self.nc:
                y_cls.append(xi)  # (b, 80, 80, 80), (b, 80, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")

        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)

        y = [
            torch.cat((yi_det, yi_cls), dim=1).flatten(2)
            for (yi_det, yi_cls) in zip(y_det, y_cls)
        ]

        return y

    def decode(self, x):
        batch_box_cls = torch.cat(x, dim=-1)  # (b, 144, 8400)

        if self.device.type == "cpu":
            with ThreadPool(NUM_THREADS) as pool:
                return pool.map(
                    self.process_box_cls, batch_box_cls
                )  # [(*, 84), (*, 84), (*, 84), ...]

        return [self.process_box_cls(box_cls) for box_cls in batch_box_cls]

    def process_box_cls(self, box_cls):
        if self.n_extra == 0:
            ic = torch.amax(box_cls[-self.nc :, :], dim=0) > self.invconf_thres
        else:
            ic = (
                torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0)
                > self.invconf_thres
            )
        box_cls = box_cls[:, ic]  # (144, *)
        if box_cls.numel() == 0:
            return torch.zeros(
                (0, 4 + self.nc + self.n_extra), dtype=torch.float32
            )  # (0, 84)

        box, scores, extra = torch.split(
            box_cls[None], [self.reg_max * 4, self.nc, self.n_extra], dim=1
        )  # (1, 64, *), (1, 80, *), (1, 32, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                self.anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * self.strides[:, ic]
        )

        return (
            torch.cat([dbox, scores.sigmoid(), extra], dim=1).squeeze(0).transpose(0, 1)
        )

    def nms(
        self,
        prediction,
        agnostic=False,
        max_det=300,
        max_nms=30000,
        max_wh=7680,
    ):
        """
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L162
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (np.array): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            max_det (int): The maximum number of boxes to keep after NMS.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into NMS.
            max_wh (int): The maximum box width and height in pixels

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        # Checks
        assert (
            0 <= self.conf_thres <= 1
        ), f"Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= self.iou_thres <= 1
        ), f"Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0"

        def nms_single(x):
            box, conf, mask = torch.split(x, [4, self.nc, self.n_extra], dim=1)

            i, j = torch.nonzero(conf > self.conf_thres).T  # use multi-label as default
            x = torch.cat(
                (box[i], x[i, 4 + j, None], j[:, None].to(torch.float32), mask[i]),
                dim=1,
            )

            if x.numel() == 0:  # no boxes
                return torch.zeros((0, 6 + self.n_extra), dtype=torch.float32)

            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence with descending order and remove excess boxes

            c = x[:, 5:6] * (0.0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = non_max_suppression(boxes, scores, self.iou_thres, max_det)  # NMS

            return x[i, :]

        if self.device.type == "cpu":
            with ThreadPool(NUM_THREADS) as pool:
                return pool.map(nms_single, prediction)

        return [nms_single(p) for p in prediction]


class YoloPosePostProcessAnchorless(YoloPostProcessAnchorless):
    def __init__(self, conf_thres=0.5, iou_thres=0.5):
        super().__init__(conf_thres, iou_thres)
        self.nc = 1
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor
        self.n_extra = 51

    def rearrange_npu_out(self, x):
        y_det = []
        y_cls = []
        y_kpt = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")

            if xi.shape[1] == self.reg_max * 4:
                y_det.append(xi)  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[1] == self.nc:
                y_cls.append(xi)  # (b, 1, 80, 80), (b, 1, 40, 40), ...
            elif xi.shape[1] == self.n_extra:
                y_kpt.append(xi.flatten(2))  # (b, 51, 80, 80), (b, 1, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")

        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        y_kpt = sorted(
            y_kpt, key=lambda x: x.numel(), reverse=True
        )  # (b, 51, 6400), (b, 51, 1600), (b, 51, 400)

        y_tmp = [
            torch.cat((yi_det, yi_cls), dim=1).flatten(2)
            for (yi_det, yi_cls) in zip(
                y_det, y_cls
            )  # (b, 65, 6400), (b, 65, 1600), (b, 65, 400)
        ]
        y = [
            torch.cat((yi_tmp, yi_kpt), dim=1) for (yi_tmp, yi_kpt) in zip(y_tmp, y_kpt)
        ]  # (b, 116, 6400), (b, 116, 1600), (b, 116, 400)

        return y

    def process_box_cls(self, box_cls):
        ic = (
            torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0)
            > self.invconf_thres
        )
        box_cls = box_cls[:, ic]  # (116, *)
        if box_cls.numel() == 0:
            return torch.zeros(
                (0, 4 + self.nc + self.n_extra), dtype=torch.float32
            )  # (0, 56)

        box, scores, keypoints = torch.split(
            box_cls[None], [self.reg_max * 4, self.nc, self.n_extra], dim=1
        )  # (1, 64, *), (1, 1, *), (1, 51, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                self.anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * self.strides[:, ic]
        )

        keypoints = keypoints.view(1, 17, 3, -1)
        key_coord, key_conf = torch.split(
            keypoints, [2, 1], dim=2
        )  # (1, 17, 2, 8400), (1, 17, 1, 8400)

        key_coord = (key_coord * 2 + self.anchors[:, ic] - 0.5) * self.strides[
            :, ic
        ]  # (1, 17, 2, *)
        keypoints = torch.cat([key_coord, key_conf.sigmoid()], dim=2).view(
            1, self.n_extra, -1
        )  # (1, 51, *)

        return (
            torch.cat([dbox, scores.sigmoid(), keypoints], dim=1)
            .squeeze(0)
            .transpose(0, 1)
        )  # (*, 56)
