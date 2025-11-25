from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from coco import get_coco_det_palette, get_coco_label


def compute_ratio_pad(input_shape, img_shape, ratio_pad=None):
    """Compute ratio and pad which were used to resize image to input_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            input_shape[0] / img_shape[0], input_shape[1] / img_shape[1]
        )  # gain  = old / new
        pad = round((input_shape[1] - img_shape[1] * gain) / 2 - 0.1), round(
            (input_shape[0] - img_shape[0] * gain) / 2 - 0.1
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    return gain, pad


def clip_boxes(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])  # y1, y2
    return boxes


def scale_boxes(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    gain, pad = compute_ratio_pad(img1_shape, img0_shape, ratio_pad)

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    return clip_boxes(coords, img0_shape)


def scale_masks(img1_shape, masks, img0_shape, ratio_pad=None):
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L377
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
        img1_shape[0] - img0_shape[0] * gain
    ) / 2

    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(masks.shape[1] - pad[1]), int(masks.shape[2] - pad[0])

    masks = masks[:, top:bottom, left:right]
    masks = F.interpolate(
        masks.unsqueeze(0), scale_factor=1 / gain, mode="bilinear", align_corners=False
    ).squeeze(0)

    masks = torch.cat(
        [
            torch.from_numpy(cv2.resize(m.cpu().numpy(), img0_shape[:2][::-1])[None])
            for m in masks
        ],
        axis=0,
    )

    return masks.gt_(0.5).cpu().numpy()


def draw_boxes(img, xyxy, desc, cls_color):
    """plot bounding boxes on image"""
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(
        img, (x1, y1), (x2, y2), cls_color, thickness=tl, lineType=cv2.LINE_AA
    )

    tf = max(tl - 1, 1)  # font thickness
    cv2.putText(
        img,
        desc,
        (x1, y1 - 2),
        0,
        tl / 2,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


def draw_masks(img, masks, colors, alpha=0.3):
    colors = np.array(colors, dtype=np.float32) / 255
    colors = colors[:, None, None]  # shape(n,1,1,3)
    masks = np.expand_dims(masks, 3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

    img = img.astype(np.float32) / 255
    img = img * inv_alph_masks[-1] + mcs
    img = (img * 255).astype(np.uint8)

    return img


class BaseVisualizer(ABC):
    def __init__(self, dataset: str = "coco"):
        self.dataset = dataset
        self.writer = None

        if self.dataset == "coco":
            self.get_label = get_coco_label
            self.get_color = get_coco_det_palette
        else:
            raise NotImplementedError(f"Got unsupported dataset: ", self.dataset)

    @abstractmethod
    def save(self, out_post_processed, **kwargs):
        pass

    def set_video_writer(
        self, output_path: str, fps: float, video_size: Tuple[int, int]
    ) -> None:
        self.writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size
        )
        self.fps = fps

    def release(self):
        if self.writer is None:
            print("Video writer is not set yet.")
        else:
            self.writer.release()


class YoloVisualizer(BaseVisualizer):
    def __init__(self) -> None:
        super().__init__("coco")
        self.model_input_size = [640, 640]

    def save(
        self,
        out_post_processed: List[torch.Tensor],
        input_path: str,
        output_path: str = None,
        is_yolox: bool = False,
        masks: List[torch.Tensor] = None,
        kpts: List[torch.Tensor] = None,
    ):
        assert not (
            masks is not None and kpts is not None
        ), "masks and kpts cannot be used together."

        img = cv2.imread(input_path)

        img, col_list = self.draw_det(out_post_processed, img, is_yolox)

        if masks is not None:
            masks = masks[0]
            img = self.add_mask(img, masks, col_list)

        if output_path is not None:  # image demo
            cv2.imwrite(output_path, img)

        return img

    def draw_det(
        self,
        out_post_processed: List[torch.Tensor],
        img: np.ndarray,
        is_yolox: bool = False,
    ):
        det = out_post_processed[0]
        num_det = det.shape[0]
        if not is_yolox:
            det[:, :4] = scale_boxes(self.model_input_size, det[:, :4], img.shape)

        col_list = []
        for j in range(num_det):
            xyxy = det[j, :4].view(-1).tolist()
            conf = det[j, 4].cpu().numpy()
            cls = det[j, 5].cpu().numpy().item()
            cls_name = self.get_label(int(cls))
            cls_color = self.get_color(int(cls))
            col_list.append(cls_color)

            desc = f"{cls_name}: {round(100 * conf.item(), 1)}%"
            img = draw_boxes(img, xyxy, desc, cls_color)

        return img, col_list

    def add_mask(self, img, masks, col_list):
        masks = scale_masks(self.model_input_size, masks, img.shape)
        img = draw_masks(img, masks, col_list)

        return img
