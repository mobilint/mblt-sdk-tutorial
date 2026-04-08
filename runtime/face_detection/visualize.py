from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from face_metadata import get_face_det_palette, get_face_label


def compute_ratio_pad(input_shape, img_shape, ratio_pad=None):
    """Compute ratio and padding used by the letterbox resize."""
    if ratio_pad is None:
        gain = min(input_shape[0] / img_shape[0], input_shape[1] / img_shape[1])
        pad = (
            round((input_shape[1] - img_shape[1] * gain) / 2 - 0.1),
            round((input_shape[0] - img_shape[0] * gain) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    return gain, pad


def clip_boxes(boxes, img_shape):
    """Clip xyxy boxes to image shape (height, width)."""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])
    return boxes


def scale_boxes(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    gain, pad = compute_ratio_pad(img1_shape, img0_shape, ratio_pad)

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain

    return clip_boxes(coords, img0_shape)


def draw_boxes(img: np.ndarray, xyxy: list[float], desc: str, cls_color: tuple[int, int, int]) -> np.ndarray:
    """Draw a single bounding box and label in place."""
    h, w = img.shape[:2]
    tl = max(round(0.002 * (h + w) / 2) + 1, 1)
    x1, y1, x2, y2 = (int(v) for v in xyxy)

    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)
    cv2.putText(
        img,
        desc,
        (x1, max(y1 - 4, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        tl / 3,
        (225, 255, 255),
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


class BaseVisualizer(ABC):
    def __init__(self):
        self.writer = None
        self.get_label = get_face_label
        self.get_color = get_face_det_palette

    @abstractmethod
    def save(self, out_post_processed, **kwargs):
        raise NotImplementedError

    def set_video_writer(self, output_path: str, fps: float, video_size: tuple[int, int]) -> None:
        self.writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size)
        self.fps = fps

    def release(self):
        if self.writer is None:
            print("Video writer is not set yet.")
        else:
            self.writer.release()


class YoloVisualizer(BaseVisualizer):
    def __init__(self, model_input_size: tuple[int, int] = (640, 640)) -> None:
        super().__init__()
        self.model_input_size = model_input_size

    def save(
        self,
        out_post_processed: list[torch.Tensor] | None,
        input_path: str,
        output_path: str | None = None,
        is_yolox: bool = False,
    ) -> np.ndarray:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {input_path}")

        if out_post_processed is not None:
            img = self.draw_det(out_post_processed, img, is_yolox)

        if output_path is not None:
            cv2.imwrite(output_path, img)

        return img

    def draw_det(self, out_post_processed: list[torch.Tensor], img: np.ndarray, is_yolox: bool = False) -> np.ndarray:
        det = out_post_processed[0].detach().cpu().clone()
        if det.numel() == 0:
            return img

        if not is_yolox:
            det[:, :4] = scale_boxes(self.model_input_size, det[:, :4], img.shape)

        for row in det:
            xyxy = row[:4].tolist()
            conf = float(row[4].item())
            cls = int(row[5].item())
            desc = f"{self.get_label(cls)}: {conf * 100:.1f}%"
            draw_boxes(img, xyxy, desc, self.get_color(cls))

        return img
