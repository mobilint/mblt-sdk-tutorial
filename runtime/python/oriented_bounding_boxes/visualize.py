import cv2
import numpy as np
import torch
from dota import get_dota_label, get_dota_palette
from utils import scale_rboxes, xywhr2xyxyxyxy


class YoloObbVisualizer:
    def __init__(self) -> None:
        self.model_input_size = (1024, 1024)

    def save(self, detections: list[torch.Tensor] | None, input_path: str, output_path: str) -> np.ndarray:
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")

        rendered = self.draw(detections, image)
        cv2.imwrite(output_path, rendered)
        return rendered

    def draw(self, detections: list[torch.Tensor] | None, image: np.ndarray) -> np.ndarray:
        if detections is None or not detections or detections[0].numel() == 0:
            return image

        output = image.copy()
        det = detections[0].detach().cpu()
        image_shape: tuple[int, int] = (int(image.shape[0]), int(image.shape[1]))
        rboxes = scale_rboxes(self.model_input_size, torch.cat([det[:, :4], det[:, 6:7]], dim=-1), image_shape)
        polygons = xywhr2xyxyxyxy(rboxes).to(torch.int32).numpy()

        for polygon, score, cls_idx in zip(polygons, det[:, 4].tolist(), det[:, 5].tolist()):
            class_index = int(cls_idx)
            color = get_dota_palette(class_index)
            text_anchor = polygon.min(axis=0)
            label = f"{get_dota_label(class_index)} {score * 100:.1f}%"
            cv2.putText(
                output,
                label,
                (int(text_anchor[0]), max(int(text_anchor[1]) - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.drawContours(output, [polygon.reshape(-1, 1, 2)], -1, color, 2)

        return output
