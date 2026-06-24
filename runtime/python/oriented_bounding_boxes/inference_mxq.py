import os
from argparse import ArgumentParser

import cv2
import numpy as np
import qbruntime
from postprocess import YoloObbPostProcess
from visualize import YoloObbVisualizer


def preprocess_yolo_obb(img: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    if input_shape[-1] == 3:
        target_h, target_w, is_hwc = input_shape[0], input_shape[1], True
    else:
        target_h, target_w, is_hwc = input_shape[1], input_shape[2], False

    h0, w0 = img.shape[:2]
    ratio = min(target_h / h0, target_w / w0)
    new_unpad = int(round(w0 * ratio)), int(round(h0 * ratio))

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = target_h - new_unpad[1], target_w - new_unpad[0]
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    if not is_hwc:
        img = np.transpose(img, (2, 0, 1))

    return np.expand_dims(img, 0).astype(np.uint8, copy=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run OBB inference with a compiled MXQ model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq",
        help="Path to the compiled MXQ model",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="../rc/airport.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./tmp/airport_demo.jpg",
        help="Path to the rendered output image",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")
    args = parser.parse_args()

    acc = qbruntime.Accelerator()
    model_config = qbruntime.ModelConfig()
    model_config.set_single_core_mode(
        None,
        [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)],
    )

    model = qbruntime.Model(args.model_path, model_config)
    model.launch(acc)

    input_shape = model.get_model_input_shape()[0]
    image_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {args.image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = preprocess_yolo_obb(image_rgb, input_shape)
    outputs = model.infer([image])
    if outputs is None:
        raise RuntimeError("Model inference returned no outputs.")

    if input_shape[-1] == 3:
        outputs = [
            np.transpose(output, (0, 3, 1, 2)) if output.ndim == 4 else np.transpose(output, (2, 0, 1))
            for output in outputs
        ]

    postprocess = YoloObbPostProcess(args.conf_thres, args.iou_thres)
    visualizer = YoloObbVisualizer()
    detections = postprocess(outputs)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    visualizer.save(detections, input_path=args.image_path, output_path=args.output_path)
    model.dispose()
