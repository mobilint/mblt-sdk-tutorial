import os
from argparse import ArgumentParser

import cv2
import numpy as np
import qbruntime
from postprocess import YoloPostProcessAnchorless
from visualize import YoloVisualizer

MODEL_INPUT_SIZE = (640, 640)


def preprocess_yolo(img_path: str, img_size: tuple[int, int] = MODEL_INPUT_SIZE) -> np.ndarray:
    """Load an image and apply the same letterbox preprocessing used at compile time."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    r = min(img_size[0] / h0, img_size[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference with a compiled YOLO face MXQ model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../../compilation/face_detection/yolov11m-face.mxq",
        help="Path to the compiled MXQ model",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="../rc/cr7.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./tmp/cr_demo.jpg",
        help="Path to the output image",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")

    args = parser.parse_args()

    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])

    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    try:
        postprocess = YoloPostProcessAnchorless(args.conf_thres, args.iou_thres, img_size=MODEL_INPUT_SIZE[0])
        visualizer = YoloVisualizer(model_input_size=MODEL_INPUT_SIZE)

        img = preprocess_yolo(args.image_path)
        outputs = model.infer([img])
        result = postprocess(outputs)

        output_path = args.output_path or os.path.join(os.path.dirname(args.image_path), "output.jpg")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        visualizer.save(result, input_path=args.image_path, output_path=output_path)
    finally:
        model.dispose()
