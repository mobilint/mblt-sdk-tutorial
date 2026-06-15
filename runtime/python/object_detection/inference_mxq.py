import os
from argparse import ArgumentParser

import cv2
import numpy as np
import qbruntime
from postprocess import YoloPostProcessAnchorless
from visualize import YoloVisualizer


def preprocess_yolo(img, input_shape):
    """Letterbox an RGB image into the shape/layout the MXQ model expects.

    Args:
        img: RGB image (BGR->RGB conversion must be done by the caller).
        input_shape: model.get_model_input_shape()[0], e.g. (640, 640, 3) for HWC
            or (3, 640, 640) for CHW. The position of the channel (==3) decides the layout.

    Returns:
        A batched uint8 array ready for model.infer: (1, H, W, 3) for HWC models,
        (1, 3, H, W) for CHW models.
    """
    # Decide target size and layout straight from the model's input shape.
    if input_shape[-1] == 3:  # channel last -> HWC, e.g. (640, 640, 3)
        target_h, target_w, is_hwc = input_shape[0], input_shape[1], True
    else:  # channel first -> CHW, e.g. (3, 640, 640)
        target_h, target_w, is_hwc = input_shape[1], input_shape[2], False

    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    h0, w0 = img.shape[:2]  # orig hw
    r = min(target_h / h0, target_w / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = target_h - new_unpad[1], target_w - new_unpad[0]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2  # to center the image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

    if not is_hwc:  # CHW model -> move channel to front
        img = np.transpose(img, (2, 0, 1))

    return np.expand_dims(img, 0)  # add batch dim


if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference with compiled model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../../../compilation/object_detection/yolo11m.mxq",
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

    postprocess = YoloPostProcessAnchorless(args.conf_thres, args.iou_thres)
    visualizer = YoloVisualizer()

    # Read the image, then let preprocess match the model's expected input layout/shape.
    input_shape = model.get_model_input_shape()[0]  # e.g. (640, 640, 3) HWC or (3, 640, 640) CHW
    img_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = preprocess_yolo(img_rgb, input_shape)
    outputs = model.infer([img])

    # postprocess expects channel-first (BCHW). When the model runs in HWC, the NPU
    # returns channel-last outputs, so transpose them here and leave postprocess.py untouched.
    if input_shape[-1] == 3:
        outputs = [
            np.transpose(o, (0, 3, 1, 2)) if o.ndim == 4 else np.transpose(o, (2, 0, 1))
            for o in outputs
        ]
    result = postprocess(outputs)

    output_path = args.output_path or os.path.join(os.path.dirname(args.image_path), "output.jpg")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    visualizer.save(result, input_path=args.image_path, output_path=output_path)
    model.dispose()
