import os
from argparse import ArgumentParser

import cv2
import numpy as np
import qbruntime
from postprocess import YoloPosePostProcessAnchorless
from visualize import YoloVisualizer


def preprocess_yolo(img_path: str, img_size=(640, 640)):
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  # orig hw
    r = min(img_size[0] / h0, img_size[1] / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2  # to center the image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border

    return img


if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference with compiled model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the compiled MXQ model"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to the output image"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument("--iou_thres", type=float, default=0.45, help="IoU threshold")

    args = parser.parse_args()

    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(1)
    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    postprocess = YoloPosePostProcessAnchorless(args.conf_thres, args.iou_thres)
    visualizer = YoloVisualizer()

    img = preprocess_yolo(args.image_path)
    img = np.expand_dims(np.transpose(img, [2, 0, 1]), 0)
    outputs = model.infer([img])
    result = postprocess(outputs)

    output_path = args.output_path or os.path.join(
        os.path.dirname(args.image_path), "output.jpg"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visualizer.save(
        [result[0][..., :6]],
        input_path=args.image_path,
        output_path=output_path,
        kpts=[result[0][..., 6:]],
    )
