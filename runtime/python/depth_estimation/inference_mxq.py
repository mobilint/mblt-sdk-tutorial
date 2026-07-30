"""Run YOLO26 depth estimation with a compiled MXQ model."""

from argparse import ArgumentParser

import cv2
import numpy as np
import qbruntime
from postprocess import postprocess_depth
from visualize import save_depth_overlay


def preprocess_yolo(
    image: np.ndarray,
    input_shape: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Letterbox an RGB image for an HWC or CHW uint8 MXQ input."""
    if len(input_shape) != 3:
        raise ValueError(f"Expected a three-dimensional model input shape, got {input_shape}")
    if input_shape[-1] == 3:
        target_height, target_width = input_shape[:2]
        is_hwc = True
    elif input_shape[0] == 3:
        target_height, target_width = input_shape[1:]
        is_hwc = False
    else:
        raise ValueError(f"Could not determine the channel axis from input shape {input_shape}")

    source_height, source_width = image.shape[:2]
    scale = min(target_height / source_height, target_width / source_width)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    if (source_width, source_height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    if not is_hwc:
        image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0), (top, bottom, left, right)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run depth estimation with a compiled MXQ model")
    parser.add_argument(
        "--model-path",
        default="../../../compilation/depth_estimation/yolo26m-depth.mxq",
        help="Path to the compiled MXQ model",
    )
    parser.add_argument("--image-path", default="../rc/bus.jpg", help="Path to the input image")
    parser.add_argument("--output-path", default="./tmp/bus_depth_demo.jpg", help="Path for the output image")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.7,
        help="Depth-map opacity between 0 and 1",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")

    image_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {args.image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    accelerator = qbruntime.Accelerator()
    model_config = qbruntime.ModelConfig()
    model_config.set_single_core_mode(
        None,
        [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)],
    )
    model = qbruntime.Model(args.model_path, model_config)
    model.launch(accelerator)

    try:
        input_shape = tuple(model.get_model_input_shape()[0])
        model_input, borders = preprocess_yolo(image_rgb, input_shape)
        outputs = model.infer([model_input])
        if outputs is None:
            raise RuntimeError("Model inference returned no outputs")
    finally:
        model.dispose()

    depth = postprocess_depth(
        outputs,
        input_shape=input_shape,
        original_shape=image_bgr.shape[:2],
        letterbox_borders=borders,
    )
    save_depth_overlay(
        image_bgr,
        depth,
        output_path=args.output_path,
        alpha=args.overlay_alpha,
    )
    print(f"Saved the depth visualization to {args.output_path}")


if __name__ == "__main__":
    main()
