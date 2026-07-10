"""Compile the Qwen3-VL vision encoder (HF -> MBLT) with ModelParser.

The vision feature entry point is hooked to capture pixel_values / image_grid_thw,
which are reshaped to the NPU layout and run through ModelParser.
"""

import argparse
import os

from qbcompiler.model_dict.common import DataFormat, LayerType
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen3vl import (
    VisionModelForQwen3VL,
    repreprocess_pixel_values,
)
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from qbcompiler.model_dict.parser.parser import ModelParser
from utils import create_sample_messages, load_model_and_processor, prepare_inputs, serialize_to_mblt

TARGET_DEVICES = ["regulus-ra", "aries-rb", "regulus-rb"]
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"


def capture_vision_inputs(model, inputs):
    """Capture pixel_values / image_grid_thw at the vision feature entry point."""
    container = DefaultInputsCaptureContainer()
    with InputCaptureCtxManager(model.model.get_image_feature_class, 1, container):
        model(**inputs)
    if not container.captured_kwargs:
        raise RuntimeError("Vision feature hook captured no inputs.")
    captured = container.captured_kwargs[0]
    return captured["pixel_values"], captured["image_grid_thw"]


def build_parser(vision_model, target_device):
    """Configure a torch-backend ModelParser for the vision encoder."""
    parser = ModelParser(
        model=vision_model,
        backend="torch",
        target_device=target_device,
        yolo_decode_include=True,
    )
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True
    return parser


def set_input_constants_nhwc(model_dict):
    """Vision input constants use the NHWC layout on the NPU."""
    for subgraph in model_dict.subgraphs:
        for op in subgraph.operators:
            if op.layertype == LayerType.InputConstant:
                subgraph.activations[op.options.outputs[0]].dataformat = DataFormat.NHWC


def main():
    args = _parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    model, processor = load_model_and_processor(args.model)
    messages = create_sample_messages(IMAGE_URL, "Describe this image in detail.")
    inputs = prepare_inputs(processor, messages, model.device, tuple(args.image_size))

    pixel_values, grid_thw = capture_vision_inputs(model, inputs)
    images = repreprocess_pixel_values(pixel_values, grid_thw[0])
    feed_dict = {"images": wrap_tensor("images", images.to(model.device))}

    vision_model = VisionModelForQwen3VL(model.model)
    vision_model.set_grid_thw(grid_thw.to(model.device))
    vision_model.to(model.device)

    parser = build_parser(vision_model, args.target_device)
    parser.parse(feed_dict=feed_dict, save_subgraph_type=1, debug=False)

    model_dict, weight_dict = parser.get_md_wd(body_only=False)
    set_input_constants_nhwc(model_dict)
    serialize_to_mblt(model_dict, weight_dict, args.save_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Compile Qwen3-VL vision encoder to MBLT")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--save-path", default="mblt/Qwen3-VL-4B-Instruct_vision_transformer.mblt")
    parser.add_argument("--target-device", required=True, choices=TARGET_DEVICES)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    return parser.parse_args()


if __name__ == "__main__":
    main()
