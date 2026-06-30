"""Qwen2-VL Vision Encoder Compilation to MBLT Format (via mblt_compile)."""

import argparse
import os

from qbcompiler import mblt_compile
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen2vl import (
    VisionModelForQwen2VL,
    repreprocess_pixel_values,
)
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from utils import create_sample_messages, load_model_and_processor, prepare_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Qwen2-VL vision encoder to MBLT")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--save-path", type=str, default="mblt/Qwen2-VL-2B-Instruct_vision_transformer.mblt")
    parser.add_argument("--target-device", type=str, required=True, help="Target NPU (e.g. aries-rb, regulus-rb)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    model, processor = load_model_and_processor(args.model)
    messages = create_sample_messages(
        image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true",
        text_prompt="Describe this image in detail with document format.",
    )

    inputs = prepare_inputs(processor, messages, model.device, tuple(args.image_size))

    # Capture vision encoder pixel values + grid.
    container = DefaultInputsCaptureContainer()
    with InputCaptureCtxManager(model.visual, 1, container):
        model.generate(**inputs, max_new_tokens=20)
    pixel_values = container.captured_args[0][0]
    grid_thw = container.captured_kwargs[0]["grid_thw"]

    # Reprocess pixel values to the NPU layout, then wrap for the parser.
    images = repreprocess_pixel_values(pixel_values, grid_thw[0])
    fd_inputs = {"images": wrap_tensor("images", images.to(model.device))}

    vision_model = VisionModelForQwen2VL(model)
    vision_model.set_grid_thw(grid_thw.to(model.device))
    vision_model.to(model.device)

    mblt_compile(
        model=vision_model,
        mblt_save_path=args.save_path,
        target_device=args.target_device,
        backend="torch",
        feed_dict=fd_inputs,
    )
    print(f"Vision encoder compiled: {args.save_path}")
