"""Qwen2-VL Vision Encoder Compilation to MBLT Format"""

from typing import Dict, List, Tuple

import torch
from qbcompiler.model_dict.common import DataFormat, LayerType
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen2vl import (
    VisionModelForQwen2VL,
    repreprocess_pixel_values,
)
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from qbcompiler.model_dict.parser.parser import ModelParser
from utils import (
    prepare_inputs,
    print_compilation_summary,
    serialize_to_mblt,
    validate_compiled_model,
)


def compile_vision_encoder(
    model,
    processor,
    messages: List[Dict],
    image_size: Tuple[int, int] = (224, 224),
    output_path: str = "mblt/Qwen2-VL-2B-Instruct_vision_transformer.mblt",
    target_device: str = "aries2",
    ignore_weight: bool = False,
    debug: bool = True,
) -> Tuple[str, str, str]:
    """
    Compile Qwen2-VL vision encoder to MBLT format.

    This function:
    1. Captures vision encoder inputs during a sample inference
    2. Reprocesses pixel values to Aries 2-compatible format
    3. Applies architectural patches (3D→2D conv, split QKV, etc.)
    4. Compiles the model using qbcompiler ModelParser
    5. Serializes to MBLT binary format
    6. Validates output by comparing with original model

    Args:
        model: Qwen2VLForConditionalGeneration model
        processor: AutoProcessor for Qwen2-VL
        messages: Sample messages for input capture
        image_size: Target image size for preprocessing
        output_path: Path to save the compiled MBLT file
        target_device: Target device architecture (default: "aries2")
        ignore_weight: If True, don't save weights (graph structure only)
        debug: Enable debug logging during compilation

    Returns:
        Tuple of (mblt_path, inference_values_path, comparison_path)
    """

    print("=" * 80)
    print("VISION ENCODER COMPILATION")
    print("=" * 80)

    inputs = prepare_inputs(processor, messages, model.device, image_size)

    # STEP 1: Capture vision encoder inputs
    print("\n[1/6] Capturing vision encoder inputs...")

    inputs_container = DefaultInputsCaptureContainer()
    max_call_limit = 1  # Only capture the first call

    with InputCaptureCtxManager(model.visual, max_call_limit, inputs_container):
        _ = model.generate(**inputs, max_new_tokens=20)

    pixel_values = inputs_container.captured_args[0][0]
    grid_thw = inputs_container.captured_kwargs[0]["grid_thw"]

    print(f"   Captured pixel_values: {pixel_values.shape}")
    print(f"   Captured grid_thw: {grid_thw}")

    # STEP 2: Reprocess pixel values for Aries 2
    print("\n[2/6] Reprocessing pixel values to Aries 2 format...")

    images = repreprocess_pixel_values(pixel_values, grid_thw[0])

    print(f"   Reprocessed shape: {images.shape}")

    fd_inputs = {"images": wrap_tensor("images", images.to(model.device))}
    grid_thw = grid_thw.to(model.device)

    # STEP 3: Create patched vision model
    print("\n[3/6] Applying Aries 2 architectural patches...")

    vision_model = VisionModelForQwen2VL(model)
    vision_model.set_grid_thw(grid_thw)
    vision_model.to(model.device)

    # STEP 4: Compile with qbcompiler parser
    print(
        f"\n[4/6] Compiling to MBLT with qbcompiler parser (target: {target_device})..."
    )

    parser = ModelParser(
        model=vision_model,
        backend="torch",
        target_device=target_device,
        # Prevents YOLO pattern detection from false-matching VLM operators
        yolo_decode_include=True,
    )

    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True

    parser.parse(
        feed_dict=fd_inputs,
        save_subgraph_type=1,
        debug=debug,
    )

    md, wd = parser.get_md_wd(body_only=False)

    # STEP 5: Serialize to MBLT format
    print("\n[5/6] Serializing to MBLT binary format...")

    # Set data format for input constants to NHWC (Aries 2 layout)
    for sg in md.subgraphs:
        for op in sg.operators:
            if op.layertype == LayerType.InputConstant:
                sg.activations[op.options.outputs[0]].dataformat = DataFormat.NHWC

    serialize_to_mblt(
        md,
        wd,
        output_path,
        ignore_weight=ignore_weight,
        sort_operators=True,
    )

    # STEP 6: Validation
    print("\n[6/6] Validating compiled model...")

    inference_values_path, comparison_path = validate_compiled_model(
        parser, model.device, output_path
    )

    print_compilation_summary(
        "VISION ENCODER",
        output_path,
        inference_values_path,
        comparison_path,
    )

    return output_path, inference_values_path, comparison_path


if __name__ == "__main__":
    from utils import create_sample_messages, load_model_and_processor

    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    output_path = "mblt/Qwen2-VL-2B-Instruct_vision_transformer.mblt"

    model, processor = load_model_and_processor(model_name)

    messages = create_sample_messages(
        image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true",
        text_prompt="Describe this image in detail with document format.",
    )

    compile_vision_encoder(
        model=model,
        processor=processor,
        messages=messages,
        image_size=(224, 224),
        output_path=output_path,
        target_device="aries2",
        ignore_weight=False,
        debug=False,
    )
